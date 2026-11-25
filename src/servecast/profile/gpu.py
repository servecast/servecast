"""The GPU as `profile` sees it, behind one interface (`Backend`), so that another vendor's GPU is a class beside
NVIDIA's and the commands do not change: which registry entry the device is, whether it can be profiled, its SM clock
while a load point runs, the clock lock, and the attention kernel the engine is pinned to.

This release implements NVIDIA's, through NVML (the nvidia-ml-py package, installed with the `profile` extra),
imported only when a command needs it. Profiling needs compute capability 8.0 or newer: FlashAttention 2, the kernel
every calibration and measurement uses, needs it. A computer without such a GPU (a Mac, another vendor's GPU, an older
NVIDIA GPU) is refused with the reason; every other command runs there. Nothing here records a UUID, a serial number,
a bus id or a host name.

The device is the one vLLM runs on: the first of CUDA_VISIBLE_DEVICES (an index or a UUID), else the first GPU, with
CUDA numbering the devices as NVML does (by PCI bus), so the clock is read from the GPU that serves. The clock is
sampled once a second in a background thread. A load point's clock is the mean of the samples inside its steady-state
window, the same window its metrics come from, as in the paper's measurements; the spread of those samples is
recorded with it.

The clock lock is the paper's. Locking needs the right to set clocks: NVML is tried first, then `nvidia-smi`, then
`sudo -n nvidia-smi` (which never asks for a password). A level counts as locked only if the clock holds it while a
dense GEMM loads the GPU (`burn.py`, at least the power serving draws): the settled mean within 1.5 % of the level,
never above it, steady, and not held down by the power cap or slowed by the driver. `sweep` tries levels from the
top down and keeps the first that holds. Where the clock cannot be locked, or no level holds, the run measures the
clock instead and says why.
"""

from __future__ import annotations

import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from servecast.core import gpus

SAMPLE_EVERY_S = 1.0
#: how CUDA must number the devices for an index in CUDA_VISIBLE_DEVICES to name the GPU NVML reads
DEVICE_ORDER = "PCI_BUS_ID"
#: a level holds when, under load, the settled clock averages within this fraction of it, never exceeds it by more,
#: and stays within twice it (the paper's lock check)
LOCK_TOL = 0.015
#: the check of one level: samples of the clock under load, after the load's ramp (10 s; the paper's took 4)
LOCK_SAMPLES = 50
LOCK_PERIOD_S = 0.2
LOCK_RAMP_S = 1.0
#: how long the load may take to start before the check gives up
LOCK_LOAD_START_S = 120.0
#: the sweep tries the driver's levels from the top down to this fraction of the top
SWEEP_FLOOR = 0.5
#: a level the power cap holds down in more than this fraction of the samples is not held (the paper's run gate)
POWER_CAP_MAX = 0.05
#: NVML's clock event reasons: the power cap, and the slowdowns that fail a level on any sample
SW_POWER_CAP = 0x4
SLOWDOWNS = {0x8: "hw_slowdown", 0x20: "sw_thermal_slowdown", 0x40: "hw_thermal_slowdown",
             0x80: "hw_power_brake_slowdown"}


class GpuError(RuntimeError):
    """The GPU cannot serve this manifest: the message names why, and the manifest or the registry can fix it."""


class CannotProfileError(RuntimeError):
    """This computer cannot profile: the message names why."""


class Backend(Protocol):
    """What the profile commands need from a GPU vendor's library. `Nvidia` implements it; `detect` picks it."""

    vendor: str
    #: the engine's attention settings: the kernel every calibration and measurement was made with
    attention: Mapping[str, Any]

    def describe(self) -> dict[str, Any]: ...

    def unsupported(self, described: Mapping[str, Any]) -> str | None: ...

    def busy(self) -> list[int]: ...

    def sm_clock_mhz(self) -> float: ...

    def sample(self) -> dict[str, Any]: ...

    def supported_sm_clocks(self) -> list[int]: ...

    def lock(self, mhz: int) -> str | None: ...

    def unlock(self) -> str | None: ...

    def attention_mismatch(self, engine_config: Any) -> str | None: ...


#: the oldest NVIDIA driver of each CUDA major version, to name in the refusal
MIN_DRIVER = {12: 525, 13: 580}


def torch_cuda() -> str | None:
    """The CUDA version the installed PyTorch was built for, such as '13.0'; None without PyTorch or for a CPU build.
    Importing PyTorch creates no CUDA context, so the engine that starts later keeps all the GPU's memory."""
    try:
        import torch
    except (ImportError, OSError):
        return None
    return None if torch.version.cuda is None else str(torch.version.cuda)


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


class Nvidia:
    """An NVIDIA GPU through NVML."""

    vendor = "NVIDIA"
    #: FlashAttention 2 needs compute capability 8.0 or newer
    MIN_CAPABILITY = (8, 0)
    FLASH_ATTN_VERSION = 2
    attention: Mapping[str, Any] = {"backend": "FLASH_ATTN", "flash_attn_version": FLASH_ATTN_VERSION}

    def __init__(self, nv: ModuleType, handle: Any) -> None:
        self.nv, self.handle = nv, handle
        #: how the clock was locked, so that the same way releases it: "NVML", "nvidia-smi", or None
        self.locked_by: str | None = None

    def describe(self) -> dict[str, Any]:
        """The device's name and memory, its registry id (None for a GPU the registry does not know), the driver, the
        top SM clock and the compute capability."""
        nv, h = self.nv, self.handle
        name = _text(nv.nvmlDeviceGetName(h))
        match = next((g for g in gpus.registry().values() if g.device_name == name), None)
        major, minor = nv.nvmlDeviceGetCudaComputeCapability(h)
        return {"vendor": self.vendor, "name": name, "id": match.id if match else None,
                "memory_bytes": int(nv.nvmlDeviceGetMemoryInfo(h).total),
                "driver": _text(nv.nvmlSystemGetDriverVersion()),
                "max_sm_clock_mhz": int(nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_SM)),
                "compute_capability": f"{int(major)}.{int(minor)}"}

    def unsupported(self, described: Mapping[str, Any]) -> str | None:
        """Why this device cannot be profiled, or None."""
        major, minor = (int(x) for x in str(described["compute_capability"]).split("."))
        if (major, minor) < self.MIN_CAPABILITY:
            return (f"{described['name']} has compute capability {major}.{minor}; FlashAttention 2, the attention "
                    "kernel every calibration and measurement uses, needs 8.0 or newer")
        supported, built_for = self.cuda_driver(), torch_cuda()
        if supported is not None and built_for is not None:
            needed = int(built_for.split(".")[0])
            if supported[0] < needed:  # within one major version, CUDA runs on any driver of that version
                newer = f" ({MIN_DRIVER[needed]} or newer)" if needed in MIN_DRIVER else ""
                return (f"the NVIDIA driver {described['driver']} supports CUDA up to {supported[0]}.{supported[1]}, "
                        f"and the installed PyTorch is built for CUDA {built_for}, so vLLM could not start; install "
                        f"a driver that supports CUDA {needed}{newer}")
        return None

    def cuda_driver(self) -> tuple[int, int] | None:
        """The newest CUDA version the driver supports, as (major, minor); None when NVML does not say."""
        try:
            version = int(self.nv.nvmlSystemGetCudaDriverVersion())
        except (AttributeError, TypeError, ValueError, self.nv.NVMLError):
            return None
        return version // 1000, version % 1000 // 10

    def busy(self) -> list[int]:
        """The process ids holding the GPU (another process would share its memory and its clock); none when NVML
        cannot list them."""
        try:
            return [int(p.pid) for p in self.nv.nvmlDeviceGetComputeRunningProcesses(self.handle)]
        except self.nv.NVMLError:
            return []

    def sm_clock_mhz(self) -> float:
        return float(self.nv.nvmlDeviceGetClockInfo(self.handle, self.nv.NVML_CLOCK_SM))

    def sample(self) -> dict[str, Any]:
        """The SM clock now, whether the power cap holds it down, and the driver's slowdowns."""
        nv = self.nv
        reasons = (getattr(nv, "nvmlDeviceGetCurrentClocksEventReasons", None)
                   or getattr(nv, "nvmlDeviceGetCurrentClocksThrottleReasons", None))
        try:
            bits = int(reasons(self.handle)) if reasons is not None else 0
        except nv.NVMLError:
            bits = 0
        return {"sm_mhz": self.sm_clock_mhz(), "power_capped": bool(bits & SW_POWER_CAP),
                "slowdowns": [name for bit, name in SLOWDOWNS.items() if bits & bit]}

    def supported_sm_clocks(self) -> list[int]:
        """The SM clock levels the driver accepts, at the top memory clock, from the top down; none if it will not
        say."""
        nv, h = self.nv, self.handle
        try:
            memory = max(int(m) for m in nv.nvmlDeviceGetSupportedMemoryClocks(h))
            return sorted({int(c) for c in nv.nvmlDeviceGetSupportedGraphicsClocks(h, memory)}, reverse=True)
        except (nv.NVMLError, ValueError):
            return []

    def _uuid(self) -> str:
        return _text(self.nv.nvmlDeviceGetUUID(self.handle))

    def lock(self, mhz: int) -> str | None:
        """Lock the SM clock at `mhz`: through NVML, else nvidia-smi (plain, then `sudo -n`). None when it took,
        else why not. Persistence mode keeps the lock between the processes of a run."""
        nv = self.nv
        try:
            try:
                nv.nvmlDeviceSetPersistenceMode(self.handle, 1)
            except nv.NVMLError:
                pass  # nvidia-persistenced may hold it already; the lock below is what decides
            nv.nvmlDeviceSetGpuLockedClocks(self.handle, int(mhz), int(mhz))
        except nv.NVMLError as error:
            uuid = self._uuid()
            nvidia_smi(["-i", uuid, "-pm", "1"])
            refused = nvidia_smi(["-i", uuid, "-lgc", f"{int(mhz)},{int(mhz)}"])
            if refused is not None:
                return (f"the clock cannot be locked here: NVML refused ({error}) and so did nvidia-smi ({refused}); "
                        "locking needs root, or sudo without a password")
            self.locked_by = "nvidia-smi"
            return None
        self.locked_by = "NVML"
        return None

    def unlock(self) -> str | None:
        """Release the lock the way it was set; None when it was released, else why not (the run reports it)."""
        if self.locked_by == "nvidia-smi":
            refused = nvidia_smi(["-i", self._uuid(), "-rgc"])
            self.locked_by = None
            return None if refused is None else (f"nvidia-smi did not release the clock lock ({refused}); reset it "
                                                 "with sudo nvidia-smi -rgc")
        try:
            self.nv.nvmlDeviceResetGpuLockedClocks(self.handle)
        except self.nv.NVMLError as error:
            return f"NVML did not release the clock lock ({error}); reset it with nvidia-smi -rgc"
        self.locked_by = None
        return None

    def attention_mismatch(self, engine_config: Any) -> str | None:
        """Read back from the engine that was built: vLLM picks the FlashAttention version by architecture (3 on
        H100-class GPUs), and a kernel nobody checked is not a pinned one."""
        got = getattr(getattr(engine_config, "attention_config", None), "flash_attn_version", None)
        if got is None or int(got) != self.FLASH_ATTN_VERSION:
            return (f"the engine runs FlashAttention {got}, not {self.FLASH_ATTN_VERSION}: the pin did not take, and a "
                    "measurement on another kernel would not match any calibration")
        return None


def _import_nvml() -> ModuleType:  # pragma: no cover - needs the profile extra
    import pynvml

    return pynvml


def _amd_present() -> bool:
    return Path("/dev/kfd").exists() or any(shutil.which(tool) for tool in ("rocm-smi", "amd-smi"))


def detect(env: Mapping[str, str] | None = None, *, system: str | None = None,
           import_nvml: Callable[[], ModuleType] = _import_nvml, amd: Callable[[], bool] = _amd_present) -> Backend:
    """The GPU vLLM will run on, as a `Backend`; `CannotProfileError` names why this computer cannot profile."""
    env = os.environ if env is None else env
    system = platform.system() if system is None else system
    if system == "Darwin":
        raise CannotProfileError("macOS has no CUDA, and the measurement runs vLLM's CUDA engine on an NVIDIA GPU "
                                 "(Linux)")
    if system != "Linux":
        raise CannotProfileError(f"{system}: the measurement runs vLLM's CUDA engine on Linux")
    try:
        nv = import_nvml()
    except ImportError:
        if amd():
            raise CannotProfileError("an AMD GPU (ROCm): this release profiles NVIDIA GPUs only") from None
        raise CannotProfileError("NVML is not installed; on the GPU host, install the profile extra: pip install "
                                 "'servecast[profile]'") from None
    try:
        nv.nvmlInit()
    except nv.NVMLError as error:
        if amd():
            raise CannotProfileError("an AMD GPU (ROCm): this release profiles NVIDIA GPUs only") from None
        raise CannotProfileError(f"no NVIDIA driver answers (NVML: {error})") from None
    count = int(nv.nvmlDeviceGetCount())
    if count == 0:
        raise CannotProfileError("NVML finds no NVIDIA GPU")
    visible = env.get("CUDA_VISIBLE_DEVICES")
    first = None if visible is None else visible.split(",")[0].strip()
    if first == "":
        raise CannotProfileError("CUDA_VISIBLE_DEVICES is empty, so no GPU is visible to vLLM")
    if first is not None and first.startswith("MIG-"):
        raise CannotProfileError("CUDA_VISIBLE_DEVICES names a MIG instance; the cost laws model a whole GPU")
    order = env.get("CUDA_DEVICE_ORDER", DEVICE_ORDER)
    if count > 1 and order != DEVICE_ORDER and (first is None or first.isdigit()):
        raise CannotProfileError(f"CUDA_DEVICE_ORDER={order} on a computer with {count} GPUs: NVML numbers them by "
                                 f"PCI bus, so the clock would be read from another GPU than the one vLLM runs on; "
                                 f"unset it or set {DEVICE_ORDER}")
    try:
        if first is None or first.isdigit():
            index = int(first or 0)
            if index >= count:
                raise CannotProfileError(f"CUDA_VISIBLE_DEVICES={visible}: NVML finds {count} GPU(s)")
            handle = nv.nvmlDeviceGetHandleByIndex(index)
        else:
            handle = nv.nvmlDeviceGetHandleByUUID(first)
    except nv.NVMLError as error:
        raise CannotProfileError(f"CUDA_VISIBLE_DEVICES={visible}: NVML cannot open that GPU ({error})") from None
    return Nvidia(nv, handle)


def cuda_environment(env: Mapping[str, str]) -> dict[str, str]:
    """The variables to set before vLLM starts, so CUDA numbers the GPUs as `detect` did."""
    return {} if "CUDA_DEVICE_ORDER" in env else {"CUDA_DEVICE_ORDER": DEVICE_ORDER}


def check_hardware(described: Mapping[str, Any], hardware: str | None) -> str:
    """The registry id of the GPU in use; refused when the registry does not know it or the manifest names another."""
    if described["id"] is None:
        known = ", ".join(f"{g.id} ({g.device_name})" for g in gpus.registry().values())
        raise GpuError(f"{described['name']} is not in ServeCast's GPU registry (known: {known}); add it to "
                       f"{gpus.user_registry()}")
    if hardware is not None and hardware != described["id"]:
        raise GpuError(f"engine.hardware: {hardware}, but this machine's GPU is {described['id']} "
                       f"({described['name']})")
    return str(described["id"])


class ClockSampler:
    """The SM clock, sampled in a background thread, as (time, MHz) pairs on the caller's clock."""

    def __init__(self, read: Callable[[], float], every_s: float = SAMPLE_EVERY_S,
                 now: Callable[[], float] = time.perf_counter) -> None:
        self.read, self.every_s, self.now = read, every_s, now
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append((self.now(), float(self.read())))
            self._stop.wait(self.every_s)

    def __enter__(self) -> ClockSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def window(self, t0: float, t1: float) -> dict[str, Any]:
        """The samples in (t0, t1]: their mean, spread and count. No sample, no clock."""
        inside = [mhz for t, mhz in self.samples if t0 < t <= t1]
        if not inside:
            return {"mean_mhz": None, "min_mhz": None, "max_mhz": None, "n": 0}
        return {"mean_mhz": statistics.fmean(inside), "min_mhz": min(inside), "max_mhz": max(inside), "n": len(inside)}


def sampler(backend: Backend) -> Callable[[Callable[[], float]], ClockSampler]:
    """What `serving.serve` takes: given the run's clock, a sampler of this GPU's SM clock on it."""
    return lambda now: ClockSampler(backend.sm_clock_mhz, now=now)


def nvidia_smi(args: Sequence[str]) -> str | None:
    """Run nvidia-smi with `args`, then again through `sudo -n` (never a password prompt) if it was refused: None when
    it succeeded, else why not."""
    why = "nvidia-smi is not installed"
    for command in (["nvidia-smi", *args], ["sudo", "-n", "nvidia-smi", *args]):
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            why = f"{command[0]}: {error}"
            continue
        if done.returncode == 0:
            return None
        lines = (done.stderr or done.stdout or "").strip().splitlines()
        why = lines[-1] if lines else f"{command[0]}: exit status {done.returncode}"
    return why


def lock_verdict(mhz: int, samples: Sequence[Mapping[str, Any]], loaded: bool) -> dict[str, Any]:
    """Whether the clock held `mhz` over `samples` taken under load (the paper's lock check): the first quarter is the
    load's ramp; the rest must average within LOCK_TOL of the level, never exceed it by more and stay within twice
    it; the power cap may hold the clock down in at most POWER_CAP_MAX of the samples, and no sample may show a
    driver slowdown."""
    sm = [float(s["sm_mhz"]) for s in samples]
    settled = sm[max(1, len(sm) // 4):] if len(sm) >= 8 else sm
    tol = LOCK_TOL * mhz
    slowdowns = sorted({name for s in samples for name in s.get("slowdowns", ())})
    report: dict[str, Any] = {
        "mhz": int(mhz), "samples": len(sm), "mean_mhz": statistics.fmean(settled) if settled else None,
        "min_mhz": min(settled) if settled else None, "max_mhz": max(settled) if settled else None,
        "power_capped_fraction": (sum(bool(s.get("power_capped")) for s in samples) / len(samples)
                                  if samples else None),
        "slowdowns": slowdowns}
    why: list[str] = []
    if not loaded:
        why.append("the load did not run through the check, so the level was not tested under load")
    elif not settled:
        why.append("no clock sample")
    else:
        mean, low, high = report["mean_mhz"], report["min_mhz"], report["max_mhz"]
        if abs(mean - mhz) > tol:
            why.append(f"the clock averaged {mean:.0f} MHz under load, more than {100 * LOCK_TOL:g} % off {mhz}")
        if high > mhz + tol:
            why.append(f"the clock went above the lock, to {high:.0f} MHz")
        if high - low > 2 * tol:
            why.append(f"the clock was not steady under load ({low:.0f} to {high:.0f} MHz)")
        if report["power_capped_fraction"] > POWER_CAP_MAX:
            why.append(f"the power cap held the clock down in {100 * report['power_capped_fraction']:.0f} % of the "
                       "samples")
        if slowdowns:
            why.append(f"the driver slowed the GPU down ({', '.join(slowdowns)})")
    report["held"] = not why
    report["why"] = why
    return report


def under_load(backend: Backend) -> tuple[list[dict[str, Any]], bool]:  # pragma: no cover - needs a GPU and torch
    """The GPU's clock samples while a dense GEMM loads it, and whether the load ran through them. The load runs in a
    process of its own (`python -m servecast.profile.burn`), so that this process never holds a CUDA context, which
    would take memory from the engine that starts after it."""
    env = {**os.environ, **cuda_environment(os.environ)}
    child = subprocess.Popen([sys.executable, "-m", "servecast.profile.burn"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True, env=env)
    try:
        started: list[str] = []
        reader = threading.Thread(target=lambda: started.append(child.stdout.readline() if child.stdout else ""),
                                  daemon=True)
        reader.start()
        reader.join(LOCK_LOAD_START_S)
        if not started or not started[0].startswith("ready"):
            return [], False
        time.sleep(LOCK_RAMP_S)
        samples = []
        for _ in range(LOCK_SAMPLES):
            samples.append(backend.sample())
            time.sleep(LOCK_PERIOD_S)
        return samples, child.poll() is None
    finally:
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()


def sweep(backend: Backend, levels: Sequence[int],
          measure: Callable[[Backend], tuple[list[dict[str, Any]], bool]] = under_load) -> dict[str, Any]:
    """The first of `levels` (from the top down) that the clock holds under load, left locked; the verdict of each
    level tried; and, if the clock cannot be locked at all, why. With no level held, the clock is released."""
    tried: list[dict[str, Any]] = []
    for mhz in levels:
        refused = backend.lock(int(mhz))
        if refused is not None:
            return {"lock_mhz": None, "levels": tried, "refused": refused}
        samples, loaded = measure(backend)
        tried.append(lock_verdict(int(mhz), samples, loaded))
        if tried[-1]["held"]:
            return {"lock_mhz": int(mhz), "levels": tried, "refused": None}
    released = backend.unlock() if tried else None
    return {"lock_mhz": None, "levels": tried, "refused": None, **({"release": released} if released else {})}
