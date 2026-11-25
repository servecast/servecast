"""The GPU through a fake NVML: which computers can profile and why the others cannot, the device vLLM runs on,
registry identity, the hardware check, the clock window, a refused lock, and the attention pin read back."""

from __future__ import annotations

import time
import types
from collections.abc import Sequence
from typing import Any

import pytest

from servecast.core.deployment import Deployment
from servecast.profile import gpu as G
from servecast.profile import vllm_engine

A100 = ("NVIDIA A100-SXM4-40GB", (8, 0))
H100 = ("NVIDIA H100 80GB HBM3", (9, 0))
T4 = ("Tesla T4", (7, 5))


def fake_nvml(devices: Sequence[tuple[str, tuple[int, int]]] = (A100,), clocks: Sequence[float] = (1200.0,),
              lockable: bool = True, driver: bool = True, processes: Sequence[int] = (),
              hold_mhz: float | None = None, driver_version: str = "580.82",
              cuda: int = 13000) -> types.SimpleNamespace:
    """NVML as the profile commands call it; `calls` records the clock locks and releases. With `hold_mhz`, the GPU
    holds a lock up to that level under load; locked above it, the clock reads hold_mhz and the power cap is on.
    `cuda` is the newest CUDA version the driver supports, as NVML gives it (13000 for 13.0)."""
    class NVMLError(Exception):
        pass

    it = iter(clocks)
    state: dict[str, Any] = {"last": clocks[0], "locked": None}
    calls: list[tuple[str, int]] = []

    def capped() -> bool:
        return hold_mhz is not None and state["locked"] is not None and state["locked"] > hold_mhz

    def init() -> None:
        if not driver:
            raise NVMLError("Driver Not Loaded")

    def by_index(i: int) -> int:
        if not 0 <= i < len(devices):
            raise NVMLError("Invalid Argument")
        return i

    def by_uuid(uuid: str) -> int:
        if not uuid.startswith("GPU-") or not uuid[4:].isdigit():
            raise NVMLError("Not Found")
        return by_index(int(uuid[4:]))

    def clock(_h: int, _kind: int) -> float:
        if hold_mhz is not None and state["locked"] is not None:
            return float(min(state["locked"], hold_mhz))
        state["last"] = next(it, state["last"])
        return state["last"]

    def set_locked(_h: int, lo: int, hi: int) -> None:
        if not lockable:
            raise NVMLError("Insufficient Permissions")
        calls.append(("lock", lo))
        state["locked"] = lo

    def reset(_h: int) -> None:
        calls.append(("unlock", 0))
        state["locked"] = None

    def persistence(_h: int, _on: int) -> None:
        if not lockable:
            raise NVMLError("Insufficient Permissions")

    return types.SimpleNamespace(
        NVMLError=NVMLError, NVML_CLOCK_SM=0, nvmlInit=init, nvmlDeviceGetCount=lambda: len(devices),
        nvmlDeviceGetHandleByIndex=by_index, nvmlDeviceGetHandleByUUID=by_uuid,
        nvmlDeviceGetName=lambda h: devices[h][0].encode(),
        nvmlDeviceGetCudaComputeCapability=lambda h: devices[h][1],
        nvmlDeviceGetMemoryInfo=lambda h: types.SimpleNamespace(total=42_949_672_960),
        nvmlSystemGetDriverVersion=lambda: driver_version.encode(), nvmlSystemGetCudaDriverVersion=lambda: cuda,
        nvmlDeviceGetMaxClockInfo=lambda h, k: 1410,
        nvmlDeviceGetClockInfo=clock,
        nvmlDeviceGetComputeRunningProcesses=lambda h: [types.SimpleNamespace(pid=p) for p in processes],
        nvmlDeviceSetGpuLockedClocks=set_locked, nvmlDeviceResetGpuLockedClocks=reset,
        nvmlDeviceSetPersistenceMode=persistence, nvmlDeviceGetUUID=lambda h: f"GPU-{h}".encode(),
        nvmlDeviceGetSupportedMemoryClocks=lambda h: [1215, 405],
        nvmlDeviceGetSupportedGraphicsClocks=lambda h, mem: list(range(210, 1411, 15)),
        nvmlDeviceGetCurrentClocksThrottleReasons=lambda h: G.SW_POWER_CAP if capped() else 0,
        calls=calls, state=state)


def loaded(backend: G.Backend) -> tuple[list[dict[str, Any]], bool]:
    """The check of one level on the fake: its samples, with the load running throughout."""
    return [backend.sample() for _ in range(G.LOCK_SAMPLES)], True


def smi_refused(calls: list[list[str]] | None = None) -> Any:
    """nvidia-smi as a plain user without passwordless sudo sees it: every call refused, and recorded."""
    def run(args: Sequence[str]) -> str:
        if calls is not None:
            calls.append(list(args))
        return "sudo: a password is required"
    return run


def detect(nv: Any = None, env: dict[str, str] | None = None, *, system: str = "Linux", amd: bool = False,
           missing: bool = False) -> G.Backend:
    def import_nvml() -> Any:
        if missing:
            raise ImportError("No module named 'pynvml'")
        return nv if nv is not None else fake_nvml()

    return G.detect(env or {}, system=system, import_nvml=import_nvml, amd=lambda: amd)


@pytest.mark.parametrize(("case", "why"), [
    ({"system": "Darwin"}, "macOS has no CUDA"),
    ({"system": "Windows"}, "Windows: the measurement runs vLLM's CUDA engine on Linux"),
    ({"missing": True}, "pip install 'servecast[profile]'"),
    ({"missing": True, "amd": True}, "an AMD GPU (ROCm): this release profiles NVIDIA GPUs only"),
    ({"nv": fake_nvml(driver=False)}, "no NVIDIA driver answers (NVML: Driver Not Loaded)"),
    ({"nv": fake_nvml(driver=False), "amd": True}, "an AMD GPU (ROCm)"),
    ({"nv": fake_nvml(devices=())}, "NVML finds no NVIDIA GPU"),
    ({"env": {"CUDA_VISIBLE_DEVICES": ""}}, "no GPU is visible to vLLM"),
    ({"env": {"CUDA_VISIBLE_DEVICES": "MIG-4b1c"}}, "a MIG instance"),
    ({"env": {"CUDA_VISIBLE_DEVICES": "3"}}, "NVML finds 1 GPU(s)"),
    ({"env": {"CUDA_VISIBLE_DEVICES": "GPU-x"}}, "NVML cannot open that GPU"),
    ({"nv": fake_nvml(devices=(A100, H100)), "env": {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}},
     "the clock would be read from another GPU"),
])
def test_a_computer_that_cannot_profile_is_told_why(case: dict[str, Any], why: str) -> None:
    with pytest.raises(G.CannotProfileError) as error:
        detect(**case)
    assert why in str(error.value)


def test_the_device_is_the_one_vllm_runs_on() -> None:
    two = (A100, H100)
    assert detect(fake_nvml(devices=two)).describe()["name"] == A100[0]
    assert detect(fake_nvml(devices=two), {"CUDA_VISIBLE_DEVICES": "1,0"}).describe()["name"] == H100[0]
    assert detect(fake_nvml(devices=two), {"CUDA_VISIBLE_DEVICES": "GPU-1"}).describe()["name"] == H100[0]
    # one GPU: the order CUDA numbers devices in cannot pick another
    assert detect(env={"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}).describe()["name"] == A100[0]
    assert G.cuda_environment({}) == {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    assert G.cuda_environment({"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}) == {}


def test_compute_capability_8_0_or_newer_can_profile() -> None:
    for device, refused in ((A100, False), (H100, False), (("NVIDIA GeForce RTX 3090 Ti", (8, 6)), False),
                            (T4, True)):
        backend = detect(fake_nvml(devices=(device,)))
        why = backend.unsupported(backend.describe())
        assert (why is not None) == refused, device
    assert "needs 8.0 or newer" in str(why)


@pytest.mark.parametrize(("cuda", "built_for", "refused"), [
    (13000, "13.0", False), (13020, "13.0", False), (12080, "12.8", False), (13000, None, False),
    (12080, "13.0", True), (12020, "13.0", True),
])
def test_a_driver_too_old_for_the_installed_pytorch_cannot_profile(monkeypatch: pytest.MonkeyPatch, cuda: int,
                                                                   built_for: str | None, refused: bool) -> None:
    """vLLM's PyTorch runs on a driver of its CUDA major version or newer; an older one is named before vLLM starts."""
    monkeypatch.setattr(G, "torch_cuda", lambda: built_for)
    backend = detect(fake_nvml(driver_version="570.86", cuda=cuda))
    why = backend.unsupported(backend.describe())
    assert (why is not None) == refused, why
    if refused:
        assert f"the NVIDIA driver 570.86 supports CUDA up to 12.{cuda % 1000 // 10}" in str(why)
        assert "the installed PyTorch is built for CUDA 13.0" in str(why)
        assert "install a driver that supports CUDA 13 (580 or newer)" in str(why)


def test_the_gpu_is_matched_to_the_registry_and_to_the_manifest() -> None:
    d = detect().describe()
    assert d == {"vendor": "NVIDIA", "name": "NVIDIA A100-SXM4-40GB", "id": "a100_sxm_40gb",
                 "memory_bytes": 42_949_672_960, "driver": "580.82", "max_sm_clock_mhz": 1410,
                 "compute_capability": "8.0"}
    assert G.check_hardware(d, None) == G.check_hardware(d, "a100_sxm_40gb") == "a100_sxm_40gb"
    with pytest.raises(G.GpuError, match=r"engine\.hardware: h100_sxm"):
        G.check_hardware(d, "h100_sxm")
    with pytest.raises(G.GpuError, match=r"not in ServeCast's GPU registry .*; add it to .*gpus\.json"):
        G.check_hardware(detect(fake_nvml(devices=(T4,))).describe(), None)
    assert not {"uuid", "serial", "bus_id"} & set(d)


def test_a_window_clock_uses_only_the_samples_inside_it() -> None:
    ticks = iter([0.5, 1.5, 2.5, 3.5])
    backend = detect(fake_nvml(clocks=(1100.0, 1200.0, 1210.0, 1500.0)))
    sampler = G.sampler(backend)(lambda: next(ticks))
    for _ in range(4):
        sampler.samples.append((sampler.now(), sampler.read()))
    assert sampler.window(1.0, 3.0) == {"mean_mhz": 1205.0, "min_mhz": 1200.0, "max_mhz": 1210.0, "n": 2}
    assert sampler.window(10.0, 11.0)["mean_mhz"] is None


def test_the_sampler_thread_samples_and_stops() -> None:
    with G.ClockSampler(detect().sm_clock_mhz, every_s=0.01) as sampler:
        time.sleep(0.1)
    assert len(sampler.samples) >= 3 and not sampler._thread.is_alive()


def test_a_lock_nvml_refuses_is_set_with_nvidia_smi_and_released_the_same_way(monkeypatch: pytest.MonkeyPatch) -> None:
    smi: list[list[str]] = []
    monkeypatch.setattr(G, "nvidia_smi", lambda args: smi.append(list(args)))
    backend = detect(fake_nvml(lockable=False))
    assert backend.lock(1200) is None
    assert backend.unlock() is None
    assert smi == [["-i", "GPU-0", "-pm", "1"], ["-i", "GPU-0", "-lgc", "1200,1200"], ["-i", "GPU-0", "-rgc"]]


def test_a_clock_that_cannot_be_locked_says_why(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(G, "nvidia_smi", smi_refused())
    assert detect().lock(1200) is None                      # NVML takes it: nvidia-smi is not needed
    why = detect(fake_nvml(lockable=False)).lock(1200)
    assert why is not None and "NVML refused (Insufficient Permissions)" in why
    assert "nvidia-smi (sudo: a password is required)" in why and "locking needs root" in why


def test_nvidia_smi_is_retried_through_sudo_without_a_password(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[list[str]] = []

    def run(command: list[str], **_: Any) -> Any:
        ran.append(command)
        refused = command[0] == "nvidia-smi"
        return types.SimpleNamespace(returncode=4 if refused else 0, stdout="",
                                     stderr="Insufficient Permissions\n" if refused else "")
    monkeypatch.setattr(G.subprocess, "run", run)
    assert G.nvidia_smi(["-i", "GPU-0", "-rgc"]) is None
    assert ran == [["nvidia-smi", "-i", "GPU-0", "-rgc"], ["sudo", "-n", "nvidia-smi", "-i", "GPU-0", "-rgc"]]


def test_the_sweep_locks_the_highest_level_the_clock_holds_under_load() -> None:
    nv = fake_nvml(hold_mhz=1275)
    backend = G.Nvidia(nv, 0)
    levels = backend.supported_sm_clocks()
    assert levels[0] == 1410 and levels == sorted(levels, reverse=True)
    found = G.sweep(backend, levels, loaded)
    assert found["lock_mhz"] == 1275 and nv.state["locked"] == 1275       # left locked for the run
    assert [v["mhz"] for v in found["levels"]] == list(range(1410, 1274, -15))
    assert not any(v["held"] for v in found["levels"][:-1]) and found["levels"][-1]["held"]
    # 1290 reads 1275, within 1.5 %, but the power cap holds it down: not a level the clock holds
    assert found["levels"][-2]["why"] == ["the power cap held the clock down in 100 % of the samples"]


def test_with_no_level_held_the_clock_is_released() -> None:
    nv = fake_nvml(hold_mhz=1000)
    found = G.sweep(G.Nvidia(nv, 0), [1410, 1395], loaded)
    assert found["lock_mhz"] is None and nv.state["locked"] is None and nv.calls[-1] == ("unlock", 0)


def test_a_level_is_judged_only_under_load_and_after_the_ramp() -> None:
    def at(*mhz: float, capped: bool = False, slow: tuple[str, ...] = ()) -> list[dict[str, Any]]:
        return [{"sm_mhz": f, "power_capped": capped, "slowdowns": list(slow)} for f in mhz]

    ramp = at(*([900.0] * 4), *([1200.0] * 12))
    assert G.lock_verdict(1200, ramp, loaded=True)["held"]                    # the first quarter is the ramp
    assert G.lock_verdict(1200, ramp, loaded=False)["why"] == [
        "the load did not run through the check, so the level was not tested under load"]
    assert "above the lock" in G.lock_verdict(1200, at(*([1200.0] * 15), 1240.0), loaded=True)["why"][0]
    # 1.5 % of 1200 is 18 MHz: a 40 MHz swing around a mean on target is not steady
    assert G.lock_verdict(1200, at(*([1170.0, 1210.0] * 8)), loaded=True)["why"] == [
        "the clock was not steady under load (1170 to 1210 MHz)"]
    assert "sw_thermal_slowdown" in G.lock_verdict(1200, at(*([1200.0] * 16), slow=("sw_thermal_slowdown",)),
                                                   loaded=True)["why"][0]


def test_the_engine_is_pinned_to_the_backends_attention_and_it_is_read_back() -> None:
    backend = detect()
    d = Deployment(block_size=16, max_batched_tokens=2048, max_sequences=256, kv_blocks=9965, context_length=8192,
                   memory_fraction=0.9)
    args = vllm_engine.engine_args("Qwen/Qwen2.5-7B", d, backend.attention)
    assert args["attention_config"] == {"backend": "FLASH_ATTN", "flash_attn_version": 2}
    assert (args["num_gpu_blocks_override"], args["max_num_batched_tokens"], args["max_num_seqs"],
            args["max_model_len"], args["enable_prefix_caching"]) == (9965, 2048, 256, 8192, False)

    def built(version: Any) -> Any:
        return types.SimpleNamespace(attention_config=types.SimpleNamespace(flash_attn_version=version))

    assert backend.attention_mismatch(built(2)) is None
    assert "FlashAttention 3, not 2" in str(backend.attention_mismatch(built(3)))
    assert "FlashAttention None" in str(backend.attention_mismatch(types.SimpleNamespace()))


def test_vllms_own_block_count_is_read_from_its_log() -> None:
    """The line as vLLM 0.25.1 printed it in the paper's serving runs; the last one counts (an engine started again)."""
    line = ("(EngineCore pid=4745) INFO 10-11 14:56:12 [kv_cache_utils.py:2084] Overriding num_gpu_blocks={} with "
            "num_gpu_blocks_override=9965\n")
    assert vllm_engine.computed_blocks(line.format(11520) + "...\n" + line.format(11167)) == 11167
    assert vllm_engine.computed_blocks("Overriding num_gpu_blocks=11167 with 9965") is None
    assert vllm_engine.computed_blocks("no such line") is None
