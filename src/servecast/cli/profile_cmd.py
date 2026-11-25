"""`servecast profile serving` and `servecast profile calibration`: measurements on this GPU with vLLM.

`profile serving` serves the manifest's workload and writes measurement.json for `servecast compare`
(`profile.serving`). `profile calibration` runs the calibration probe of the manifest's cost law and writes the
profiling data `servecast calibrate` fits (`profile.probe`).

Both check what can be checked without the GPU first (the manifest, the output directory); then the computer, through
the interface `doctor` uses (`profile.gpu`): an NVIDIA GPU of compute capability 8.0 or newer, in the registry, the one
the manifest names, and free; then vLLM's version. Only then is the clock locked (if the manifest asks) and the
engine started.

Exit statuses: 0 measured; 1 the measurement failed, or a probe's audit has flags (no profiling data); 2 the manifest or
the options cannot be run (every problem named, with its fix); 3 this computer cannot profile, or a resource refused
(the GPU in use, an output directory in use); 124 a load point reached --timeout before converging; 130 interrupted.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, NoReturn

import typer

from servecast import __version__

COMMAND = "profile serving"
CALIBRATION = "profile calibration"
#: what a run writes: the measurement's files, and vLLM's own output (engine.log), kept to diagnose the engine
KNOWN_OUTPUTS = ("measurement.json", "pool.jsonl", "requests.jsonl", "environment.json", "SHA256SUMS", "engine.log")
#: what a probe writes: the profiling data (only from a clean probe), every beat with the audit, the environment,
#: their checksums, and vLLM's own output; a per-GPU law's probe writes a probe record per synthetic model and the
#: GEMM grid instead of probe.json
PROBE_OUTPUTS = ("profiling.json", "probe.json", "environment.json", "SHA256SUMS", "engine.log")
GPU_PROBE_OUTPUTS = ("profiling.json", "gemm.json", "environment.json", "SHA256SUMS", "engine.log")
#: how --poweroff-when-safe powers the machine off: without a password prompt, or not at all
POWEROFF = ("sudo", "-n", "shutdown", "-h", "now")
#: what the engine raises when it cannot be built or stops serving
ENGINE_FAILURES = (RuntimeError, ValueError, OSError)
#: the event loop the measurement runs on: the default on the GPU host (the tests give one with a simulated clock)
loop_factory: Callable[[], asyncio.AbstractEventLoop] | None = None


def fail(status: int, message: str, command: str = COMMAND) -> NoReturn:
    typer.echo(f"servecast {command}: {message}", err=True)
    raise typer.Exit(status)


def say(message: str, command: str = COMMAND) -> None:
    typer.echo(f"servecast {command}: {message}", err=True)


def read_config(model: str) -> dict[str, Any]:
    """The LLM's config.json: from its local directory, or from the hub (with HF_TOKEN for a gated model)."""
    from servecast.core import hub

    path = Path(model)
    if path.exists():
        if not (path / "config.json").is_file():
            raise ValueError(f"{model}: give the LLM's directory, holding config.json and the weights, or a Hugging "
                             "Face id")
        return dict(json.loads((path / "config.json").read_text(encoding="utf-8")))
    return hub.fetch_config(model)


def vllm_version() -> str | None:
    from servecast.cli.doctor_cmd import _version

    return _version("vllm")


def software() -> dict[str, Any]:
    """The versions the measurement runs on, CUDA's as torch reports it."""
    from servecast.cli.doctor_cmd import _version

    found: dict[str, Any] = {"servecast": __version__, "python": platform.python_version(),
                             "platform": f"{platform.system()} {platform.machine()}"}
    found |= {dist: _version(dist) for dist in ("torch", "transformers", "nvidia-ml-py")}
    try:
        import torch

        found["cuda"] = torch.version.cuda
    except ImportError:
        found["cuda"] = None
    return found


def engine(model: str, deployment: Any, backend: Any) -> Any:  # pragma: no cover - needs a GPU and vLLM
    from servecast.profile.vllm_engine import VllmEngine

    return VllmEngine(model, deployment, backend)


def prompt(model: str, n_tokens: int) -> list[int]:  # pragma: no cover - needs the LLM's tokenizer
    from servecast.profile.vllm_engine import prompt_ids

    return prompt_ids(model, n_tokens)


def probe_engine(model: str, deployment: Any, backend: Any, *, kv_tokens: int,
                 sample_clock: Callable[[], float]) -> Any:  # pragma: no cover - needs a GPU and vLLM
    from servecast.profile.probe_vllm import VllmProbeEngine

    return VllmProbeEngine(model, deployment, backend, kv_tokens=kv_tokens, sample_clock=sample_clock)


def synthetic_model(name: str) -> Path:  # pragma: no cover - downloads the base model's config and tokenizer
    """A synthetic model's directory, in ServeCast's cache (no weights: it is served with random ones)."""
    from servecast.core.paths import cache_dir
    from servecast.profile.probe_vllm import materialize_synthetic

    return materialize_synthetic(name, cache_dir() / "synthetic")


def probe_synthetic(job: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - needs a GPU and vLLM
    """One synthetic model's probe in a fresh process (`profile.probe_child`), so its engine's memory is freed
    before the next model's engine starts."""
    import sys
    import tempfile

    from servecast.profile.probe import ProbeError

    with tempfile.TemporaryDirectory() as tmp:
        job_file, record_file = Path(tmp) / "job.json", Path(tmp) / "record.json"
        job_file.write_text(json.dumps(job), encoding="utf-8")
        done = subprocess.run([sys.executable, "-m", "servecast.profile.probe_child", str(job_file),
                               str(record_file)], check=False)
        if not record_file.is_file():
            raise ProbeError(f"the probe process of {job['model']} ended with status {done.returncode} before "
                             "writing its record")
        return dict(json.loads(record_file.read_text(encoding="utf-8")))


def gemm_grid(shapes: Sequence[tuple[str, Any]], sample_clock: Callable[[], float]) -> list[dict[str, Any]]:
    """The GEMM grid, timed on this GPU: per layer shape, the repetitions of the sweep over the rows."""  # GPU only
    from servecast.profile import probe_gpu as PG
    from servecast.profile.probe_vllm import time_layer

    return [{"model": name, "reps": [time_layer(sh, PG.GRID_ROWS, PG.GRID_SECONDS, sample_clock)
                                     for _ in range(PG.GRID_REPEATS)]}
            for name, sh in shapes]  # pragma: no cover - needs a GPU


def verify(out_dir: Path) -> list[str]:
    """The files SHA256SUMS lists that are missing or whose bytes no longer match it."""
    sums = out_dir / "SHA256SUMS"
    if not sums.is_file():
        return ["SHA256SUMS"]
    bad = []
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        path = out_dir / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            bad.append(name)
    return bad


def power_off() -> str | None:
    """Power the machine off; None when the command was accepted, else why not."""
    try:
        subprocess.run(POWEROFF, check=True, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        return f"{' '.join(POWEROFF)} failed ({error})"
    return None


def power_off_when_safe(output_dir: Path, command: str) -> None:
    """Power off once every file matches SHA256SUMS; say why not otherwise."""
    bad = verify(output_dir)
    if bad:
        say(f"--poweroff-when-safe: not powered off: {', '.join(bad)} do not match SHA256SUMS", command)
        return
    say("--poweroff-when-safe: every file matches SHA256SUMS; powering off", command)
    refused = power_off()
    if refused is not None:
        say(f"--poweroff-when-safe: {refused}; power the machine off yourself", command)


def write_sums(out_dir: Path, names: Sequence[str]) -> None:
    """SHA256SUMS of the named files present in `out_dir`."""
    lines = [f"{hashlib.sha256((out_dir / n).read_bytes()).hexdigest()}  {n}" for n in names if (out_dir / n).is_file()]
    (out_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def measure_under_load(backend: Any) -> tuple[list[dict[str, Any]], bool]:  # pragma: no cover - needs a GPU
    """The clock's samples while a dense GEMM loads the GPU, and whether the load ran through them (`gpu.under_load`;
    the tests replace it)."""
    from servecast.profile import gpu as G

    return G.under_load(backend)


def calibration_clock(m: Any) -> float | None:
    """The clock the manifest's calibration was made at, the level `profile serving` locks at; None when the manifest
    names no calibration. A calibration it names but that cannot be found or read ends the command, as it ends
    predict: the serving would otherwise lock at a level of its own, and be compared with a prediction made at
    another clock."""
    if m.calibration.use:  # where predict finds it: shipped, the user's (`servecast calibrate`), or a file
        from servecast.core import calibration as calibrations
        from servecast.core.paths import data_dir
        from servecast.model import predict as P

        name = str(m.calibration.use)
        shipped = P.shipped_calibrations()
        user = data_dir() / "calibrations" / f"{name}.json"
        path = Path(shipped.get(name) or (user if user.is_file() else name))
        if not path.is_file():
            fail(2, f"calibration.use: no calibration {name!r}; shipped: {', '.join(shipped)} (servecast list "
                    "calibrations shows yours too). profile serving locks the clock at the level the calibration "
                    "was made at")
        try:
            doc = calibrations.parse(json.loads(path.read_text(encoding="utf-8")), path)
            return float(doc["conditions"]["clock_mhz"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            fail(2, f"calibration.use: {path} is not a calibration ServeCast can read ({error})")
    if m.calibration.data:
        from servecast.calibrate import fit

        try:
            return float(fit.read_profiling(Path(str(m.calibration.data)))[0]["reference_mhz"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            fail(2, f"calibration.data: {error}")
    return None


#: the manifest field a deployment's memory refusal names, which the per-GPU probe's engine does not take from it
CAPACITY_FIELD = "engine.workers.aggregated.kv_cache.capacity.blocks: "
#: how a locked level was chosen, as the run reports it
CHOSEN_BY = {"sweep": "the highest level it holds under load", "calibration": "the level its calibration was made at",
             "manifest": "the level the manifest asks for"}


@contextlib.contextmanager
def clock_policy(backend: Any, wanted: Any, command: str = COMMAND, *,
                 target_mhz: float | None = None) -> Iterator[dict[str, Any]]:
    """The clock as the manifest asks, locked for the run and released after it, even on a failure. measured: never
    locked. {lock: MHz}: that level. auto: `target_mhz` (profile serving: the level its calibration was made at), else
    the highest level the clock holds under load, the driver's levels tried from the top down (profile calibration).
    A level is kept only if the clock holds it under load (`gpu.sweep`); otherwise, or where the clock cannot be locked
    at all, the clock is measured, and the run says why."""
    from servecast.profile import gpu as G

    if wanted == "measured":
        yield {"policy": "measured"}
        return
    if not isinstance(wanted, str):
        levels, chosen_by = [int(wanted.lock)], "manifest"
    elif target_mhz is not None:
        levels, chosen_by = [round(target_mhz)], "calibration"
    else:
        from servecast.profile.probe import CLOCK_SNAP

        # the probe reads its data at its steps' clocks snapped to CLOCK_SNAP (25 MHz), so only a level on that grid
        # is read at the clock it was locked at (on an A100, whose levels step by 15 MHz: 1350, 1275, 1200, ...)
        top = backend.supported_sm_clocks()
        levels, chosen_by = ([c for c in top if c >= G.SWEEP_FLOOR * top[0] and c % CLOCK_SNAP == 0]
                             if top else []), "sweep"
    if not levels:
        found: dict[str, Any] = {"lock_mhz": None, "levels": [], "refused": "the driver lists no clock level to lock"}
    else:
        found = G.sweep(backend, levels, measure_under_load)
    if found["lock_mhz"] is None:
        why = found["refused"] or (
            f"the clock did not hold {levels[0]} MHz under load ({'; '.join(found['levels'][-1]['why'])})"
            if len(levels) == 1 else f"the clock held no level from {levels[0]} to {levels[-1]} MHz under load")
        say(f"warning: {why}; the clock is measured instead", command)
        yield {"policy": "measured", "lock_wanted": chosen_by, "lock_refused": why, "levels": found["levels"]}
        return
    say(f"the clock is locked at {found['lock_mhz']} MHz, {CHOSEN_BY[chosen_by]}", command)
    try:
        yield {"policy": "locked", "lock_mhz": found["lock_mhz"], "chosen_by": chosen_by, "levels": found["levels"]}
    finally:
        why = backend.unlock()
        if why is not None:
            say(f"warning: {why}", command)


def machine(m: Any, command: str) -> tuple[Any, dict[str, Any], str, str]:
    """The computer, as `doctor` sees it: the GPU backend, the device, its registry id and vLLM's version. Every
    reason it cannot measure this manifest ends the command."""
    from servecast.core.deployment import VLLM_VERSION
    from servecast.profile import gpu as G

    try:
        backend = G.detect()
    except G.CannotProfileError as error:
        fail(3, f"this computer cannot profile: {error}. Every other command runs on it.", command)
    described = backend.describe()
    why = backend.unsupported(described)
    if why is not None:
        fail(3, f"this computer cannot profile: {why}. Every other command runs on it.", command)
    try:
        gpu_id = G.check_hardware(described, m.engine.hardware)
    except G.GpuError as error:
        fail(2, str(error), command)
    holders = backend.busy()
    if holders:
        fail(3, f"the GPU is in use by process(es) {', '.join(map(str, holders))}; a measurement needs it to itself, "
                "since another process would share its memory and its clock", command)
    vllm = vllm_version()
    if vllm != VLLM_VERSION:
        fail(3, f"vLLM {vllm} is installed; the measurement runs {VLLM_VERSION}, the version every calibration uses"
             if vllm else f"vLLM {VLLM_VERSION} is not installed; install the profile extra: pip install "
             "'servecast[profile]'", command)
    return backend, described, gpu_id, str(vllm)


def writable(output_dir: Path, known: Sequence[str], command: str) -> None:
    """Make the output directory, refuse one that cannot be written, and remove an earlier run's known files."""
    import tempfile

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output_dir):  # an existing read-only directory passes mkdir
            pass
    except OSError as error:
        fail(3, f"--output-dir: cannot write {output_dir}: {error.strerror}", command)
    for name in known:  # an earlier run's files would mix with this one's
        (output_dir / name).unlink(missing_ok=True)


def print_table(points: Sequence[dict[str, Any]]) -> None:
    typer.echo(f"{'clients':>8} {'think ms':>9} {'TTFT ms':>9} {'P95 TTFT':>9} {'TPOT ms':>8} {'P95 TPOT':>9} "
               f"{'req/s':>8} {'tok/s':>9} {'MHz':>6}  converged")

    def f(v: float | None, fmt: str) -> str:
        return format(v, fmt) if v is not None else "-".rjust(int(fmt.split(".")[0]))

    for point in points:
        s, load = point["summary"], point["load"]
        typer.echo(f"{load['concurrency']:>8} {load['think_time_ms']:>9.0f} {f(s['mean_ttft_ms'], '9.1f')} "
                   f"{f(s['p95_ttft_ms'], '9.1f')} {f(s['mean_tpot_ms'], '8.2f')} {f(s['p95_tpot_ms'], '9.2f')} "
                   f"{f(s['request_throughput_rps'], '8.3f')} {f(s['output_throughput_tok_s'], '9.1f')} "
                   f"{f(point['measured_clock_mhz'], '6.0f')}  "
                   f"{'yes' if point['converged'] else 'NO (' + point['stop_reason'] + ')'}")


def serving(config: Path, *, sets: list[str], sweep: list[str] | None, output_dir: Path, overwrite: bool,
            output_format: str, timeout: float | None, poweroff_when_safe: bool) -> None:
    from servecast.cli.predict_cmd import load_manifest, parse_sweep
    from servecast.core.manifest import ManifestError
    from servecast.profile import gpu as G
    from servecast.profile import serving as S
    from servecast.profile.vllm_engine import EngineError

    m = load_manifest(config, sets, [], command=COMMAND)
    swept = parse_sweep(sweep, command=COMMAND)
    poweroff = poweroff_when_safe or m.profile.poweroff_when_safe
    if output_dir.is_dir() and any(output_dir.iterdir()) and not overwrite:
        fail(3, f"--output-dir: {output_dir} is not empty; give --overwrite to replace {', '.join(KNOWN_OUTPUTS)}")
    target_mhz = calibration_clock(m)  # before the GPU: a calibration that cannot be found ends the command here

    backend, described, gpu_id, vllm = machine(m, COMMAND)

    try:  # the experiment, resolved as predict resolves it
        if not m.engine.model:
            raise ManifestError("engine.model: required: the LLM to serve, a Hugging Face id or a local directory")
        try:
            llm_config = read_config(str(m.engine.model))
        except ValueError as error:
            raise ManifestError(f"engine.model: {error}") from error
        setup = S.prepare(m, gpu_id, llm_config, sweep=swept)
    except ManifestError as error:
        fail(2, str(error))
    for warning in setup.pool.warnings:
        say(f"warning: {warning}")

    writable(output_dir, KNOWN_OUTPUTS, COMMAND)
    os.environ.update(G.cuda_environment(os.environ))
    log = output_dir / "engine.log"
    points: list[dict[str, Any]] = []
    try:
        with clock_policy(backend, m.profile.clock, target_mhz=target_mhz) as clock:
            environment = {"gpu": {k: v for k, v in described.items() if k != "processes"}, "vllm": vllm,
                           "software": software(), "clock": clock, "attention": dict(backend.attention)}
            say(f"{len(setup.points)} load point(s) of {setup.model} on {described['name']}; vLLM's output goes to "
                f"{log}")
            try:
                ids = prompt(setup.model, max(setup.pool.input_tokens))
            except (EngineError, *ENGINE_FAILURES) as error:
                fail(1, f"the prompt could not be built with {setup.model}'s tokenizer ({type(error).__name__}: "
                        f"{error}); a gated LLM needs HF_TOKEN set to a token that has accepted its licence")
            with S.output_to(log):
                try:
                    served = engine(setup.model, setup.deployment, backend)
                except (EngineError, *ENGINE_FAILURES) as error:
                    fail(1, f"vLLM could not start ({type(error).__name__}: {error}); its output is in {log}")
                try:
                    points = S.serve(setup, output_dir, engine=served, sampler=G.sampler(backend), prompt=ids,
                                     environment=environment, engine_log=log, timeout_s=timeout,
                                     loop_factory=loop_factory, progress=say)
                except ENGINE_FAILURES as error:
                    fail(1, f"the measurement stopped ({type(error).__name__}: {error}); the load points measured "
                            f"before it are in {output_dir}, vLLM's output in {log}")
                finally:
                    served.shutdown()
    except KeyboardInterrupt:
        fail(130, f"interrupted; the load points measured before it are in {output_dir}")

    if output_format == "json":
        typer.echo((output_dir / "measurement.json").read_text(encoding="utf-8"), nl=False)
    else:
        print_table(points)
        say(f"wrote {output_dir / 'measurement.json'}")
    status = 0 if all(p["converged"] for p in points) else (124 if timeout else 1)
    if poweroff:
        power_off_when_safe(output_dir, COMMAND)
    if status:
        raise typer.Exit(status)


def print_probe(record: dict[str, Any], profiling: dict[str, Any] | None, output_dir: Path) -> None:
    a = record["audit"]
    designed = a["compositions_designed"]
    drift = "unmeasured" if a["reference_drift"] is None else f"{100 * a['reference_drift']:.2f} %"
    clock = [float(t["sm_clock_mhz"]) for t in record["clock"]]
    typer.echo(f"compositions     {a['compositions']} of {sum(designed.values())} designed ({designed['decode']} "
               f"decode, {designed['prefill']} prefill, {designed['mixed']} mixed)")
    typer.echo(f"beats kept       {a['beats_kept']} of {a['beats_run']} timed, {a['beat_gpu_s']:.0f} s on the GPU")
    typer.echo(f"reference drift  {drift}")
    typer.echo(f"SM clock         {min(clock):.0f}-{max(clock):.0f} MHz over {len(clock)} samples" if clock
               else "SM clock         not sampled")
    if profiling is None:
        typer.echo("profiling data   none: " + "; ".join(a["flags"]))
    else:
        typer.echo(f"profiling data   {len(profiling['cells'])} cells at {profiling['reference_mhz']:.0f} MHz, in "
                   f"{output_dir / 'profiling.json'}")


def calibration(config: Path, *, sets: list[str], output_dir: Path, overwrite: bool, output_format: str,
                poweroff_when_safe: bool) -> None:
    from servecast.cli.predict_cmd import load_manifest
    from servecast.core import versions

    m = load_manifest(config, sets, [], command=CALIBRATION)
    try:
        law = str(versions.resolve(m.versions)["cost_law"])
    except ValueError as error:
        fail(2, f"versions: {error}", CALIBRATION)
    per_gpu = versions.law(law).calibrated_per == "gpu"
    if not per_gpu and not m.engine.model:
        fail(2, "engine.model: required: the LLM to probe, a Hugging Face id or a local directory", CALIBRATION)
    if m.engine.gpu_clock_mhz is not None:
        fail(2, "engine.gpu_clock_mhz: the probe runs at the GPU's own clock; lock it with profile.clock: {lock: "
                "MHz}, or remove the field", CALIBRATION)
    poweroff = poweroff_when_safe or m.profile.poweroff_when_safe
    known = GPU_PROBE_OUTPUTS if per_gpu else PROBE_OUTPUTS
    if output_dir.is_dir() and any(output_dir.iterdir()) and not overwrite:
        fail(3, f"--output-dir: {output_dir} is not empty; give --overwrite to replace {', '.join(known)}",
             CALIBRATION)
    backend, described, gpu_id, vllm = machine(m, CALIBRATION)
    if per_gpu:
        _calibrate_gpu(m, law, backend=backend, described=described, gpu_id=gpu_id, vllm=vllm, output_dir=output_dir,
                       output_format=output_format, poweroff=poweroff)
    else:
        _calibrate_llm(m, law, backend=backend, described=described, gpu_id=gpu_id, vllm=vllm, output_dir=output_dir,
                       output_format=output_format, poweroff=poweroff)


def _calibrate_llm(m: Any, law: str, *, backend: Any, described: dict[str, Any], gpu_id: str, vllm: str,
                   output_dir: Path, output_format: str, poweroff: bool) -> None:
    """The per-LLM law's probe: the manifest's LLM, in this process, on the paper's probe engine
    (`probe_design.engine`), whatever the manifest's serving settings; the manifest's KV block count, vLLM's own for
    this LLM on this GPU, caps the engine's, as the paper's probes were capped where an LLM did not fit."""
    from servecast.core import deployment, formats, gpus
    from servecast.core.manifest import ManifestError
    from servecast.profile import gpu as G
    from servecast.profile import probe as PR
    from servecast.profile import probe_design as PD
    from servecast.profile import serving as S
    from servecast.profile.vllm_engine import EngineError

    model = str(m.engine.model)
    try:  # the paper's probe engine, refused as predict and profile serving refuse a deployment
        try:
            arch = deployment.architecture_from_config(read_config(model))
        except ValueError as error:
            raise ManifestError(f"engine.model: {model}: {error}") from error
        deployment.check_architecture(m, arch)
        capacity = m.engine.workers.aggregated.kv_cache.capacity
        blocks = PD.ENGINE_KV_BLOCKS if capacity is None else min(PD.ENGINE_KV_BLOCKS, int(capacity.blocks))
        d = PD.engine(gpus.get(gpu_id), arch, model, kv_blocks=blocks)
    except ManifestError as error:
        fail(2, str(error), CALIBRATION)
    design = PD.design(d.kv_blocks * d.block_size, d.context_length, d.max_batched_tokens, d.max_sequences)
    unfit = PD.gates(design)
    if unfit:
        fail(2, f"the probe does not fit this engine ({d.kv_blocks} KV blocks of {d.block_size} tokens, context "
                f"{d.context_length}, budget {d.max_batched_tokens}, {d.max_sequences} sequences): {'; '.join(unfit)}",
             CALIBRATION)

    writable(output_dir, PROBE_OUTPUTS, CALIBRATION)
    os.environ.update(G.cuda_environment(os.environ))
    log = output_dir / "engine.log"
    spans: list[dict[str, Any]] = []
    aborted: str | None = None
    interrupted = False
    with clock_policy(backend, m.profile.clock, CALIBRATION) as clock:
        environment = {"gpu": {k: v for k, v in described.items() if k != "processes"}, "vllm": vllm,
                       "software": software(), "clock": clock, "attention": dict(backend.attention),
                       "engine": dataclasses.asdict(d)}
        n = PD.n_cells(design)
        say(f"probing {model} on {described['name']} for the cost law {law}: {sum(n.values())} compositions; "
            f"vLLM's output goes to {log}", CALIBRATION)
        with S.output_to(log):
            try:
                eng = probe_engine(model, d, backend, kv_tokens=design["kv_tokens"],
                                   sample_clock=backend.sm_clock_mhz)
            except (EngineError, PR.ProbeError, *ENGINE_FAILURES) as error:
                fail(1, f"vLLM could not start ({type(error).__name__}: {error}); its output is in {log}", CALIBRATION)

            def progress(done: int, total: int) -> None:
                if done == 1 and not eng.beats:
                    raise PR.ProbeError("the step timer recorded no engine step: it is not on vLLM's call path")
                if done and done % 20 == 0:
                    say(f"{done} of {total} tasks", CALIBRATION)
            try:
                PR.drive(eng, design, lambda: len(eng.beats), spans, progress=progress)
            except (PR.ProbeError, *ENGINE_FAILURES) as error:
                aborted = f"{type(error).__name__}: {error}"
            except KeyboardInterrupt:
                aborted, interrupted = "interrupted", True
            finally:
                eng.shutdown()
        lock_mhz = clock.get("lock_mhz")
        beats = list(eng.beats)
        samples = [dict(t) for t in eng.clock]
        record: dict[str, Any] = {
            "schema": "servecast-probe/1", "gpu": gpu_id, "llm": model, "cost_law": law, "design": design,
            "spans": spans, "beats": [dataclasses.asdict(b) for b in beats], "clock": samples,
            "audit": PR.audit(beats, spans, design, samples, lock_mhz=lock_mhz, aborted=aborted,
                              degraded=bool(getattr(eng, "degraded", False)))}
    formats.Probe.model_validate(record)
    (output_dir / "probe.json").write_text(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=1, ensure_ascii=False) + "\n",
                                                 encoding="utf-8")
    profiling = None
    if record["audit"]["clean"]:
        try:
            profiling = PR.profiling_document(PR.steps(beats, spans, samples), gpu=gpu_id, llm=model, cost_law=law,
                                              declared_mhz=lock_mhz)
        except PR.ProbeError as error:
            record["audit"]["flags"].append(str(error))
        else:
            formats.ProfilingData.model_validate(profiling)
            (output_dir / "profiling.json").write_text(
                json.dumps(profiling, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
    write_sums(output_dir, ("profiling.json", "probe.json", "environment.json"))
    if output_format == "json":
        typer.echo(json.dumps({"audit": record["audit"], "profiling": None if profiling is None else
                               {"cells": len(profiling["cells"]), "reference_mhz": profiling["reference_mhz"]}},
                              indent=1))
    else:
        print_probe(record, profiling, output_dir)
    if interrupted:
        fail(130, f"interrupted; what was measured is in {output_dir / 'probe.json'}", CALIBRATION)
    if profiling is None:
        fail(1, f"the probe's audit has flags, so no profiling data was written; every beat is in "
                f"{output_dir / 'probe.json'}", CALIBRATION)
    say(f"wrote {output_dir / 'profiling.json'}; fit it with: servecast calibrate --data {output_dir}", CALIBRATION)
    if poweroff:
        power_off_when_safe(output_dir, CALIBRATION)


def _calibrate_gpu(m: Any, law: str, *, backend: Any, described: dict[str, Any], gpu_id: str, vllm: str,
                   output_dir: Path, output_format: str, poweroff: bool) -> None:
    """The per-GPU law's probe: each synthetic model of the pool in a process of its own, then the GEMM grid, at one
    clock. Every synthetic model runs on the paper's probe engine with its 9,965 KV blocks, as the paper's did: the
    manifest's serving settings belong to the LLM it serves, not to these models."""
    from servecast.core import deployment, formats, gpus
    from servecast.core.manifest import ManifestError
    from servecast.profile import gpu as G
    from servecast.profile import probe as PR
    from servecast.profile import probe_design as PD
    from servecast.profile import probe_gpu as PG
    from servecast.profile import serving as S

    jobs: list[dict[str, Any]] = []
    for name in PG.POOL:
        try:
            model_dir = synthetic_model(name)
            arch = deployment.architecture_from_config(json.loads((model_dir / "config.json").read_text("utf-8")))
            d = PD.engine(gpus.get(gpu_id), arch, name)
        except ManifestError as error:  # the engine's KV cache and the model's weights do not fit this GPU
            fail(3, f"{name}: this GPU cannot hold the per-GPU probe, which runs every synthetic model with "
                    f"{PD.ENGINE_KV_BLOCKS} KV blocks as the paper did: "
                    f"{str(error).split('. Give the count')[0].removeprefix(CAPACITY_FIELD)}",
                 CALIBRATION)
        except (ValueError, PR.ProbeError) as error:
            fail(2, f"{name}: {error}", CALIBRATION)
        design = PD.design(d.kv_blocks * d.block_size, d.context_length, d.max_batched_tokens, d.max_sequences)
        unfit = PD.gates(design)
        if unfit:
            fail(2, f"{name}: the probe does not fit this engine: {'; '.join(unfit)}", CALIBRATION)
        jobs.append({"model": name, "dir": str(model_dir), "deployment": dataclasses.asdict(d), "design": design})

    writable(output_dir, (*GPU_PROBE_OUTPUTS, *(f"probe-{n.removeprefix('synthetic/')}.json" for n in PG.POOL)),
             CALIBRATION)
    os.environ.update(G.cuda_environment(os.environ))
    log = output_dir / "engine.log"
    probed: list[tuple[dict[str, Any], dict[str, Any]]] = []
    failure: str | None = None
    interrupted = False
    runs: list[dict[str, Any]] = []
    with clock_policy(backend, m.profile.clock, CALIBRATION) as clock:
        lock_mhz = clock.get("lock_mhz")
        environment = {"gpu": {k: v for k, v in described.items() if k != "processes"}, "vllm": vllm,
                       "software": software(), "clock": clock, "attention": dict(backend.attention),
                       "engine": jobs[0]["deployment"]}
        say(f"probing {len(jobs)} synthetic models on {described['name']} for the cost law {law}, then its GEMM "
            f"grid; vLLM's output goes to {log}", CALIBRATION)
        with S.output_to(log):
            try:
                for i, job in enumerate(jobs, 1):
                    say(f"{i} of {len(jobs)}: {job['model']}", CALIBRATION)
                    try:
                        record = probe_synthetic({**job, "lock_mhz": lock_mhz})
                    except (PR.ProbeError, *ENGINE_FAILURES) as error:
                        failure = f"{job['model']}: {type(error).__name__}: {error}"
                        break
                    probed.append((job, record))
                    if not record["audit"]["clean"]:
                        failure = f"{job['model']}: the probe's audit has flags"
                        break
                if failure is None:
                    shapes = PG.grid_shapes()
                    say(f"timing the GEMM grid: {len(shapes)} layer shapes, {len(PG.GRID_ROWS)} row counts, "
                        f"{PG.GRID_REPEATS} times", CALIBRATION)
                    runs = gemm_grid(shapes, backend.sm_clock_mhz)
            except KeyboardInterrupt:
                interrupted, failure = True, "interrupted"
    written = []
    for job, record in probed:
        doc = {"schema": "servecast-probe/1", "gpu": gpu_id, "llm": job["model"], "cost_law": law,
               "design": job["design"], **record}
        formats.Probe.model_validate(doc)
        path = output_dir / f"probe-{job['model'].removeprefix('synthetic/')}.json"
        path.write_text(json.dumps(doc, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(path.name)
    grid_clock: float | None = None
    grid: dict[str, Any] | None = None
    if runs:
        clocks = [float(p["sm_mhz"]) for r in runs for rep in r["reps"] for p in rep if p.get("sm_mhz")]
        grid_clock = float(lock_mhz) if lock_mhz else (PR.step_clock(clocks) or None)
        flags = PG.grid_flags(runs, gpus.get(gpu_id).peak_flops * (grid_clock or 1.0) / gpus.get(gpu_id).boost_sm_mhz)
        if grid_clock is None:
            flags.append("the grid's clock did not hold: its points' SM clocks differ by more than 2 %")
        elif lock_mhz and any(abs(c / lock_mhz - 1) > PR.CLOCK_TOL for c in clocks):
            flags.append(f"the grid's clock did not hold within {100 * PR.CLOCK_TOL:.1f} % of {lock_mhz:g} MHz")
        grid = {"schema": "servecast-gemm/1", "gpu": gpu_id, "clock_mhz": grid_clock, "seconds": PG.GRID_SECONDS,
                "runs": runs, "clean": not flags, "flags": flags}
        formats.Gemm.model_validate(grid)
        (output_dir / "gemm.json").write_text(json.dumps(grid, separators=(",", ":"), ensure_ascii=False) + "\n",
                                              encoding="utf-8")
        written.append("gemm.json")
        if flags:
            failure = "the GEMM grid's audit has flags: " + "; ".join(flags)
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=1, ensure_ascii=False) + "\n",
                                                 encoding="utf-8")
    profiling = None
    if failure is None and grid is not None and grid_clock is not None:
        try:
            profiling = PG.profiling_document(
                [(job["model"], PR.steps([PR.Beat(**b) for b in rec["beats"]], rec["spans"], rec["clock"]))
                 for job, rec in probed], runs, gpu=gpu_id, cost_law=law, declared_mhz=lock_mhz,
                grid_clock_mhz=grid_clock)
        except PR.ProbeError as error:
            failure = str(error)
        else:
            formats.ProfilingData.model_validate(profiling)
            (output_dir / "profiling.json").write_text(
                json.dumps(profiling, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
    write_sums(output_dir, ("profiling.json", *written, "environment.json"))
    audit_fields = ("clean", "flags", "compositions", "beats_kept", "reference_drift")
    summary: dict[str, Any] = {
        "models": [{"llm": job["model"], **{k: rec["audit"][k] for k in audit_fields}} for job, rec in probed],
               "gemm": None if grid is None else {k: grid[k] for k in ("clock_mhz", "clean", "flags")},
               "profiling": None if profiling is None else {"models": len(profiling["models"]),
                                                            "reference_mhz": profiling["reference_mhz"]},
               "failure": failure}
    if output_format == "json":
        typer.echo(json.dumps(summary, indent=1))
    else:
        for row in summary["models"]:
            drift = "-" if row["reference_drift"] is None else f"{100 * row['reference_drift']:.2f} %"
            typer.echo(f"{row['llm']:<42} {row['compositions']:>4} compositions, {row['beats_kept']:>5} beats kept, "
                       f"drift {drift}{'' if row['clean'] else ', FLAGS: ' + '; '.join(row['flags'])}")
        if grid is not None:
            typer.echo(f"GEMM grid: {len(runs)} layer shapes at {grid_clock or 0:.0f} MHz"
                       f"{'' if grid['clean'] else ', FLAGS: ' + '; '.join(grid['flags'])}")
        if profiling is not None:
            typer.echo(f"profiling data: {len(profiling['models'])} synthetic models and the grid at "
                       f"{profiling['reference_mhz']:.0f} MHz, in {output_dir / 'profiling.json'}")
    if interrupted:
        fail(130, f"interrupted; what was measured is in {output_dir}", CALIBRATION)
    if profiling is None:
        fail(1, f"no profiling data was written ({failure}); what was measured is in {output_dir}", CALIBRATION)
    say(f"wrote {output_dir / 'profiling.json'}; fit it with: servecast calibrate --data {output_dir}", CALIBRATION)
    if poweroff:
        power_off_when_safe(output_dir, CALIBRATION)
