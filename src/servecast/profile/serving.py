"""`servecast profile serving`: the manifest's workload served by vLLM on this GPU, measured, and written.

In order: everything a measurement needs is resolved by the resolver `predict` uses (`core.deployment`), so the
measurement describes the same experiment, and what cannot run is refused before the GPU is touched; the GPU is
checked against the manifest and must be free; the pool and the environment are written; the engine is built, with
what it prints in engine.log; then each load point is served until the estimator's rule stops it, with the clock
sampled, and written at once.

The engine, the clock sampler and the prompt builder are parameters: the command passes vLLM, NVML and the LLM's
tokenizer, the tests a GPU-free mock.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import os
import sys
import threading
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from pathlib import Path
from typing import Any

from servecast.core import deployment, gpus
from servecast.core import estimator as estimators
from servecast.core.deployment import Deployment, Pool
from servecast.core.manifest import Manifest, ManifestError
from servecast.profile import closed_loop, measurement
from servecast.profile.vllm_engine import computed_blocks

MS = 1e-3
#: the stop rule's series, as the progress line names them
SERIES_NAMES = {"inv_X": "the throughput", "R": "the end-to-end latency", "TTFT": "mean TTFT", "TPOT": "mean TPOT",
                "TTFT_p95": "P95 TTFT", "TPOT_p95": "P95 TPOT"}


def checkpoint_line(n_clients: int, checkpoint: Mapping[str, Any]) -> str:
    """What a checkpoint of the stop rule tells the user: how far the run is, and the 95 % interval that is furthest
    from its target, which is the one the run waits for."""
    done = f"{n_clients} clients: {checkpoint['n_at']:,} requests after {checkpoint['elapsed_s'] / 60:.1f} min"
    width, target = checkpoint.get("worst_rel_hw"), checkpoint.get("target")
    if checkpoint.get("worst") is None or width is None or width != width or width == float("inf"):
        return f"{done}; too few requests for a 95 % interval yet"
    name = SERIES_NAMES.get(str(checkpoint["worst"]), str(checkpoint["worst"]))
    goal = f" (target ±{100 * target:g} %)" if target else ""
    return f"{done}; the widest 95 % interval is that of {name}, ±{100 * width:.1f} %{goal}"


@dataclasses.dataclass(frozen=True)
class Setup:
    """A measurement, resolved: what `serve` runs and what measurement.json records."""

    manifest: Manifest
    gpu_id: str
    model: str
    architecture: dict[str, int]
    deployment: Deployment
    components: dict[str, str]
    targets: dict[str, float]
    pool: Pool
    points: list[tuple[int, float]]
    sections: dict[str, Any]


def prepare(m: Manifest, gpu_id: str, config: Mapping[str, Any], *,
            sweep: tuple[str, list[str]] | None = None) -> Setup:
    """Resolve the manifest for this GPU and the LLM's config.json; refuse what cannot be measured."""
    if m.traffic is None:
        raise ManifestError("traffic: required by profile serving: the source of the lengths, the load and the stop "
                            "rule")
    if not m.engine.model:
        raise ManifestError("engine.model: required: the LLM to serve, a Hugging Face id or a local directory")
    if m.engine.gpu_clock_mhz is not None:
        raise ManifestError("engine.gpu_clock_mhz: a measurement runs at the GPU's own clock; lock it with "
                            "profile.clock: {lock: MHz}, or remove the field")
    try:
        arch = deployment.architecture_from_config(config)
    except ValueError as error:
        raise ManifestError(f"engine.model: {m.engine.model}: {error}") from error
    deployment.check_architecture(m, arch)
    d = deployment.resolve(m, gpus.get(gpu_id), arch["max_context"], str(m.engine.model), architecture=arch)
    components = deployment.measured_components(m)
    targets = deployment.stop_targets(m, components["estimator"])
    pool = deployment.build_pool(m, components, d.context_length)
    sections = deployment.resolved_sections(m, model=str(m.engine.model), architecture=arch, hardware=gpu_id,
                                            deployment=d, components=components)
    sections["engine"].pop("gpu_clock_mhz", None)
    sections["profile"] = m.profile.model_dump(mode="json")
    return Setup(manifest=m, gpu_id=gpu_id, model=str(m.engine.model), architecture=arch, deployment=d,
                 components=components, targets=targets, pool=pool, points=deployment.load_points(m, sweep),
                 sections=sections)


@contextlib.contextmanager
def output_to(path: Path, fds: Sequence[int] = (1, 2)) -> Generator[Path]:
    """Copy what this process, and every process it starts (vLLM's engine core among them), writes to standard output
    and standard error into `path`, while still showing it. vLLM logs to standard output, its warnings go to standard
    error, and the line that gives its own KV block count comes from the engine core."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    log = open(path, "ab")  # closed below, once every pump has stopped
    copies: list[tuple[int, int, int, threading.Thread]] = []

    def pump(read_end: int, shown: int) -> None:
        while chunk := os.read(read_end, 65536):
            with lock:
                log.write(chunk)
                log.flush()
            os.write(shown, chunk)

    sys.stdout.flush()
    sys.stderr.flush()
    for fd in fds:
        saved = os.dup(fd)
        read_end, write_end = os.pipe()
        os.dup2(write_end, fd)
        os.close(write_end)
        thread = threading.Thread(target=pump, args=(read_end, saved), daemon=True)
        thread.start()
        copies.append((fd, saved, read_end, thread))
    try:
        yield path
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        for fd, saved, _, _ in copies:
            os.dup2(saved, fd)
        for _, saved, read_end, thread in copies:
            thread.join(timeout=5)
            os.close(read_end)
            os.close(saved)
        log.close()


def serve(setup: Setup, out_dir: Path, *, engine: Any, sampler: Any, prompt: Sequence[int],
          environment: Mapping[str, Any], engine_log: Path | None = None, timeout_s: float | None = None,
          loop_factory: Callable[[], asyncio.AbstractEventLoop] | None = None,
          progress: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
    """Serve every load point on one event loop and write each as soon as it is measured. The loop's clock times the
    requests; the GPU clock is sampled on the monotonic clock, which is the default loop's. `loop_factory` makes the
    loop: the default on the GPU host, a loop with a simulated clock for the GPU-free mock."""
    est = estimators.get(setup.components["estimator"])
    traffic = setup.manifest.traffic
    assert traffic is not None, "prepare() refuses a manifest without traffic"
    writer = measurement.Writer(out_dir, setup.pool.input_tokens, setup.pool.output_tokens, setup.pool.header,
                                environment)
    points: list[dict[str, Any]] = []
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        for n, z_ms in setup.points:
            if progress is not None:
                progress(f"serving {n} clients, think time {z_ms:g} ms")
            stream = closed_loop.lengths(setup.pool.input_tokens, setup.pool.output_tokens, traffic.seed)
            def report(checkpoint: dict[str, Any], n: int = n) -> None:  # every checkpoint but the last, one line each
                if progress is not None and not checkpoint["stop"]:
                    progress(checkpoint_line(n, checkpoint))

            with sampler(time.monotonic) as clock:
                result = runner.run(closed_loop.serve(engine, prompt, stream, n, z_ms * MS, est, setup.targets,
                                                      seed=traffic.seed, seconds_budget=timeout_s, progress=report))
            window = result["assessment"]["window"]
            clock_window = clock.window(result["t0"] + window["t0"], result["t0"] + window["t1"])
            log_text = engine_log.read_text(encoding="utf-8", errors="replace") if engine_log and \
                engine_log.is_file() else ""
            point = measurement.point(result, n, z_ms, est, clock_mhz=clock_window["mean_mhz"],
                                      kv_blocks=engine.kv_blocks(), kv_blocks_computed=computed_blocks(log_text))
            point["diagnostics"]["clock_window"] = clock_window
            points.append(point)
            doc = measurement.document(points, setup.targets, components=setup.components,
                                       seed=traffic.seed, pool_sha256=writer.pool_sha256,
                                       vllm_version=str(environment.get("vllm") or deployment.VLLM_VERSION),
                                       resolved_manifest=setup.sections)
            kept = {q["rid"] for q in result["assessment"]["kept"]}
            writer.add(doc, result["records"], kept, n, z_ms)
    return points
