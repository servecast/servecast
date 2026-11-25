"""The calibration probe on the GPU: vLLM 0.25.1's in-process engine, with a timer around each engine step.

Ported unchanged from the probe behind the paper's calibrations:

* the engine is vLLM's synchronous `LLM`, in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0), so the step timer can
  wrap its model runner; it is built with the serving engine's settings (`vllm_engine.engine_args`: the same KV
  blocks, token budget, sequence cap and context length) and two pins, both READ BACK from the engine that was built:
  the GPU backend's attention kernel (FlashAttention 2 on NVIDIA) and asynchronous scheduling, whose one-step
  admission lag the population model stages;
* the timer wraps `execute_model` of every model-runner class on the call path; each step is timed with CUDA events
  and its composition read from the scheduler's output (`composition`): the prefill chunks and, for each decode
  context, the requests decoding at it;
* each prompt is a prefix of one fixed text, after the LLM's chat template when it has one, of exactly the tokens a
  cell asks for.

The per-GPU law adds two things, also ported unchanged: a synthetic model is a local directory (the base model's
config.json with the geometry replaced, and its tokenizer, downloaded without weights) served with random weights
(load_format "dummy"); and the GEMM grid times one layer's five matrix multiplications on random bf16 matrices, each
point for `seconds` of wall clock after three warm-up calls, with the SM clock sampled throughout.

UNVERIFIED in this repository (its CI has no GPU; the release's A100 run checks it): everything below
`composition`, which is pure and tested on scheduler outputs built by hand.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from servecast.core.deployment import Deployment
from servecast.profile.gpu import Backend
from servecast.profile.probe import CLOCK_EVERY, Beat, ProbeError
from servecast.profile.vllm_engine import PROMPT_TEXT, EngineError, engine_args

#: the runner, reached from the synchronous engine when it runs in this process
RUNNER_PATHS = ("llm_engine.model_executor.driver_worker.model_runner",
                "llm_engine.model_executor.driver_worker.worker.model_runner")
#: the model-runner classes the timer wraps, imported by name (vLLM 0.25.1 has both layouts)
RUNNER_MODULES = ("vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner")
#: the KV blocks the engine holds may fall short of the design's by this much, no more
KV_SHORTFALL = 0.02


def composition(runner: object, scheduler_output: object) -> tuple[list[list[int]], list[list[int]], bool]:
    """(prefill, decode, degraded) of one engine step. prefill: [tokens, tokens already computed] per request with a
    prefill chunk, largest first; decode: [context, requests] per decode context, ascending; degraded: a request's
    state could not be read and was classified by its token count alone.

    A request vLLM has seen before is decoding when it is granted exactly one token and prefilling otherwise (its
    next chunk), which speculative decoding would break (`assert_no_speculation`)."""
    num_sched: Any = getattr(scheduler_output, "num_scheduled_tokens", None)
    if num_sched is None:
        num_sched = getattr(scheduler_output, "scheduled_tokens", None)
    items = list(num_sched.items()) if hasattr(num_sched, "items") else []
    meta: dict[Any, tuple[int, int | None]] = {}
    for nr in (getattr(scheduler_output, "scheduled_new_reqs", None) or []):
        rid = getattr(nr, "req_id", None)
        toks = getattr(nr, "prompt_token_ids", None)
        npr = len(toks) if toks is not None else getattr(nr, "num_prompt_tokens", None)
        comp = getattr(nr, "num_computed_tokens", 0)
        meta[rid] = (int(comp or 0), int(npr) if npr is not None else None)
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    if cached is not None:
        rids = getattr(cached, "req_ids", None) or []
        comps = getattr(cached, "num_computed_tokens", None) or []
        for i, rid in enumerate(rids):
            meta[rid] = (int(comps[i]) if i < len(comps) else 0, -1)  # -1: seen before
    req_states = getattr(runner, "requests", None) or {}
    prefill: list[list[int]] = []
    decode: dict[int, int] = defaultdict(int)
    degraded = False
    for req_id, n in items:
        n_tok = int(n)
        if n_tok <= 0:
            continue
        computed: Any
        n_prompt: Any
        computed, n_prompt = meta.get(req_id, (None, None))
        if computed is None:
            st = req_states.get(req_id) if hasattr(req_states, "get") else None
            computed = getattr(st, "num_computed_tokens", None)
            n_prompt = getattr(st, "num_prompt_tokens", None)
            if n_prompt is None:
                toks = getattr(st, "prompt_token_ids", None)
                n_prompt = len(toks) if toks is not None else None
        if n_prompt == -1:
            if n_tok == 1:
                decode[int(computed)] += 1
            else:
                prefill.append([n_tok, int(computed)])
        elif computed is None or n_prompt is None:
            degraded = True
            if n_tok == 1:
                decode[0] += 1
            else:
                prefill.append([n_tok, 0])
        elif int(computed) < int(n_prompt):
            q = min(n_tok, int(n_prompt) - int(computed))
            prefill.append([q, int(computed)])
            extra = n_tok - q
            if extra > 0:
                decode[int(n_prompt)] += extra
        else:
            decode[int(computed)] += n_tok
    prefill.sort(key=lambda e: -e[0])
    dec = [[kv, c] for kv, c in sorted(decode.items())]
    if not prefill and not dec:
        total = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
        if total > 0:
            return [[total, 0]], [], True
    return prefill, dec, degraded


class BeatRecorder:
    """The beats the step timer records: one per engine step after the first `warmup_beats`, empty steps left out,
    the SM clock sampled every `clock_every` beats."""

    def __init__(self, sample_clock: Callable[[], float] | None, warmup_beats: int = 0,
                 clock_every: int = CLOCK_EVERY) -> None:
        self.sample_clock, self.warmup_beats, self.clock_every = sample_clock, warmup_beats, max(1, clock_every)
        self.beats: list[Beat] = []
        self.clock: list[dict[str, Any]] = []
        self.degraded = False
        self.hook_calls = 0
        self._seen = 0
        self._t0: float | None = None
        self._pending: tuple[list[list[int]], list[list[int]], bool] | None = None

    def begin(self, prefill: list[list[int]], decode: list[list[int]], degraded: bool) -> None:
        self._pending = (prefill, decode, degraded)

    def finish(self, dur_ms: float) -> None:
        if self._pending is None:
            raise ProbeError("a step finished that never began")
        prefill, decode, degraded = self._pending
        self._pending = None
        self._seen += 1
        if self._seen <= self.warmup_beats or (not prefill and not decode):
            return
        now = time.perf_counter()
        if self._t0 is None:
            self._t0 = now
        self.degraded = self.degraded or degraded
        self.beats.append(Beat(prefill=prefill, decode=decode, dur_ms=float(dur_ms), t_ms=1e3 * (now - self._t0)))
        if self.sample_clock is not None and len(self.beats) % self.clock_every == 1:
            mhz = float(self.sample_clock())
            self.clock.append({"beat_index": len(self.beats) - 1, "sm_clock_mhz": mhz,
                               "t_ms": 1e3 * (time.perf_counter() - self._t0)})


def _resolve(obj: object, path: str) -> object | None:
    for attr in path.split("."):
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


def _runner_classes(llm: object) -> list[type]:  # pragma: no cover - needs vLLM
    """The model-runner classes on the call path: those vLLM names, and the live runner's own class."""
    import importlib

    found: list[type] = []
    for mod in RUNNER_MODULES:
        try:
            found.append(importlib.import_module(mod).GPUModelRunner)
        except (ImportError, AttributeError):
            pass
    for path in RUNNER_PATHS:
        runner = _resolve(llm, path)
        if runner is not None and hasattr(type(runner), "execute_model"):
            found.append(type(runner))
    unique: list[type] = []
    for cls in found:
        if cls not in unique and hasattr(cls, "execute_model"):
            unique.append(cls)
    return unique


def install_timer(recorder: BeatRecorder, llm: object) -> tuple[Callable[[], None], list[str]]:  # pragma: no cover
    """Wrap `execute_model` of every model-runner class on the call path: each step's composition is read before it
    runs and its duration timed with CUDA events. Returns the function that removes the timer, and the classes."""
    import torch

    classes = _runner_classes(llm)
    if not classes:
        raise ProbeError("no vLLM model-runner class was found: the step timer cannot be installed")

    def wrap(original: Callable[..., Any]) -> Callable[..., Any]:
        def timed(self: object, *args: Any, **kwargs: Any) -> Any:
            recorder.hook_calls += 1
            sched = args[0] if args else kwargs.get("scheduler_output")
            if sched is None or not torch.cuda.is_available():
                return original(self, *args, **kwargs)
            recorder.begin(*composition(self, sched))
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = original(self, *args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            recorder.finish(start.elapsed_time(end))
            return out
        return timed

    originals = [(cls, getattr(cls, "execute_model")) for cls in classes]  # noqa: B009 - every original first
    for cls, original in originals:
        setattr(cls, "execute_model", wrap(original))  # noqa: B010 - the class is only known at run time

    def restore() -> None:
        for cls, original in originals:
            setattr(cls, "execute_model", original)  # noqa: B010
    return restore, [f"{c.__module__}.{c.__name__}" for c in classes]


def live_attention_backend(llm: object) -> str | None:  # pragma: no cover - needs a live engine
    """The attention backend the engine resolved, read from its runner's attention groups (FlashAttention's is named
    FLASH_ATTN); None when it cannot be read or the groups disagree."""
    for path in RUNNER_PATHS:
        runner = _resolve(llm, path)
        groups = getattr(runner, "attn_groups", None) if runner is not None else None
        if not groups:
            continue
        names = set()
        for per_kv_group in groups:
            for g in (per_kv_group if isinstance(per_kv_group, list | tuple) else [per_kv_group]):
                backend = getattr(g, "backend", None)
                name = getattr(backend, "__name__", None) or type(backend).__name__
                if name:
                    names.add(name)
        if len(names) == 1:
            name = names.pop()
            return "FLASH_ATTN" if name.startswith("FlashAttention") else name
        return None
    return None


def kv_capacity_tokens(llm: object) -> int | None:  # pragma: no cover - needs a live engine
    """The KV tokens the engine holds (blocks times block size), read from its cache configuration."""
    for base in ("llm_engine", "engine", None):
        eng = getattr(llm, base, None) if base else llm
        if eng is None:
            continue
        cc = getattr(eng, "cache_config", None) or getattr(getattr(eng, "vllm_config", None), "cache_config", None)
        if cc is None:
            continue
        blocks = getattr(cc, "num_gpu_blocks", None) or getattr(cc, "num_gpu_blocks_override", None)
        size = getattr(cc, "block_size", None)
        if blocks and size:
            return int(blocks) * int(size)
    return None


def fixed_prompt(tok: Any, n_tokens: int) -> list[int]:  # pragma: no cover - needs the LLM's tokenizer
    """Exactly `n_tokens` token ids after the chat template (when the LLM has one): the fixed text, padded with
    "word " until it is long enough (a tokenizer can merge the padding, so it is extended until it reaches)."""
    if getattr(tok, "chat_template", None):
        out = tok.apply_chat_template([{"role": "user", "content": PROMPT_TEXT}], add_generation_prompt=True,
                                      tokenize=True)
        if hasattr(out, "keys"):
            out = out["input_ids"]
        out = list(out)
        if out and isinstance(out[0], list | tuple):
            out = list(out[0])
        ids = [int(i) for i in out]
    else:
        ids = list(tok(PROMPT_TEXT)["input_ids"])
    for _ in range(64):
        if len(ids) >= n_tokens:
            break
        pad = tok("word " * (n_tokens - len(ids)))["input_ids"]
        if not pad:
            break
        ids += list(pad)
    ids = ids[:n_tokens]
    if len(ids) != n_tokens:
        raise ProbeError(f"a prompt of exactly {n_tokens} tokens could not be built (got {len(ids)})")
    return ids


class VllmProbeEngine:  # pragma: no cover - needs a GPU and vLLM
    """vLLM's in-process engine behind the driver's four calls, with the step timer around it."""

    def __init__(self, model: str, d: Deployment, backend: Backend, *, kv_tokens: int,
                 sample_clock: Callable[[], float] | None, load_format: str | None = None) -> None:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        import torch
        import vllm
        from vllm import LLM, SamplingParams, TokensPrompt

        if not torch.cuda.is_available():
            raise EngineError("no CUDA device: profile calibration runs on the GPU host")
        self.vllm_version = str(vllm.__version__)
        self.SP, self.TP = SamplingParams, TokensPrompt
        extra = {"load_format": load_format} if load_format else {}
        self.llm = LLM(**engine_args(model, d, backend.attention), async_scheduling=True, **extra)
        self.core = self.llm.llm_engine
        config = getattr(self.core, "vllm_config", None)
        problems = [why for why in (backend.attention_mismatch(config),) if why]
        attn = live_attention_backend(self.llm)
        if attn != backend.attention.get("backend"):
            problems.append(f"the engine's attention backend is {attn}, not {backend.attention.get('backend')}")
        if not bool(getattr(getattr(config, "scheduler_config", None), "async_scheduling", False)):
            problems.append("the engine does not schedule asynchronously; the calibration must, as the serving "
                            "engine does")
        if getattr(config, "speculative_config", None):
            problems.append("speculative decoding is on: the step timer reads a step's composition by the tokens "
                            "each request is granted, which it would misread")
        held = kv_capacity_tokens(self.llm)
        if held is None:
            problems.append("the engine's KV capacity could not be read back")
        elif held < (1 - KV_SHORTFALL) * kv_tokens:
            problems.append(f"the engine holds {held} KV tokens, the probe was designed for {kv_tokens}")
        if problems:
            raise EngineError("; ".join(problems))
        self.tok = self.llm.get_tokenizer()
        self._ids: dict[int, list[int]] = {}
        # warm the engine before the timer, so no recorded step is a warm-up step
        self.llm.generate([self.TP(prompt_token_ids=self.ids_of(64))],
                          self.SP(ignore_eos=True, max_tokens=4, temperature=0.0), use_tqdm=False)
        self.recorder = BeatRecorder(sample_clock)
        self.restore, self.runners = install_timer(self.recorder, self.llm)

    def ids_of(self, n_tokens: int) -> list[int]:
        if n_tokens not in self._ids:
            self._ids[n_tokens] = fixed_prompt(self.tok, n_tokens)
        return self._ids[n_tokens]

    def add(self, request_id: str, n_tokens: int, max_tokens: int) -> None:
        self.core.add_request(request_id, self.TP(prompt_token_ids=list(self.ids_of(n_tokens))),
                              self.SP(ignore_eos=True, max_tokens=max_tokens, temperature=0.0))

    def step(self) -> list[tuple[str, bool, int]]:
        return [(o.request_id, bool(o.finished), len(o.outputs[0].token_ids) if o.outputs else 0)
                for o in self.core.step()]

    def abort(self, request_ids: list[str]) -> None:
        self.core.abort_request(list(request_ids))

    def unfinished(self) -> bool:
        return bool(self.core.has_unfinished_requests())

    @property
    def beats(self) -> Sequence[Beat]:
        return self.recorder.beats

    @property
    def clock(self) -> Sequence[dict[str, Any]]:
        return self.recorder.clock

    @property
    def degraded(self) -> bool:
        return self.recorder.degraded

    def shutdown(self) -> None:
        self.restore()


def materialize_synthetic(name: str, root: Path) -> Path:  # pragma: no cover - downloads from the hub
    """The synthetic model's directory under `root`: the base model's config.json with the geometry replaced, and its
    tokenizer files, downloaded without the weights (HF_TOKEN is used when set)."""
    from huggingface_hub import snapshot_download

    from servecast.profile import probe_gpu as PG

    snap = Path(snapshot_download(PG.BASE, allow_patterns=list(PG.BASE_FILES), token=os.environ.get("HF_TOKEN")))
    config = PG.synthetic_config(name, json.loads((snap / "config.json").read_text(encoding="utf-8")))
    out = root / name.removeprefix("synthetic/")
    out.mkdir(parents=True, exist_ok=True)
    for f in PG.BASE_FILES:
        if f != "config.json" and (snap / f).exists():
            shutil.copyfile(snap / f, out / f)
    (out / "config.json").write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8")
    return out


#: the SM clock is read this often while a grid point runs
GRID_CLOCK_EVERY_S = 0.02


def _timed(fn: Callable[[], Any], seconds: float, torch: Any,
           sample_clock: Callable[[], float] | None) -> tuple[float, int, list[float]]:  # pragma: no cover - GPU
    """Seconds per call of `fn`: three warm-up calls, then calls for `seconds` of wall clock, synchronized; the SM
    clock is sampled from the first timed call until the GPU has finished."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            if sample_clock is not None:
                samples.append(float(sample_clock()))
            stop.wait(GRID_CLOCK_EVERY_S)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        t0, n = time.time(), 0
        while time.time() - t0 < seconds:
            fn()
            n += 1
        torch.cuda.synchronize()
        dt = time.time() - t0
    finally:
        stop.set()
        thread.join(timeout=1.0)
    return dt / max(n, 1), n, samples


def time_layer(shapes: Sequence[tuple[str, int, int]], rows: Sequence[int], seconds: float,
               sample_clock: Callable[[], float] | None) -> list[dict[str, Any]]:  # pragma: no cover - needs a GPU
    """One layer's five matrix multiplications on random bf16 matrices, timed at each row count."""
    import torch

    from servecast.profile.probe_gpu import layer_flops

    def mk(*size: int) -> Any:
        return torch.randn(*size, device="cuda", dtype=torch.bfloat16)
    weights = [mk(k, m) for _, k, m in shapes]
    points = []
    for n in rows:
        x, h = mk(n, shapes[0][1]), mk(n, shapes[4][1])

        def work(x: Any = x, h: Any = h) -> Any:
            wqkv, wo, wg, wu, wd = weights
            return x @ wqkv, x @ wo, x @ wg, x @ wu, h @ wd
        per, iters, clock = _timed(work, seconds, torch, sample_clock)
        points.append({"n": int(n), "s_per_iter": per, "iters": iters, "flops": layer_flops(shapes, n),
                       "sm_mhz": (sum(clock) / len(clock)) if clock else None})
    return points
