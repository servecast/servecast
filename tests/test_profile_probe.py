"""The calibration probe of the paper's law, off the GPU: its design, the driver on a simulated engine with vLLM's
scheduling policy, the audit, the step decoder and recorder of the GPU path, and the profiling data.

The strongest check is the one the paper's probe was designed against: a simulated engine whose every step is priced
by a calibration gives profiling data whose fit returns that calibration's parameters. That the port is the paper's
probe is checked on the research side, against what the paper's probes ran (the same designs, the same beats from
the same driver, the shipped profiling data byte for byte).
"""

from __future__ import annotations

import hashlib
import json
import types
from typing import Any

import numpy as np
import pytest

from servecast.calibrate import fit
from servecast.core import formats, gpus
from servecast.model import geometry
from servecast.model import predict as P
from servecast.profile import probe as PR
from servecast.profile import probe_design as PD
from servecast.profile import probe_gpu as PG
from servecast.profile import probe_vllm as PV
from tests.mock_probe_engine import SimEngine

#: the engine of the paper's A100 probes: 9,965 KV blocks of 16 tokens, context 4,096, budget 2,048, 512 sequences
PAPER_A100 = (9965 * 16, 4096, 2048, 512)
CALIBRATION = "a100_sxm_40gb/d9cbf1d3/llama-3-8b"
LLM = "meta-llama/Meta-Llama-3-8B"


def run_probe(price: Any, design: dict[str, Any], clock_mhz: float = 1200.0,
              seed: int = 0) -> tuple[SimEngine, list[dict[str, Any]]]:
    eng = SimEngine(price, np.random.default_rng(seed), design["budget"], design["max_num_seqs"], clock_mhz)
    spans: list[dict[str, Any]] = []
    PR.drive(eng, design, lambda: len(eng.beats), spans)
    return eng, spans


@pytest.fixture(scope="module")
def probed() -> dict[str, Any]:
    """The probe of the paper's A100 engine, each step priced by the shipped calibration of Llama-3-8B."""
    calibration = P.load_calibration(CALIBRATION)
    design = PD.design(*PAPER_A100)
    eng, spans = run_probe(P.clock(calibration, geometry.resolve(LLM)), design)
    audit = PR.audit(eng.beats, spans, design, eng.clock, lock_mhz=1200.0)
    doc = PR.profiling_document(PR.steps(eng.beats, spans, eng.clock), gpu="a100_sxm_40gb", llm=LLM,
                                cost_law="d9cbf1d3", declared_mhz=1200.0)
    return {"calibration": calibration, "design": design, "eng": eng, "spans": spans, "audit": audit, "doc": doc}


def test_the_design_of_the_paper_engine() -> None:
    design = PD.design(*PAPER_A100)
    assert PD.n_cells(design) == {"decode": 40, "prefill": 6, "mixed": 132}
    assert len(design["order"]) == 72 and sum(o.startswith("reference") for o in design["order"]) == 4
    assert PD.gates(design) == []


def test_an_engine_too_small_for_the_probe_is_told_why() -> None:
    assert PD.gates(PD.design(kv_tokens=2000, max_model_len=1024, budget=512)) == [
        "only 20 mixed compositions fit this engine; the paper's calibrations had 113 or more"]
    why = PD.gates(PD.design(kv_tokens=159440, max_model_len=100))
    assert "no decode composition fits this engine" in why and len(why) >= 3


def test_the_probe_of_a_simulated_engine_is_clean_and_has_the_papers_cells(probed: dict[str, Any]) -> None:
    audit, doc = probed["audit"], probed["doc"]
    assert audit["clean"], audit["flags"]
    assert audit["compositions"] == 178 and audit["reference_drift"] < PR.DRIFT_MAX
    formats.ProfilingData.model_validate(doc)
    count: dict[str, int] = {}
    for c in doc["cells"]:
        count[c["cls"]] = count.get(c["cls"], 0) + 1
    # as in the shipped profiling data of the paper's A100 probes: 10 cells per decode composition, one per prefill
    # repeat, one per mixed injection
    assert count == {"decode": 400, "prefill": 48, "mixed": 528} and doc["reference_mhz"] == 1200.0
    assert abs(sum(c["w"] for c in doc["cells"]) - 178) < 1e-9      # each composition weighs 1


def test_a_fit_of_the_simulated_probe_gives_the_law_back(probed: dict[str, Any]) -> None:
    """The design's identifiability test: the probe's profiling data, fitted, return the law that priced them."""
    body = json.dumps(probed["doc"]).encode()
    fitted = fit.calibrate(probed["doc"], hashlib.sha256(body).hexdigest())
    want = probed["calibration"]["law"]["parameters"]
    got = fitted["law"]["parameters"]
    for name, value in want.items():
        assert abs(got[name] / value - 1) < 0.01, (name, got[name], value)


def test_a_drifting_engine_or_a_clock_off_its_lock_is_flagged(probed: dict[str, Any]) -> None:
    design, eng, spans = probed["design"], probed["eng"], probed["spans"]
    slower = [PR.Beat(b.prefill, b.decode, b.dur_ms * (1.0 + 0.1 * i / len(eng.beats)), b.t_ms)
              for i, b in enumerate(eng.beats)]
    assert any("reference drift" in f for f in PR.audit(slower, spans, design, eng.clock, lock_mhz=1200.0)["flags"])
    hot = [{**t, "sm_clock_mhz": 1100.0} if i % 2 else t for i, t in enumerate(eng.clock)]
    assert any("did not hold" in f for f in PR.audit(eng.beats, spans, design, hot, lock_mhz=1200.0)["flags"])
    assert any("aborted" in f for f in PR.audit(eng.beats, spans[:-3], design, eng.clock, lock_mhz=1200.0,
                                                aborted="ProbeError: x")["flags"])
    assert any("guessed" in f for f in PR.audit(eng.beats, spans, design, eng.clock, lock_mhz=None,
                                                degraded=True)["flags"])


def test_a_composition_is_kept_only_in_the_cell_made_for_it() -> None:
    decode = {"kind": "decode", "nd": 4}
    mixed = {"kind": "mixed", "nd": 4, "q": 128}
    assert PR.intended(decode, PR.Beat([], [[512, 3], [600, 1]], 1.0))
    assert not PR.intended(decode, PR.Beat([[64, 0]], [[512, 4]], 1.0))
    assert PR.intended(mixed, PR.Beat([[128, 0]], [[512, 4]], 1.0))
    assert not PR.intended(mixed, PR.Beat([[64, 0], [64, 64]], [[512, 4]], 1.0))
    assert PR.intended({"kind": "prefill", "q": 64}, PR.Beat([[64, 0]], [], 1.0))


def test_a_steps_clock_is_snapped_and_the_reference_is_the_lock_or_the_commonest() -> None:
    assert PR.step_clock([1199.0, 1201.0]) == 1200.0 and PR.step_clock([1376.0, 1380.0]) == 1375.0
    assert PR.step_clock([1200.0, 1300.0]) == 0.0 and PR.step_clock([]) == 0.0
    recs = [{"f_mhz": f} for f in (1200.0, 1200.0, 1410.0, 0.0)]
    assert PR.reference_clock(recs, 1200.0) == 1200.0 and PR.reference_clock(recs, 1300.0) == 0.0
    assert PR.reference_clock(recs, None) == 1200.0 and PR.reference_clock([], None) == 0.0


def test_the_cells_of_a_step_share_a_weight_of_one() -> None:
    beats = [PR.Beat([], [[100 + i, 2]], 10.0 + i) for i in range(25)]
    cells = PR.build_cells(PR.records([PR.Step("step000", beats, [1200.0])]))
    assert len(cells) == 10 and abs(sum(c["w"] for c in cells) - 1.0) < 1e-12
    assert [len(c["K"]) for c in cells] == [2, 3, 2, 3, 2, 3, 2, 3, 2, 3]
    assert cells[0]["K"] == [200.0, 202.0] and cells[0]["T"] == 10.5 and cells[0]["cls"] == "decode"


def _sched(new: list[tuple[str, int, int]], cached: list[tuple[str, int]], tokens: dict[str, int]) -> Any:
    return types.SimpleNamespace(
        num_scheduled_tokens=tokens,
        scheduled_new_reqs=[types.SimpleNamespace(req_id=r, prompt_token_ids=[0] * p, num_computed_tokens=c)
                            for r, p, c in new],
        scheduled_cached_reqs=types.SimpleNamespace(req_ids=[r for r, _ in cached],
                                                    num_computed_tokens=[c for _, c in cached]))


def test_a_steps_composition_is_read_from_vllms_scheduler_output() -> None:
    # a new request's first chunk, a cached one continuing its prompt, two decoding at context 700, one at 900
    sched = _sched([("a", 1000, 0)], [("b", 512), ("c", 700), ("d", 700), ("e", 900)],
                   {"a": 512, "b": 256, "c": 1, "d": 1, "e": 1})
    assert PV.composition(object(), sched) == ([[512, 0], [256, 512]], [[700, 2], [900, 1]], False)
    # a request whose state cannot be read is classified by its token count, and said so
    unknown = _sched([], [], {"z": 1, "y": 64})
    assert PV.composition(object(), unknown) == ([[64, 0]], [[0, 1]], True)


def test_the_recorder_keeps_the_timed_steps_and_samples_the_clock() -> None:
    clock = iter([1200.0, 1201.0, 1199.0])
    rec = PV.BeatRecorder(lambda: next(clock), warmup_beats=1, clock_every=2)
    for i in range(6):
        rec.begin([[64, 0]] if i != 3 else [], [], False)
        rec.finish(float(i))
    # the first step is warm-up, the empty fourth is left out: four beats, the clock read after the 1st and 3rd
    assert [b.dur_ms for b in rec.beats] == [1.0, 2.0, 4.0, 5.0]
    assert [(t["beat_index"], t["sm_clock_mhz"]) for t in rec.clock] == [(0, 1200.0), (2, 1201.0)]
    with pytest.raises(PR.ProbeError):
        rec.finish(1.0)


# the per-GPU law (c62ebe75) ---------------------------------------------------------------------------------

def shipped_per_gpu() -> dict[str, Any]:
    return json.loads(P.shipped_profiling()["a100_sxm_40gb/c62ebe75"].read_text(encoding="utf-8"))


BASE_CONFIG = {"architectures": ["LlamaForCausalLM"], "vocab_size": 32256, "max_position_embeddings": 16384,
               "hidden_size": 2048, "intermediate_size": 5504, "num_hidden_layers": 24, "num_attention_heads": 16,
               "num_key_value_heads": 16, "model_type": "llama"}


def test_every_synthetic_model_runs_on_the_papers_probe_engine_where_it_fits() -> None:
    """The engine all 26 of the paper's probes ran on: on the A100 it holds each of the 8 synthetic models and gives
    the design of the paper's A100 probes; a 24 GB GPU cannot hold the widest KV cache of the pool."""
    from servecast.core import deployment
    from servecast.core.manifest import ManifestError

    a100 = gpus.get("a100_sxm_40gb")
    for name in PG.POOL:
        d = PD.engine(a100, deployment.architecture_from_config(PG.synthetic_config(name, BASE_CONFIG)), name)
        assert (d.kv_blocks * d.block_size, d.context_length, d.max_batched_tokens, d.max_sequences) == PAPER_A100
        assert d.memory_fraction == 0.9
    widest = "synthetic/llama-l24-d2048-f5504-h16-kv16"
    with pytest.raises(ManifestError, match=r"at most [0-9]+ blocks fit"):
        PD.engine(gpus.get("rtx_3090_ti"), deployment.architecture_from_config(
            PG.synthetic_config(widest, BASE_CONFIG)), widest)


def test_a_synthetic_model_is_the_base_models_config_with_its_geometry() -> None:
    config = PG.synthetic_config("synthetic/llama-l36-d3072-f8256-h24-kv3", BASE_CONFIG)
    assert (config["num_hidden_layers"], config["hidden_size"], config["intermediate_size"],
            config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]) == (36, 3072, 8256, 24, 3,
                                                                                                128)
    assert config["vocab_size"] == 32256 and config["tie_word_embeddings"] is False
    assert [m["llm"] for m in shipped_per_gpu()["models"]] == list(PG.POOL)
    with pytest.raises(PR.ProbeError, match="head width"):
        PG.synthetic_geometry("synthetic/llama-l12-d2000-f5504-h16-kv2")
    with pytest.raises(PR.ProbeError, match="not the base model's"):
        PG.synthetic_config(PG.POOL[0], {**BASE_CONFIG, "vocab_size": 32000})


def test_the_grid_is_the_shipped_grids_shapes_and_flops() -> None:
    shapes = dict(PG.grid_shapes())
    assert len(shapes) == 24 and list(shapes)[:3] == list(PG.GRID_LLMS)
    assert "synthetic-d8192-f44288-k2048" in shapes and "synthetic-d1024-f2816-k256" in shapes
    for run in shipped_per_gpu()["gemm_grid"]["runs"]:
        for rep in run["reps"]:
            for point in rep:
                assert PG.layer_flops(shapes[run["model"]], point["n"]) == point["flops"], (run["model"], point["n"])


def test_the_grid_audit_passes_the_shipped_grid_and_flags_an_impossible_one() -> None:
    grid = shipped_per_gpu()["gemm_grid"]
    a100 = gpus.get("a100_sxm_40gb")
    peak = a100.peak_flops * grid["clock_mhz"] / a100.boost_sm_mhz
    assert PG.grid_flags(grid["runs"], peak) == []
    fast = [{"model": r["model"], "reps": [[{**p, "s_per_iter": p["s_per_iter"] / 3} for p in rep]
                                           for rep in r["reps"]]} for r in grid["runs"][:1]]
    assert any("of the GPU's peak" in f for f in PG.grid_flags(fast, peak))
    flat = [{"model": "x", "reps": [[{"n": n, "s_per_iter": 1e-3, "flops": 1.0} for n in PG.GRID_ROWS]]}]
    assert any("do not grow on a line" in f for f in PG.grid_flags(flat, peak))


def test_the_per_gpu_profiling_data_share_one_clock(probed: dict[str, Any]) -> None:
    steps = PR.steps(probed["eng"].beats, probed["spans"], probed["eng"].clock)
    runs = shipped_per_gpu()["gemm_grid"]["runs"]
    doc = PG.profiling_document([(PG.POOL[0], steps)], runs, gpu="a100_sxm_40gb", cost_law="c62ebe75",
                                declared_mhz=1200.0, grid_clock_mhz=1200.0)
    formats.ProfilingData.model_validate(doc)
    assert doc["models"][0]["llm"] == PG.POOL[0] and len(doc["gemm_grid"]["runs"]) == 24
    with pytest.raises(PR.ProbeError, match="the GEMM grid ran at 1375 MHz"):
        PG.profiling_document([(PG.POOL[0], steps)], runs, gpu="a100_sxm_40gb", cost_law="c62ebe75",
                              declared_mhz=1200.0, grid_clock_mhz=1375.0)
