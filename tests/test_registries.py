"""The registries ServeCast reads its datasheet and architecture numbers from: the shipped files keep the numbers the
shipped calibrations were fitted with, and a file of your own adds entries without changing a shipped one."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from servecast.core import formats, gpus
from servecast.model import geometry

#: the numbers core/gpus.py held in code before they moved to data/registry/gpus.json: every shipped calibration was
#: fitted with them, so they must not move by a single bit
GPUS_BEFORE = {
    "a100_sxm_40gb": ("NVIDIA A100-SXM4-40GB", 40, 108, 312e12, 1.555e12, 1410),
    "h100_sxm": ("NVIDIA H100 80GB HBM3", 80, 132, 989.5e12, 3.35e12, 1830),
    "rtx_3090_ti": ("NVIDIA GeForce RTX 3090 Ti", 24, 84, 80e12, 1.008e12, 1860),
}


def test_the_shipped_gpus_keep_their_numbers() -> None:
    assert {g.id: (g.device_name, g.memory_gib, g.n_sm, g.peak_flops, g.bandwidth, g.boost_sm_mhz)
            for g in gpus.SHIPPED.values()} == GPUS_BEFORE


def _yours(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entries: dict) -> None:
    monkeypatch.setenv("SERVECAST_DATA_DIR", str(tmp_path))
    (tmp_path / "gpus.json").write_text(json.dumps({"schema": "servecast-gpus/1", "gpus": entries}), encoding="utf-8")


L40S = {"device_name": "NVIDIA L40S", "memory_gib": 48, "sm_count": 142, "peak_flop_per_s": 362e12,
        "memory_bandwidth_bytes_per_s": 0.864e12, "boost_clock_mhz": 2520}


def test_your_gpus_json_adds_a_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _yours(tmp_path, monkeypatch, {"l40s": L40S})
    assert gpus.get("l40s").n_sm == 142 and gpus.get("a100_sxm_40gb").n_sm == 108


def test_your_gpus_json_cannot_change_a_shipped_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _yours(tmp_path, monkeypatch, {"a100_sxm_40gb": L40S})
    with pytest.raises(ValueError, match="ship with ServeCast"):
        gpus.get("a100_sxm_40gb")


@pytest.mark.parametrize("entry", [{**L40S, "sm_count": 0}, {k: v for k, v in L40S.items() if k != "memory_gib"}])
def test_a_gpu_entry_that_is_not_one_is_refused_naming_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                entry: dict) -> None:
    _yours(tmp_path, monkeypatch, {"l40s": entry})
    with pytest.raises(ValueError, match=r"gpus\.json: not a GPU registry"):
        gpus.get("l40s")


#: the geometries model/geometry.py held in code before they moved to data/registry/models.json
MODELS_BEFORE = {
    "meta-llama/Meta-Llama-3-8B": (32, 4096, 14336, 1024, 8, 32, 128256, 8192),
    "meta-llama/Llama-3.2-3B": (28, 3072, 8192, 1024, 8, 24, 128256, 131072),
    "Qwen/Qwen2.5-7B": (28, 3584, 18944, 512, 4, 28, 152064, 131072),
    "Qwen/Qwen2.5-3B": (36, 2048, 11008, 256, 2, 16, 151936, 32768),
    "deepseek-ai/deepseek-coder-1.3b-base": (24, 2048, 5504, 2048, 16, 16, 32256, 16384),
    "deepseek-ai/deepseek-llm-7b-base": (30, 4096, 11008, 4096, 32, 32, 102400, 4096),
}


def test_the_shipped_models_keep_their_geometry() -> None:
    assert {g.name: (g.n_layers, g.d_model, g.d_ff, g.d_kv, g.h_kv, g.n_heads, g.vocab, g.max_context)
            for g in geometry.KNOWN.values()} == MODELS_BEFORE
    assert all(g.bytes_per_elt == 2 and g.kv_bytes_per_elt == 2 for g in geometry.KNOWN.values())


def test_the_shipped_files_are_valid_registries() -> None:
    root = Path(gpus.__file__).parents[1] / "data" / "registry"
    formats.GpuRegistry.model_validate(json.loads((root / "gpus.json").read_text(encoding="utf-8")))
    formats.ModelRegistry.model_validate(json.loads((root / "models.json").read_text(encoding="utf-8")))


SMALL = {"num_hidden_layers": 16, "hidden_size": 2048, "intermediate_size": 8192, "num_attention_heads": 16,
         "num_key_value_heads": 8, "vocab_size": 32000, "max_position_embeddings": 4096}


def _models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entries: dict) -> None:
    monkeypatch.setenv("SERVECAST_DATA_DIR", str(tmp_path))
    (tmp_path / "models.json").write_text(json.dumps({"schema": "servecast-models/1", "models": entries}),
                                          encoding="utf-8")


def test_your_models_json_adds_an_llm_known_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _models(tmp_path, monkeypatch, {"my-org/small": SMALL})
    g = geometry.resolve("my-org/small")
    assert (g.n_layers, g.d_kv) == (16, 1024) and "Qwen/Qwen2.5-7B" in geometry.known()


@pytest.mark.parametrize(("entries", "refusal"), [
    ({"Qwen/Qwen2.5-7B": SMALL}, "ships with ServeCast"),
    ({"my-org/moe": {**SMALL, "num_local_experts": 8}}, "mixture-of-experts"),
    ({"my-org/neox": {**SMALL, "model_type": "gpt_neox"}}, "feed-forward"),
])
def test_your_models_json_is_refused_where_a_config_json_would_be(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                  entries: dict, refusal: str) -> None:
    _models(tmp_path, monkeypatch, entries)
    with pytest.raises(ValueError, match=refusal):
        geometry.known()


def test_predict_finds_an_llm_of_your_models_json_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from servecast.model import run

    _models(tmp_path, monkeypatch, {"my-org/small": SMALL})
    assert run.resolve_llm("my-org/small", offline=True).n_layers == 16


def test_a_broken_models_json_is_an_input_error_naming_the_file(tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    from servecast.core.manifest import ManifestError
    from servecast.model import run

    monkeypatch.setenv("SERVECAST_DATA_DIR", str(tmp_path))
    (tmp_path / "models.json").write_text("{", encoding="utf-8")
    with pytest.raises(ManifestError, match=r"engine\.model: .*models\.json"):
        run.resolve_llm("Qwen/Qwen2.5-7B", offline=True)
