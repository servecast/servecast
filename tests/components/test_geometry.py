"""The LLM geometry: the paper's six LLMs and synthetic names, field by field against the reference."""

from __future__ import annotations

import pytest

from servecast.model import geometry
from tests.components.inputs import load

G = load("geometry")


@pytest.mark.parametrize("name", sorted(G))
def test_geometry_is_the_reference_one(name: str) -> None:
    g = geometry.resolve(name)
    got = {k: getattr(g, k) for k in G[name]}
    assert got == G[name]


def test_a_config_json_gives_the_same_geometry(tmp_path) -> None:
    import json
    cfg = {"num_hidden_layers": 32, "hidden_size": 4096, "intermediate_size": 14336, "num_attention_heads": 32,
           "num_key_value_heads": 8, "vocab_size": 128256, "max_position_embeddings": 8192, "torch_dtype": "bfloat16"}
    d = tmp_path / "Meta-Llama-3-8B"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg))
    got, want = geometry.resolve(str(d)), geometry.KNOWN["meta-llama/Meta-Llama-3-8B"]
    assert {k: v for k, v in got.as_dict().items() if k != "name"} == {k: v for k, v in want.as_dict().items()
                                                                      if k != "name"}


@pytest.mark.parametrize("change, message", [({"num_local_experts": 8}, "mixture-of-experts"),
                                             ({"torch_dtype": "float8_e4m3fn"}, "16-bit")])
def test_what_the_laws_do_not_model_is_refused(change: dict, message: str) -> None:
    cfg = {"num_hidden_layers": 2, "hidden_size": 256, "intermediate_size": 512, "num_attention_heads": 2,
           "vocab_size": 1000, **change}
    with pytest.raises(ValueError, match=message):
        geometry.from_config("x", cfg)
