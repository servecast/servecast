"""An LLM's geometry: the counted inputs of the cost laws (layers, widths, heads, vocabulary), read from its
config.json.

Only dense decoder-only transformers with bf16 weights and a bf16 KV cache are modelled. The weights the beat reads
are counted as L [d (d + 2 d_kv) + d^2 + 3 d d_ff] (attention projections and a gated MLP), plus the vocabulary
matrix of the output layer, vocab x d, read on every beat.
"""

from __future__ import annotations

import dataclasses
import json
import re
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from servecast.core.formats import ModelRegistry
from servecast.core.paths import data_dir


@dataclasses.dataclass(frozen=True)
class Geometry:
    name: str
    n_layers: int
    d_model: int
    d_ff: int
    d_kv: int                       # per-layer KV width: KV heads x head size
    h_kv: int                       # KV heads
    n_heads: int                    # query heads
    vocab: int
    max_context: int
    bytes_per_elt: int = 2          # weights
    kv_bytes_per_elt: int = 2       # KV cache

    @property
    def kv_bytes_per_token(self) -> int:
        """KV-cache bytes per token of context: a key and a value of width d_kv per layer."""
        return 2 * self.kv_bytes_per_elt * self.n_layers * self.d_kv

    @property
    def gqa_group(self) -> int:
        """Query heads per KV head."""
        if self.n_heads % self.h_kv:
            raise ValueError(f"{self.name}: {self.n_heads} query heads are not a multiple of {self.h_kv} KV heads")
        return self.n_heads // self.h_kv

    @property
    def head_params(self) -> int:
        return self.vocab * self.d_model

    @property
    def head_bytes(self) -> float:
        return float(self.bytes_per_elt * self.head_params)

    @property
    def w_params(self) -> int:
        return self.n_layers * (self.d_model * (self.d_model + 2 * self.d_kv) + self.d_model * self.d_model
                                + 3 * self.d_model * self.d_ff)

    @property
    def w_bytes(self) -> float:
        return float(self.bytes_per_elt * self.w_params)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


#: a synthetic model declares its geometry in its name: synthetic/<family>-l<layers>-d<width>-f<ffn>-h<heads>-kv<kv>.
#: Its vocabulary and context are its family's base model's; the head size is 128.
SYNTHETIC = re.compile(r"synthetic/(?P<family>[a-z0-9]+)-l(?P<l>\d+)-d(?P<d>\d+)-f(?P<f>\d+)-h(?P<h>\d+)-kv(?P<kv>\d+)")
SYNTHETIC_FAMILIES = {"llama": {"vocab": 32256, "max_context": 16384}}
HEAD_DIM = 128

#: config.json fields read, by name; anything else in the file is ignored
CONFIG_FIELDS = ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads",
                 "num_key_value_heads", "vocab_size", "max_position_embeddings", "head_dim", "torch_dtype")


def synthetic(name: str) -> Geometry:
    m = SYNTHETIC.fullmatch(name)
    if not m:
        raise ValueError(f"{name!r} is not a synthetic model name: synthetic/<family>-l<L>-d<D>-f<F>-h<H>-kv<KV>")
    fam = SYNTHETIC_FAMILIES.get(m["family"])
    if fam is None:
        raise ValueError(f"{name}: unknown family {m['family']!r}; known: {sorted(SYNTHETIC_FAMILIES)}")
    layers, d, f, h, kv = (int(m[k]) for k in ("l", "d", "f", "h", "kv"))
    if d != h * HEAD_DIM or h % kv:
        raise ValueError(f"{name}: the head size must be {HEAD_DIM} and kv must divide h")
    return Geometry(name, n_layers=layers, d_model=d, d_ff=f, d_kv=kv * HEAD_DIM, h_kv=kv, n_heads=h,
                    vocab=int(fam["vocab"]), max_context=int(fam["max_context"]))


#: one synthetic transformer layer of a GEMM grid: width d, FFN width f, KV width k (head size 128, no vocabulary)
LAYER = re.compile(r"synthetic-d(\d+)-f(\d+)-k(\d+)")


def layer(name: str) -> Geometry:
    """The shapes of a GEMM grid's synthetic layer, `synthetic-d<d>-f<d_ff>-k<d_kv>`: one layer, no vocabulary."""
    m = LAYER.fullmatch(name)
    if not m:
        raise ValueError(f"{name!r} is not a synthetic layer: synthetic-d<d>-f<d_ff>-k<d_kv>")
    d, dff, dkv = (int(x) for x in m.groups())
    if d % HEAD_DIM or dkv % HEAD_DIM or dff % HEAD_DIM or not 0 < dkv <= d:
        raise ValueError(f"{name}: d, d_ff and d_kv must be multiples of {HEAD_DIM}, with d_kv <= d")
    return Geometry(name, n_layers=1, d_model=d, d_ff=dff, d_kv=dkv, h_kv=dkv // HEAD_DIM, n_heads=d // HEAD_DIM,
                    vocab=0, max_context=4096)


def from_config(name: str, config: dict[str, Any]) -> Geometry:
    """The geometry of a dense LLM from its config.json. Refuses what the cost laws do not model."""
    missing = [k for k in ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads",
                           "vocab_size") if k not in config]
    if missing:
        raise ValueError(f"{name}: config.json lacks {', '.join(missing)}")
    for key in ("num_experts", "num_local_experts", "n_routed_experts"):
        if config.get(key):
            raise ValueError(f"{name}: a mixture-of-experts model is not supported")
    dtype = str(config.get("torch_dtype") or "bfloat16")
    if dtype not in ("bfloat16", "float16"):
        raise ValueError(f"{name}: weights in {dtype}; only 16-bit weights are modelled")
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads") or heads)
    head_dim = int(config.get("head_dim") or int(config["hidden_size"]) // heads)
    return Geometry(name, n_layers=int(config["num_hidden_layers"]), d_model=int(config["hidden_size"]),
                    d_ff=int(config["intermediate_size"]), d_kv=kv_heads * head_dim, h_kv=kv_heads, n_heads=heads,
                    vocab=int(config["vocab_size"]), max_context=int(config.get("max_position_embeddings") or 0))


def _read_models(path: Path) -> dict[str, dict[str, Any]]:
    """The entries of a models.json, each as the config.json fields it holds (all of them, so the refusals of a
    config.json see what they need); ValueError names the file and the problem."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        ModelRegistry.model_validate(doc)
    except (OSError, ValueError, ValidationError) as error:
        raise ValueError(f"{path}: not a model registry: {error}") from None
    return {str(name): dict(entry) for name, entry in doc["models"].items()}


#: the LLMs whose calibrations ship (data/registry/models.json): each entry holds the config.json fields the cost
#: laws read, and is read as a config.json is
KNOWN: dict[str, Geometry] = {
    name: from_config(name, entry)
    for name, entry in _read_models(Path(str(resources.files("servecast").joinpath("data", "registry",
                                                                                  "models.json")))).items()}


def user_models() -> Path:
    return data_dir() / "models.json"


def known() -> dict[str, Geometry]:
    """The shipped LLMs, then yours (models.json in your data directory): yours go through the same reader as a
    config.json, with its refusals, and cannot change a shipped one."""
    path = user_models()
    if not path.is_file():
        return dict(KNOWN)
    from servecast.core import deployment  # only for a file of yours: its refusals are those of a config.json

    yours = {}
    for name, entry in _read_models(path).items():
        if name in KNOWN:
            raise ValueError(f"{path}: {name} ships with ServeCast and cannot be changed; give your LLM another name")
        try:
            deployment.architecture_from_config(entry)
            yours[name] = from_config(name, entry)
        except ValueError as error:
            raise ValueError(f"{path}: {name}: {error}") from None
    return {**KNOWN, **yours}


def resolve(model: str) -> Geometry:
    """A Hugging Face id of a known LLM, a synthetic model's name, or a local directory holding config.json."""
    llms = known()
    if model in llms:
        return llms[model]
    if model.startswith("synthetic/"):
        return synthetic(model)
    path = Path(model)
    cfg = path / "config.json" if path.is_dir() else path
    if cfg.is_file():
        return from_config(path.name if path.is_dir() else path.parent.name, json.loads(cfg.read_text("utf-8")))
    raise ValueError(f"no geometry for {model!r}: give a local directory holding its config.json "
                     f"(known without one: {', '.join(sorted(llms))})")
