"""Trained checkpoints: a run directory or a Hub repo holding a LoRA adapter (or, for a full-weight run, the whole bf16
backbone), `head.pt` and the tokenizer.

Loader rule: `adapter_config.json` present -> a LoRA adapter on `meta.base` at `meta.base_revision`; no adapter and
`config.json` + `model*.safetensors` (save_pretrained of the backbone, `meta.weights == "full"`) -> the backbone is loaded from
the checkpoint directory itself, nothing is merged, in the dtype head.pt's `weights_dtype` names (it must match the `dtype`
save_pretrained wrote to config.json). The tokenizer always comes from the base (both layouts carry a copy).

This is the one place that knows the layout of `head.pt` and how a checkpoint becomes a `DecisionModel`:
`kev.serve`, `kev.benchmark`, `kev.train --init_from`, `kev.publish`, the scripts and the Hugging Face Space all go
through it. The Space vendors this file next to `model.py` and `api.py` (scripts/publish_space.sh), so it must not
import the data or suite modules at import time.

    ck = Checkpoint("jaredpalmer/kev-4b")          # or a local run directory; `@tag` pins a Hub revision
    tok, model = ck.load("mps", LoadOptions.from_env())
    ck.meta.temperature                             # the calibration the checkpoint carries
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .model import DecisionModel, load_tokenizer

from .ids import is_hub_id   # vendored here: no torch needed to answer a question about a name


def resolve_run(run):
    """Local run directory as given, or a Hub repo id like jaredpalmer/kev-4b, optionally pinned to a revision or tag
    with `@` (jaredpalmer/kev-4b@qwen3), downloaded to the HF cache. Returns a str path."""
    if os.path.isdir(run):
        return str(run)
    from huggingface_hub import snapshot_download
    repo, _, revision = str(run).partition("@")
    pats = ["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"]
    try:   # the cache first: a load must not ask the network which revision is current (inventio: `update` does)
        return snapshot_download(repo, revision=revision or None, allow_patterns=pats, local_files_only=True)
    except Exception:   # not downloaded yet: the first load fetches it
        return snapshot_download(repo, revision=revision or None, allow_patterns=pats)


@dataclass
class Meta:
    """Contents of `head.pt`. Every reader gets the same defaults for fields older checkpoints did not write.
    `extra` keeps the rest of the file (training args, suite hash, init provenance, temperature fit) so a
    read-modify-write round trip loses nothing."""
    base: str
    head: dict | None = None
    base_revision: str | None = None
    lora: int = 0
    head_dim: int = 256
    option_isolation: bool = False
    special_embeddings: bool = False
    weights_dtype: str = "fp32"
    temperature: float = 1.0
    holdout: list = field(default_factory=list)
    weights: str = "lora"          # "lora": an adapter on the base; "full": the whole backbone is in the checkpoint (kev.train --full_ft)
    extra: dict = field(default_factory=dict)

    KNOWN = ("base", "head", "base_revision", "lora", "head_dim", "option_isolation", "special_embeddings", "weights_dtype", "temperature", "holdout", "weights")

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in cls.KNOWN if k in d}, extra={k: v for k, v in d.items() if k not in cls.KNOWN})

    def to_dict(self):
        return {**self.extra, **{k: getattr(self, k) for k in self.KNOWN}}   # known fields win over a stray key in extra


def read_meta(run):
    return Meta.from_dict(torch.load(f"{run}/head.pt", map_location="cpu"))


def write_meta(run, meta):
    torch.save(meta.to_dict(), f"{run}/head.pt")


@dataclass(frozen=True)
class LoadOptions:
    """How a checkpoint is turned into a model. Defaults are the exact path every reported number uses.

    dtype        None = fp32, the exact path every reported number uses (bf16 when the checkpoint was trained with a bf16
                 backbone). Serving defaults to bf16 on CUDA and MPS instead: half the memory, 2-4.5x lower latency on an
                 L4 (Kev-4B: 209 -> 118 ms at 101 tokens, 850 -> 189 ms at 330 tokens), probabilities within ~0.01 and
                 the same argmax on the checks run so far. The exact path is INVENTIO_SYSTEMONE_DTYPE=fp32.
    merge        fold the LoRA into the base weights: the delta is computed from the fp32 adapter and added in fp32 with
                 one rounding to the load dtype, so a bf16 model holds exactly round(W + delta), the same bits as merging
                 an fp32 copy and casting, without the fp32 copy (Kev-9B needed 36 GB of GPU memory to load for that).
                 Identical because the Qwen bases are stored in bf16; a base stored in fp32 would be rounded twice.
                 Exact in fp32; in bf16 it is faster (~15%) and closer to the fp32 numbers than the unmerged adapter
                 (kev-4b, 24 dev records: max |dp| 0.017 vs 0.029, 0 vs 1 argmax flips). Ignored for adapters that carry
                 trained token embeddings. A checkpoint trained on a bf16 backbone keeps its adapter unmerged, as it was
                 trained (the fused serving path that folded it is not vendored).
    attn         attention backend; None = the model default (SDPA on CUDA, eager elsewhere). "sdpa" on MPS measured
                 parity with eager and is a few percent faster.
    lora_scale   WiSE-FT-style interpolation between base (0) and fine-tuned weights (1), at inference.
    temperature  None = the temperature the checkpoint carries (fitted by scripts/calibrate_checkpoint.py); 1.0 = raw logits.
    """
    dtype: torch.dtype | None = None
    merge: bool = True
    attn: str | None = None
    lora_scale: float = 1.0
    temperature: float | None = None



class Checkpoint:
    def __init__(self, run):
        self.requested = str(run)                    # what the caller asked for (a Hub id stays a Hub id in labels)
        self.path = resolve_run(run)
        self.meta = read_meta(self.path)

    def file(self, name):
        return Path(self.path) / name

    def adapter_config(self):
        return json.loads(self.file("adapter_config.json").read_text(encoding="utf-8"))

    @property
    def full(self):
        """The loader rule (module docstring): True for a full-weight checkpoint, False for a LoRA adapter; head.pt's
        `weights` must agree with the files."""
        found = "lora" if self.file("adapter_config.json").exists() else "full" if self.file("config.json").exists() and self.shards() else None
        if found != self.meta.weights:
            raise ValueError(f"{self.path}: head.pt says weights={self.meta.weights!r} but the directory holds "
                             f"{ {'lora': 'an adapter', 'full': 'backbone weights'}.get(found, 'neither an adapter nor backbone weights') }")
        return found == "full"

    def shards(self):
        """The backbone's safetensors files of a full-weight checkpoint (model.safetensors or model-*-of-*.safetensors)."""
        return sorted(Path(self.path).glob("model*.safetensors"))

    def load(self, device, opts=LoadOptions()):
        """-> (tokenizer, model) in eval mode with the LoRA applied (or the full backbone loaded) and the pointer head
        loaded. The model is a DecisionModel, ready for encode()/probs()."""
        meta = self.meta
        # a checkpoint that carries its tokenizer is read from itself: a published full-weight model then loads with
        # no second repository and no network (inventio's offline promise); otherwise the base's, as Kev does
        own = self.file("tokenizer.json").exists() and self.full
        tok = load_tokenizer(self.path) if own else load_tokenizer(meta.base, revision=meta.base_revision)
        m = self._load_torch(tok, device, opts)
        m.head.load_state_dict(meta.head); m.eval()
        m.head.temperature = meta.temperature if opts.temperature is None else opts.temperature
        return tok, m

    def _load_torch(self, tok, device, opts):
        return self._full_torch(tok, device, opts)[0] if self.full else self._adapted_torch(tok, device, opts)[0]

    SAVED_DTYPES = {"bf16": "bfloat16", "fp32": "float32"}   # head.pt weights_dtype -> the dtype save_pretrained writes to config.json

    def _full_torch(self, tok, device, opts):
        """-> (model, True). Full weights load in the dtype head.pt's `weights_dtype` names (bf16 for every kev.train
        --full_ft run: the dtype they were trained in), which must be the dtype save_pretrained recorded in config.json;
        otherwise a mislabelled export would be silently cast (fp32 weights rounded to bf16, or bf16 upcast to twice the
        memory). An explicit dtype still casts on purpose (fp32: the same values computed in fp32). Nothing to merge."""
        if opts.lora_scale != 1: raise ValueError("lora_scale interpolates an adapter; a full-weight checkpoint has none")
        meta = self.meta
        cfg = json.loads(self.file("config.json").read_text(encoding="utf-8"))
        expected, saved = self.SAVED_DTYPES.get(meta.weights_dtype), cfg.get("dtype") or cfg.get("torch_dtype")
        if expected is None or saved not in (None, expected):
            raise ValueError(f"{self.path}: config.json records the weights as {saved} but head.pt says weights_dtype={meta.weights_dtype!r}")
        return DecisionModel(meta.base, tok, device, head_dim=meta.head_dim, option_isolation=meta.option_isolation,
                             dtype=opts.dtype or getattr(torch, expected), attn=opts.attn, weights=self.path), True

    def _adapted_torch(self, tok, device, opts):
        """-> (model, whether the adapter was merged): the base with this checkpoint's LoRA."""
        from peft import PeftModel
        meta = self.meta
        dtype, merge = opts.dtype or torch.float32, opts.merge
        if meta.weights_dtype == "bf16":
            # trained with a bf16 backbone: load it the same way. The exact path keeps the fp32 adapter unmerged.
            dtype, merge = torch.bfloat16, merge
        merge = merge and not self.adapter_config().get("trainable_token_indices")   # token-trained adapters stay unmerged
        m = DecisionModel(meta.base, tok, device, lora=None, revision=meta.base_revision, head_dim=meta.head_dim,
                          option_isolation=meta.option_isolation, dtype=dtype, attn=opts.attn)
        m.lm = PeftModel.from_pretrained(m.lm, self.path, torch_device=str(device)).to(device)   # trainable token embeddings, if any, live in the adapter
        if opts.lora_scale != 1:
            for module in m.lm.modules():
                if isinstance(getattr(module, "scaling", None), dict):
                    for k in module.scaling: module.scaling[k] *= opts.lora_scale
            m.lora_scale = opts.lora_scale
        if merge: m.lm = m.lm.merge_and_unload()     # W += delta: fp32 math, one rounding (see LoadOptions.merge)
        if dtype != torch.float32: m.lm = m.lm.to(dtype)
        return m, merge

    COMPAT_FIELDS = ("base", "base_revision", "lora", "head_dim", "option_isolation", "special_embeddings", "weights")

def load(run, device, opts=LoadOptions()):
    """Convenience: Checkpoint(run).load(device, opts)."""
    return Checkpoint(run).load(device, opts)
