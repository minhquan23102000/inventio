"""Kev's serving path, vendored so a System One checkpoint needs one install and no second repository.

Source: https://github.com/jaredpalmer/kev at f1535963cea021439370c23127bc970b6788e730 (2026-09-26),
Apache-2.0, (c) the Kev authors. Copied: `model.py`, `checkpoint.py`, `api.py`, `device.py` — the four
files `kev.serve` turns a checkpoint into answered questions with (Kev's own Hugging Face Space vendors
the first three next to each other, so the layout is Kev's, not ours). Everything a serving pass
touches is their code unchanged: the packed block-causal encode, the pointer head and its temperature,
the readouts, the request/answer schema.

Changed here, and nothing else:

- `api.py`: the option cap is 512, Kev ships 255 (a code pool's median line count is 407; see
  `inventio/systemone.py` for what 255 cost on SWE-bench Lite).
- `model.py`: the CUDA-graph serving batch path (`kev.cuda_graphs`, `EAGER_STATES`) is gone;
  `probs_batch` runs Kev's eager path, the one every measured number in this repository used.
- `checkpoint.py`: the trainer-only and other-backend parts are gone — `mlx_available`,
  `fused_available`, `backend`, `_load_mlx`, `warm_start`, `_load_backbone_into`,
  `_load_adapter_into`, `weights_sha256`, `release_date`, `hybrid_base`, `LoadOptions.from_env` and
  the `backend`/`cuda_graphs`/`fused` fields. `_load_torch` is the torch path alone. `resolve_run` reads a
  Hub checkpoint from the cache when it is there instead of asking the Hub for the current revision on every
  load (inventio's release check is `inventio update`'s, once a day). `Checkpoint.load` reads a full-weight
  checkpoint's own tokenizer when it carries one, so the published model needs neither the base repo nor the network.
- `device.py`: as published.
- This file: `load()` (the device and dtype defaults `kev.serve` serves with, the Windows attention
  fallback, and the dependency check that names the missing extra instead of raising an ImportError
  from inside a forward pass).

`load()` is the only entry point inventio uses; a checkpoint answers to any model name, so what
identifies weights is the run directory or Hub id it was loaded from.
"""

import os
import re

__all__ = ["Checkpoint", "LoadOptions", "REQUIRES", "is_hub_id", "load", "missing", "patch_attention"]

# What a checkpoint needs, and what an environment without it has to be told. `transformers>=5.17` is
# the version the hybrid (Qwen3.5) cache classes came from; an older one imports and then fails inside
# the first forward pass, which is a worse error than this one.
REQUIRES = ("torch>=2.6", "transformers>=5.17,<6", "peft>=0.21", "pydantic>=2.9")


def missing() -> str | None:
    """What this environment still has to install to load a checkpoint, or None when it can.

    Reads installed distributions, never imports them: `default_ranker()` asks this while the command line
    is being built, and importing torch there would make every `inventio` command pay for a model runtime
    it may not use. A package that answers here and then fails to import is caught at load time, where the
    same message comes back.
    """
    import importlib.util
    from importlib.metadata import version

    for name, need in (("torch", "2.6"), ("transformers", "5.17"), ("peft", "0.21"), ("pydantic", "2.9")):
        if importlib.util.find_spec(name) is None:
            return f"{name} is not installed — pip install 'inventio[systemone]'"
    try:
        if tuple(int(x) for x in re.findall(r"\d+", version("transformers"))[:2]) < (5, 17):
            return (f"transformers {version('transformers')} is older than 5.17 (the hybrid cache classes "
                    f"the encoder uses) — pip install -U 'inventio[systemone]'")
    except Exception:   # a distribution without version metadata: let the import speak
        pass
    return None


def patch_attention() -> None:
    """Windows: torch's SDPA GQA path falls back to the math kernel for the hybrid (Qwen3.5) bases, which
    holds the whole L x L score matrix — out of memory past ~8k tokens, twice the time at 3k. Repeating
    the kv heads instead lets SDPA take its memory-efficient kernel; measured on this machine as the
    same numbers, a different kernel (benchmarks/kev_win.py, where the patch was found)."""
    if os.name == "nt":
        import transformers.integrations.sdpa_attention as sdpa

        sdpa.use_gqa_in_sdpa = lambda *a, **k: False


def load(run, device=None, *, dtype=None, merge: bool = True, attn: str | None = None,
         lora_scale: float = 1.0, temperature: float | None = None):
    """A checkpoint as `kev.serve` loads it -> (tokenizer, model).

    What `kev.serve` serves with: bf16 on an accelerator (half the memory; probabilities within ~0.01 of
    the exact path, the same argmax), fp32 on CPU; a checkpoint trained on a bf16 backbone loads in bf16
    wherever it runs, as it was trained. INVENTIO_SYSTEMONE_DTYPE=fp32 asks for the exact path — the one
    every number in this repository was reported from — which costs about twice the wall time.
    INVENTIO_SYSTEMONE_DEVICE=cpu|cuda|mps leaves the accelerator to something else.
    """
    from .device import default_device

    dev = str(device or os.environ.get("INVENTIO_SYSTEMONE_DEVICE") or default_device())
    if dev == "cpu":
        # the hybrid (Qwen3.5) DeltaNet layers take a compiled kernel from the Hugging Face kernel hub when
        # one is installed, and that kernel needs a GPU: on CPU it raises inside a Triton launch. The
        # pure-torch fallback is the only path that runs there, so ask for it before transformers is
        # imported (the flag is read at its import time).
        os.environ.setdefault("USE_HUB_KERNELS", "NO")
    miss = missing()
    if miss:
        raise RuntimeError(miss)
    from .checkpoint import Checkpoint, LoadOptions

    import torch

    patch_attention()
    device = dev
    if dtype is None and device != "cpu" and os.environ.get("INVENTIO_SYSTEMONE_DTYPE", "").lower() != "fp32":
        dtype = torch.bfloat16
    opts = LoadOptions(dtype=dtype, merge=merge, attn=attn, lora_scale=lora_scale, temperature=temperature)
    return Checkpoint(run).load(device, opts)


from .ids import is_hub_id   # noqa: E402  (torch-free: `inventio systemone --use` validates a name without the extra)


def __getattr__(name):   # PEP 562: Checkpoint/LoadOptions/resolve_run pull torch in, so they load on first use
    if name in ("Checkpoint", "LoadOptions", "resolve_run"):
        from . import checkpoint

        return getattr(checkpoint, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
