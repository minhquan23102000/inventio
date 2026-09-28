"""The System One training recipe's pinned trainer flags, in one place: `systemone.py train` (this laptop)
and `modal_train.py` (Modal) both build Kev's command from here, so a run is the same recipe wherever
it runs. No imports beyond the standard library: the Modal client reads it without inventio's
dependencies."""

# the Kev checkpoint we fine-tune from, and the base it was trained on (its own training_config.json)
INIT_FROM = "jaredpalmer/kev-0.8b"
BASE = "Qwen/Qwen3.5-0.8B-Base"
BASE_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"


STEP_RECORDS = 8   # records per optimizer step; how they split into micro-batches is the machine's business


def trainer_args(records: str, out: str, *, max_state: int, lr: float, epochs: int, seed: int,
                 row_budget: int = 0, max_steps: int = 0, batch: int = 1, init_from: str = INIT_FROM) -> list[str]:
    """The arguments after `kev_win.py train`. bf16 weights and compute, gradient checkpointing, the shared
    state prefix, 8 records per optimizer step; no none-option or distractor augmentation, since the options
    are exhaustive passage and line ids. `batch` records per forward pass (1 on the 8 GB laptop, more on a
    big card): Kev divides every step's loss by its record count, so the gradient is the same up to
    summation order, and `--length_sort` deals a step's records into micro-batches that pad little."""
    if STEP_RECORDS % batch:
        raise ValueError(f"batch {batch} does not divide {STEP_RECORDS} records per step")
    args = ["--data", records, "--init_from", init_from, "--base", BASE, "--base_revision", BASE_REVISION,
            "--max_state", str(max_state), "--device", "cuda", "--dtype", "bf16", "--weights_dtype", "bf16",
            "--checkpointing", "1", "--shared_prefix", "1", "--batch", str(batch), "--accum", str(STEP_RECORDS // batch),
            "--lr", str(lr), "--epochs", str(epochs), "--seed", str(seed),
            "--p_none", "0", "--p_none_distract", "0", "--p_distract", "0",
            "--row_budget", str(row_budget), "--out", out]
    if batch > 1:
        args += ["--length_sort", "1"]
    if max_steps:
        args += ["--max_steps", str(max_steps)]
    return args
