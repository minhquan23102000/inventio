"""Run Kev's server or trainer on Windows, where torch's wheels have no flash-attention kernel and
kev.train imports two Unix-only modules. Two substitutions, neither of which changes a number:

- SDPA's GQA path (`use_gqa_in_sdpa`) falls back to the math kernel, which holds the whole L x L score
  matrix: OOM on an 8 GB card past ~8k tokens, twice the time at 3k. Repeating the kv heads instead
  lets SDPA take the memory-efficient kernel.
- `resource` (peak RSS in training_metrics.json) and `fcntl` (suite file locks, unused by a --data run)
  do not exist on Windows.

    python benchmarks/kev_win.py serve --run runs/s1-v1 --port 8008
    python benchmarks/kev_win.py train --data <records.jsonl> --init_from jaredpalmer/kev-0.8b ...

Run it with the Kev environment's interpreter (`C:/Users/LEGION/kev/.venv/Scripts/python.exe`).
"""
import sys
import types

sys.modules.setdefault("resource", types.SimpleNamespace(
    RUSAGE_SELF=0, getrusage=lambda _: types.SimpleNamespace(ru_maxrss=0)))
sys.modules.setdefault("fcntl", types.SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=lambda *a: None))

import transformers.integrations.sdpa_attention as sdpa  # noqa: E402

sdpa.use_gqa_in_sdpa = lambda *a, **k: False

# `where_line` is a choice over every line of the 15 passages, and kev/api.py caps a choice at 255 options
# (`Choice._check` reads that module global at validation time; its pointer head has no cap of its own). Code
# pools have a median 407 lines, so 255 loses 1,279 of the 1,386 SWE issues whose answer BM25 already found.
# Raising it changes no number, only how many options a question may offer; the served context's branch limit
# (SERVE_MAX_BRANCH, 73,728 tokens) takes a 512-key branch without noticing, and training's `--max_state 6656`
# lifts max_branch to 7,296.
import kev.api as _api  # noqa: E402

_api.MAX_OPTIONS = 512

if len(sys.argv) > 1 and sys.argv[1] == "fits":
    # `fits --data <records.jsonl> --base <hf id> --base_revision <sha> --max_state N --out <file>`:
    # which records the trainer's own admission rule keeps, and the file it would then train on. Run this
    # before drawing a balanced subset: `train --bal` draws from the file, so a draw made first would admit
    # records only for the trainer to drop them, and the composition it prints would be a lie (measured
    # 2026-09-26: at `--max_state 6656` the code arm keeps 429 of 5,308 records, at 8192 1,000).
    import argparse
    import json
    from collections import Counter
    from pathlib import Path

    from kev.data import load_records
    from kev.model import fits, load_tokenizer, training_context
    from kev.train import materialize

    ap = argparse.ArgumentParser(prog="kev_win.py fits")
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--base_revision", default="")
    ap.add_argument("--max_state", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(sys.argv[2:])
    tok = load_tokenizer(a.base, revision=a.base_revision or None)
    c = training_context(a.max_state)
    reqs = load_records(a.data)
    kept = [r for r in reqs if fits(materialize(r), tok, **c)]
    by = lambda rows: Counter((r["_meta"].get("source", "?"), r["_meta"].get("arm", "?")) for r in rows)  # noqa: E731
    print(f"{a.data}: {len(kept)} of {len(reqs)} records fit "
          f"({c['max_state']} state / {c['max_branch']} branch / {c['max_packed']} packed tokens)")
    for k in sorted(set(by(reqs)) | set(by(kept))):
        print(f"  {k[0]:12s} {k[1]:4s} kept {by(kept)[k]:5d}  dropped {by(reqs)[k] - by(kept)[k]:5d}")
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    sys.exit(0)

if len(sys.argv) < 2 or sys.argv[1] not in ("serve", "train"):
    raise SystemExit(__doc__.strip().splitlines()[-1])
mod = __import__(f"kev.{sys.argv.pop(1)}", fromlist=["main"])
sys.exit(mod.main())
