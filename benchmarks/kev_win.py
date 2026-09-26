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

if len(sys.argv) < 2 or sys.argv[1] not in ("serve", "train"):
    raise SystemExit(__doc__.strip().splitlines()[-1])
mod = __import__(f"kev.{sys.argv.pop(1)}", fromlist=["main"])
sys.exit(mod.main())
