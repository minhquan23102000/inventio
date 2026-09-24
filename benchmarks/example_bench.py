"""Where each question of an example lands, per ranker: the check a public benchmark cannot make.

    python benchmarks/example_bench.py none dispositio laya typesafe
    python benchmarks/example_bench.py --example examples/webshop dispositio <checkpoint dir>

examples/webshop is a small team's knowledge in the shape Inventio is built for: code, a policy,
a runbook, an incident report, one source per subfolder. Its 13 questions (questions.jsonl) are
asked the way someone on call asks them, most without the words of the section that answers;
each names the one section, as `source:path:line`, that answers it. No ranker was trained or
tuned on them. A ranker argument is `none` (BM25 order), `laya`, `typesafe`, `dispositio`, or a
checkpoint directory, loaded as dispositio.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inventio.ingest import ingest_source  # noqa: E402
from inventio.links import rebuild_links  # noqa: E402
from inventio.rankers import RANKERS, make_ranker  # noqa: E402
from inventio.search import search  # noqa: E402
from inventio.store import connect  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rankers", nargs="+", help=f"{', '.join(RANKERS)}, or a checkpoint directory")
    ap.add_argument("--example", default=str(Path(__file__).resolve().parent.parent / "examples" / "webshop"))
    args = ap.parse_args()
    root = Path(args.example)
    con = connect(Path(tempfile.mkdtemp()) / "example.db")
    for src in sorted(p for p in root.iterdir() if p.is_dir()):
        ingest_source(con, src.name, src, True, [])
    rebuild_links(con)
    con.commit()
    qs = [json.loads(line) for line in (root / "questions.jsonl").open(encoding="utf-8")]
    for arg in args.rankers:
        name = arg if arg in RANKERS else "dispositio"
        if name != arg:
            os.environ["INVENTIO_DISPOSITIO_MODEL"] = arg
        ranker = make_ranker(name)
        ranks = []
        for q in qs:
            coords = [f"{h.source}:{h.path}:{h.start_line}" for h in search(con, q["query"], k=10, pool=30, ranker=ranker)]
            ranks.append(coords.index(q["answer"]) + 1 if q["answer"] in coords else None)
        mrr = sum(1 / r for r in ranks if r) / len(qs)
        print(f"{Path(arg).name}: first {sum(r == 1 for r in ranks)}/{len(qs)}, MRR@10 {mrr:.2f}")
        for r, q in zip(ranks, qs):
            print(f"  {r or '-':>2}  {q['query']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
