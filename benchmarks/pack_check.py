"""Does packing ten neighbours into one Jev call judge them as one call per pair would?

Pre-registered before any full run: 10 SciFact chunks x their 10 BM25 neighbours = 100 pairs,
neighbour order shuffled in the packed call. Pass: the true/false decision (p >= 0.5) agrees on at
least 90% of pairs and the mean |p_packed - p_single| is at most 0.15. Results go to
results/pack-check.json.

    python benchmarks/pack_check.py
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import data_dir  # noqa: E402
from inventio.facts import JevJudge, passage, same_question  # noqa: E402
from inventio.search import bm25, fts_query  # noqa: E402
from inventio.store import connect  # noqa: E402

AGREE_MIN, DIFF_MAX = 0.90, 0.15


def main() -> int:
    con = connect(data_dir(None) / "beir" / "scifact" / "inventio.db")
    rng = random.Random(20260923)
    ids = [r["id"] for r in con.execute("SELECT id FROM chunks ORDER BY id")]
    judge = JevJudge()
    packed_jobs, single_jobs, pairs = [], [], []
    for a in rng.sample(ids, 40):
        ra = con.execute("SELECT c.id, c.file_id, f.path, c.heading_path, c.text FROM chunks c JOIN files f ON f.id = c.file_id WHERE c.id = ?", (a,)).fetchone()
        words = fts_query(ra["text"]).split(" OR ")[:24]
        hs = [h for h in bm25(con, " ".join(w.strip('"') for w in words), 30) if h.id != a][:10]
        if len(hs) < 10:
            continue
        pa = passage(ra["path"], ra["heading_path"], ra["text"])
        pbs = [h.passage() for h in hs]
        order = list(range(10))
        rng.shuffle(order)
        state = {"passage": pa, **{f"n{j}": pbs[i] for j, i in enumerate(order)}}
        packed_jobs.append((state, {f"n{j}": same_question(f"n{j}") for j in range(10)}))
        for j, i in enumerate(order):
            pairs.append((len(packed_jobs) - 1, f"n{j}"))
            single_jobs.append(({"passage": pa, "n0": pbs[i]}, {"n0": same_question("n0")}))
        if len(packed_jobs) == 10:
            break
    packed = dict(judge.batch(iter(packed_jobs)))
    single = dict(judge.batch(iter(single_jobs)))
    rows = []
    for k, (pj, slot) in enumerate(pairs):
        if packed.get(pj) is None or single.get(k) is None:
            continue
        rows.append((packed[pj][slot], single[k]["n0"]))
    agree = sum((p >= 0.5) == (s >= 0.5) for p, s in rows) / len(rows)
    diff = sum(abs(p - s) for p, s in rows) / len(rows)
    res = {"pairs": len(rows), "agree": round(agree, 3), "mean_abs_diff": round(diff, 3),
           "true_packed": sum(p >= 0.5 for p, _ in rows), "true_single": sum(s >= 0.5 for _, s in rows),
           "pass": agree >= AGREE_MIN and diff <= DIFF_MAX, "bar": {"agree_min": AGREE_MIN, "diff_max": DIFF_MAX},
           "refused": judge.refused, "failed": judge.failed}
    out = Path(__file__).resolve().parent / "results" / "pack-check.json"
    out.write_text(json.dumps({**res, "p": rows}, indent=1))
    print(res)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
