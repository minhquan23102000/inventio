"""Run Inventio on SWE-bench Lite file retrieval, the CodeRAG-Bench repo-level setting.

    python benchmarks/swe_bench.py --rankers none,laya,typesafe [--limit N]

Reads `<data>/swe-lite` (fetch it with `benchmarks/data.py swe-lite`): lite.jsonl plus one
clone per repo. For every instance the repo is exported at base_commit and indexed with
Inventio's own ingest, twice:

  code   non-test .py files only: the CodeRAG-Bench corpus (published BM25 nDCG@10 = 43.0)
  mixed  the whole repository as Inventio would index it: code, tests, .rst/.txt/.md docs,
         configs. The gold files are the same; everything else is realistic noise.

The query is the issue text; the gold documents are the files the fix patch touches. Ranked
chunks are collapsed to files (first occurrence wins) and scored with binary nDCG@10.
Per-instance results are appended to benchmarks/results/swe-lite/results.jsonl, so an
interrupted run resumes where it stopped.
"""

import argparse
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import data_dir  # noqa: E402
from inventio.ingest import ingest_source  # noqa: E402
from inventio.rankers import make_ranker  # noqa: E402
from inventio.search import bm25, rank_key  # noqa: E402
from inventio.store import connect  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results" / "swe-lite"


def is_test(name: str) -> bool:  # CodeRAG-Bench's rule, create/swebench.py
    return bool({"test", "tests", "testing"} & set(re.split(r" |_|/|\.", name.lower())))


def gold_files(patch: str) -> set[str]:
    return {m.group(1) for m in re.finditer(r"^--- a/(\S+)", patch, re.M)}


def export(repo: Path, commit: str, dest: Path, code_only: bool) -> None:
    data = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", commit],
                          capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            if code_only and (not m.name.endswith(".py") or is_test(m.name)):
                continue
            out = dest / m.name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(tf.extractfile(m).read())


def ndcg10(ranked: list[str], gold: set[str]) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in gold)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    return dcg / idcg if idcg else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="data directory (default: user cache)")
    ap.add_argument("--rankers", default="none")
    ap.add_argument("--variants", default="code,mixed")
    ap.add_argument("--pool", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    swe = data_dir(args.data) / "swe-lite"
    out_dir = RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path, cache_path = out_dir / "results.jsonl", out_dir / "scores.jsonl"
    done = {(r["iid"], r["variant"], r["ranker"]) for r in map(json.loads, res_path.open(encoding="utf-8"))} \
        if res_path.exists() else set()
    cache = {r["k"]: r["p"] for r in map(json.loads, cache_path.open(encoding="utf-8"))} \
        if cache_path.exists() else {}
    rows = [json.loads(l) for l in (swe / "lite.jsonl").open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    rnames = args.rankers.split(",")
    rankers = {n: make_ranker(n, None) for n in rnames}
    work = swe / "work"
    resf, cachef = res_path.open("a", encoding="utf-8"), cache_path.open("a", encoding="utf-8")
    for n, inst in enumerate(rows, 1):
        iid, gold = inst["instance_id"], gold_files(inst["patch"])
        for variant in args.variants.split(","):
            todo = [r for r in rnames if (iid, variant, r) not in done]
            if not todo:
                continue
            shutil.rmtree(work, ignore_errors=True)
            tree = work / "tree"
            tree.mkdir(parents=True)
            t = time.time()
            export(swe / inst["repo"].replace("/", "__"), inst["base_commit"], tree, variant == "code")
            con = connect(work / "map.db")
            with con:  # links are not used by BM25 or the rankers, so they are not built here
                st = ingest_source(con, iid, tree, True, [])
            t_index = time.time() - t
            q = inst["problem_statement"]
            t = time.time()
            base = bm25(con, q, args.pool, [iid])
            t_bm25 = time.time() - t
            for rname in todo:
                hits, t = list(base), time.time()
                ranker = rankers[rname]
                if ranker is not None:
                    key = lambda h: hashlib.sha1(f"{rname}|{iid}|{h.coord}|{h.text}".encode()).hexdigest()
                    miss = [h for h in hits if key(h) not in cache]
                    if miss:
                        for h, p in zip(miss, ranker.score(q, miss)):
                            cache[key(h)] = p
                            cachef.write(json.dumps({"k": key(h), "p": p}) + "\n")
                        cachef.flush()
                    for h in hits:
                        h.score = cache[key(h)]
                    hits.sort(key=rank_key)
                ranked = list(dict.fromkeys(h.path for h in hits))
                r = {
                    "iid": iid, "variant": variant, "ranker": rname, "files": st["files"], "chunks": st["chunks"],
                    "gold": sorted(gold), "ranked": ranked[:10], "ndcg10": ndcg10(ranked, gold),
                    "top1": bool(ranked[:1] and ranked[0] in gold), "top5": bool(gold & set(ranked[:5])),
                    "in_pool": bool(gold & set(ranked)),
                    "sec_index": round(t_index, 2), "sec_query": round(t_bm25 + time.time() - t, 3),
                }
                resf.write(json.dumps(r) + "\n")
                resf.flush()
            con.close()
        print(f"{n}/{len(rows)} {iid}", flush=True)
    shutil.rmtree(work, ignore_errors=True)

    allr = [json.loads(l) for l in res_path.open(encoding="utf-8")]
    summary = {}
    for variant in args.variants.split(","):
        for rname in rnames:
            rs = [r for r in allr if r["variant"] == variant and r["ranker"] == rname]
            if not rs:
                continue
            m = lambda k: round(sum(r[k] for r in rs) / len(rs), 4)
            summary[f"{variant}/{rname}"] = {
                "n": len(rs), "ndcg@10": m("ndcg10"), "top1": m("top1"), "top5": m("top5"),
                f"in_pool@{args.pool}chunks": m("in_pool"), "sec_query": m("sec_query"), "sec_index": m("sec_index"),
            }
            print(variant, rname, summary[f"{variant}/{rname}"], flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
