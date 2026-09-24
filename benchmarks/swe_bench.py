"""Run Inventio on SWE-bench Lite file retrieval, the CodeRAG-Bench repo-level setting.

    python benchmarks/swe_bench.py --rankers none,dispositio,typesafe [--limit N]

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

    python benchmarks/swe_bench.py --types --variants mixed --rankers none,typesafe

measures document-type widening instead, in three arms over the same instances: `base` (BM25
pool), `types` (base plus BM25's best chunks of the types TypeSafe predicts, as `query --types`
does) and `control` (plain BM25 with a pool as large as the widened one, so a gain from types is
not just a gain from more candidates). Results go to benchmarks/results/swe-lite-types/.
"""

import argparse
import hashlib
import io
import json
import os
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
from inventio.ingest import DOC_TYPES, ingest_source  # noqa: E402
from inventio.rankers import make_ranker, ranker_tag  # noqa: E402
from inventio.search import bm25, rank_key, scope_types, widen_by_symbols, widen_by_type  # noqa: E402
from inventio.store import connect  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results" / "swe-lite"
TYPES_RESULTS = Path(__file__).resolve().parent / "results" / "swe-lite-types"
STRAT_RESULTS = Path(__file__).resolve().parent / "results" / "swe-lite-strat"
# --strat: widenings code decides, each against BM25 with as many candidates (ctl-*)
STRAT_ARMS = ("base", "all", "ctl-all", "code", "ctl-code", "under", "ctl-under", "sym", "ctl-sym",
              "code+sym", "ctl-code+sym", "symfirst")


class Fixed:
    """A type predictor that returns the same probabilities for every query."""
    cloud = False

    def __init__(self, probs):
        self.probs = probs

    def types(self, q, types):
        return self.probs


def strat_pools(con, q: str, base: list, src: str, depth: int) -> dict:
    present = scope_types(con, [src])
    corpus = dict(con.execute("SELECT f.type, count(*) FROM chunks c JOIN files f ON f.id = c.file_id "
                              "GROUP BY f.type").fetchall())
    total = sum(corpus.values()) or 1
    in_base = {t: sum(h.type == t for h in base) / max(len(base), 1) for t in present}
    choose = {
        "all": {t: 1.0 for t in present},
        "code": {"SoftwareSourceCode": 1.0} if "SoftwareSourceCode" in present else {},
        "under": {t: 1.0 for t in present if in_base[t] < corpus.get(t, 0) / total},
    }
    pools = {"base": base}
    for arm, probs in choose.items():
        pools[arm] = base + (widen_by_type(con, q, base, Fixed(probs), depth, [src]) if probs else [])
    pools["sym"] = base + widen_by_symbols(con, q, base, depth, [src])
    pools["symfirst"] = pools["sym"][len(base):] + base  # the same pool, symbol hits ahead of BM25's
    pools["code+sym"] = pools["code"] + widen_by_symbols(con, q, pools["code"], depth, [src])
    for arm in ("all", "code", "under", "sym", "code+sym"):
        pools[f"ctl-{arm}"] = bm25(con, q, len(pools[arm]), [src])
    return pools


def is_test(name: str) -> bool:  # CodeRAG-Bench's rule, create/swebench.py
    return bool({"test", "tests", "testing"} & set(re.split(r" |_|/|\.", name.lower())))


def gold_files(patch: str) -> set[str]:
    return {m.group(1) for m in re.finditer(r"^--- a/(\S+)", patch, re.M)}


def keep(name: str, code_only: bool) -> bool:
    return not code_only or (name.endswith(".py") and not is_test(name))


def export(repo: Path, commit: str, dest: Path, code_only: bool, paths: list[str] | None = None) -> None:
    """Write the repository's files at `commit` under `dest` (only `paths`, when given)."""
    for i in range(0, len(paths), 200) if paths is not None else [None]:
        part = paths[i:i + 200] if paths is not None else []
        data = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", commit, "--", *part],
                              capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            for m in tf.getmembers():
                if m.isfile() and keep(m.name, code_only):
                    out = dest / m.name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(tf.extractfile(m).read())


def checkout(repo: Path, commit: str, work: Path, code_only: bool) -> Path:
    """One tree per repository and variant, moved from commit to commit by rewriting only the files
    that differ: a full export writes 30,000 files per django issue, most of them unchanged. The
    map beside it is updated the same way by the incremental ingest."""
    tree, mark = work / "tree", work / "commit"
    prev = mark.read_text().strip() if mark.exists() else None
    if prev is None:
        shutil.rmtree(work, ignore_errors=True)
        tree.mkdir(parents=True)
        export(repo, commit, tree, code_only)
    elif prev != commit:
        diff = subprocess.run(["git", "-C", str(repo), "diff", "--name-status", "--no-renames", "-z", prev, commit],
                              capture_output=True, check=True, text=True, encoding="utf-8").stdout.split("\0")
        changed = []
        for status, name in zip(diff[::2], diff[1::2]):
            if status == "D" or not keep(name, code_only):
                (tree / name).unlink(missing_ok=True)
            else:
                changed.append(name)
        if changed:
            export(repo, commit, tree, code_only, changed)
    mark.write_text(commit)
    return tree


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
    ap.add_argument("--types", action="store_true", help="measure type widening: arms base, types, control")
    ap.add_argument("--strat", action="store_true", help="measure widenings code decides (STRAT_ARMS)")
    ap.add_argument("--type-limit", type=int, default=None, help="chunks added per predicted type (default: --pool)")
    args = ap.parse_args()
    swe = data_dir(args.data) / "swe-lite"
    out_dir = STRAT_RESULTS if args.strat else TYPES_RESULTS if args.types else RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = STRAT_ARMS if args.strat else ("base", "types", "control") if args.types else ("base",)
    armed = args.types or args.strat
    res_path, cache_path = out_dir / "results.jsonl", out_dir / "scores.jsonl"
    done = {(r["iid"], r["variant"], r["ranker"], r.get("arm", "base"))
            for r in map(json.loads, res_path.open(encoding="utf-8"))} if res_path.exists() else set()
    # one score cache for every mode: a pair scored once is never paid for again
    caches = [RESULTS / "scores.jsonl", cache_path]
    cache = {r["k"]: r["p"] for c in dict.fromkeys(caches) if c.exists()
             for r in map(json.loads, c.open(encoding="utf-8"))}
    rows = [json.loads(l) for l in (swe / "lite.jsonl").open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    rankers = {ranker_tag(n): make_ranker(n) for n in args.rankers.split(",")}
    rnames = list(rankers)
    predictor = (rankers.get("typesafe") or make_ranker("typesafe")) if args.types else None
    src = "swe"  # one source per tree; issue ids still key the score cache and the results
    resf, cachef = res_path.open("a", encoding="utf-8"), cache_path.open("a", encoding="utf-8")
    for n, inst in enumerate(rows, 1):
        iid, gold = inst["instance_id"], gold_files(inst["patch"])
        for variant in args.variants.split(","):
            todo = [(r, a) for r in rnames for a in arms if (iid, variant, r, a) not in done]
            if not todo:
                continue
            t = time.time()
            repo_dir = swe / inst["repo"].replace("/", "__")
            wdir = swe / "work" / f"{inst['repo'].replace('/', '__')}-{variant}"
            tree = checkout(repo_dir, inst["base_commit"], wdir, variant == "code")
            con = connect(wdir / "map.db")
            with con:  # links are not used by BM25 or the rankers, so they are not built here
                st = ingest_source(con, src, tree, True, [])
            t_index = time.time() - t
            q = inst["problem_statement"]
            t = time.time()
            pools = {"base": bm25(con, q, args.pool, [src])}
            t_bm25, added, probs = time.time() - t, [], {}
            if args.types:
                t = time.time()
                probs = predictor.types(q, {k: DOC_TYPES[k] for k in scope_types(con, [src])})
                fixed = type("Fixed", (), {"cloud": True, "types": lambda self, q, types: probs})()
                added = widen_by_type(con, q, pools["base"], fixed, args.type_limit or args.pool, [src])
                pools["types"] = pools["base"] + added
                t_types = time.time() - t
                pools["control"] = bm25(con, q, len(pools["types"]), [src])
            if args.strat:
                t = time.time()
                pools = strat_pools(con, q, pools["base"], src, args.type_limit or args.pool)
                t_types = time.time() - t
            for rname, arm in todo:
                hits, t = [h for h in pools[arm]], time.time()
                for h in hits:
                    h.score = None
                ranker = rankers[rname]
                if ranker is not None:
                    key = lambda h: hashlib.sha1(f"{rname}|{iid}|{h.coord}|{h.text}".encode()).hexdigest()
                    miss = list({key(h): h for h in hits if key(h) not in cache}.values())
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
                    "sec_index": round(t_index, 2),
                    "sec_query": round(t_bm25 + (t_types if arm not in ("base", "control") and armed else 0)
                                       + time.time() - t, 3),
                }
                if armed:
                    r |= {"arm": arm, "pool": len(hits)}
                if args.types:
                    r |= {"type_probs": {k: round(v, 3) for k, v in probs.items()},
                          "types_added": sorted({h.via[len("type:"):] for h in added})}
                resf.write(json.dumps(r) + "\n")
                resf.flush()
            con.close()
        print(f"{n}/{len(rows)} {iid}", flush=True)

    allr = [json.loads(l) for l in res_path.open(encoding="utf-8")]
    summary = {}  # every variant and ranker on record, not only this run's
    for variant in dict.fromkeys(r["variant"] for r in allr):
        for rname in dict.fromkeys(r["ranker"] for r in allr):
            for arm in arms:
                rs = [r for r in allr if r["variant"] == variant and r["ranker"] == rname and r.get("arm", "base") == arm]
                if not rs:
                    continue
                m = lambda k: round(sum(r[k] for r in rs) / len(rs), 4)
                name = f"{variant}/{rname}" + (f"/{arm}" if armed else "")
                summary[name] = {
                    "n": len(rs), "ndcg@10": m("ndcg10"), "top1": m("top1"), "top5": m("top5"), "in_pool": m("in_pool"),
                    "sec_query": m("sec_query"), "sec_index": m("sec_index"),
                }
                if armed:
                    summary[name]["mean_pool"] = m("pool")
                print(name, summary[name], flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
