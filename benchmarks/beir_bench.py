"""Run Inventio on a BEIR-format dataset through its real ingest and search path; score nDCG@10.

    python benchmarks/beir_bench.py scifact --rankers none,laya,typesafe
    python benchmarks/beir_bench.py coir-stackoverflow-qa --rankers laya --limit 300
    python benchmarks/beir_bench.py scifact --arms --rankers none,typesafe

`--arms` first has Jev judge the content categories and fact links of every chunk (facts.py;
judgments are stored in the dataset's map and reused on a rerun), then compares three pools on
the same queries: `base` (BM25's 30), `facts` (plus what the query's predicted categories and the
judged links add) and `control` (BM25 grown to the facts pool's size), each reordered by every
ranker. Per-query rows go to results/beir-<name>/arms.jsonl, totals to summary.json["arms"].

The dataset is read from `<data>/beir/<name>` (fetch it with `benchmarks/data.py`). Each document
becomes one Markdown file (`# title` + text) in a scratch tree next to the dataset, indexed with
Inventio's own ingest into a scratch map. Ranked chunks are collapsed to documents (first
occurrence wins) before scoring, the way BEIR scores documents.

Ranker scores are cached per (query, chunk) in benchmarks/results/beir-<name>/ so a rerun or an
interrupted run does not pay twice. Time per query is measured only over queries that were
scored fresh, so a cached rerun does not report itself as fast.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import data_dir  # noqa: E402
from inventio.bench import ARMS, arm_pools, paired, rank_arms  # noqa: E402
from inventio.ingest import ingest_source  # noqa: E402
from inventio.rankers import make_ranker  # noqa: E402
from inventio.search import bm25, rank_key  # noqa: E402
from inventio.store import connect  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


def load_beir(d: Path, split: str = "test"):
    corpus = {}
    for line in (d / "corpus.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        corpus[r["_id"]] = (r.get("title") or "", r.get("text") or "")
    queries = {}
    for line in (d / "queries.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        queries[r["_id"]] = r["text"]
    qrels: dict[str, dict[str, int]] = {}
    for i, line in enumerate((d / "qrels" / f"{split}.tsv").open(encoding="utf-8")):
        if i == 0:
            continue
        q, doc, s = line.rstrip("\n").split("\t")
        if int(s) > 0:
            qrels.setdefault(q, {})[doc] = int(s)
    return corpus, {q: queries[q] for q in qrels}, qrels


def teach(con, args, queries) -> None:
    """Teacher labels on the train split, for fine-tuning Laya (benchmarks/finetune_laya.py):
    Jev's query categories and its relevance p for BM25's 30 candidates of every train query,
    stored in the dataset's map. Test queries are never asked here."""
    from inventio.facts import JevJudge, warm_query_categories
    from inventio.rankers import TypeSafeRanker

    t = time.time()
    print(f"query categories: {warm_query_categories(con, JevJudge(), list(queries.values()))} asked", flush=True)
    ranker = TypeSafeRanker(con)  # writes every (query, passage, p) into the map's labels table
    done = {(r[0], r[1]) for r in con.execute("SELECT query, passage FROM labels WHERE model = ?", (ranker.model,))}
    for n, q in enumerate(queries.values(), 1):
        todo = [h for h in bm25(con, q, args.pool, [args.dataset]) if (q, h.passage()) not in done]
        if todo:  # a rerun after an interruption asks only what was never answered
            ranker.score(q, todo)
        if n % 50 == 0:
            print(f"  relevance {n}/{len(queries)}  {time.time() - t:.0f}s", flush=True)


def safe_name(doc_id: str) -> str:
    return doc_id.replace("/", "_")


def materialize(corpus, root: Path) -> None:
    if (root / ".done").exists():
        return
    (root / "docs").mkdir(parents=True, exist_ok=True)
    for did, (title, text) in corpus.items():
        (root / "docs" / f"{safe_name(did)}.md").write_text(f"# {title or did}\n\n{text}\n", encoding="utf-8")
    (root / ".done").write_text("ok")


def ndcg10(ranked: list[str], rel: dict[str, int]) -> float:
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def load_cache(path: Path) -> dict:
    cache = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            cache[(r["q"], r["chunk"])] = r["p"]
    return cache


def run_arms(con, args, queries, qrels, safe, out_dir, summary) -> None:
    from inventio.facts import JevJudge, build, warm_query_categories

    judge = JevJudge()
    t = time.time()
    res = build(con, judge, [args.dataset], relink=True)
    print(f"facts: {json.dumps(res)} in {time.time() - t:.0f}s", flush=True)
    t = time.time()
    warm_query_categories(con, judge, list(queries.values()))
    print(f"query categories in {time.time() - t:.0f}s", flush=True)
    pools = {qid: arm_pools(con, q, judge, args.pool, [args.dataset]) for qid, q in queries.items()}
    rows_path = out_dir / "arms.jsonl"
    arms_summary = summary.setdefault("arms", {})
    arms_summary["facts"] = {"categories": res["categories"], "links": res["links"], "kept": res["kept"]}
    for rname in args.rankers.split(","):
        ranker = make_ranker(rname, None)
        cache_path = out_dir / f"scores-{rname}.jsonl"
        cache = load_cache(cache_path)
        nd = {a: [] for a in ARMS}
        rec = {a: [] for a in ARMS}
        size = {a: [] for a in ARMS}
        unscored = 0
        with cache_path.open("a", encoding="utf-8") as cf, rows_path.open("a", encoding="utf-8") as rf:
            for n, (qid, q) in enumerate(queries.items(), 1):
                scores = None
                if ranker is not None:
                    uniq = list({h.id: h for hs in pools[qid].values() for h in hs}.values())
                    todo = [h for h in uniq if (qid, h.id) not in cache]
                    if todo:
                        for h, p in zip(todo, ranker.score(q, todo)):
                            cache[(qid, h.id)] = p
                            cf.write(json.dumps({"q": qid, "chunk": h.id, "p": p}) + "\n")
                        cf.flush()
                    scores = {h.id: cache[(qid, h.id)] for h in uniq}
                    unscored += sum(p is None for p in scores.values())
                for arm, hits in rank_arms(pools[qid], scores).items():
                    ranked = list(dict.fromkeys(safe[Path(h.path).stem] for h in hits))
                    nd[arm].append(ndcg10(ranked, qrels[qid]))
                    rec[arm].append(len(set(ranked) & set(qrels[qid])) / len(qrels[qid]))
                    size[arm].append(len(hits))
                    rf.write(json.dumps({"q": qid, "ranker": rname, "arm": arm, "ndcg10": round(nd[arm][-1], 4),
                                         "recall": round(rec[arm][-1], 4), "pool": len(hits),
                                         "via": sorted({h.via.split(":")[0] for h in hits if h.via})}) + "\n")
                if n % 100 == 0:
                    print(f"  {rname} {n}/{len(queries)} " + " ".join(
                        f"{a} {sum(nd[a]) / len(nd[a]):.3f}" for a in ARMS), flush=True)
        arms_summary[rname] = {
            **{a: {"queries": len(queries), "ndcg@10": round(sum(nd[a]) / len(nd[a]), 4),
                   "recall@pool": round(sum(rec[a]) / len(rec[a]), 4),
                   "mean_pool": round(sum(size[a]) / len(size[a]), 1)} for a in ARMS},
            "facts_vs_control": paired(nd["control"], nd["facts"]),
            "facts_vs_base": paired(nd["base"], nd["facts"]),
            "unscored_pairs": unscored,  # firewall refusals and repeated timeouts: keep their BM25 place
        }
        print(rname, json.dumps(arms_summary[rname]), flush=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="name under <data>/beir, e.g. scifact or coir-stackoverflow-qa")
    ap.add_argument("--rankers", default="none")
    ap.add_argument("--pool", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="first N test queries only")
    ap.add_argument("--data", help="data directory (default: user cache)")
    ap.add_argument("--arms", action="store_true", help="base / facts / control pools (see module docstring)")
    ap.add_argument("--teach", action="store_true", help="Jev labels on the train split (see teach())")
    args = ap.parse_args()
    ds = data_dir(args.data) / "beir" / args.dataset
    out_dir = RESULTS / f"beir-{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus, queries, qrels = load_beir(ds, "train" if args.teach else "test")
    safe = {safe_name(d): d for d in corpus}
    if args.limit:
        queries = dict(list(queries.items())[: args.limit])
    scratch = ds / "inventio-tree"
    materialize(corpus, scratch)
    con = connect(ds / "inventio.db")
    if not con.execute("SELECT 1 FROM sources WHERE name = ?", (args.dataset,)).fetchone():
        t = time.time()
        with con:  # links are not used by BM25 or the rankers, so they are not built here
            st = ingest_source(con, args.dataset, scratch, True, [])
        print(f"indexed {st['files']} docs / {st['chunks']} chunks in {time.time() - t:.1f}s", flush=True)

    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if args.teach:
        teach(con, args, queries)
        return 0
    if args.arms:
        (out_dir / "arms.jsonl").unlink(missing_ok=True)
        run_arms(con, args, queries, qrels, safe, out_dir, summary)
        return 0
    for rname in args.rankers.split(","):
        cache_path = out_dir / f"scores-{rname}.jsonl"
        cache = load_cache(cache_path)
        ranker = make_ranker(rname, None)
        nd, rec, fresh_secs, fresh_n = [], [], 0.0, 0
        with cache_path.open("a", encoding="utf-8") as cf:
            for n, (qid, q) in enumerate(queries.items(), 1):
                t = time.time()
                hits = bm25(con, q, args.pool, [args.dataset])
                fresh = True
                if ranker is not None:
                    todo = [h for h in hits if (qid, h.id) not in cache]
                    fresh = len(todo) == len(hits)
                    if todo:
                        for h, p in zip(todo, ranker.score(q, todo)):
                            cache[(qid, h.id)] = p
                            cf.write(json.dumps({"q": qid, "chunk": h.id, "p": p}) + "\n")
                        cf.flush()
                    for h in hits:
                        h.score = cache[(qid, h.id)]
                    hits.sort(key=rank_key)
                if fresh:
                    fresh_secs, fresh_n = fresh_secs + time.time() - t, fresh_n + 1
                ranked = list(dict.fromkeys(safe[Path(h.path).stem] for h in hits))
                nd.append(ndcg10(ranked, qrels[qid]))
                rec.append(len(set(ranked) & set(qrels[qid])) / len(qrels[qid]))
                if n % 50 == 0:
                    print(f"  {rname} {n}/{len(queries)} nDCG@10 so far {sum(nd) / len(nd):.3f}", flush=True)
        key = f"{rname}" + (f"@first{args.limit}" if args.limit else "")
        summary[key] = {
            "queries": len(queries),
            "ndcg@10": round(sum(nd) / len(nd), 4),
            f"recall@{args.pool}chunks": round(sum(rec) / len(rec), 4),
            "sec_per_query": round(fresh_secs / fresh_n, 3) if fresh_n else None,
            "timed_queries": fresh_n,
        }
        print(rname, summary[key], flush=True)
        summary_path.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
