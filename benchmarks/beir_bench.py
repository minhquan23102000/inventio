"""Run Inventio on a BEIR-format dataset through its real ingest and search path; score nDCG@10.

    python benchmarks/beir_bench.py scifact --rankers none,laya,typesafe
    python benchmarks/beir_bench.py coir-stackoverflow-qa --rankers laya --limit 300

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
from inventio.ingest import ingest_source  # noqa: E402
from inventio.rankers import make_ranker  # noqa: E402
from inventio.search import bm25, rank_key  # noqa: E402
from inventio.store import connect  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


def load_beir(d: Path):
    corpus = {}
    for line in (d / "corpus.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        corpus[r["_id"]] = (r.get("title") or "", r.get("text") or "")
    queries = {}
    for line in (d / "queries.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        queries[r["_id"]] = r["text"]
    qrels: dict[str, dict[str, int]] = {}
    for i, line in enumerate((d / "qrels" / "test.tsv").open(encoding="utf-8")):
        if i == 0:
            continue
        q, doc, s = line.rstrip("\n").split("\t")
        if int(s) > 0:
            qrels.setdefault(q, {})[doc] = int(s)
    return corpus, {q: queries[q] for q in qrels}, qrels


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="name under <data>/beir, e.g. scifact or coir-stackoverflow-qa")
    ap.add_argument("--rankers", default="none")
    ap.add_argument("--pool", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="first N test queries only")
    ap.add_argument("--data", help="data directory (default: user cache)")
    args = ap.parse_args()
    ds = data_dir(args.data) / "beir" / args.dataset
    out_dir = RESULTS / f"beir-{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus, queries, qrels = load_beir(ds)
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
    for rname in args.rankers.split(","):
        cache_path = out_dir / f"scores-{rname}.jsonl"
        cache = {}
        if cache_path.exists():
            for line in cache_path.open(encoding="utf-8"):
                r = json.loads(line)
                cache[(r["q"], r["chunk"])] = r["p"]
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
