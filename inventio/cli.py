import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import __version__
from .facts import JUDGES, make_judge
from .rankers import RANKERS, CloudRefused, default_ranker, make_ranker
from .store import connect, default_db


def _db(args):
    return connect(Path(args.db) if args.db else None)


def cmd_init(args) -> int:
    from .ingest import ingest_source
    from .links import rebuild_links

    root = Path(args.path)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    con = _db(args)
    t = time.time()
    name = args.name or root.resolve().name
    with con:
        stats = ingest_source(con, name, root, args.public, args.exclude or [], full=args.full)
        links = rebuild_links(con)
    print(
        f"{stats['source']}: {stats['files']} files, {stats['chunks']} chunks "
        f"(+{stats['added']} new, ~{stats['changed']} edited, -{stats['removed']} removed, "
        f"{stats['unchanged']} unchanged); "
        f"map links {', '.join(f'{k} {v}' for k, v in sorted(links.items())) or 'none'}; "
        f"{time.time() - t:.1f}s -> {args.db or default_db()}"
    )
    if args.facts:
        args.source = [name]
        return cmd_facts(args)
    return 0


def cmd_facts(args) -> int:
    from .facts import build

    con = _db(args)
    t = time.time()
    try:
        res = build(con, make_judge(args.judge), args.source, relink=getattr(args, "relink", False))
    except CloudRefused as e:
        print(str(e), file=sys.stderr)
        return 3
    c, l = res["categories"], res["links"]
    print(
        f"categories: {c['chunks']} chunks ({c['asked']} judged, {c['cached']} from earlier judgments, "
        f"{c['refused']} refused, {c['failed']} failed); kept {', '.join(f'{k} {v}' for k, v in res['kept'].items()) or 'none'}"
    )
    print(f"fact links: {l['chunks']} chunks, {l['pairs']} pairs ({l['asked']} judged, {l['cached']} earlier), "
          f"{l['links']} about links; {time.time() - t:.1f}s")
    return 0


def cmd_sources(args) -> int:
    con = _db(args)
    rows = con.execute(
        "SELECT s.name, s.root, s.public, s.indexed_at, count(DISTINCT f.id) files, count(c.id) chunks "
        "FROM sources s LEFT JOIN files f ON f.source_id = s.id LEFT JOIN chunks c ON c.file_id = f.id "
        "GROUP BY s.id ORDER BY s.name"
    ).fetchall()
    for r in rows:
        print(f"{r['name']:<16} {'public ' if r['public'] else 'private'} {r['files']:>5} files {r['chunks']:>6} chunks  {r['root']}  ({r['indexed_at']})")
    if not rows:
        print("no sources; run `inventio init <path>`")
    return 0


def cmd_drop(args) -> int:
    from .links import rebuild_links
    from .store import drop_source

    con = _db(args)
    row = con.execute("SELECT id FROM sources WHERE name = ?", (args.name,)).fetchone()
    if not row:
        print(f"no source named {args.name}", file=sys.stderr)
        return 2
    with con:
        drop_source(con, row["id"])
        con.execute("DELETE FROM sources WHERE id = ?", (row["id"],))
        rebuild_links(con)
    print(f"dropped {args.name}")
    return 0


def _snippet(text: str, width: int = 160) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    s = " ".join(lines)
    return s[:width] + ("…" if len(s) > width else "")


def cmd_query(args) -> int:
    from .search import search

    con = _db(args)
    try:
        ranker = make_ranker(args.ranker)
        hits = search(con, args.text, k=args.k, pool=args.pool, ranker=ranker, expand_links=args.links,
                      by_type=args.types, facts=make_judge(args.judge) if args.facts else None,
                      symbols=not args.no_symbols, neighbours=args.neighbours, sources=args.source)
    except CloudRefused as e:
        print(str(e), file=sys.stderr)
        return 3
    if args.json:
        print(json.dumps([h.as_dict() for h in hits], ensure_ascii=False, indent=2))
        return 0
    if not hits:
        print("no match")
        return 1
    # grouped by document type, groups in order of their best hit; the number is the overall rank
    groups: dict[str, list] = {}
    for i, h in enumerate(hits, 1):
        groups.setdefault(h.type or "(untyped)", []).append((i, h))
    for t, members in groups.items():
        print(f"== {t}")
        for i, h in members:
            score = f"  p={h.score:.2f}" if h.score is not None else ""
            entered = f"  [via {h.via}]" if h.via else ""
            print(f"{i}. {h.source}:{h.coord}  {h.heading_path}{score}{entered}")
            print(f"   {_snippet(h.text)}")
            for l in h.links[:3]:
                arrow = "->" if l["dir"] == "out" else "<-"
                print(f"   {arrow} {l['rel']} {l['source']}:{l['coord']}  ({l['via']})")
    return 0


def cmd_bench(args) -> int:
    from . import bench

    con = _db(args)
    rows = bench.load(Path(args.file))
    try:
        ranker = make_ranker(args.ranker)
        if args.arms:
            res = bench.run_arms(con, rows, ranker, make_judge(args.judge), pool=args.pool)
        else:
            res = bench.run(con, rows, ranker, pool=args.pool, expand_links=args.links, by_type=args.types,
                            facts=make_judge(args.judge) if args.facts else None,
                            symbols=not args.no_symbols, neighbours=args.neighbours)
    except CloudRefused as e:
        print(str(e), file=sys.stderr)
        return 3
    res["config"] = {"ranker": args.ranker, "links": args.links, "types": args.types, "pool": args.pool,
                     "arms": args.arms, "judge": args.judge if args.arms else None}
    if args.json:
        print(json.dumps(res))
    elif args.arms:
        print(f"ranker={args.ranker} judge={args.judge} pool={args.pool}")
        for arm in bench.ARMS:
            r = res[arm]
            n = r["n"]
            print(f"  {arm:<8} pool {r['mean_pool']:>5}  in pool {r['in_pool']}/{n}  top-1 {r['top1']}/{n}  "
                  f"top-5 {r['top5']}/{n}  top-10 {r['top10']}/{n}  nDCG@10 {r['ndcg@10']}")
        print(f"  facts vs control: {res['facts_vs_control']}")
    else:
        n = res["n"]
        print(f"ranker={args.ranker} links={'on' if args.links else 'off'} types={'on' if args.types else 'off'} "
              f"pool={args.pool}  n={n}")
        print(f"  in pool {res['in_pool']}/{n}  top-1 {res['top1']}/{n}  top-5 {res['top5']}/{n}  "
              f"top-10 {res['top10']}/{n}  {res['sec_per_query']}s/query")
    return 0


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="inventio", description="Local retrieval map: BM25 + structure + links, reranked by dispositio or TypeSafe.")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--db", help=f"map file (default {default_db()}, or $INVENTIO_DB)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="index a directory as a source; re-running updates only what changed")
    s.add_argument("path")
    s.add_argument("--name", help="source name (default: directory name)")
    s.add_argument("--public", action="store_true", help="allow this source's text to be sent to a cloud ranker")
    s.add_argument("--exclude", action="append", metavar="GLOB", help="skip paths matching GLOB (repeatable)")
    s.add_argument("--full", action="store_true", help="drop the source and rebuild it from scratch")
    s.add_argument("--facts", action="store_true", help="then judge content categories and fact links (see `facts`)")
    judge = os.environ.get("INVENTIO_JUDGE", "dispositio")
    judges = "dispositio (local), laya (local, as published), typesafe (cloud, public sources only)"
    s.add_argument("--judge", choices=JUDGES, default=judge, help=f"model for --facts: {judges}")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("facts", help="judge content categories and fact links for chunks that have none yet")
    s.add_argument("--source", action="append", metavar="NAME", help="only this source (repeatable)")
    s.add_argument("--judge", choices=JUDGES, default=judge, help=judges)
    s.add_argument("--relink", action="store_true",
                   help="look for neighbours of every categorized chunk, not only new ones (judged pairs are reused)")
    s.set_defaults(fn=cmd_facts)

    s = sub.add_parser("sources", help="list indexed sources")
    s.set_defaults(fn=cmd_sources)

    s = sub.add_parser("drop", help="remove a source from the map")
    s.add_argument("name")
    s.set_defaults(fn=cmd_drop)

    def ranking(s):
        s.add_argument("--ranker", choices=RANKERS, default=default_ranker(),
                       help="reorder the pool: dispositio (local; the default when the laya extra is installed), "
                            "none (BM25 order), laya (local, as published), typesafe (cloud, public sources only)")
        s.add_argument("--pool", type=int, default=30, help="BM25 candidates handed to the ranker")
        s.add_argument("--links", action="store_true",
                       help="also hand the ranker chunks linked to the top BM25 hits (off: measured no gain yet)")
        s.add_argument("--no-symbols", action="store_true",
                       help="do not add the files and definitions the question names (on by default)")
        s.add_argument("--neighbours", action="store_true",
                       help="add the chunks of other files that share the most distinctive words of the top "
                            "BM25 hits (code only, no model; needs --ranker to reorder them)")
        s.add_argument("--types", action="store_true",
                       help="ask the ranker which document types hold the answer and add BM25's best chunks "
                            "of those types to the pool (needs --ranker)")
        s.add_argument("--facts", action="store_true",
                       help="add BM25's best chunks of the query's predicted content categories and the chunks "
                            "its top hits are linked to by judged fact links (needs `facts` run on the map)")
        s.add_argument("--judge", choices=JUDGES, default=judge,
                       help="model that predicts the query's categories for --facts / --arms")

    s = sub.add_parser("query", help="find the passages that answer a question")
    s.add_argument("text")
    s.add_argument("-k", type=int, default=5, help="results to print")
    s.add_argument("--source", action="append", metavar="NAME", help="only search this source (repeatable)")
    s.add_argument("--json", action="store_true")
    ranking(s)
    s.set_defaults(fn=cmd_query)

    s = sub.add_parser("bench", help="measure a configuration on questions with known answers")
    s.add_argument("file", help="JSON Lines: question, source, path, start_line, end_line")
    s.add_argument("--json", action="store_true")
    s.add_argument("--arms", action="store_true",
                   help="compare base BM25, BM25 + --facts, and BM25 with a pool as large as the facts one")
    ranking(s)
    s.set_defaults(fn=cmd_bench)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
