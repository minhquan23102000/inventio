import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import __version__
from .rankers import RANKERS, CloudRefused, make_ranker
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
    with con:
        stats = ingest_source(con, args.name or root.resolve().name, root, args.public, args.exclude or [])
        links = rebuild_links(con)
    print(
        f"{stats['source']}: {stats['files']} files, {stats['chunks']} chunks; "
        f"map links {', '.join(f'{k} {v}' for k, v in sorted(links.items())) or 'none'}; "
        f"{time.time() - t:.1f}s -> {args.db or default_db()}"
    )
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
        print("no sources; run `lacthu init <path>`")
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
        ranker = make_ranker(args.ranker, con)
        hits = search(con, args.text, k=args.k, pool=args.pool, ranker=ranker,
                      expand_links=args.links, sources=args.source)
    except CloudRefused as e:
        print(str(e), file=sys.stderr)
        return 3
    if args.json:
        print(json.dumps([h.as_dict() for h in hits], ensure_ascii=False, indent=2))
        return 0
    if not hits:
        print("no match")
        return 1
    for i, h in enumerate(hits, 1):
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
        res = bench.run(con, rows, make_ranker(args.ranker, con), pool=args.pool, expand_links=args.links)
    except CloudRefused as e:
        print(str(e), file=sys.stderr)
        return 3
    res["config"] = {"ranker": args.ranker, "links": args.links, "pool": args.pool}
    if args.json:
        print(json.dumps(res))
    else:
        n = res["n"]
        print(f"ranker={args.ranker} links={'on' if args.links else 'off'} pool={args.pool}  n={n}")
        print(f"  in pool {res['in_pool']}/{n}  top-1 {res['top1']}/{n}  top-5 {res['top5']}/{n}  "
              f"top-10 {res['top10']}/{n}  {res['sec_per_query']}s/query")
    return 0


def cmd_labels(args) -> int:
    con = _db(args)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for r in con.execute("SELECT query, passage, noul, model, source FROM labels ORDER BY id"):
            f.write(json.dumps(dict(r), ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} labels -> {args.out}")
    return 0


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="lacthu", description="Local retrieval map: BM25 + structure + links, reranked by Laya or TypeSafe.")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--db", help=f"map file (default {default_db()}, or $LACTHU_DB)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="index a directory as a source (re-running rebuilds it)")
    s.add_argument("path")
    s.add_argument("--name", help="source name (default: directory name)")
    s.add_argument("--public", action="store_true", help="allow this source's text to be sent to a cloud ranker")
    s.add_argument("--exclude", action="append", metavar="GLOB", help="skip paths matching GLOB (repeatable)")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("sources", help="list indexed sources")
    s.set_defaults(fn=cmd_sources)

    s = sub.add_parser("drop", help="remove a source from the map")
    s.add_argument("name")
    s.set_defaults(fn=cmd_drop)

    def ranking(s):
        s.add_argument("--ranker", choices=RANKERS, default=os.environ.get("LACTHU_RANKER", "none"),
                       help="reorder the pool: none (BM25 order), laya (local), typesafe (cloud, public sources only)")
        s.add_argument("--pool", type=int, default=30, help="BM25 candidates handed to the ranker")
        s.add_argument("--links", action="store_true",
                       help="also hand the ranker chunks linked to the top BM25 hits (off: measured no gain yet)")

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
    ranking(s)
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("labels", help="export TypeSafe judgments as fine-tuning data for Laya")
    s.add_argument("out")
    s.set_defaults(fn=cmd_labels)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
