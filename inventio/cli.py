import argparse
import json
import errno
import os
import shutil
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

    con = _db(args)
    t = time.time()
    if "://" in args.path:
        return _init_remote(con, args, t)
    if args.jql:
        print("--jql applies to a Jira URL", file=sys.stderr)
        return 2
    root = Path(args.path)
    if not root.is_dir():
        print(f"not a directory or a supported URL: {root}", file=sys.stderr)
        return 2
    name = args.name or root.resolve().name
    row = con.execute("SELECT root FROM sources WHERE name = ?", (name,)).fetchone()
    if row and Path(row["root"]).resolve() != root.resolve() and not args.full:
        # one map holds every source: a second `docs` folder must not silently replace the first
        print(f"source {name!r} already indexes {row['root']}; give this one another --name, "
              f"or pass --full to re-point {name!r} here", file=sys.stderr)
        return 2
    with con:
        stats = ingest_source(con, name, root, args.public, args.exclude or [], full=args.full)
        links = rebuild_links(con)
    _report(stats, links, t, args)
    if args.facts:
        args.source = [name]
        return cmd_facts(args)
    return 0


def _init_remote(con, args, t: float) -> int:
    from . import connectors
    from .links import rebuild_links

    try:
        found = connectors.for_url(args.path, args.jql)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    if not found:
        print(f"not a Confluence/Jira URL, a database URL (postgresql://, mysql://, sqlite:///...), "
              f"kafka:// or s3://: {args.path}", file=sys.stderr)
        return 2
    kind, origin = found
    try:
        with con:
            stats = connectors.sync(con, args.name, kind, origin, args.public)
            links = rebuild_links(con)
    except connectors.RemoteError as e:
        print(str(e), file=sys.stderr)
        return 2
    _report(stats, links, t, args)
    if args.facts:
        args.source = [stats["source"]]
        return cmd_facts(args)
    return 0


def _report(stats: dict, links: dict, t: float, args) -> None:
    mirrored = (f"; mirror {stats['fetched']} fetched, {stats['moved']} moved, {stats['deleted']} deleted"
                if "fetched" in stats else "")
    print(
        f"{stats['source']}: {stats['files']} files, {stats['chunks']} chunks "
        f"(+{stats['added']} new, ~{stats['changed']} edited, -{stats['removed']} removed, "
        f"{stats['unchanged']} unchanged){mirrored}; "
        f"map links {', '.join(f'{k} {v}' for k, v in sorted(links.items())) or 'none'}; "
        f"{time.time() - t:.1f}s -> {args.db or default_db()}"
    )


def cmd_sync(args) -> int:
    """Bring every source (or the named ones) up to date: directories re-read, remote sources
    re-listed and only what changed fetched."""
    from . import connectors
    from .ingest import ingest_source
    from .links import rebuild_links

    con = _db(args)
    rows = con.execute("SELECT * FROM sources ORDER BY name").fetchall()
    if args.source:
        unknown = set(args.source) - {r["name"] for r in rows}
        if unknown:
            print(f"no source named {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        rows = [r for r in rows if r["name"] in args.source]
    for r in rows:
        t = time.time()
        try:
            with con:
                if r["kind"] == "dir":
                    excludes = [x for x in r["excludes"].split("\n") if x]
                    stats = ingest_source(con, r["name"], Path(r["root"]), bool(r["public"]), excludes)
                else:
                    stats = connectors.sync(con, r["name"], r["kind"], r["origin"], bool(r["public"]))
                links = rebuild_links(con)
        except connectors.RemoteError as e:
            print(f"{r['name']}: {e}", file=sys.stderr)
            return 2
        _report(stats, links, t, args)
    if args.facts:
        args.source = [r["name"] for r in rows]
        return cmd_facts(args)
    return 0


def cmd_read(args) -> int:
    from .connectors import RemoteError
    from .read import resolve

    con = _db(args)
    try:
        p = resolve(con, args.target)
    except RemoteError as e:
        print(str(e), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(p.as_dict(), ensure_ascii=False, indent=2))
        return 0
    where = f"{p.source}:{p.path}:{p.start}-{p.end}" if p.source else f"{p.path} (read live, not in the map)"
    print("  ".join(x for x in (where, p.url or "", f"version {p.version}" if p.version else "") if x))
    width = len(str(p.end))
    for n, line in enumerate(p.lines, p.start):
        print(f"{n:>{width}}  {line}")
    return 0


def cmd_show(args) -> int:
    from .connectors import RemoteError
    from .show import as_json, render, show

    con = _db(args)
    try:
        node = show(con, args.target)
    except RemoteError as e:
        print(str(e), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(as_json(node), ensure_ascii=False, indent=2))
    else:
        print("\n".join(render(node)))
    return 0


def cmd_grep(args) -> int:
    """Every line that matches, in the files the map indexes: exhaustive where `query` ranks a
    few. Files are read where `read` reads them (the directory, or the mirror of a remote source),
    so every printed coordinate opens with `read`."""
    import re

    from .ingest import file_text

    con = _db(args)
    try:
        rx = re.compile(args.pattern, re.IGNORECASE if args.ignore_case else 0)
    except re.error as e:
        print(f"not a regular expression: {e}", file=sys.stderr)
        return 2
    where, params = "", []
    if args.source:
        where = f" WHERE s.name IN ({','.join('?' * len(args.source))})"
        params = args.source
    rows = con.execute("SELECT s.name, s.root, f.path FROM files f JOIN sources s ON s.id = f.source_id"
                       + where + " ORDER BY s.name, f.path", params).fetchall()
    hits, total, files = [], 0, 0
    for r in rows:
        try:
            text = file_text(Path(r["root"]) / r["path"])
        except OSError:
            continue
        if not rx.search(text):
            continue
        files += 1
        for n, line in enumerate(text.split("\n"), 1):
            if rx.search(line):
                total += 1
                if len(hits) < args.max:
                    hits.append({"source": r["name"], "path": r["path"], "line": n, "text": line.rstrip()})
    if args.json:
        print(json.dumps({"matches": hits, "total": total, "files": files}, ensure_ascii=False, indent=2))
        return 0 if total else 1
    for h in hits:
        print(f"{h['source']}:{h['path']}:{h['line']}  {h['text'].strip()[:200]}")
    if total > len(hits):
        print(f"... {total - len(hits)} more ({total} lines in {files} files); narrow with --source or the pattern, or raise -m")
    if not total:
        print("no match")
    return 0 if total else 1


def cmd_ls(args) -> int:
    """Browse a source like a folder; a file lists its chunks, each with the coordinate `read` takes."""
    if not args.target:
        return cmd_sources(args)
    con = _db(args)
    source, _, prefix = args.target.partition(":")
    row = con.execute("SELECT id FROM sources WHERE name = ?", (source,)).fetchone()
    if not row:
        print(f"no source named {source!r}; see `inventio sources`", file=sys.stderr)
        return 2
    prefix = prefix.strip("/")
    f = con.execute("SELECT id FROM files WHERE source_id = ? AND path = ?", (row["id"], prefix)).fetchone()
    if f:
        for c in con.execute("SELECT start_line, end_line, kind, heading_path FROM chunks WHERE file_id = ? "
                             "ORDER BY start_line", (f["id"],)):
            print(f"{source}:{prefix}:{c['start_line']}-{c['end_line']}  {c['kind']:<8} {c['heading_path']}")
        return 0
    under = prefix + "/" if prefix else ""
    entries: dict[str, list] = {}  # name -> [files or None for a file, chunks]
    for r in con.execute("SELECT f.path, count(c.id) n FROM files f LEFT JOIN chunks c ON c.file_id = f.id "
                         "WHERE f.source_id = ? GROUP BY f.id", (row["id"],)):
        if not r["path"].startswith(under):
            continue
        head, sep, _ = r["path"][len(under):].partition("/")
        if sep:
            e = entries.setdefault(head + "/", [0, 0])
            e[0] += 1
            e[1] += r["n"]
        else:
            entries[head] = [None, r["n"]]
    if not entries:
        print(f"{args.target}: no such file or folder in the map", file=sys.stderr)
        return 2
    plural = lambda n, w: f"{n} {w}" + ("" if n == 1 else "s")  # noqa: E731
    for name, (n_files, n_chunks) in sorted(entries.items()):
        size = plural(n_chunks, "chunk") if n_files is None else f"{plural(n_files, 'file')}, {plural(n_chunks, 'chunk')}"
        print(f"{source}:{under}{name}  {size}")
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
        "SELECT s.name, s.root, s.kind, s.origin, s.public, s.indexed_at, count(DISTINCT f.id) files, count(c.id) chunks "
        "FROM sources s LEFT JOIN files f ON f.source_id = s.id LEFT JOIN chunks c ON c.file_id = f.id "
        "GROUP BY s.id ORDER BY s.name"
    ).fetchall()
    for r in rows:
        where = r["root"] if r["kind"] == "dir" else f"{r['kind']} {r['origin']}"
        print(f"{r['name']:<16} {'public ' if r['public'] else 'private'} {r['files']:>5} files {r['chunks']:>6} chunks  {where}  ({r['indexed_at']})")
    if not rows:
        print("no sources; run `inventio init <path>`")
    return 0


def cmd_drop(args) -> int:
    from .links import rebuild_links
    from .store import drop_source

    con = _db(args)
    row = con.execute("SELECT id, kind, root FROM sources WHERE name = ?", (args.name,)).fetchone()
    if not row:
        print(f"no source named {args.name}", file=sys.stderr)
        return 2
    with con:
        drop_source(con, row["id"])
        con.execute("DELETE FROM sources WHERE id = ?", (row["id"],))
        rebuild_links(con)
    if row["kind"] != "dir":  # the mirror is Inventio's own copy; a directory source is the user's
        shutil.rmtree(row["root"], ignore_errors=True)
    print(f"dropped {args.name}")
    return 0


def _snippet(text: str, width: int = 160) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.lstrip().startswith(("#", "<!--"))]
    s = " ".join(lines)
    return s[:width] + ("…" if len(s) > width else "")


def cmd_query(args) -> int:
    from .connectors import web_url
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
        rows = [{**h.as_dict(), "url": web_url(h.root, h.path, h.heading_path)} for h in hits]
        print(json.dumps(rows, ensure_ascii=False, indent=2))
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
            url = web_url(h.root, h.path, h.heading_path)
            if url:
                print(f"   {url}")
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
    p = argparse.ArgumentParser(
        prog="inventio", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Local retrieval map: BM25 + structure + links, reranked by dispositio or TypeSafe.",
        epilog="""\
finding an answer (a person or an agent):
  inventio query "why was there no backup?"          ranked passages, each with a coordinate
  inventio read wiki:runbook.md:8-13                 the lines behind it
  inventio read https://<site>/browse/SHOP-812       or a Confluence/Jira URL met on the way
  inventio show wiki:runbook.md:8-13                 what the map knows about it: type, category,
                                                     links in and out, the sections around it,
                                                     similar passages elsewhere
  inventio grep "nightly_backup"                     every line that says it, not just the best few
  inventio ls wiki:Ops/                              what a source holds, like a folder; a file
                                                     lists its sections

query when you have a question, show to see where a result leads, grep when you have an
exact name (all callers, every page that cites a ticket), ls when you need to see what is
there. Every command prints source:path:start-end coordinates that read and show open; add
--json to query, read, show and grep for machine output. `inventio <command> -h` for options.""")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--db", help=f"map file (default {default_db()}, or $INVENTIO_DB)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="index a directory, a Confluence space, a Jira project/search, or the schema "
                                    "of a database, Kafka or S3 as a source; re-running updates only what changed")
    s.add_argument("path", help="a directory, or a URL: https://<site>/wiki/spaces/KEY, https://<site>/browse/PROJ, "
                                "https://<site>/issues/?jql=..., postgresql://user@host/db (any SQLAlchemy URL), "
                                "kafka://broker:9092[?registry=URL], s3://bucket/prefix/")
    s.add_argument("--name", help="source name (default: directory name, wiki-KEY, jira-PROJ)")
    s.add_argument("--jql", help="for Jira: narrow the tickets, e.g. \"updated >= -365d\"")
    s.add_argument("--public", action="store_true", help="allow this source's text to be sent to a cloud ranker")
    s.add_argument("--exclude", action="append", metavar="GLOB", help="skip paths matching GLOB (repeatable)")
    s.add_argument("--full", action="store_true", help="drop the source and rebuild it from scratch")
    s.add_argument("--facts", action="store_true", help="then judge content categories and fact links (see `facts`)")
    judge = os.environ.get("INVENTIO_JUDGE", "dispositio")
    judges = "dispositio (local), laya (local, as published), typesafe (cloud, public sources only)"
    s.add_argument("--judge", choices=JUDGES, default=judge, help=f"model for --facts: {judges}")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("sync", help="bring every source up to date: directories re-read, Confluence and Jira "
                                    "re-listed, only changed pages and tickets fetched")
    s.add_argument("--source", action="append", metavar="NAME", help="only this source (repeatable)")
    s.add_argument("--facts", action="store_true", help="then judge content categories and fact links")
    s.add_argument("--judge", choices=JUDGES, default=judge, help=f"model for --facts: {judges}")
    s.set_defaults(fn=cmd_sync)

    s = sub.add_parser("read", help="print the lines behind a coordinate or a Confluence/Jira URL")
    s.add_argument("target", help="source:path[:start[-end]], or a page or ticket URL (#heading and "
                                  "focusedCommentId narrow it to that section)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_read)

    s = sub.add_parser("show", help="what the map knows about a coordinate or URL: type, category, names, "
                                    "links in and out, the sections around it, similar passages")
    s.add_argument("target", help="source:path[:start[-end]] (a file, or the sections these lines overlap), "
                                  "or a mirrored page or ticket URL")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("grep", help="every line matching a regular expression, in every indexed file "
                                    "(Confluence and Jira too); exhaustive where query ranks")
    s.add_argument("pattern", help="Python regular expression, e.g. \"nightly_backup\" or \"SHOP-8\\d\\d\"")
    s.add_argument("-i", "--ignore-case", action="store_true")
    s.add_argument("--source", action="append", metavar="NAME", help="only this source (repeatable)")
    s.add_argument("-m", "--max", type=int, default=50, help="lines to print (the total is always counted)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_grep)

    s = sub.add_parser("ls", help="browse a source like a folder: its folders and files, or a file's sections")
    s.add_argument("target", nargs="?", help="source, source:folder/ or source:path/to/file (none: list sources)")
    s.set_defaults(fn=cmd_ls)

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
    try:
        return args.fn(args)
    except OSError as e:
        # the reader stopped reading (`inventio read ... | head`): what it wanted is already out.
        # POSIX says EPIPE, Windows EINVAL; anything on a terminal is a real error
        if e.errno not in (errno.EPIPE, errno.EINVAL) or sys.stdout.isatty():
            raise
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())  # the exit flush would fail again
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
