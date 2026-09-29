"""Sources that are not a directory on disk: a Confluence space, a Jira query, a GitHub
repository's issues and pull requests, a database's schema; later mail or chat.

Each is mirrored into a directory of Markdown files (mirror.py), one per page, ticket, thread or
table, and that directory is indexed like any other source. Chunking, BM25, links, judgments
and the incremental re-index are therefore shared; a connector only says what exists and turns
one item into Markdown. A new kind is one module that provides the names below. Built-in kinds
are listed in KINDS here; a kind from another package registers its module under the entry-point
group `inventio.connectors` (pyproject: [project.entry-points."inventio.connectors"]
notion = "inventio_notion"), so `pip install` it and `inventio init <its URL>` works. This list
is the plugin interface (CONNECTOR_API below counts its changes):

    KIND                        the value kept in sources.kind (and the entry point's name)
    DOC_TYPE                    optional: the document type of every file (Dataset for schema
                                cards); otherwise decided from the path, as for a directory
    origin(url, query)          the canonical URL a source of this kind syncs from, or None when
                                the URL is not this kind's (`query` narrows it, e.g. a JQL)
    locate(url)                 (item id, fragment) of the item a URL points at, or None
    heading_url(item, heads)    where a chunk under these headings lives on the web, or None
    fetch_url(url)              (web URL, Doc) of one item read live, for items not mirrored
    Remote(origin)              .origin, .name (a default source name), .FORMAT (an int raised
                                when the Markdown written changes, so every item is written again),
                                .listing() -> {item id: mirror.Entry}, every item without its body,
                                .fetch(ids) -> yields (item id, mirror.Doc) for the ids asked for;
                                Doc.meta holds the fields a query filters on (status, labels, updated)

  optional, for a kind from another package that signs in (`inventio login <url>` asks it):
    login(url, ask, ask_secret) ask for the credential, try it once, keep it; returns what to say
    logout(url)                 forget it; returns what to say
    credential(origin)          where the credential comes from: "env", "keyring" or "missing"

  helpers a plugin may import, kept as stable as the list above: mirror.Doc and mirror.Entry;
  http.Client (get, post, raw, retrying where the server asks) and RemoteError; markdown's
  segment, safe, fence, table and tidy; ingest.slug (a heading's anchor key, for Doc.anchors);
  credentials.get, put, delete and Missing (the keychain, one JSON secret per key).
"""

from importlib.metadata import entry_points
from pathlib import Path

from . import confluence, github, jira, kafka, mirror, s3, sql
from .http import RemoteError

CONNECTOR_API = 1  # raised when a name above changes meaning; a plugin may check it


_BUILTIN = (confluence, jira, sql, kafka, s3, github)  # github last: it may ask `gh` about a host


def _plugins() -> list[tuple[str, object]]:
    """(kind, module) of every connector other packages registered. One that fails to import is
    reported, not fatal; one naming a built-in kind is refused, so installing a package never
    reroutes Jira."""
    import sys

    taken, out = {m.KIND for m in _BUILTIN}, []
    for ep in entry_points(group="inventio.connectors"):
        early = sys.modules.get(ep.module)
        if early is not None and not hasattr(early, "KIND"):
            # the plugin was imported first (a test, a script) and its own `from inventio...` led
            # here: the module is half-built, so it is registered by the entry point's name, which
            # must be its KIND, and it finishes building once this import returns
            kind, m = ep.name, early
        else:
            try:
                m = ep.load()
                kind = m.KIND
            except Exception as e:  # a plugin must not take the CLI down
                print(f"inventio: connector plugin {ep.name} not loaded: {type(e).__name__}: {e}", file=sys.stderr)
                continue
        if kind in taken:
            print(f"inventio: connector plugin {ep.name} not loaded: kind {kind!r} is taken", file=sys.stderr)
            continue
        taken.add(kind)
        out.append((kind, m))
    return out


KINDS = {m.KIND: m for m in _BUILTIN[:-1]} | dict(_plugins()) | {_BUILTIN[-1].KIND: _BUILTIN[-1]}

__all__ = ["KINDS", "RemoteError", "for_url", "sync", "web_url"]


def for_url(url: str, query: str | None = None) -> tuple[str, str] | None:
    """(kind, origin) of a URL naming something that can be synced: a space, a project, a query."""
    for kind, m in KINDS.items():
        o = m.origin(url, query)
        if o:
            return kind, o
    return None


def sync(con, name: str | None, kind: str, origin: str, public: bool, log=print) -> dict:
    """Mirror a remote source, then index the mirror. Returns ingest's counts plus the mirror's."""
    from ..ingest import ingest_source

    remote = KINDS[kind].Remote(origin)
    name = name or remote.name
    row = con.execute("SELECT kind, origin, root FROM sources WHERE name = ?", (name,)).fetchone()
    if row and (row["kind"], row["origin"]) != (kind, origin):  # one map, many sources: never replace one silently
        raise RemoteError(f"source {name!r} already indexes {row['origin'] or row['root']}; give this one another --name")
    root = mirror.mirror_dir(name)
    counts = mirror.sync(kind, remote, root, log)
    stats = ingest_source(con, name, root, public, [], dtype=getattr(KINDS[kind], "DOC_TYPE", None))
    con.execute("UPDATE sources SET kind = ?, origin = ? WHERE name = ?", (kind, origin, name))
    write_meta(con, name, root)
    return {**stats, **counts}


def write_meta(con, name: str, root: Path) -> None:
    """The mirror's per-item fields as file_meta rows, replacing the source's old ones."""
    files = {r["path"]: r["id"] for r in con.execute(
        "SELECT f.id, f.path FROM files f JOIN sources s ON s.id = f.source_id WHERE s.name = ?", (name,))}
    con.execute("DELETE FROM file_meta WHERE file_id IN (SELECT f.id FROM files f JOIN sources s "
                "ON s.id = f.source_id WHERE s.name = ?)", (name,))
    rows = []
    for item in mirror.load(root)["items"].values():
        fid = files.get(item["path"])
        for key, value in (item.get("meta") or {}).items() if fid else ():
            rows += [(fid, key, v) for v in ([value] if isinstance(value, str) else value) if v]
    con.executemany("INSERT INTO file_meta (file_id, key, value) VALUES (?, ?, ?)", rows)


def web_url(root: str, path: str, heading_path: str) -> str | None:
    """Where a chunk of a mirrored file lives on the web; None for a file of a directory source."""
    hit = mirror.item_at(root, path)
    if hit is None:
        return None
    kind, item = hit
    return KINDS[kind].heading_url(item, [h for h in heading_path.split(" > ") if h])


def locate(url: str) -> tuple[str, str, str] | None:
    """(kind, item id, fragment) of a URL naming one item."""
    for kind, m in KINDS.items():
        hit = m.locate(url)
        if hit:
            return kind, *hit
    return None


def mirrored(con, kind: str, item_id: str) -> tuple[str, Path, dict] | None:
    """(source name, file, manifest item) of an item some source mirrors."""
    for r in con.execute("SELECT name, root FROM sources WHERE kind = ?", (kind,)):
        item = mirror.load(Path(r["root"]))["items"].get(item_id)
        if item:
            return r["name"], Path(r["root"]) / item["path"], item
    return None
