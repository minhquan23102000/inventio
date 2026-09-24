"""`inventio show`: what the map knows about a coordinate, and where it leads.

`read` prints what a passage says; `show` prints the passage as a node of the map: what it is
(document type, content category, the names it defines and mentions, where it lives on the web),
its links in and out, its place in the file (the section around it, the one before and after),
and the passages of other files that share its most distinctive words. Every line names who
decided it: code (the path, a Markdown link, an identifier) or a model with its probability, so
a reader can tell what is certain from what was judged. Every printed coordinate opens with
`read` or `show`."""

from dataclasses import dataclass, field

from .connectors import KINDS, RemoteError
from .read import resolve
from .search import Hit, neighbours_of

SIMILAR = 3       # passages of other files printed under `similar`
MAX_NAMES = 8     # names printed per chunk (every one with --json)

NODE_SQL = """
SELECT c.id, s.name source, f.path, f.type, c.kind, c.parent_id, c.start_line, c.end_line, c.heading_path,
       cc.category, cc.p, cc.model
FROM chunks c JOIN files f ON f.id = c.file_id JOIN sources s ON s.id = f.source_id
LEFT JOIN chunk_categories cc ON cc.chunk_id = c.id AND cc.kept = 1
"""


@dataclass
class Node:
    """A chunk as a line of `show`: its coordinate, heading, type and kept category."""
    id: int
    source: str
    path: str
    start: int
    end: int
    heading: str
    type: str | None
    kind: str
    category: str | None = None
    p: float | None = None
    by: str | None = None  # the model that chose the category, or "code"
    extra: dict = field(default_factory=dict)

    @property
    def coord(self) -> str:
        return f"{self.source}:{self.path}:{self.start}-{self.end}"

    def label(self) -> str:
        cat = ""
        if self.category:
            cat = f" · {self.category}" + (" (code)" if self.by == "code" else f" p={self.p:.2f} ({self.by})")
        return f"{self.coord}  {self.heading or '(top)'}  · {self.type or '?'}{cat}"

    def as_dict(self) -> dict:
        return {"coord": self.coord, "heading": self.heading, "type": self.type, "kind": self.kind,
                "category": self.category, "p": self.p, "category_by": self.by, **self.extra}


def _node(r) -> Node:
    return Node(r["id"], r["source"], r["path"], r["start_line"], r["end_line"], r["heading_path"], r["type"],
                r["kind"], r["category"], r["p"], r["model"])


def _one(con, chunk_id: int) -> Node | None:
    r = con.execute(NODE_SQL + " WHERE c.id = ?", (chunk_id,)).fetchone()
    return _node(r) if r else None


def show(con, target: str) -> dict:
    """The node behind a coordinate (a file, or the chunks its lines overlap) or a mirrored URL."""
    p = resolve(con, target)
    if p.source is None:
        raise RemoteError(f"{target} is not in the map (no source mirrors it); `inventio read` prints it live")
    src = con.execute("SELECT kind, public FROM sources WHERE name = ?", (p.source,)).fetchone()
    rows = con.execute(NODE_SQL + " WHERE s.name = ? AND f.path = ? AND c.end_line >= ? AND c.start_line <= ? "
                                  "ORDER BY c.start_line", (p.source, p.path, p.start, p.end)).fetchall()
    if not rows:
        raise RemoteError(f"{p.source}:{p.path} has no indexed chunk in lines {p.start}-{p.end}")
    chunks = [_node(r) for r in rows]
    ids = {c.id for c in chunks}
    f = con.execute("SELECT f.lang, f.type FROM files f JOIN sources s ON s.id = f.source_id "
                    "WHERE s.name = ? AND f.path = ?", (p.source, p.path)).fetchone()
    by_connector = src["kind"] != "dir" and getattr(KINDS.get(src["kind"]), "DOC_TYPE", None)
    for c in chunks:
        names = con.execute("SELECT ident, role FROM idents WHERE chunk_id = ? ORDER BY ident", (c.id,)).fetchall()
        c.extra = {role: [r["ident"] for r in names if r["role"] == role] for role in ("defines", "mentions")}
    return {
        "coord": f"{p.source}:{p.path}:{p.start}-{p.end}",
        "heading": chunks[0].heading,
        "type": f["type"], "type_by": "connector" if by_connector else "path",
        "lang": f["lang"], "source_kind": src["kind"], "public": bool(src["public"]),
        "url": p.url, "version": p.version,
        "chunks": chunks,
        "links": _links(con, ids),
        "structure": _structure(con, p, chunks, ids),
        "similar": _similar(con, p, chunks),
    }


def _links(con, ids: set[int]) -> list[tuple[str, str, str, dict, Node]]:
    """(direction, rel, via, judgment, node) for every link leaving the node's chunks; links
    between two of its own chunks are left out, and a pair linked both ways is one entry
    (`both`). `about` links carry the model's p."""
    from .facts import SAME, key, passage

    ph = ",".join("?" * len(ids))
    rows = con.execute(f"SELECT src, dst, rel, via, 'out' dir FROM links WHERE src IN ({ph}) "
                       f"UNION ALL SELECT src, dst, rel, via, 'in' dir FROM links WHERE dst IN ({ph})",
                       [*ids, *ids]).fetchall()
    order = {"citation": 0, "mentions": 1, "about": 2}
    found: dict[tuple[str, int], list] = {}
    for r in sorted(rows, key=lambda r: (order.get(r["rel"], 3), r["dir"] != "out")):
        other = r["dst"] if r["dir"] == "out" else r["src"]
        if other in ids:
            continue
        d = "~" if r["rel"] == "about" else r["dir"]
        if (r["rel"], other) in found:
            e = found[(r["rel"], other)]
            if e[0] != d and d != "~":
                e[0] = "both"
                e[2] = ", ".join(dict.fromkeys((*e[2].split(", "), r["via"])))
            continue
        node = _one(con, other)
        if node is None:
            continue
        judged = {}
        if r["rel"] == "about":  # the pair was judged with the seeking chunk (src) as the passage
            a, b = (con.execute("SELECT f.path, c.heading_path, c.text FROM chunks c JOIN files f ON f.id = c.file_id "
                                "WHERE c.id = ?", (i,)).fetchone() for i in (r["src"], r["dst"]))
            j = con.execute("SELECT p, model FROM judgments WHERE key = ?",
                            (key(SAME, SAME, passage(*a), passage(*b)),)).fetchone()
            judged = {"p": j["p"], "model": j["model"]} if j else {}
        found[(r["rel"], other)] = [d, r["rel"], r["via"], judged, node]
    return [tuple(e) for e in found.values()]


def _structure(con, p, chunks: list[Node], ids: set[int]) -> list[tuple[str, Node]]:
    """The section around the node, and the chunks just before and after it in the same file."""
    out = []
    parent = con.execute("SELECT parent_id FROM chunks WHERE id = ?", (chunks[0].id,)).fetchone()[0]
    if parent and parent not in ids and (n := _one(con, parent)):
        out.append(("parent", n))
    near = " WHERE s.name = ? AND f.path = ? AND c.{} ORDER BY c.start_line {} LIMIT 1"
    for rel, cond, order in (("before", "end_line < ?", "DESC"), ("after", "start_line > ?", "ASC")):
        r = con.execute(NODE_SQL + near.format(cond, order),
                        (p.source, p.path, p.start if rel == "before" else p.end)).fetchone()
        if r:
            out.append((rel, _node(r)))
    return out


def _similar(con, p, chunks: list[Node]) -> list[Node]:
    """Chunks of other files sharing the node's most distinctive words (search.neighbours_of)."""
    text = "\n".join(r["text"] for r in con.execute(
        f"SELECT text FROM chunks WHERE id IN ({','.join('?' * len(chunks))})", [c.id for c in chunks]))
    seed = Hit(id=chunks[0].id, source=p.source, public=False, root="", path=p.path, start_line=p.start,
               end_line=p.end, heading_path="", text=text)
    return [n for i in neighbours_of(con, seed)[:SIMILAR] if (n := _one(con, i))]


def render(node: dict) -> list[str]:
    """The text `inventio show` prints."""
    about = [f"{node['type']} (by {node['type_by']})", node["lang"], "public" if node["public"] else "private"]
    if node["url"]:
        about.append(node["url"])
    if node["version"]:
        about.append(f"version {node['version']}")
    one = len(node["chunks"]) == 1
    lines = [f"{node['coord']}  {node['heading'] or '(top)'}" if one else node["coord"],
             "  " + " · ".join(x for x in about if x)]
    for c in node["chunks"]:
        facts = [f"{c.kind} {c.start}-{c.end}"]
        if not one:
            facts.append(c.heading or "(top)")
        if c.category:
            facts.append(c.category + (" (code)" if c.by == "code" else f" p={c.p:.2f} ({c.by})"))
        defines, mentions = c.extra.get("defines", []), c.extra.get("mentions", [])
        if defines:
            more = f" +{len(defines) - MAX_NAMES}" if len(defines) > MAX_NAMES else ""
            facts.append(f"defines {', '.join(defines[:MAX_NAMES])}{more}")
        if mentions:  # the ones defined somewhere are the links below; every one is in --json
            facts.append(f"mentions {len(mentions)} name{'s' * (len(mentions) != 1)}")
        lines.append("  " + " · ".join(facts))
    if node["links"]:
        lines.append("links")
        arrow = {"out": "->", "in": "<-", "both": "<>", "~": "~ "}
        for d, rel, via, judged, n in node["links"]:
            why = f"p={judged['p']:.2f} ({judged['model']})" if judged else f"({via})"
            lines.append(f"  {arrow[d]} {rel:<8} {n.label()}  {why}")
    if node["structure"]:
        lines.append("structure")
        mark = {"parent": "^", "before": "<", "after": ">"}
        lines += [f"  {mark[rel]} {n.label()}" for rel, n in node["structure"]]
    if node["similar"]:
        lines.append("similar")
        lines += [f"  {n.label()}" for n in node["similar"]]
    return lines


def as_json(node: dict) -> dict:
    return {**node, "chunks": [c.as_dict() for c in node["chunks"]],
            "links": [{"dir": d, "rel": rel, "via": via, **judged, **n.as_dict()} for d, rel, via, judged, n in node["links"]],
            "structure": [{"rel": rel, **n.as_dict()} for rel, n in node["structure"]],
            "similar": [n.as_dict() for n in node["similar"]]}
