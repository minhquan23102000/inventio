"""Query time: BM25 finds seeds, the predicted document types and links widen the pool, a ranker
orders it.

The ranker only ever reads the pool (tens of chunks), never the whole map.
"""

import re
from dataclasses import dataclass, field

WORD = re.compile(r"\w+", re.UNICODE)


@dataclass
class Hit:
    id: int
    source: str
    public: bool
    root: str
    path: str
    start_line: int
    end_line: int
    heading_path: str
    text: str
    type: str | None = None        # document type of the file (DOC_TYPES in ingest)
    bm25_rank: int | None = None   # 1-based; None when the chunk entered through a link
    via: str = ""                  # how a chunk entered the pool beyond plain BM25: type:..., rel:ident
    score: float | None = None
    links: list[dict] = field(default_factory=list)

    @property
    def coord(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    def passage(self) -> str:
        head = f"{self.path} > {self.heading_path}" if self.heading_path else self.path
        return f"[{head}]\n{self.text}"

    def as_dict(self) -> dict:
        return {
            "source": self.source, "path": self.path, "root": self.root,
            "start_line": self.start_line, "end_line": self.end_line,
            "heading_path": self.heading_path, "type": self.type, "score": self.score,
            "bm25_rank": self.bm25_rank, "via": self.via, "text": self.text, "links": self.links,
        }


def fts_query(q: str) -> str:
    words = [w for w in WORD.findall(q.lower()) if len(w) > 1]
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words))


HIT_SQL = """
SELECT c.id, s.name source, s.public, s.root, f.path, f.type, c.start_line, c.end_line, c.heading_path, c.text
FROM chunks c JOIN files f ON f.id = c.file_id JOIN sources s ON s.id = f.source_id
"""


def _source_filter(sources: list[str] | None) -> tuple[str, list]:
    if not sources:
        return "", []
    return f" AND s.name IN ({','.join('?' * len(sources))})", list(sources)


def bm25(con, q: str, k: int, sources: list[str] | None = None, head_weight: float = 1.0,
         types: list[str] | None = None) -> list[Hit]:
    match = fts_query(q)
    if not match:
        return []
    where, args = _source_filter(sources)
    if types:
        where += f" AND f.type IN ({','.join('?' * len(types))})"
        args += list(types)
    rows = con.execute(
        "SELECT c.id, s.name source, s.public, s.root, f.path, f.type, c.start_line, c.end_line, c.heading_path, c.text "
        "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
        "JOIN files f ON f.id = c.file_id JOIN sources s ON s.id = f.source_id "
        f"WHERE chunks_fts MATCH ?{where} ORDER BY bm25(chunks_fts, ?, 1.0) LIMIT ?",
        [match, *args, head_weight, k],
    ).fetchall()
    return [Hit(**{**dict(r), "public": bool(r["public"])}, bm25_rank=i + 1) for i, r in enumerate(rows)]


TYPE_MIN_P = 0.25  # a type the ranker gives at least this probability widens the pool


def scope_types(con, sources: list[str] | None = None) -> list[str]:
    where, args = _source_filter(sources)
    rows = con.execute(
        "SELECT DISTINCT f.type FROM files f JOIN sources s ON s.id = f.source_id "
        f"WHERE f.type IS NOT NULL{where} ORDER BY f.type",
        args,
    ).fetchall()
    return [r["type"] for r in rows]


def widen_by_type(con, q: str, pool: list[Hit], ranker, per_type: int,
                  sources: list[str] | None = None) -> list[Hit]:
    """BM25's best chunks inside each document type the ranker thinks the answer is.

    Only ever adds to the pool: a wrong guess costs a few extra candidates, never an answer
    plain BM25 had already found.
    """
    from .ingest import DOC_TYPES
    from .rankers import CloudRefused

    present = scope_types(con, sources)
    if len(present) < 2:
        return []
    if getattr(ranker, "cloud", False):  # widening may add any chunk in scope, so all of it must be public
        where, args = _source_filter(sources)
        private = [r["name"] for r in con.execute(f"SELECT name FROM sources s WHERE public = 0{where}", args)]
        if private:
            raise CloudRefused(
                f"--types with ranker 'typesafe' could send text from non-public source(s) {', '.join(private)}; "
                "restrict with --source, re-init them with --public, or use --ranker laya"
            )
    probs = ranker.types(q, {t: DOC_TYPES.get(t, t) for t in present})
    if not probs:
        return []
    chosen = [t for t, p in sorted(probs.items(), key=lambda kv: -kv[1]) if p >= TYPE_MIN_P] or [max(probs, key=probs.get)]
    have = {h.id for h in pool}
    out = []
    for t in chosen:
        for h in bm25(con, q, per_type, sources, types=[t]):
            if h.id not in have:
                have.add(h.id)
                h.via = f"type:{t}"
                out.append(h)
    return out


def expand(con, pool: list[Hit], seeds: int, limit: int, sources: list[str] | None = None) -> list[Hit]:
    """Chunks linked to the top seeds that BM25 did not already put in the pool."""
    seed_ids = [h.id for h in pool[:seeds]]
    if not seed_ids or limit <= 0:
        return []
    have = {h.id for h in pool}
    ph = ",".join("?" * len(seed_ids))
    rows = con.execute(
        f"""
        SELECT other, count(DISTINCT seed) n, min(rel) rel, group_concat(DISTINCT via) via FROM (
            SELECT dst other, src seed, rel, via FROM links WHERE src IN ({ph})
            UNION ALL
            SELECT src other, dst seed, rel, via FROM links WHERE dst IN ({ph})
        ) GROUP BY other ORDER BY n DESC, rel ASC
        """,
        seed_ids + seed_ids,
    ).fetchall()
    where, args = _source_filter(sources)
    out = []
    for r in rows:
        if r["other"] in have:
            continue
        row = con.execute(HIT_SQL + f" WHERE c.id = ?{where}", [r["other"], *args]).fetchone()
        if row is None:
            continue
        out.append(Hit(**{**dict(row), "public": bool(row["public"])}, via=f"{r['rel']}:{r['via']}"))
        if len(out) >= limit:
            break
    return out


LINK_SQL = """
SELECT l.rel, l.via, s.name source, f.path, c.start_line, c.end_line, '{dir}' dir
FROM links l JOIN chunks c ON c.id = l.{other} JOIN files f ON f.id = c.file_id JOIN sources s ON s.id = f.source_id
WHERE l.{me} = ?
"""


def attach_links(con, hits: list[Hit], per_hit: int = 5) -> None:
    """Where each hit leads: outgoing links first, then incoming; one entry per target."""
    for h in hits:
        rows = con.execute(
            LINK_SQL.format(dir="out", other="dst", me="src") + " UNION ALL "
            + LINK_SQL.format(dir="in", other="src", me="dst"),
            (h.id, h.id),
        ).fetchall()
        seen, h.links = set(), []
        for r in rows:
            coord = f"{r['path']}:{r['start_line']}-{r['end_line']}"
            if (r["source"], coord) in seen:
                continue
            seen.add((r["source"], coord))
            h.links.append({"rel": r["rel"], "dir": r["dir"], "via": r["via"], "source": r["source"], "coord": coord})
            if len(h.links) >= per_hit:
                break


def rank_key(h: Hit) -> tuple[bool, float]:
    return (h.score is None, -(h.score or 0.0))


def search(con, q: str, *, k: int = 5, pool: int = 30, ranker=None, expand_links: bool = False,
           by_type: bool = False, type_limit: int | None = None, seeds: int = 5, expand_limit: int = 10,
           sources: list[str] | None = None) -> list[Hit]:
    hits = bm25(con, q, pool, sources)
    if by_type and ranker is not None:
        # as deep inside each predicted type as the pool goes overall: at depth 10, BM25's best
        # chunks of the predicted type were nearly always in the pool already (SWE-bench smoke)
        hits += widen_by_type(con, q, hits, ranker, type_limit or pool, sources)
    if expand_links:
        hits += expand(con, hits, seeds, expand_limit, sources)
    if ranker is not None and hits:
        scores = ranker.score(q, hits)
        for h, s in zip(hits, scores):
            h.score = s
        # stable sort: ties keep BM25 order, linked chunks after BM25 hits, unscored chunks last
        hits.sort(key=rank_key)
    top = hits[:k]
    attach_links(con, top)
    return top
