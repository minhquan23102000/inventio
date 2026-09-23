"""Query time: BM25 finds seeds, links widen the pool, a ranker orders it.

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
    bm25_rank: int | None = None   # 1-based; None when the chunk entered through a link
    via: str = ""                  # how a linked chunk entered the pool
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
            "heading_path": self.heading_path, "score": self.score,
            "bm25_rank": self.bm25_rank, "via": self.via, "text": self.text, "links": self.links,
        }


def fts_query(q: str) -> str:
    words = [w for w in WORD.findall(q.lower()) if len(w) > 1]
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words))


HIT_SQL = """
SELECT c.id, s.name source, s.public, s.root, f.path, c.start_line, c.end_line, c.heading_path, c.text
FROM chunks c JOIN files f ON f.id = c.file_id JOIN sources s ON s.id = f.source_id
"""


def _source_filter(sources: list[str] | None) -> tuple[str, list]:
    if not sources:
        return "", []
    return f" AND s.name IN ({','.join('?' * len(sources))})", list(sources)


def bm25(con, q: str, k: int, sources: list[str] | None = None, head_weight: float = 1.0) -> list[Hit]:
    match = fts_query(q)
    if not match:
        return []
    where, args = _source_filter(sources)
    rows = con.execute(
        "SELECT c.id, s.name source, s.public, s.root, f.path, c.start_line, c.end_line, c.heading_path, c.text "
        "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
        "JOIN files f ON f.id = c.file_id JOIN sources s ON s.id = f.source_id "
        f"WHERE chunks_fts MATCH ?{where} ORDER BY bm25(chunks_fts, ?, 1.0) LIMIT ?",
        [match, *args, head_weight, k],
    ).fetchall()
    return [Hit(**{**dict(r), "public": bool(r["public"])}, bm25_rank=i + 1) for i, r in enumerate(rows)]


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


def search(con, q: str, *, k: int = 5, pool: int = 30, ranker=None, expand_links: bool = False,
           seeds: int = 5, expand_limit: int = 10, sources: list[str] | None = None) -> list[Hit]:
    hits = bm25(con, q, pool, sources)
    if expand_links:
        hits += expand(con, hits, seeds, expand_limit, sources)
    if ranker is not None and hits:
        scores = ranker.score(q, hits)
        for h, s in zip(hits, scores):
            h.score = s
        # stable sort: ties keep BM25 order, linked chunks after BM25 hits
        hits.sort(key=lambda h: -h.score)
    top = hits[:k]
    attach_links(con, top)
    return top
