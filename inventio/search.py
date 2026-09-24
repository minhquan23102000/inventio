"""Query time: BM25 finds seeds; predicted document types, predicted content categories and links
widen the pool; a ranker orders it. Widening only ever adds candidates, never removes one.

The ranker only ever reads the pool (tens of chunks), never the whole map.
"""

import math
import re
from dataclasses import dataclass, field

WORD = re.compile(r"\w+", re.UNICODE)
# letters only Vietnamese writes: đ, ơ, ư, ă, and the hook-above and dot-below tone marks
VIETNAMESE = re.compile("[đĐơƠưƯăĂ\u1ea0-\u1ef9]")


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


def fts_query(q: str, phrases: bool | None = None) -> str:
    """Every word of the query, OR-ed, plus each pair of adjacent words as a phrase when `phrases`
    (default: when the query is Vietnamese). Vietnamese writes a word as syllables apart, so
    `hợp đồng` (contract) is two index terms; the phrase restores the word. Measured with BM25
    alone: Zalo legal nDCG@10 0.543 -> 0.756, while English SciFact falls 0.670 -> 0.618, where
    a pair of words is rarely one word."""
    import unicodedata

    toks = WORD.findall(q.lower())
    terms = [f'"{w}"' for w in toks if len(w) > 1]
    if phrases is None:
        phrases = bool(VIETNAMESE.search(unicodedata.normalize("NFC", q)))
    if phrases:
        terms += [f'"{a} {b}"' for a, b in zip(toks, toks[1:])]
    return " OR ".join(dict.fromkeys(terms))


HIT_SQL = """
SELECT c.id, s.name source, s.public, s.root, f.path, f.type, c.start_line, c.end_line, c.heading_path, c.text
FROM chunks c JOIN files f ON f.id = c.file_id JOIN sources s ON s.id = f.source_id
"""


def _source_filter(sources: list[str] | None) -> tuple[str, list]:
    if not sources:
        return "", []
    return f" AND s.name IN ({','.join('?' * len(sources))})", list(sources)


def bm25(con, q: str, k: int, sources: list[str] | None = None, head_weight: float = 1.0,
         types: list[str] | None = None, categories: list[str] | None = None,
         files: list[int] | None = None, phrases: bool | None = None) -> list[Hit]:
    match = fts_query(q, phrases)
    if not match:
        return []
    where, args = _source_filter(sources)
    if types:
        where += f" AND f.type IN ({','.join('?' * len(types))})"
        args += list(types)
    if categories:  # chunks that keep one of these content categories (facts.py)
        where += (" AND c.id IN (SELECT chunk_id FROM chunk_categories WHERE kept = 1 "
                  f"AND category IN ({','.join('?' * len(categories))}))")
        args += list(categories)
    if files:
        where += f" AND f.id IN ({','.join('?' * len(files))})"
        args += list(files)
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
                "restrict with --source, re-init them with --public, or use --ranker dispositio"
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


def widen_by_category(con, q: str, pool: list[Hit], categories: list[str], limit: int,
                      sources: list[str] | None = None) -> list[Hit]:
    """BM25's best chunks among those that keep one of the query's predicted categories."""
    if not categories or limit <= 0:
        return []
    have = {h.id for h in pool}
    out = []
    for h in bm25(con, q, len(pool) + limit, sources, categories=categories):
        if h.id not in have:
            h.bm25_rank, h.via = None, "category:" + ",".join(categories)
            out.append(h)
            if len(out) >= limit:
                break
    return out


PATHLIKE = re.compile(r"[\w.\-]+(?:/[\w.\-]+)+|\b[a-z_][\w]*(?:\.[a-z_]\w*){2,}\b")
MAX_DEFINERS = 3  # a name defined in more places than this names nothing in particular


def _stem(path: str) -> str:
    return re.sub(r"\.\w+$", "", path.lower().replace("\\", "/").strip("./"))


def query_symbols(q: str) -> tuple[set[str], set[str]]:
    """Identifiers and path fragments a query names: `nightly_backup`, `Header.fromstring`,
    a traceback's `astropy/io/fits/header.py` and `line 5, in fromstring`, a dotted module
    `astropy.io.fits.header`."""
    from .ingest import mentions_in, norm_ident

    idents = set()
    for x in mentions_in(q) | {norm_ident(m) for m in re.findall(r"line \d+, in (\w{3,})", q)}:
        idents.add(x)
        idents.update(p for p in re.split(r"[./:]", x) if len(p) >= 3)
    paths = set()
    for m in PATHLIKE.findall(q):
        p = m.strip("./")
        if "/" not in p:  # dotted module: astropy.io.fits.header -> astropy/io/fits/header
            p = p.replace(".", "/")
        if "/" in p:
            paths.add(_stem(p))
    return idents, paths


def widen_by_symbols(con, q: str, pool: list[Hit], limit: int, sources: list[str] | None = None) -> list[Hit]:
    """BM25's best chunks of files whose path the query names, then chunks that define a name the
    query uses (a name defined in more than MAX_DEFINERS places is skipped). Code decides both."""
    idents, paths = query_symbols(q)
    have, out = {h.id for h in pool}, []
    where, args = _source_filter(sources)
    if paths:
        files = [r["id"] for r in con.execute(
            f"SELECT f.id, f.path FROM files f JOIN sources s ON s.id = f.source_id WHERE 1 = 1{where}", args)
            if any((s := _stem(r["path"])) == p or s.endswith("/" + p) for p in paths)][:20]
        for h in bm25(con, q, 3 * len(files), sources, files=files) if files else []:
            if h.id not in have:
                have.add(h.id)
                h.bm25_rank, h.via = None, "path"
                out.append(h)
    if idents:
        ph = ",".join("?" * len(idents))
        rows = con.execute(
            f"""SELECT i.chunk_id, i.ident FROM idents i JOIN chunks c ON c.id = i.chunk_id
                JOIN files f ON f.id = c.file_id JOIN sources s ON s.id = f.source_id
                WHERE i.role = 'defines' AND i.ident IN ({ph}){where}
                  AND i.ident IN (SELECT ident FROM idents WHERE role = 'defines' AND ident IN ({ph})
                                  GROUP BY ident HAVING count(*) <= {MAX_DEFINERS})
                ORDER BY length(i.ident) DESC""",
            [*idents, *args, *idents],
        ).fetchall()
        for r in rows:
            if r["chunk_id"] not in have:
                have.add(r["chunk_id"])
                row = con.execute(HIT_SQL + " WHERE c.id = ?", [r["chunk_id"]]).fetchone()
                out.append(Hit(**{**dict(row), "public": bool(row["public"])}, via=f"defines:{r['ident']}"))
    return out[:limit]


NEIGHBOUR_TERMS, NEIGHBOUR_FETCH, NEIGHBOURS_PER_SEED = 24, 50, 10


def _fold(w: str) -> str:
    """A word as the index stores it (unicode61 remove_diacritics 2): `hợp` -> `hop`."""
    import unicodedata

    return "".join(ch for ch in unicodedata.normalize("NFKD", w) if not unicodedata.combining(ch))


def neighbours_of(con, seed: Hit, sources: list[str] | None = None) -> list[int]:
    """Chunks of other files that share this chunk's most distinctive words: its 24 words with the
    highest tf-idf, as one BM25 query."""
    from collections import Counter

    tf = Counter(_fold(w) for w in WORD.findall(seed.text.lower()) if len(w) > 1)
    if not tf:
        return []
    n = con.execute("SELECT count(*) FROM chunks").fetchone()[0] or 1
    df, words = {}, list(tf)
    for i in range(0, len(words), 500):
        part = words[i:i + 500]
        df.update(con.execute(f"SELECT term, doc FROM chunks_vocab WHERE col = 'body' AND term IN "
                              f"({','.join('?' * len(part))})", part).fetchall())
    score = {w: c * math.log(n / df[w]) for w, c in tf.items() if df.get(w, 0) >= 2}
    q = " ".join(sorted(score, key=lambda w: -score[w])[:NEIGHBOUR_TERMS])
    hits = bm25(con, q, NEIGHBOUR_FETCH, sources) if q else []
    return [h.id for h in hits if (h.source, h.path) != (seed.source, seed.path)][:NEIGHBOURS_PER_SEED]


def widen_by_neighbours(con, pool: list[Hit], seeds: int = 5, limit: int = 10,
                        sources: list[str] | None = None) -> list[Hit]:
    """The neighbours of the top seeds that BM25 did not already put in the pool, those shared by
    the most seeds first. A pseudo-relevance feedback per seed, decided by code: on SciFact it
    finds more answers than the same neighbours pruned by a model (`about` links, facts.py)."""
    from collections import Counter

    have, votes = {h.id for h in pool}, Counter()
    for h in pool[:seeds]:
        votes.update(b for b in neighbours_of(con, h, sources) if b not in have)
    out = []
    for b, _ in votes.most_common(limit):
        row = con.execute(HIT_SQL + " WHERE c.id = ?", [b]).fetchone()
        out.append(Hit(**{**dict(row), "public": bool(row["public"])}, via="neighbour"))
    return out


def expand(con, pool: list[Hit], seeds: int, limit: int, sources: list[str] | None = None,
           rels: tuple[str, ...] | None = None) -> list[Hit]:
    """Chunks linked to the top seeds that BM25 did not already put in the pool."""
    seed_ids = [h.id for h in pool[:seeds]]
    if not seed_ids or limit <= 0:
        return []
    have = {h.id for h in pool}
    ph = ",".join("?" * len(seed_ids))
    only = f" AND rel IN ({','.join('?' * len(rels))})" if rels else ""
    rel_args = list(rels or ())
    rows = con.execute(
        f"""
        SELECT other, count(DISTINCT seed) n, min(rel) rel, group_concat(DISTINCT via) via FROM (
            SELECT dst other, src seed, rel, via FROM links WHERE src IN ({ph}){only}
            UNION ALL
            SELECT src other, dst seed, rel, via FROM links WHERE dst IN ({ph}){only}
        ) GROUP BY other ORDER BY n DESC, rel ASC
        """,
        seed_ids + rel_args + seed_ids + rel_args,
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


def widen_by_facts(con, q: str, pool: list[Hit], judge, *, limit: int = 10, seeds: int = 5,
                   sources: list[str] | None = None) -> list[Hit]:
    """The judge predicts which content categories the query is about; BM25's best chunks of those
    categories join the pool, then the chunks the top seeds are linked to by judged `about` links."""
    from .facts import kept_categories, query_categories

    cats = kept_categories(query_categories(con, judge, q))
    added = widen_by_category(con, q, pool, cats, limit, sources)
    return added + expand(con, pool + added, seeds, limit, sources, rels=("about",))


def search(con, q: str, *, k: int = 5, pool: int = 30, ranker=None, expand_links: bool = False,
           by_type: bool = False, type_limit: int | None = None, facts=None, facts_limit: int = 10,
           symbols: bool = True, neighbours: bool = False, seeds: int = 5, expand_limit: int = 10,
           sources: list[str] | None = None) -> list[Hit]:
    """`facts` is a judge (facts.make_judge) that predicts the query's content categories."""
    hits = bm25(con, q, pool, sources)
    named = []
    if symbols:
        # files and definitions the question names. SWE-bench Lite `mixed`: the answer file in the
        # pool 63% -> 73% for 3 more candidates; nDCG@10 against a BM25 pool of the same size
        # 0.430 -> 0.495 ranked by an earlier dispositio, 0.401 -> 0.456 unranked with these hits first
        named = widen_by_symbols(con, q, hits, pool, sources)
        hits += named
    if by_type and ranker is not None:
        # as deep inside each predicted type as the pool goes overall: at depth 10, BM25's best
        # chunks of the predicted type were nearly always in the pool already (SWE-bench smoke)
        hits += widen_by_type(con, q, hits, ranker, type_limit or pool, sources)
    if facts is not None:
        hits += widen_by_facts(con, q, hits, facts, limit=facts_limit, seeds=seeds, sources=sources)
    if neighbours:
        hits += widen_by_neighbours(con, hits, seeds, expand_limit, sources)
    if expand_links:
        hits += expand(con, hits, seeds, expand_limit, sources)
    if ranker is not None and hits:
        scores = ranker.score(q, hits)
        for h, s in zip(hits, scores):
            h.score = s
        # stable sort: ties keep BM25 order, linked chunks after BM25 hits, unscored chunks last
        hits.sort(key=rank_key)
    elif named:  # no ranker to reorder: what the question names goes ahead of BM25's guesses
        first = {h.id for h in named}
        hits = named + [h for h in hits if h.id not in first]
    top = hits[:k]
    attach_links(con, top)
    return top
