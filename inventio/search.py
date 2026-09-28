"""Query time: BM25 finds seeds; predicted document types, predicted content categories and links
widen the pool; a ranker orders it. Widening only ever adds candidates, never removes one.

The ranker only ever reads the pool (tens of chunks), never the whole map.
"""

import math
import re
from dataclasses import dataclass, field

from .scope import Scope
from .scope import where as scope_where

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
            "bm25_rank": self.bm25_rank, "via": self.via, "text": self.text,
            "links": [{k: v for k, v in l.items() if k != "id"} for l in self.links],   # a chunk id is the map's own
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


def bm25(con, q: str, k: int, scope: Scope | None = None, head_weight: float = 1.0,
         types: list[str] | None = None, categories: list[str] | None = None,
         files: list[int] | None = None, phrases: bool | None = None) -> list[Hit]:
    match = fts_query(q, phrases)
    if not match:
        return []
    where, args = scope_where(scope)
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


def scope_types(con, scope: Scope | None = None) -> list[str]:
    where, args = scope_where(scope, chunks=False)
    rows = con.execute(
        "SELECT DISTINCT f.type FROM files f JOIN sources s ON s.id = f.source_id "
        f"WHERE f.type IS NOT NULL{where} ORDER BY f.type",
        args,
    ).fetchall()
    return [r["type"] for r in rows]


def widen_by_type(con, q: str, pool: list[Hit], ranker, per_type: int,
                  scope: Scope | None = None) -> list[Hit]:
    """BM25's best chunks inside each document type the ranker thinks the answer is.

    Only ever adds to the pool: a wrong guess costs a few extra candidates, never an answer
    plain BM25 had already found.
    """
    from .ingest import DOC_TYPES
    from .rankers import CloudRefused

    present = scope_types(con, scope)
    if len(present) < 2:
        return []
    if getattr(ranker, "cloud", False):  # widening may add any chunk in scope, so all of it must be public
        where, args = scope_where(scope, chunks=False)
        private = [r["name"] for r in con.execute(
            "SELECT DISTINCT s.name FROM files f JOIN sources s ON s.id = f.source_id "
            f"WHERE s.public = 0{where}", args)]
        if private:
            raise CloudRefused(
                f"--types with ranker 'typesafe' could send text from non-public source(s) {', '.join(private)}; "
                "narrow with --source or -w, re-init them with --public, or use --ranker dispositio"
            )
    probs = ranker.types(q, {t: DOC_TYPES.get(t, t) for t in present})
    if not probs:
        return []
    chosen = [t for t, p in sorted(probs.items(), key=lambda kv: -kv[1]) if p >= TYPE_MIN_P] or [max(probs, key=probs.get)]
    have = {h.id for h in pool}
    out = []
    for t in chosen:
        for h in bm25(con, q, per_type, scope, types=[t]):
            if h.id not in have:
                have.add(h.id)
                h.via = f"type:{t}"
                out.append(h)
    return out


def widen_by_category(con, q: str, pool: list[Hit], categories: list[str], limit: int,
                      scope: Scope | None = None) -> list[Hit]:
    """BM25's best chunks among those that keep one of the query's predicted categories."""
    if not categories or limit <= 0:
        return []
    have = {h.id for h in pool}
    out = []
    for h in bm25(con, q, len(pool) + limit, scope, categories=categories):
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


def widen_by_symbols(con, q: str, pool: list[Hit], limit: int, scope: Scope | None = None) -> list[Hit]:
    """BM25's best chunks of files whose path the query names, then chunks that define a name the
    query uses (a name defined in more than MAX_DEFINERS places is skipped). Code decides both."""
    idents, paths = query_symbols(q)
    have, out = {h.id for h in pool}, []
    where, args = scope_where(scope)
    if paths:
        fwhere, fargs = scope_where(scope, chunks=False)
        files = [r["id"] for r in con.execute(
            f"SELECT f.id, f.path FROM files f JOIN sources s ON s.id = f.source_id WHERE 1 = 1{fwhere}", fargs)
            if any((s := _stem(r["path"])) == p or s.endswith("/" + p) for p in paths)][:20]
        for h in bm25(con, q, 3 * len(files), scope, files=files) if files else []:
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


def neighbours_of(con, seed: Hit, scope: Scope | None = None) -> list[int]:
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
    hits = bm25(con, q, NEIGHBOUR_FETCH, scope) if q else []
    return [h.id for h in hits if (h.source, h.path) != (seed.source, seed.path)][:NEIGHBOURS_PER_SEED]


def widen_by_neighbours(con, pool: list[Hit], seeds: int = 5, limit: int = 10,
                        scope: Scope | None = None) -> list[Hit]:
    """The neighbours of the top seeds that BM25 did not already put in the pool, those shared by
    the most seeds first. A pseudo-relevance feedback per seed, decided by code: on SciFact it
    finds more answers than the same neighbours pruned by a model (`about` links, facts.py)."""
    from collections import Counter

    have, votes = {h.id for h in pool}, Counter()
    for h in pool[:seeds]:
        votes.update(b for b in neighbours_of(con, h, scope) if b not in have)
    out = []
    for b, _ in votes.most_common(limit):
        row = con.execute(HIT_SQL + " WHERE c.id = ?", [b]).fetchone()
        out.append(Hit(**{**dict(row), "public": bool(row["public"])}, via="neighbour"))
    return out


def expand(con, pool: list[Hit], seeds: int, limit: int, scope: Scope | None = None,
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
    where, args = scope_where(scope)
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
SELECT l.rel, l.via, c.id other_id, s.name source, f.path, c.start_line, c.end_line, '{dir}' dir
FROM links l JOIN chunks c ON c.id = l.{other} JOIN files f ON f.id = c.file_id JOIN sources s ON s.id = f.source_id
WHERE l.{me} = ?
"""


def attach_links(con, hits: list[Hit], per_hit: int = 5) -> None:
    """Where each hit leads, named by what the map knows for certain (`name_link`): links the sources draw
    (citation, mentions) only, since a model-judged `about` link can be wrong where a shown link must not be;
    links to another kind of thing first (a runbook's code before another runbook), one of each kind before
    a second; one entry per target."""
    for h in hits:
        rows = con.execute(
            LINK_SQL.format(dir="out", other="dst", me="src") + " UNION ALL "
            + LINK_SQL.format(dir="in", other="src", me="dst"),
            (h.id, h.id),
        ).fetchall()
        seen, links = set(), []
        for r in rows:
            if r["rel"] == "about":
                continue
            coord = f"{r['path']}:{r['start_line']}-{r['end_line']}"
            if (r["source"], coord) in seen:
                continue
            seen.add((r["source"], coord))
            links.append({"rel": r["rel"], "dir": r["dir"], "via": r["via"], "source": r["source"], "coord": coord,
                          "id": r["other_id"]})
        kinds = chunk_kinds(con, [h.id] + [l["id"] for l in links])
        for l in links:
            l.update(name_link(con, h, l, kinds))
        mine, taken = kinds[h.id]["kind"], set()
        order = []
        # another kind of thing first; within that, a link one side states (defines, uses, a ticket's verb, a
        # Markdown link) before two pages that only name the same thing; then the sources' order
        for l in sorted(links, key=lambda l: (l["kind"] == mine, l["fact"].endswith(" too"))):
            order.append((l["kind"] in taken, len(order), l))
            taken.add(l["kind"])
        h.links = [l for *_, l in sorted(order, key=lambda x: x[:2])][:per_hit]


# The lowest probability at which a judged category is printed as a link's noun: precision >= 0.95 at >= 40%
# coverage of its predictions, overall and out of domain, on the 380 held-out passages (`systemone.py judge
# v4-calib`, calibration). Rule, Reference, Explanation and Other never reach it (Rule: 1.0 at 0.85 but 6 of
# 18 out-of-domain predictions kept), so they print as `page`. Finding clears at every threshold with one
# out-of-domain example; 0.8 keeps 30 of its 34 at precision 1.0.
KIND_TAU = {"Procedure": 0.8, "Record": 0.7, "Finding": 0.8}
CATEGORY_NOUN = {"Procedure": "steps", "Record": "record", "Finding": "finding"}
TYPE_NOUN = {"SoftwareSourceCode": "code", "Test": "test", "Configuration": "config", "Dataset": "table"}


def chunk_kinds(con, ids: list[int]) -> dict[int, dict]:
    """What each chunk is, from facts the map holds: `kind` (a noun: code, test, a ticket's type, a pull
    request, a confidently judged category, else page), and `title` (its heading path)."""
    ids = list(dict.fromkeys(ids))
    ph = ",".join("?" * len(ids))
    out = {r["id"]: {"type": r["type"], "title": r["heading_path"], "file_id": r["file_id"], "category": r["category"],
                     "p": r["p"]}
           for r in con.execute(
               f"""SELECT c.id, c.heading_path, c.file_id, f.type, cc.category, cc.p FROM chunks c
                   JOIN files f ON f.id = c.file_id
                   LEFT JOIN chunk_categories cc ON cc.chunk_id = c.id AND cc.kept = 1
                   WHERE c.id IN ({ph})""", ids)}
    meta: dict[int, dict] = {}
    fids = list({v["file_id"] for v in out.values()})
    for r in con.execute(f"SELECT file_id, key, value FROM file_meta WHERE key IN ('issuetype', 'item', 'state') "
                         f"AND file_id IN ({','.join('?' * len(fids))})", fids):
        meta.setdefault(r["file_id"], {})[r["key"]] = r["value"]
    for v in out.values():
        m = meta.get(v["file_id"], {})
        v["ticket"] = bool(m.get("issuetype"))
        if v["type"] in TYPE_NOUN:
            v["kind"] = TYPE_NOUN[v["type"]]
        elif m.get("issuetype"):
            v["kind"] = m["issuetype"].lower()
        elif m.get("item"):
            v["kind"] = ("pull request" if m["item"] == "pull" else "issue") + (f" ({m['state']})" if m.get("state") else "")
        elif v["category"] in KIND_TAU and (v["p"] or 0.0) >= KIND_TAU[v["category"]]:
            v["kind"] = CATEGORY_NOUN[v["category"]]
        else:
            v["kind"] = "page"
    return out


CLOSES = re.compile(r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+(?:[\w.-]+/[\w.-]+)?#(\d+)", re.I)


def name_link(con, h: Hit, l: dict, kinds: dict) -> dict:
    """The link's `kind` (what the target is), `fact` (what makes the link, in the sources' own words where
    they have them) and `title`. Nothing here is guessed: a Jira link verb is the line the ticket carries,
    `defines`/`uses` is which side defines the name, `closes` is a closing keyword naming the item."""
    t, tickets = kinds[l["id"]], kinds[h.id]["ticket"] and kinds[l["id"]]["ticket"]
    fact = ""
    if l["rel"] == "mentions":
        name = l["via"]
        # a ticket's `Links:` lines (`- is caused by SHOP-9 ...`, connectors/jira.py): only between two tickets,
        # where a bullet naming the other's key is that line and not a step of a runbook
        link_line = rf"^\s*-\s+(.+?)\s+{re.escape(name)}\b"
        if l["dir"] == "out":   # this chunk names it: the ticket's own link line, else who defines it
            verb = re.search(link_line, h.text, re.I | re.M) if tickets else None
            defines = con.execute("SELECT 1 FROM idents WHERE chunk_id = ? AND ident = ? AND role = 'defines'",
                                  (l["id"], name)).fetchone()
            fact = verb.group(1) if verb else f"defines {name}" if defines else f"names {name} too"
        else:                   # the target names it
            other = con.execute("SELECT text FROM chunks WHERE id = ?", (l["id"],)).fetchone()["text"]
            verb = re.search(link_line, other, re.I | re.M) if tickets else None
            mine = con.execute("SELECT 1 FROM idents WHERE chunk_id = ? AND ident = ? AND role = 'defines'",
                               (h.id, name)).fetchone()
            fact = f"{verb.group(1)} this" if verb else f"uses {name}" if mine else f"names {name} too"
    elif l["rel"] == "citation":
        fact = "links to" if l["dir"] == "out" else "links here"
        m = CLOSES.search(h.text) if l["dir"] == "out" else None
        if m and re.search(rf"#{m.group(1)}\b", t["title"] or ""):
            fact = "closes"
    else:
        fact = l["rel"]
    return {"kind": t["kind"], "fact": fact, "title": t["title"]}


def rank_key(h: Hit) -> tuple[bool, float]:
    return (h.score is None, -(h.score or 0.0))


def widen_by_facts(con, q: str, pool: list[Hit], judge, *, limit: int = 10, seeds: int = 5,
                   scope: Scope | None = None) -> list[Hit]:
    """The judge predicts which content categories the query is about; BM25's best chunks of those
    categories join the pool, then the chunks the top seeds are linked to by judged `about` links."""
    from .facts import kept_categories, query_categories

    cats = kept_categories(query_categories(con, judge, q))
    added = widen_by_category(con, q, pool, cats, limit, scope)
    return added + expand(con, pool + added, seeds, limit, scope, rels=("about",))


def _flat(groups: list[list[Hit]]) -> list[Hit]:
    return [h for g in groups for h in g]


def _interleave(groups: list[list[Hit]]) -> list[Hit]:
    """One from each group in turn, each chunk once: a ranker reading only the first few still sees
    every kind of widening."""
    out, seen = [], set()
    for i in range(max((len(g) for g in groups), default=0)):
        for g in groups:
            if i < len(g) and g[i].id not in seen:
                seen.add(g[i].id)
                out.append(g[i])
    return out


def search(con, q: str, *, k: int = 5, pool: int = 15, ranker=None, expand_links: bool = False,
           by_type: bool = False, type_limit: int | None = None, facts=None, facts_limit: int = 10,
           symbols: bool = True, neighbours: bool = False, seeds: int = 5, expand_limit: int = 10,
           scope: Scope | None = None) -> list[Hit]:
    """`facts` is a judge (facts.make_judge) that predicts the query's content categories."""
    hits = bm25(con, q, pool, scope)
    first, added = len(hits), []   # what each widening step adds, kept apart so none crowds the others out
    if symbols:
        # files and definitions the question names. SWE-bench Lite `mixed`: the answer file in the
        # pool 63% -> 73% for 3 more candidates; nDCG@10 against a BM25 pool of the same size
        # 0.430 -> 0.495 ranked by an earlier dispositio, 0.401 -> 0.456 unranked with these hits first
        added.append(widen_by_symbols(con, q, hits, pool, scope))
    if by_type and ranker is not None:
        # as deep inside each predicted type as the pool goes overall: at depth 10, BM25's best
        # chunks of the predicted type were nearly always in the pool already (SWE-bench smoke)
        added.append(widen_by_type(con, q, hits + _flat(added), ranker, type_limit or pool, scope))
    if facts is not None:
        added.append(widen_by_facts(con, q, hits + _flat(added), facts, limit=facts_limit, seeds=seeds, scope=scope))
    if expand_links:
        added.append(expand(con, hits + _flat(added), seeds, expand_limit, scope))
    if neighbours:
        added.append(widen_by_neighbours(con, hits + _flat(added), seeds, expand_limit, scope))
    named = added[0] if symbols else []
    hits += _interleave(added)
    if ranker is not None and hits:
        # dispositio reads BM25's pool in passes of 15, and what widening added takes up to 7 seats in one
        # more pass beside the pool's best 8 (rankers.SystemOneRanker.score); interleaved, so the named
        # files, the links and the neighbours each get seats
        scores = ranker.score(q, hits, first) if getattr(ranker, "passes", False) else ranker.score(q, hits)
        for h, s in zip(hits, scores):
            h.score = s
        # stable sort: ties keep BM25 order, linked chunks after BM25 hits, unscored chunks last
        hits.sort(key=rank_key)
    elif named:  # no ranker to reorder: what the question names goes ahead of BM25's guesses
        first_ids = {h.id for h in named}
        hits = named + [h for h in hits if h.id not in first_ids]
    top = hits[:k]
    attach_links(con, top)
    return top
