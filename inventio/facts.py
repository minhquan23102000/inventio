"""Content categories and fact links, judged by a decision model when a source is indexed.

A category says what a passage does for its reader: states a rule, walks through a procedure,
describes something for lookup, explains why, reports a finding, or records what happened. One
choice question per chunk of prose, so every chunk keeps exactly one category (none for Other)
and the categories cut a corpus into regions instead of all covering it. A question asks for
the category of passage that would answer it, the same way.

Fact links: each chunk's BM25 neighbours in other files, of a different category, are asked
whether the two passages state something about the same specific thing; pairs judged true become
`about` links. They join the rule to the procedure that carries it out, the incident where it
failed, the decision that changed it; two passages of one category about one thing are most
often near-copies, and BM25 already finds those. Structure (headings, definitions, Markdown
links) is drawn by code in ingest and links; this module is the one place a model decides what
the map says.

Every judgment is stored in `judgments` with the text it read, the model and the source: a cache,
so a rebuilt map pays nothing twice.
"""

import hashlib
import math
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from .rankers import CloudRefused, load_laya

CATEGORIES = {
    "Rule": "states what must, may or must not be done: a requirement, a policy, a law, a limit or "
            "threshold, a convention",
    "Procedure": "tells the reader how to do something: steps, commands to run, a setup or usage guide, "
                 "a runbook",
    "Reference": "describes what something is or contains, for lookup: an API, a signature, parameters, "
                 "fields, options, a schema, a definition",
    "Explanation": "explains why or how something works: a mechanism, a rationale, a design, background, "
                   "a comparison of approaches",
    "Finding": "reports what was measured or observed: results, a benchmark, an experiment, evidence for "
               "a claim",
    "Record": "tells what happened or was decided at a point in time: an incident, a bug report, a change "
              "log, release notes, meeting notes, a decision",
    "Other": "does none of these: a table of contents, a list of links or names, credits, boilerplate",
}
NONE = "Other"    # a chunk whose category is Other keeps none, and is never linked or routed to
KEEP_P = 0.5      # a pair becomes a link at p >= this
NEIGHBOURS = 10   # neighbours judged per chunk
FETCH = 50        # BM25 candidates read to find them, before the type and same-file filters
MLT_TERMS = 24    # the chunk's most distinctive terms form its neighbour query
SAME = "same_thing"
WORD = re.compile(r"\w+", re.UNICODE)


def category_question() -> dict:
    return {"type": "choice", "instructions": "What does the `passage` do for its reader?", "criteria": CATEGORIES}


def query_category_question() -> dict:
    return {"type": "choice", "instructions": "What kind of passage would answer the `query`?",
            "criteria": CATEGORIES}


def same_question(slot: str) -> dict:
    return {
        "type": "noul",
        "instructions": f"Do the `passage` and `{slot}` each state something about the same specific thing?",
        "criteria": {
            "true": "Both passages make a statement about one and the same specific thing: the same event, "
                    "person, organization, place, product, procedure, concept or work.",
            "false": "They share only a topic, a general word or a kind of thing, or one of them names the "
                     "thing only in passing.",
        },
    }


def key(kind: str, question: str, passage: str, other: str = "") -> str:
    return hashlib.sha1("\x1f".join((kind, question, passage, other)).encode("utf-8")).hexdigest()


def kept_categories(probs: dict[str, float]) -> list[str]:
    """The most likely category, unless it is Other: at most one."""
    if not probs:
        return []
    top = max(probs, key=probs.get)
    return [] if top == NONE else [top]


def passage(path: str, heading_path: str, text: str) -> str:
    """The text a judge reads for a chunk; the same form rankers read (search.Hit.passage)."""
    head = f"{path} > {heading_path}" if heading_path else path
    return f"[{head}]\n{text}"


# --------------------------------------------------------------------------------------- judges
# A judge answers a yes/no question (`type` noul) with p, a choice question with {label: p}.

class JevJudge:
    """TypeSafe Jev in the cloud. Packs every question about one chunk into one call."""

    cloud = True
    packs = True

    def __init__(self, model: str | None = None, workers: int = 16, timeout: float = 120.0, retries: int = 5):
        from typesafe_sdk import TypeSafeClient

        if not os.environ.get("TYPESAFE_API_KEY"):
            raise RuntimeError("TYPESAFE_API_KEY is not set")
        self.client = TypeSafeClient(timeout=timeout)
        self.model = model or os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-latest")
        self.name = self.model
        self.workers = workers
        self.retries = retries
        self.refused = 0   # texts the API's edge firewall rejected (HTTP 403 "Attention Required")
        self.failed = 0    # calls that still failed after every retry; a rerun asks again

    def _call(self, state: dict, questions: dict[str, dict]) -> dict | None:
        from typesafe_sdk import Choice, Noul, NoulCriteria, TypeSafePermissionDeniedError
        from typesafe_sdk._core.errors import TypeSafeAPIError

        qs = {k: Choice(instructions=q["instructions"], criteria=q["criteria"]) if q["type"] == "choice"
              else Noul(instructions=q["instructions"], criteria=NoulCriteria(**q["criteria"])) for k, q in questions.items()}
        for attempt in range(self.retries):
            try:
                r = self.client.system_one(state=state, questions=qs, model=self.model)
                return {k: dict(r.choices[k].probabilities) if q["type"] == "choice" else float(r.nouls[k].noul)
                        for k, q in questions.items()}
            except TypeSafePermissionDeniedError as e:
                if "Attention Required" not in str(e):
                    raise  # the key itself
                self.refused += 1
                return None
            except TypeSafeAPIError as e:
                if e.status in (401, 402):
                    raise  # bad key or no credits left: retrying only turns every call into a "failed"
                if attempt == self.retries - 1:
                    self.failed += 1
                    return None
                time.sleep(2 ** attempt)
            except Exception:  # timeout, rate limit, 5xx: back off and ask again
                if attempt == self.retries - 1:
                    self.failed += 1
                    return None
                time.sleep(2 ** attempt)
        return None

    def batch(self, jobs):
        """Yield (i, answers or None) for an iterator of (state, questions) jobs, in completion
        order. Jobs are pulled lazily, so the caller may build them from the map meanwhile."""
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            live, it = {}, enumerate(jobs)
            while True:
                while len(live) < self.workers * 2:
                    nxt = next(it, None)
                    if nxt is None:
                        break
                    i, (state, qs) = nxt
                    live[pool.submit(self._call, state, qs)] = i
                if not live:
                    return
                done, _ = wait(live, return_when=FIRST_COMPLETED)
                for f in done:
                    yield live.pop(f), f.result()


class LayaJudge:
    """Laya on this machine. Asks all questions about one state in one forward pass."""

    cloud = False
    packs = False

    def __init__(self):
        self.agent, self.name = load_laya()
        self.refused = self.failed = 0

    def batch(self, jobs):
        for i, (state, qs) in enumerate(jobs):
            ans = self.agent.predict(state, qs)["answers"]
            yield i, {k: dict(ans[k]["probabilities"]) if q["type"] == "choice" else float(ans[k]["noul"])
                      for k, q in qs.items()}


JUDGES = ("laya", "typesafe")


def make_judge(name: str):
    if name == "laya":
        return LayaJudge()
    if name == "typesafe":
        return JevJudge()
    raise ValueError(f"unknown judge {name!r}; choose from {', '.join(JUDGES)}")


# ------------------------------------------------------------------------------------ index time

def _scope(sources: list[str] | None) -> tuple[str, list]:
    if not sources:
        return "", []
    return f" AND s.name IN ({','.join('?' * len(sources))})", list(sources)


CHUNK_SQL = """
SELECT c.id, c.file_id, s.name source, s.public, f.path, c.heading_path, c.text
FROM chunks c JOIN files f ON f.id = c.file_id JOIN sources s ON s.id = f.source_id
WHERE f.lang IN ('markdown', 'text')
"""


def _cache(con, kind: str, model: str) -> dict[str, float]:
    return {r["key"]: r["p"] for r in con.execute("SELECT key, p FROM judgments WHERE kind = ? AND model = ?", (kind, model))}


def _refuse_private(judge, rows) -> None:
    if judge.cloud:
        private = sorted({r["source"] for r in rows if not r["public"]})
        if private:
            raise CloudRefused(
                f"judge 'typesafe' would send text from non-public source(s) {', '.join(private)} to the cloud; "
                "restrict with --source, re-init them with --public, or use --judge laya"
            )


def _store(con, kind, question, text, other, p, model, source):
    con.execute(
        "INSERT OR IGNORE INTO judgments (kind, question, passage, other, key, p, model, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (kind, question, text, other, key(kind, question, text, other), p, model, source),
    )


def categorize(con, judge, sources: list[str] | None = None, log=print) -> dict:
    """Give every prose chunk in scope that has no category yet one p per category. Categories of
    an earlier scheme are dropped first, with the fact links they gated."""
    labels = ",".join("?" * len(CATEGORIES))
    stale = [r[0] for r in con.execute(
        f"SELECT DISTINCT chunk_id FROM chunk_categories WHERE category NOT IN ({labels})", list(CATEGORIES))]
    if stale:
        ph = ",".join(map(str, stale))
        con.execute(f"DELETE FROM chunk_categories WHERE chunk_id IN ({ph})")
        con.execute(f"DELETE FROM links WHERE rel = 'about' AND (src IN ({ph}) OR dst IN ({ph}))")
    where, args = _scope(sources)
    rows = con.execute(
        CHUNK_SQL + where + " AND NOT EXISTS (SELECT 1 FROM chunk_categories cc WHERE cc.chunk_id = c.id) ORDER BY c.id",
        args,
    ).fetchall()
    _refuse_private(judge, rows)
    cache = _cache(con, "category", judge.name)
    questions = {"category": category_question()}
    texts = [passage(r["path"], r["heading_path"], r["text"]) for r in rows]
    stats = {"chunks": len(rows), "asked": 0, "cached": 0, "refused": 0, "failed": 0, "done": []}

    def put(i, probs, fresh):
        r, text = rows[i], texts[i]
        keep = set(kept_categories(probs))
        for c, p in probs.items():
            if fresh:
                _store(con, "category", c, text, "", p, judge.name, r["source"])
            con.execute(
                "INSERT OR REPLACE INTO chunk_categories (chunk_id, category, p, kept, model) VALUES (?, ?, ?, ?, ?)",
                (r["id"], c, p, int(c in keep), judge.name),
            )
        stats["done"].append(r["id"])

    todo = []
    for i, text in enumerate(texts):
        ks = {c: key("category", c, text) for c in CATEGORIES}
        if all(k in cache for k in ks.values()):
            put(i, {c: cache[k] for c, k in ks.items()}, False)
            stats["cached"] += 1
        else:
            todo.append(i)
    con.commit()
    t0 = time.time()
    jobs = (({"passage": texts[i]}, questions) for i in todo)
    for n, (j, ans) in enumerate(judge.batch(jobs), 1):
        if ans is not None:
            put(todo[j], ans["category"], True)
            stats["asked"] += 1
        if n % 200 == 0:
            con.commit()
            log(f"  categories {n}/{len(todo)}  {time.time() - t0:.0f}s")
    con.commit()
    stats["refused"], stats["failed"] = judge.refused, judge.failed
    return stats


def _terms(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if len(w) > 1]


def link_facts(con, judge, chunk_ids: list[int] | None = None, sources: list[str] | None = None, log=print) -> dict:
    """Judge each chunk's neighbours of another category; true pairs become `about` links.

    `chunk_ids` limits which chunks look for neighbours (default: every categorized chunk in
    scope); neighbours come from the whole scope. Pairs already judged, in either direction, cost
    no call, so a rerun after a crash only repeats the BM25 lookups.
    """
    from .search import bm25

    where, args = _scope(sources)
    rows = con.execute(CHUNK_SQL + where + " ORDER BY c.id", args).fetchall()
    _refuse_private(judge, rows)
    kept = {r["chunk_id"]: r["category"] for r in con.execute("SELECT chunk_id, category FROM chunk_categories WHERE kept = 1")}
    by_id = {r["id"]: r for r in rows if r["id"] in kept}
    texts = {i: passage(r["path"], r["heading_path"], r["text"]) for i, r in by_id.items()}
    df, tfs = Counter(), {}
    for i, r in by_id.items():
        tf = Counter(_terms(r["text"]))
        tfs[i] = tf
        df.update(tf.keys())
    n_docs = max(len(by_id), 1)
    cache = _cache(con, SAME, judge.name)
    targets = [i for i in (chunk_ids if chunk_ids is not None else by_id) if i in by_id]
    planned: set[frozenset] = set()
    stats = {"chunks": len(targets), "pairs": 0, "asked": 0, "cached": 0, "links": 0, "refused": 0, "failed": 0}

    def link(a, b, p, fresh):
        ra = by_id[a]
        if fresh:
            _store(con, SAME, SAME, texts[a], texts[b], p, judge.name, ra["source"])
        if p >= KEEP_P:
            via = "~".join(sorted((kept[a], kept[b])))
            con.execute("INSERT OR IGNORE INTO links (src, dst, rel, via) VALUES (?, ?, 'about', ?)", (a, b, via))
            stats["links"] += 1

    def neighbours(a):
        tf = tfs[a]
        score = {w: c * math.log(n_docs / df[w]) for w, c in tf.items() if df[w] >= 2}
        q = " ".join(sorted(score, key=lambda w: -score[w])[:MLT_TERMS])
        out = []
        for h in bm25(con, q, FETCH, [by_id[a]["source"]]) if q else []:
            b = h.id
            if b == a or b not in by_id or by_id[b]["file_id"] == by_id[a]["file_id"] or kept[a] == kept[b]:
                continue
            pair = frozenset((a, b))
            if pair in planned:
                continue
            planned.add(pair)
            out.append(b)
            if len(out) >= NEIGHBOURS:
                break
        return out

    def jobs():
        for a in targets:
            fresh = []
            for b in neighbours(a):
                stats["pairs"] += 1
                hit = cache.get(key(SAME, SAME, texts[a], texts[b]))
                if hit is None:
                    hit = cache.get(key(SAME, SAME, texts[b], texts[a]))
                if hit is not None:
                    link(a, b, hit, False)
                    stats["cached"] += 1
                else:
                    fresh.append(b)
            if not fresh:
                continue
            if judge.packs:
                slots = {f"n{j}": b for j, b in enumerate(fresh)}
                yield (a, slots), ({"passage": texts[a], **{s: texts[b] for s, b in slots.items()}},
                                   {s: same_question(s) for s in slots})
            else:
                for b in fresh:
                    yield (a, {"n0": b}), ({"passage": texts[a], "n0": texts[b]}, {"n0": same_question("n0")})

    t0, meta = time.time(), []

    def tagged():
        for m, job in jobs():
            meta.append(m)
            yield job

    for n, (j, ans) in enumerate(judge.batch(tagged()), 1):
        a, slots = meta[j]
        if ans is not None:
            for s, b in slots.items():
                link(a, b, ans[s], True)
                stats["asked"] += 1
        if n % 200 == 0:
            con.commit()
            log(f"  links {n} calls, {stats['pairs']} pairs, {stats['links']} links  {time.time() - t0:.0f}s")
    con.commit()
    stats["refused"], stats["failed"] = judge.refused, judge.failed
    return stats


def build(con, judge, sources: list[str] | None = None, relink: bool = False, log=print) -> dict:
    """Categorize what has no categories yet, then judge the neighbours of those chunks (every
    categorized chunk with `relink`)."""
    cats = categorize(con, judge, sources, log)
    done = cats.pop("done")
    links = link_facts(con, judge, None if relink else done, sources, log)
    kept = {r["category"]: r["n"] for r in con.execute(
        "SELECT category, count(*) n FROM chunk_categories WHERE kept = 1 GROUP BY category ORDER BY n DESC")}
    return {"categories": cats, "links": links, "kept": kept}


# ------------------------------------------------------------------------------------ query time

def query_categories(con, judge, q: str) -> dict[str, float]:
    """One p per category of passage that would answer the query; cached like the rest."""
    ks = {c: key("query_category", c, q) for c in CATEGORIES}
    cache = {r["key"]: r["p"] for r in con.execute(
        f"SELECT key, p FROM judgments WHERE model = ? AND key IN ({','.join('?' * len(ks))})", [judge.name, *ks.values()])}
    if all(k in cache for k in ks.values()):
        return {c: cache[k] for c, k in ks.items()}
    _, ans = next(judge.batch(iter([({"query": q}, {"category": query_category_question()})])))
    if ans is None:
        return {}
    probs = ans["category"]
    for c, p in probs.items():
        _store(con, "query_category", c, q, "", p, judge.name, "query")
    con.commit()
    return probs


def warm_query_categories(con, judge, queries: list[str]) -> int:
    """Judge many queries' categories at once (benchmarks); query_categories then reads the cache."""
    cache = _cache(con, "query_category", judge.name)
    todo = [q for q in dict.fromkeys(queries) if any(key("query_category", c, q) not in cache for c in CATEGORIES)]
    questions = {"category": query_category_question()}
    for n, (i, ans) in enumerate(judge.batch(({"query": q}, questions) for q in todo), 1):
        if ans is not None:
            for c, p in ans["category"].items():
                _store(con, "query_category", c, todo[i], "", p, judge.name, "query")
        if n % 200 == 0:
            con.commit()
    con.commit()
    return len(todo)
