"""Rankers read (query, passage) pairs from the pool and return one probability per pair
(None for a pair the ranker refused to score; it keeps its BM25 place after the scored ones).

The question carries explicit true/false criteria: a bare "does this answer the query?" lets the
model reward passages that are merely on topic. Every ranker asks the same question.
`dispositio` is the System One model read in this process (inventio/systemone.py); `typesafe` the cloud.
"""

import os
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor

INSTRUCTIONS = "Does the `passage` answer the `query`?"
CRITERIA = {
    "true": "The passage states the specific answer, rule, or instruction the query asks for.",
    "false": "The passage is only on a related topic, or uses the same words without answering the query.",
}
TYPE_INSTRUCTIONS = "Which kind of document would contain the answer to the `query`?"

RANKERS = ("none", "dispositio", "typesafe")


class CloudRefused(RuntimeError):
    pass


def default_ranker() -> str:
    """INVENTIO_RANKER, else `dispositio` when this machine can read a checkpoint, else BM25 order.

    The check is a distribution lookup, not an import: building the command line must not cost a torch
    import.
    """
    if os.environ.get("INVENTIO_RANKER"):
        return os.environ["INVENTIO_RANKER"]
    from ._systemone import missing

    return "dispositio" if missing() is None else "none"


def ranker_tag(name: str) -> str:
    """The name results and score caches are kept under: for `dispositio`, the checkpoint's last path
    part (a run directory's name, or `dispositio@v4`), so a run of your own never shares a cache with the
    released model."""
    if name != "dispositio":
        return name
    from .systemone import run_id

    return run_id().replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


class TypeSafeRanker:
    """Sends passages to the TypeSafe API. Refuses any chunk from a source not marked public."""

    cloud = True

    def __init__(self, model: str | None = None, workers: int = 12):
        from typesafe_sdk import Noul, NoulCriteria, TypeSafeClient

        if not os.environ.get("TYPESAFE_API_KEY"):
            raise RuntimeError("TYPESAFE_API_KEY is not set")
        self.client = TypeSafeClient(timeout=30.0)
        self.model = model or os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-latest")
        self.name = f"typesafe:{self.model}"
        self.question = Noul(instructions=INSTRUCTIONS, criteria=NoulCriteria(**CRITERIA))
        self.workers = workers
        self.timeouts = 0

    def _one(self, query: str, passage: str) -> float | None:
        from typesafe_sdk import TypeSafePermissionDeniedError
        from typesafe_sdk._core.errors import (TypeSafeAPIConnectionError, TypeSafeAPITimeoutError,
                                               TypeSafeInternalServerError)

        for attempt in range(3):  # the SDK's own retries end in a timeout under long runs' load
            try:
                r = self.client.system_one(
                    state={"query": query, "passage": passage}, questions={"rel": self.question}, model=self.model
                )
                return float(r.nouls["rel"].noul)
            except TypeSafePermissionDeniedError as e:
                # The API's edge firewall rejects some texts outright with an HTML page (403 "Attention
                # Required", seen on Django source and on issue text quoting it). That pair stays
                # unscored and keeps its BM25 place; a real permission error (the key) still raises.
                if "Attention Required" not in str(e):
                    raise
                return None
            except (TypeSafeAPITimeoutError, TypeSafeAPIConnectionError, TypeSafeInternalServerError):
                # transient: timeout under load, dropped connection, Cloudflare 520 from the origin
                time.sleep(5 * (attempt + 1))
        self.timeouts += 1  # still timing out: unscored, keeps its BM25 place, counted
        return None

    def types(self, query: str, types: dict[str, str]) -> dict[str, float]:
        from typesafe_sdk import Choice, TypeSafePermissionDeniedError

        try:
            r = self.client.system_one(
                state={"query": query},
                questions={"type": Choice(instructions=TYPE_INSTRUCTIONS, criteria=types)},
                model=self.model,
            )
        except TypeSafePermissionDeniedError as e:  # edge firewall, as in _one: no widening for this query
            if "Attention Required" not in str(e):
                raise
            return {}
        return dict(r.choices["type"].probabilities)

    def score(self, query: str, hits) -> list[float]:
        private = sorted({h.source for h in hits if not h.public})
        if private:
            raise CloudRefused(
                f"ranker 'typesafe' would send text from non-public source(s) {', '.join(private)} to the cloud; "
                "restrict with --source, re-init them with --public, or use --ranker dispositio"
            )
        passages = [h.passage() for h in hits]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(lambda p: self._one(query, p), passages))


class SystemOneRanker:
    """dispositio: the local System One model.

    One pass over the state (the question and the candidate passages) answers which passage holds the
    answer, which line does, and whether the passages answer at all. `score` returns, for each hit, the
    probability the model gave its own passage (`where_passage`) — the reading the gate measures; `last`
    keeps the rest of that pass: the passage and line the model named, the `exists` head, which weights and
    temperature answered, and how many candidates had to be left out of the state.

    The checkpoint is loaded in this process (`systemone.Model`, Kev's serving path vendored): nothing to
    start, no port, and inventio's own background process is what keeps it warm between queries.
    `INVENTIO_DISPOSITIO_URL` (or an explicit `url`) reads a model on another machine over HTTP instead —
    the only shape that sends the map anywhere, and so the only shape the public-sources rule applies to.
    """

    LOCAL = ("127.0.0.1", "localhost", "::1")
    passes = True   # reads in passes of 15: search says how many leading hits are BM25's pool (search.search)

    def __init__(self, url: str | None = None, model: str = "jev-latest", timeout: float = 120,
                 run: str | None = None):
        from urllib.parse import urlparse

        from .systemone import README

        self.url = (url or os.environ.get("INVENTIO_DISPOSITIO_URL") or "").rstrip("/")
        self.host = urlparse(self.url).hostname or ""
        self.run = run
        self.model, self.timeout, self.readme, self.last = model, timeout, README, None
        self._info: dict | None = None   # which weights answered, read once: a query should not pay for it twice

    def _model(self):
        """The loaded checkpoint, or None when a URL says the model answers from elsewhere."""
        if self.url:
            return None
        from .systemone import Model, Unreachable

        try:
            return Model.cached(self.run)
        except (RuntimeError, ValueError, OSError, ImportError) as e:   # not installed, not a checkpoint, no download
            raise Unreachable(f"{e}\n{self.readme}") from e

    def _served(self) -> dict:
        """The run and temperature behind the probabilities. A checkpoint answers to any model name, so the
        run is what says which weights answered; `INVENTIO_DISPOSITIO_MODEL` pins the one you meant."""
        from .systemone import Unreachable, served

        if self._info is not None:
            return self._info
        if not self.url:
            self._info = self._model().info()
            return self._info
        try:
            info = served(self.url, timeout=30)
        except (OSError, urllib.error.URLError) as e:
            raise Unreachable(f"{self.url} did not answer ({e}).\n{self.readme}") from e
        got, want = str(info.get("run") or ""), os.environ.get("INVENTIO_DISPOSITIO_MODEL")
        if want and want.replace("\\", "/").rstrip("/") not in got.replace("\\", "/"):
            raise Unreachable(f"{self.url} serves {got!r}, not {want!r}; another run's probabilities need "
                              f"their own threshold, so refusing to read them")
        self._info = {"run": got, "temperature": info.get("temperature")}
        return self._info

    def score(self, query: str, hits, first: int | None = None) -> list[float | None]:
        """One pass for up to 15 hits. Past 15, BM25's pool (the first `first` hits, default all) is read in
        heats of 15, then a final over each heat's best, because a pass's probabilities share one pool and
        cannot be compared across passes; only the final's passages get a score, the rest keep their place in
        BM25's order. Pool 30 against the first 15 alone, same pools: MultiDoc2Dial nDCG@10 0.596 -> 0.628,
        SWE-bench Lite code 0.599 -> 0.643, TechQA 0.467 -> 0.489, SciFact and Zalo unchanged.

        Hits after the pool are what widening added (named files, links, neighbours): up to 7 of them take
        seats in the final beside the heats' best 8, so a linked chunk is read against BM25's best at the
        cost of one more pass. Each extra pass costs about what the first did."""
        from .systemone import MAX_PASSAGES

        if self.url and self.host not in self.LOCAL:
            # Reading the map on this machine is the promise (in this process, or a server on loopback); a
            # remote endpoint has to obey the rule the cloud ranker obeys — public sources only.
            private = sorted({h.source for h in hits if not h.public})
            if private:
                raise CloudRefused(
                    f"ranker 'dispositio' would send text from non-public source(s) {', '.join(private)} to "
                    f"{self.host}; restrict with --source, re-init them with --public, or serve it locally")
        if len(hits) <= MAX_PASSAGES:
            return self._pass(query, hits)
        first = min(first, len(hits)) if first else len(hits)
        extra = list(range(first, len(hits)))[:MAX_PASSAGES // 2]
        heats = [list(range(i, min(i + MAX_PASSAGES, first))) for i in range(0, first, MAX_PASSAGES)]
        seats = MAX_PASSAGES - len(extra)
        best, dropped = [], 0
        for n, heat in enumerate(heats):
            ps = self._pass(query, [hits[i] for i in heat])
            dropped += self.last["dropped_passages"]
            take = seats // len(heats) + (n < seats % len(heats))
            best += sorted((i for i, p in zip(heat, ps) if p is not None), key=lambda i: -ps[i - heat[0]])[:take]
        final = sorted(best) + extra   # BM25's order, as each heat was, then what widening added
        out: list[float | None] = [None] * len(hits)
        for i, p in zip(final, self._pass(query, [hits[i] for i in final])):
            out[i] = p
        self.last.update(heats=len(heats), dropped_passages=dropped + self.last["dropped_passages"])
        return out

    def _pass(self, query: str, hits) -> list[float | None]:
        """One state of at most 15 passages, read once; `last` is this pass's reading."""
        from .systemone import MAX_STATE_CHARS, Unreachable, ask, line_of, questions, render

        info = self._served()
        kept, dropped = list(hits), 0
        while True:   # a state the model never read is not a state to read: drop from the tail until it fits
            state, pids, lids, owner = render(kept)
            if len(state) <= MAX_STATE_CHARS or len(kept) <= 2:
                break
            kept, dropped = kept[:-1], dropped + 1
        qs = questions(query, pids, lids)
        if self.url:
            try:
                res = ask(self.url, state, qs, self.model, self.timeout)
            except urllib.error.HTTPError as e:   # a status, not a dead server: say which
                raise Unreachable(f"{self.url} answered {e.code} {e.reason} ({e.read()[:200]!r}).\n{self.readme}") from e
            except (OSError, urllib.error.URLError) as e:
                raise Unreachable(f"{self.url} did not answer ({e}).\n{self.readme}") from e
        else:
            res = self._model().ask(state, qs, self.model)
        a = res["answers"]
        about = a["where_passage"]["probabilities"]
        line, by_line = None, []
        if "where_line" in qs and "where_line" in a:
            lp = a["where_line"]["probabilities"]
            line = line_of(state, lids, owner, int(max(lp, key=lp.get)[1:]))
            by_line = [round(sum(lp.get(l, 0.0) for l, o in zip(lids, owner) if o == i), 4) for i in range(len(kept))]
        self.last = {"passages": {p: round(float(about[p]), 4) for p in pids},
                     "order": sorted(pids, key=lambda p: -about[p]), "by_line": by_line, "line": line,
                     "exists_head": float(a["exists"]["noul"]), "exists_max": round(max(about.values()), 4),
                     "kept": [f"{h.source}:{h.coord}" for h in kept], "dropped_passages": dropped, "heats": 1,
                     "served": info, "model": self.model, "latency_ms": res.get("latency_ms"),
                     "wall_ms": res.get("wall_ms")}
        out: list[float | None] = [None] * len(hits)
        for i, p in enumerate(pids):
            out[i] = float(about[p])
        return out

    def types(self, query: str, types: dict[str, str]) -> dict[str, float]:
        """Which kind of document would hold the answer: the same choice question the per-passage ranker
        asks, over the query alone — so a remote endpoint reads no map text here."""
        from .systemone import Unreachable, ask

        qs = {"type": {"type": "choice", "instructions": TYPE_INSTRUCTIONS, "criteria": types}}
        if self.url:
            try:
                res = ask(self.url, {"query": query}, qs, self.model, self.timeout)
            except (OSError, urllib.error.URLError) as e:
                raise Unreachable(f"{self.url} did not answer ({e}).\n{self.readme}") from e
        else:
            res = self._model().ask({"query": query}, qs, self.model)
        return dict(res["answers"]["type"]["probabilities"])

    def relations(self, query: str, passages, pairs: list[tuple[int, int]]) -> list[dict[str, float]]:
        """What passage t is to passage r for this question (systemone.RELATIONS), for each (r, t) of `pairs`:
        one state of at most 15 passages, every pair one choice question over it, one pass."""
        from .systemone import MAX_PASSAGES, MAX_STATE_CHARS, Unreachable, ask, relation, render

        passages = list(passages)[:MAX_PASSAGES]
        if self.url and self.host not in self.LOCAL:
            private = sorted({h.source for h in passages if not h.public})
            if private:
                raise CloudRefused(f"ranker 'dispositio' would send text from non-public source(s) "
                                   f"{', '.join(private)} to {self.host}; serve it locally")
        state, pids, _, _ = render(passages)
        while len(state) > MAX_STATE_CHARS and len(passages) > 2:   # the tail goes first, as in a ranking pass
            passages = passages[:-1]
            state, pids, _, _ = render(passages)
        asked = [(r, t) for r, t in pairs if r < len(pids) and t < len(pids)]
        if not asked:
            return [{} for _ in pairs]
        qs = {f"{pids[r]}>{pids[t]}": relation(query, pids[r], pids[t]) for r, t in asked}
        if self.url:
            try:
                res = ask(self.url, state, qs, self.model, self.timeout)
            except (OSError, urllib.error.URLError) as e:
                raise Unreachable(f"{self.url} did not answer ({e}).\n{self.readme}") from e
        else:
            res = self._model().ask(state, qs, self.model)
        got = {k: dict(v["probabilities"]) for k, v in res["answers"].items()}
        return [got.get(f"{pids[r]}>{pids[t]}", {}) if r < len(pids) and t < len(pids) else {} for r, t in pairs]


def make_ranker(name: str):
    if name == "none":
        return None
    if name == "dispositio":
        return SystemOneRanker()
    if name == "typesafe":
        return TypeSafeRanker()
    raise ValueError(f"unknown ranker {name!r}; choose from {', '.join(RANKERS)}")
