"""Rankers read (query, passage) pairs from the pool and return one probability per pair
(None for a pair the ranker refused to score; it keeps its BM25 place after the scored ones).

The question carries explicit true/false criteria: a bare "does this answer the query?" lets the
model reward passages that are merely on topic. Every ranker asks the same question.
`dispositio` is Laya fine-tuned on it (benchmarks/finetune_laya.py); `laya` is Laya as published.
"""

import json
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

RANKERS = ("none", "dispositio", "laya", "typesafe", "systemone")


class CloudRefused(RuntimeError):
    pass


DISPOSITIO = "minhquan2310/dispositio"  # Laya fine-tuned for Inventio's questions (benchmarks/finetune_laya.py)
LAYA = "convaiinnovations/laya"  # Laya multilingual as published, the base dispositio starts from


def default_ranker() -> str:
    """INVENTIO_RANKER, else `systemone` when this machine can read a checkpoint, else BM25 order.

    The System One reader is the default the moment the runtime is installed; Laya's per-passage
    fine-tune (`dispositio` v3) is the old shape and is no longer reached for — `--ranker dispositio`
    still reads it where the `laya` extra is installed. The check is a distribution lookup, not an
    import: building the command line must not cost a torch import.
    """
    if os.environ.get("INVENTIO_RANKER"):
        return os.environ["INVENTIO_RANKER"]
    from ._systemone import missing

    return "systemone" if missing() is None else "none"


def checkpoint(name: str) -> str:
    """What a local model name loads: `laya` the published checkpoint; `dispositio`
    INVENTIO_DISPOSITIO_MODEL (a directory or a Hugging Face id, e.g. a fine-tune of your own),
    else the released one."""
    if name == "laya":
        return LAYA
    if name == "dispositio":
        return os.environ.get("INVENTIO_DISPOSITIO_MODEL") or DISPOSITIO
    raise ValueError(f"not a local model: {name!r}")


def load_laya(name: str):
    """The agent for a local model name (`dispositio` or `laya`), and the name its judgments are
    stored under. The device is Laya's choice, CUDA, then Apple's MPS, then CPU, unless
    INVENTIO_DEVICE names one (`cpu` when the GPU is busy)."""
    import warnings

    import laya

    warnings.filterwarnings("ignore", module="laya")
    device = os.environ.get("INVENTIO_DEVICE") or None
    model = checkpoint(name)
    sub = "multilingual" if model == LAYA else None
    agent = laya.load(cached(model, sub) or model, device=device, subfolder=sub)
    return agent, f"laya:{model}/{sub}" if sub else f"laya:{model}"


def cached(model: str, subfolder: str | None) -> str | None:
    """The downloaded snapshot of a Hugging Face checkpoint, found without the network; None for
    a local directory or one not downloaded yet. A cached checkpoint is not refreshed here:
    `inventio update` fetches a newer release, and `update_notice` says when there is one."""
    if os.path.exists(model):
        return None
    from huggingface_hub import snapshot_download

    prefix = f"{subfolder}/" if subfolder else ""
    try:
        return snapshot_download(model, local_files_only=True, allow_patterns=[
            prefix + f for f in ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")])
    except Exception:  # not in the cache (or only part of it): laya.load downloads it
        return None


CHECK_EVERY = 86_400  # seconds between two looks at the released dispositio's latest revision


def _cached_revision() -> str | None:
    """The revision of the released dispositio this machine loads (a snapshot directory's name)."""
    path = cached(DISPOSITIO, None)
    return os.path.basename(os.path.normpath(path)) if path else None


def update_notice() -> str | None:
    """One line when Hugging Face has a newer dispositio than the one cached here, else None.
    Asks at most once a day and sends nothing but the model's name (no query, no document);
    never when INVENTIO_OFFLINE or HF_HUB_OFFLINE is set, or INVENTIO_DISPOSITIO_MODEL picks
    another checkpoint. Any failure is silent: a query must not fail over a release check."""
    if os.environ.get("INVENTIO_OFFLINE") or os.environ.get("HF_HUB_OFFLINE") or os.environ.get("INVENTIO_DISPOSITIO_MODEL"):
        return None
    from .store import data_home

    state_path = data_home() / "update.json"
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError):
        state = {}
    try:
        if time.time() - state.get("checked", 0) >= CHECK_EVERY:
            from huggingface_hub import HfApi

            state = {"checked": time.time(), "latest": HfApi().model_info(DISPOSITIO, timeout=3).sha}
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(state))
        have = _cached_revision()
    except Exception:
        return None
    if have and state.get("latest") and state["latest"] != have:
        return "inventio: a newer dispositio is released; `inventio update` fetches it (INVENTIO_OFFLINE=1 stops this check)"
    return None


def update() -> tuple[str | None, str]:
    """Fetch the released dispositio's latest revision into the Hugging Face cache; the revisions
    before and after. The earlier snapshot stays in the cache until `hf cache delete`."""
    from huggingface_hub import snapshot_download

    before = _cached_revision()
    path = snapshot_download(DISPOSITIO, allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"])
    return before, os.path.basename(os.path.normpath(path))


def ranker_tag(name: str) -> str:
    """The name results and score caches are kept under: for `dispositio`, its checkpoint's last
    path part, so a fine-tune of your own never shares a cache with the released model."""
    if name != "dispositio":
        return name
    return checkpoint(name).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


QUERY_TOKENS = 384  # Laya reads the query, then the passage, and cuts from the right: an issue of
# 1,000 tokens would leave no room for the passage it is asked about (15% of SWE-bench Lite issues)


def cap_query(tok, query: str) -> str:
    """The query's first QUERY_TOKENS tokens, so every passage keeps at least ~500 of Laya's 1,024."""
    ids = tok(query, add_special_tokens=False)["input_ids"]
    return query if len(ids) <= QUERY_TOKENS else tok.decode(ids[:QUERY_TOKENS])


def passage_room(tok, query: str, q: dict, max_len: int, head_max_len: int) -> int:
    """Tokens left for the passage once the instructions, options and (capped) query are in."""
    from laya.common import build_sequence

    return max_len - len(build_sequence(tok, {"query": query, "passage": ""}, q, max_len, head_max_len)[0]) - 8


def windows(tok, head: str, body: str, room: int) -> list[tuple[str, int, int]]:
    """`head + body` whole when Laya can read it whole; else runs of the body's consecutive lines
    that each fit in `room` tokens, every run under `head` (the `[path > heading]` line), so a long
    function is read in parts instead of losing its tail. Each comes with the body lines it covers
    (0-based, inclusive). Lengths are counted as Laya reads them, inside the JSON state."""
    cost = lambda s: len(tok(json.dumps(s, ensure_ascii=False)[1:-1], add_special_tokens=False)["input_ids"])  # noqa: E731
    lines = body.split("\n")
    if cost(head + body) <= room:
        return [(head + body, 0, len(lines) - 1)]
    left = max(16, room - cost(head))
    costs = [len(x) for x in tok([json.dumps(ln + "\n", ensure_ascii=False)[1:-1] for ln in lines],
                                 add_special_tokens=False)["input_ids"]]
    out, start, used = [], 0, 0
    for i, c in enumerate(costs):
        if used and used + c > left:
            out.append((head + "\n".join(lines[start:i]), start, i - 1))
            start, used = i, 0
        used += c
    out.append((head + "\n".join(lines[start:]), start, len(lines) - 1))
    return out


class LayaRanker:
    BATCH = 16  # pairs per forward pass; pairs are sorted by length so padding stays small
    MPS_BATCH = 4  # Apple GPUs: 3.5 s against 4.9 s for 16 on an M3, 30 pool chunks

    def __init__(self, name: str):
        self.agent, self.name = load_laya(name)
        if self.agent.device.type == "mps":
            self.BATCH = self.MPS_BATCH
        self.questions = {"rel": {"type": "noul", "instructions": INSTRUCTIONS, "criteria": CRITERIA}}

    def score(self, query: str, hits) -> list[float]:
        """The same numbers as one `agent.predict` per pair, from a few batched forward passes. A
        passage longer than Laya reads is scored in windows (see `windows`) and keeps its best."""
        import numpy as np
        import torch
        from laya.common import QTYPES, build_sequence, collate_items, temp_bucket

        a = self.agent
        q = a._to_internal(self.questions["rel"])
        max_len, head_max_len = a.cfg.get("max_len", 512), a.cfg.get("head_max_len", 192)
        items, owner = [], []
        query = cap_query(a.tok, query)
        room = passage_room(a.tok, query, q, max_len, head_max_len)
        for j, h in enumerate(hits):
            head, body = h.passage().split("\n", 1)
            for text, _, _ in windows(a.tok, head + "\n", body, room):
                seq, markers = build_sequence(a.tok, {"query": query, "passage": text}, q, max_len, head_max_len)
                items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
                owner.append(j)
        t_scale = a.temperature_by_options.get(temp_bucket(QTYPES[q["t"]], 2), a.temperature[QTYPES[q["t"]]])
        out = [0.0] * len(hits)
        order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
        with torch.no_grad():
            for s in range(0, len(order), self.BATCH):
                idx = order[s:s + self.BATCH]
                b = collate_items([[items[i]] for i in idx], a.tok.pad_token_id)
                with torch.autocast(device_type=a.device.type, dtype=a.dtype, enabled=a.device.type == "cuda"):
                    logits, _ = a.model(*(b[k].to(a.device) for k in
                                          ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")))
                z = logits.float().cpu().numpy()[:, :2] / t_scale
                p = np.exp(z - z.max(-1, keepdims=True))
                for i, row in zip(idx, p / p.sum(-1, keepdims=True)):
                    out[owner[i]] = max(out[owner[i]], round(float(row[1]), 4))
        return out

    def types(self, query: str, types: dict[str, str]) -> dict[str, float]:
        q = {"type": {"type": "choice", "instructions": TYPE_INSTRUCTIONS, "criteria": types}}
        return dict(self.agent.predict({"query": query}, q)["answers"]["type"]["probabilities"])


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
    """The local System One model — `dispositio` v4 while it is still being trained.

    One pass over the state (the question and the candidate passages) answers which passage holds the
    answer, which line does, and whether the passages answer at all. `score` returns, for each hit, the
    probability the model gave its own passage (`where_passage`) — the reading the gate measures; `last`
    keeps the rest of that pass: the passage and line the model named, the `exists` head, which weights and
    temperature answered, and how many candidates had to be left out of the state.

    The checkpoint is loaded in this process (`systemone.Model`, Kev's serving path vendored): nothing to
    start, no port, and inventio's own background process is what keeps it warm between queries.
    `INVENTIO_SYSTEMONE_URL` (or an explicit `url`) reads a model on another machine over HTTP instead —
    the only shape that sends the map anywhere, and so the only shape the public-sources rule applies to.
    """

    LOCAL = ("127.0.0.1", "localhost", "::1")

    def __init__(self, url: str | None = None, model: str = "jev-latest", timeout: float = 120,
                 run: str | None = None):
        from urllib.parse import urlparse

        from .systemone import README

        self.url = (url or os.environ.get("INVENTIO_SYSTEMONE_URL") or "").rstrip("/")
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
        run is what says which weights answered; `INVENTIO_SYSTEMONE_RUN` pins the one you meant."""
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
        got, want = str(info.get("run") or ""), os.environ.get("INVENTIO_SYSTEMONE_RUN")
        if want and want.replace("\\", "/").rstrip("/") not in got.replace("\\", "/"):
            raise Unreachable(f"{self.url} serves {got!r}, not {want!r}; another run's probabilities need "
                              f"their own threshold, so refusing to read them")
        self._info = {"run": got, "temperature": info.get("temperature")}
        return self._info

    def score(self, query: str, hits) -> list[float | None]:
        from .systemone import (MAX_PASSAGES, MAX_STATE_CHARS, Unreachable, ask, line_of, questions, render)

        if self.host not in self.LOCAL:
            # Serving the map on this machine is the promise; a remote endpoint has to obey the rule the
            # cloud ranker obeys — public sources only.
            private = sorted({h.source for h in hits if not h.public})
            if private:
                raise CloudRefused(
                    f"ranker 'systemone' would send text from non-public source(s) {', '.join(private)} to "
                    f"{self.host}; restrict with --source, re-init them with --public, or serve it locally")
        info = self._served()
        kept, dropped = list(hits[:MAX_PASSAGES]), max(0, len(hits) - MAX_PASSAGES)
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
                     "kept": [f"{h.source}:{h.coord}" for h in kept], "dropped_passages": dropped,
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


def make_ranker(name: str):
    if name == "none":
        return None
    if name in ("dispositio", "laya"):
        return LayaRanker(name)
    if name == "typesafe":
        return TypeSafeRanker()
    if name == "systemone":
        return SystemOneRanker()
    raise ValueError(f"unknown ranker {name!r}; choose from {', '.join(RANKERS)}")
