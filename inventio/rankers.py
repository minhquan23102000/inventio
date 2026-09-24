"""Rankers read (query, passage) pairs from the pool and return one probability per pair
(None for a pair the ranker refused to score; it keeps its BM25 place after the scored ones).

The question carries explicit true/false criteria: a bare "does this answer the query?" lets the
model reward passages that are merely on topic. Every ranker asks the same question.
`dispositio` is Laya fine-tuned on it (benchmarks/finetune_laya.py); `laya` is Laya as published.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

INSTRUCTIONS = "Does the `passage` answer the `query`?"
CRITERIA = {
    "true": "The passage states the specific answer, rule, or instruction the query asks for.",
    "false": "The passage is only on a related topic, or uses the same words without answering the query.",
}
TYPE_INSTRUCTIONS = "Which kind of document would contain the answer to the `query`?"

RANKERS = ("none", "dispositio", "laya", "typesafe")


class CloudRefused(RuntimeError):
    pass


DISPOSITIO = "minhquan2310/dispositio"  # Laya fine-tuned for Inventio's questions (benchmarks/finetune_laya.py)
LAYA = "convaiinnovations/laya"  # Laya multilingual as published, the base dispositio starts from


def default_ranker() -> str:
    """INVENTIO_RANKER, else dispositio when the `laya` extra is installed, else BM25 order."""
    import importlib.util

    return os.environ.get("INVENTIO_RANKER") or ("dispositio" if importlib.util.find_spec("laya") else "none")


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
    if model == LAYA:
        return laya.load(model, device=device, subfolder="multilingual"), f"laya:{model}/multilingual"
    return laya.load(model, device=device), f"laya:{model}"


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

    def __init__(self, name: str):
        self.agent, self.name = load_laya(name)
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


def make_ranker(name: str):
    if name == "none":
        return None
    if name in ("dispositio", "laya"):
        return LayaRanker(name)
    if name == "typesafe":
        return TypeSafeRanker()
    raise ValueError(f"unknown ranker {name!r}; choose from {', '.join(RANKERS)}")
