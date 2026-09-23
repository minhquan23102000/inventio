"""Rankers read (query, passage) pairs from the pool and return one probability per pair
(None for a pair the ranker refused to score; it keeps its BM25 place after the scored ones).

The question carries explicit true/false criteria: a bare "does this answer the query?" lets the
model reward passages that are merely on topic. Both rankers ask the same question, so a
fine-tuned Laya can be measured against the Jev labels it learned from.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

INSTRUCTIONS = "Does the `passage` answer the `query`?"
CRITERIA = {
    "true": "The passage states the specific answer, rule, or instruction the query asks for.",
    "false": "The passage is only on a related topic, or uses the same words without answering the query.",
}
TYPE_INSTRUCTIONS = "Which kind of document would contain the answer to the `query`?"

RANKERS = ("none", "laya", "typesafe")


class CloudRefused(RuntimeError):
    pass


def load_laya():
    """The Laya agent and its name. INVENTIO_LAYA_MODEL points at a fine-tuned checkpoint
    directory (benchmarks/finetune_laya.py writes one); without it, the published multilingual one."""
    import warnings

    import laya
    import torch

    warnings.filterwarnings("ignore", module="laya")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tuned = os.environ.get("INVENTIO_LAYA_MODEL")
    if tuned:
        return laya.load(tuned, device=device), f"laya:{tuned}"
    model, subfolder = "convaiinnovations/laya", "multilingual"
    return laya.load(model, device=device, subfolder=subfolder), f"laya:{model}/{subfolder}"


class LayaRanker:
    def __init__(self):
        self.agent, self.name = load_laya()
        self.questions = {"rel": {"type": "noul", "instructions": INSTRUCTIONS, "criteria": CRITERIA}}

    def score(self, query: str, hits) -> list[float]:
        return [
            float(self.agent.predict({"query": query, "passage": h.passage()}, self.questions)["answers"]["rel"]["noul"])
            for h in hits
        ]

    def types(self, query: str, types: dict[str, str]) -> dict[str, float]:
        q = {"type": {"type": "choice", "instructions": TYPE_INSTRUCTIONS, "criteria": types}}
        return dict(self.agent.predict({"query": query}, q)["answers"]["type"]["probabilities"])


class TypeSafeRanker:
    """Sends passages to the TypeSafe API. Refuses any chunk from a source not marked public."""

    cloud = True

    def __init__(self, con=None, model: str | None = None, workers: int = 12):
        from typesafe_sdk import Noul, NoulCriteria, TypeSafeClient

        if not os.environ.get("TYPESAFE_API_KEY"):
            raise RuntimeError("TYPESAFE_API_KEY is not set")
        self.client = TypeSafeClient(timeout=30.0)
        self.model = model or os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-latest")
        self.name = f"typesafe:{self.model}"
        self.question = Noul(instructions=INSTRUCTIONS, criteria=NoulCriteria(**CRITERIA))
        self.con = con
        self.workers = workers
        self.timeouts = 0

    def _one(self, query: str, passage: str) -> float | None:
        from typesafe_sdk import TypeSafePermissionDeniedError
        from typesafe_sdk._core.errors import TypeSafeAPITimeoutError

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
            except TypeSafeAPITimeoutError:
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
                "restrict with --source, re-init them with --public, or use --ranker laya"
            )
        passages = [h.passage() for h in hits]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            scores = list(pool.map(lambda p: self._one(query, p), passages))
        if self.con is not None:  # teacher labels for fine-tuning Laya later
            self.con.executemany(
                "INSERT OR REPLACE INTO labels (query, passage, noul, model, source) VALUES (?, ?, ?, ?, ?)",
                [(query, p, s, self.model, h.source) for p, s, h in zip(passages, scores, hits) if s is not None],
            )
            self.con.commit()
        return scores


def make_ranker(name: str, con=None):
    if name == "none":
        return None
    if name == "laya":
        return LayaRanker()
    if name == "typesafe":
        return TypeSafeRanker(con)
    raise ValueError(f"unknown ranker {name!r}; choose from {', '.join(RANKERS)}")
