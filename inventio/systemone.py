"""The System One protocol: one state — a question plus the candidate passages — and the questions a
model answers over it in one pass.

The benchmark (`benchmarks/systemone.py`) and the ranker both build the state and the questions here,
so the words a served model reads are the words it was trained on: one question per passage (the head
that carries most of the training signal), one `exists` question, and the two choice heads that name a
passage and a line. A server is expected to expose `POST /v1/systemone` and answer every question in
`questions` in a single pass (see Kev's `kev/serve.py`).
"""

import json
import os
import threading
import time
import urllib.request
from pathlib import Path

from .store import data_home

MAX_QUERY_CHARS = 1500   # the records cut the question here, so a served question is cut here too
# The line question is a choice over every line of the passages in the state. Code pools have a median
# 407 lines (p90 552), so 255 lost 1,279 of the 1,386 issues whose answer BM25 already found; 512 keeps
# 1,150. Kev's trainer has no option cap; its serving API does (kev/api.py), so `kev_win.py serve`
# raises it to this number. A state longer than this asks no line question at all.
MAX_OPTIONS = 512
# The trainer admitted 6,656 tokens of state and the served states measured 3.9 characters per token
# (md2d 12,769 characters = 3,270 tokens), so this is that length in characters: a longer state is one
# the model never read, and the card cannot hold it either. The ranker drops passages from the tail
# until the state fits and says how many it dropped.
MAX_STATE_CHARS = 26000
# The passages one state holds. Every measured number is BM25's first 15 rendered as P01..P15; a state
# with 25 passages would speak ids the model never saw (P16..) and rank differently (`pool 30` cost v1
# 0.547 -> 0.419 on md2d).
MAX_PASSAGES = 15

# A handful of wordings, not one: the project learned this the hard way with dispositio (a small model
# trained on six question kinds passed every probe asked in those wordings and failed the same questions
# worded anew, AUC 0.0). Wordings are picked by the record's key, so a rebuild is reproducible, and the
# first of each list is the exact string the harness asks at eval time.
PASSAGE_ASKS = ('Does passage {p} answer: "{q}"?',
                'In passage {p}, is the answer to "{q}" stated?',
                'Passage {p}: does it answer the question "{q}"?')
EXISTS_ASKS = ('Does any passage answer: "{q}"?',
               'Is the answer to "{q}" stated in any of these passages?',
               'These passages: does any of them answer "{q}"?')
WHERE_ASKS = ('Which passage contains the answer to: "{q}"?',
              'Where is the answer to "{q}" stated?',
              'Which of these passages answers "{q}"?')
NOUL = {"true": "At least one passage states or directly implies the answer",
        "false": "No passage addresses this"}


def render(hits):
    """-> (state text with `P01..` passage ids and `L000..` line ids, passage ids, line ids, line owner)."""
    parts, pids, lids, owner = [], [], [], []
    for i, h in enumerate(hits):
        pid = f"P{i + 1:02d}"
        pids.append(pid)
        head = f"{h.path} > {h.heading_path}" if h.heading_path else h.path
        rows = [f"{pid} [{head}]"]
        for ln in h.text.split("\n"):
            if not ln.strip():
                continue
            lid = f"L{len(lids):03d}"
            lids.append(lid)
            owner.append(i)
            rows.append(f"{lid}| {ln.rstrip()}")
        parts.append("\n".join(rows))
    return "\n\n".join(parts), pids, lids, owner


def where_line(q, lids):
    return {"type": "choice", "instructions": f'Which line contains the answer to: "{q}"?',
            "criteria": {l: None for l in lids}}


def exists(q, label):
    return {"type": "noul", "instructions": f'Does any passage answer: "{q}"?',
            "criteria": {"true": "At least one passage states or directly implies the answer",
                         "false": "No passage addresses this"}, "label": label}


def questions(q, pids, lids, with_line=True, with_passage_asks=False):
    """The questions one request asks: the shape a served model is read with.

    The served shape is the choice heads (`where_passage`, `where_line`) and `exists` — three branches.
    `with_passage_asks` adds one question per passage (`PASSAGE_ASKS[0]`), which is a *training* device
    (`records()`) and a study at evaluation time: measured on md2d it costs about 100 ms a state and
    reads the passage no better (0.620 against the choice head's 0.653, diff +0.033 [-0.007, +0.071]).
    It is not the shape dispositio scores with: those branches cannot see each other but each one reads
    the whole state, while dispositio scores one (question, passage) pair at a time.
    The query is capped exactly as `records()` caps it, so a long question meets the words it was
    trained on.
    """
    q = q[:MAX_QUERY_CHARS]
    asks = ({p: {"type": "noul", "instructions": PASSAGE_ASKS[0].format(p=p, q=q), "criteria": NOUL}
             for p in pids} if with_passage_asks else {})
    qs = {**asks,
          "where_passage": {"type": "choice", "instructions": f'Which passage contains the answer to: "{q}"?',
                            "criteria": {p: None for p in pids}},
          "exists": {"type": "noul", "instructions": f'Does any passage answer: "{q}"?',
                     "criteria": {"true": "At least one passage states or directly implies the answer",
                                  "false": "No passage addresses this"}}}
    if with_line and len(lids) <= MAX_OPTIONS:
        qs["where_line"] = where_line(q, lids)
    return qs


def ask(url, state, qs, model="jev-latest", timeout=600):
    body = json.dumps({"state": state, "model": model, "questions": qs}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", body, {"content-type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = json.load(r)
    res["wall_ms"] = round((time.time() - t) * 1000, 1)
    return res


def served(url: str, timeout: float = 10) -> dict:
    """What a server says it is serving (`GET /v1/models`): the run directory and the temperature its
    probabilities were divided by. A checkpoint answers to any model name, so the run is what identifies
    it — and a threshold belongs to one (run, temperature)."""
    with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=timeout) as r:
        info = json.load(r)
    return info[0] if isinstance(info, list) and info else info


README = """the model reads one state — a question and the candidate passages — and answers every question
about it in a single pass, inside inventio itself. Nothing to start, no port, no environment variable:

    pip install 'inventio[dispositio]'              once: torch, transformers, peft
    inventio query "your question" --ranker dispositio

The first query loads the checkpoint (a download the first time); the background process inventio already
keeps for a model ranker holds it in memory after that, so later queries pay only for the state
(`inventio serve --stop` stops that process). To read a checkpoint that is not the published one — a run
you trained yourself:

    inventio model --use <run directory, or owner/name on the Hub>

Two settings still belong to a checkpoint: INVENTIO_DISPOSITIO_MODEL says the same thing for one command,
and INVENTIO_DISPOSITIO_CAVEAT turns on the caveat line below a probability."""

# The weights: a local run directory (a LoRA adapter or merged weights, plus head.pt) or a Hub repo id. v4 is
# published as the tag `v4` of the same repository as the earlier releases (`main` keeps v3, the older
# per-passage shape, so an older inventio that loads `main` is not handed a checkpoint it cannot read).
DEFAULT_MODEL = "minhquan2310/dispositio@v4"
ENV = "INVENTIO_DISPOSITIO_MODEL"


def registry_file() -> Path:
    return data_home() / "model.json"


def run_id() -> str:
    """Which weights answer: INVENTIO_DISPOSITIO_MODEL, else what `inventio model --use` recorded, else the
    published model."""
    if os.environ.get(ENV):
        return os.environ[ENV]
    try:
        return json.loads(registry_file().read_text(encoding="utf-8"))["run"]
    except (OSError, ValueError, KeyError):
        return DEFAULT_MODEL


def use(run: str) -> str:
    """Point inventio at a checkpoint: a run directory, or a Hub repo id like `owner/name`. Returns what was
    recorded. Recorded rather than passed per command, so an ordinary query carries no path."""
    from ._systemone import is_hub_id

    run = run if is_hub_id(run) else str(Path(run).expanduser())
    if not is_hub_id(run) and not (Path(run) / "head.pt").exists():
        raise ValueError(f"{run} is not a checkpoint: no head.pt there (and it is not a Hub id like owner/name)")
    registry_file().parent.mkdir(parents=True, exist_ok=True)
    registry_file().write_text(json.dumps({"run": run}, indent=2) + "\n", encoding="utf-8")
    return run


CHECK_EVERY = 86_400   # seconds between two looks at the published model's revision


def _cached_snapshot(run: str) -> str | None:
    """The commit of a Hub checkpoint this machine would load (its snapshot directory's name), or None."""
    from huggingface_hub import snapshot_download

    repo, _, revision = run.partition("@")
    try:
        path = snapshot_download(repo, revision=revision or None, local_files_only=True,
                                 allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"])
    except Exception:
        return None
    return os.path.basename(os.path.normpath(path))


def update_notice() -> str | None:
    """One line when the published model has moved past the snapshot cached here, else None.

    Asks at most once a day and sends the model's name and nothing else (no query, no document); never when
    INVENTIO_OFFLINE or HF_HUB_OFFLINE is set, or when the weights answering are not the published ones (a run
    recorded with `inventio model --use`). Any failure is silent: a query must not fail over a release check.
    """
    if os.environ.get("INVENTIO_OFFLINE") or os.environ.get("HF_HUB_OFFLINE") or run_id() != DEFAULT_MODEL:
        return None
    path = data_home() / "update.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    try:
        if time.time() - state.get("checked", 0) >= CHECK_EVERY:
            from huggingface_hub import HfApi

            repo, _, revision = DEFAULT_MODEL.partition("@")
            state = {"checked": time.time(), "latest": HfApi().model_info(repo, revision=revision or None, timeout=3).sha}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state), encoding="utf-8")
        have = _cached_snapshot(DEFAULT_MODEL)
    except Exception:
        return None
    if have and state.get("latest") and state["latest"] != have:
        return "inventio: a newer dispositio is released; `inventio update` fetches it (INVENTIO_OFFLINE=1 stops this check)"
    return None


def update() -> tuple[str | None, str]:
    """Fetch the published model's current revision into the Hugging Face cache: the snapshots before and after.
    The earlier snapshot stays in the cache until `hf cache delete`."""
    from huggingface_hub import snapshot_download

    before = _cached_snapshot(DEFAULT_MODEL)
    repo, _, revision = DEFAULT_MODEL.partition("@")
    path = snapshot_download(repo, revision=revision or None,
                             allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"])
    return before, os.path.basename(os.path.normpath(path))


class Model:
    """A checkpoint loaded once per process, asked one state at a time.

    This is the same forward pass the local server answered, in the process that runs inventio: the loader
    is `inventio/_systemone` (Kev's serving path, vendored — see that module's docstring for the exact
    files and changes), the answers come out of Kev's own request schema. So there is no server to start,
    no port to know and no environment variable to set; `serve.py` — inventio's own background process,
    started when a ranker needs a model — is what keeps a loaded checkpoint warm between queries.
    """

    loaded: dict = {}   # (run, device, fp32?) -> Model

    def __init__(self, run: str | None = None, device: str | None = None):
        from ._systemone import is_hub_id, load

        self.run = run or run_id()
        if not is_hub_id(self.run) and not Path(self.run).exists():
            raise ValueError(f"{self.run} is not a checkpoint: no such directory (and it is not a Hub id "
                             f"like owner/name); `inventio model --use <run>` records the one you mean")
        self._lock = threading.Lock()
        self.tok, self.model = load(self.run, device)
        self.device = str(self.model.device)

    @classmethod
    def cached(cls, run: str | None = None, device: str | None = None) -> "Model":
        key = (run or run_id(), str(device or ""), os.environ.get("INVENTIO_DTYPE", ""))
        if key not in cls.loaded:
            cls.loaded[key] = cls(run, device)
        return cls.loaded[key]

    def info(self) -> dict:
        """What `GET /v1/models` calls the card, and what a threshold belongs to: the run, the temperature
        its probabilities were divided by, and how it is loaded."""
        return {"run": self.run, "temperature": self.model.head.temperature, "device": self.device,
                "dtype": self.model.dtype, "backend": "torch",
                "base": getattr(self.model.lm.config, "_name_or_path", "")}

    def ask(self, state: str | dict, qs: dict, model: str = "jev-latest") -> dict:
        """One state, every question over it, in one pass. The response keys are the served model's
        (`answers`, `usage`, `latency_ms`), so the reader of either shape is the same code.

        Kev's opt-in date-facts preprocessing is not vendored: it is off in the served path too (no
        number in this repository was measured with it), and a state is rendered here, not prepared."""
        from ._systemone.api import SystemOneRequest, output_tokens, to_answers, to_record
        from ._systemone.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE

        t = time.time()
        rec, meta = to_record(SystemOneRequest(state=state, model=model, questions=qs))
        with self._lock:
            # the serving limits, not the training ones: a state is 15 passages and a line question offers
            # one option per line, so the training default (a 1,024-token row) refuses states the server
            # answered (kev.serve passes exactly these two numbers)
            enc = self.model.encode(self.tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
            answers = to_answers(self.model.probs(enc), meta)
        return {"model": model, "answers": answers,
                "usage": {"input_tokens": len(enc["ids"]), "output_tokens": output_tokens(self.tok, answers)},
                "latency_ms": round((time.time() - t) * 1000, 1), "wall_ms": round((time.time() - t) * 1000, 1)}


# Whether the passages answer is read from the model's own `exists` head: the alternative (the max of the
# per-passage probabilities) measured the same to the third decimal (md2d 0.755 against 0.758, techqa
# 0.862 against 0.879 on the answer-absent pools) and costs fifteen more questions a query.
EXISTS_READING = "head"


def line_of(state: str, lids, owner, index: int) -> dict:
    """The line a `where_line` answer points at: its id, the passage it belongs to, and its text."""
    lid = lids[index]
    return {"id": lid, "passage": owner[index], "text": state.split(f"{lid}| ", 1)[-1].split("\n", 1)[0]}


def honesty(last: dict | None, threshold: float | None) -> str | None:
    """The caveat printed for a reader, when the operator asked for one (`INVENTIO_DISPOSITIO_CAVEAT`).

    Off by default, and not because the reading is missing: `exists` is calibrated differently per
    corpus (median on answerable pools: md2d 0.447, techqa 0.232, webshop 0.377), so one threshold
    cannot hold a stated false-alarm rate across the three, and the pools this was validated on have the
    answer *in the map*, only not in BM25's fifteen. The wording says exactly that.
    """
    if not last or threshold is None:
        return None
    p = last.get("exists_max" if EXISTS_READING == "max" else "exists_head")
    if p is None or p >= threshold:
        return None
    note = (f"the passages read do not seem to answer this (p={p:.2f}, below the {threshold:.2f} set "
            f"here); they are the closest the map has")
    line = last.get("line")
    if line and line.get("text"):
        note += f"; the model's best line was {line['text'][:90]!r}"
    return note


class Unreachable(RuntimeError):
    """The System One server did not answer. Never a reason to fall back to another order in silence."""
