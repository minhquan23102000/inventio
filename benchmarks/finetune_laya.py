"""Fine-tune Laya as Inventio's local ranker and category judge, on this machine's GPU.

    python benchmarks/finetune_laya.py --time-steps 40      # measure, print the projected run time, exit
    python benchmarks/finetune_laya.py --out <dir>          # stage 1: from the published Laya, the default MIX
    python benchmarks/finetune_laya.py --init <dir> --out <dir2> \\
        --mix coir-stackoverflow-qa=16000,multidoc2dial=4000,swe=3000,scifact=2000,zalo-legal=2000,category=2000
    set INVENTIO_DISPOSITIO_MODEL=<dir2>                    # then --ranker dispositio loads it

The released dispositio is those two stages: the second puts back the programming Q&A the first,
starting from the published model, had too little of (StackOverflow QA fell below BM25).

Laya learns the questions Inventio asks, in the exact form it reads them: the ranker's (rankers.
INSTRUCTIONS / CRITERIA: does the passage answer the query?) over the query and a passage rendered
as `[path > heading]` + text, and the category judge's (facts.category_question). Relevance labels
are written by people; category labels, for which no human set exists, by a small general model
(benchmarks/category_data.py). `--mix` sets how many training items each source gives:

- relevance: the train split of four public benchmarks with human-annotated answers, SciFact
  (English science), StackOverflow QA (English, code), Zalo legal (Vietnamese law) and
  MultiDoc2Dial (US public-service pages: rules, eligibility, procedures). Each train query gives
  its answer chunks as positives and BM25 candidates from other documents that are not answers as
  hard negatives, half from ranks 1-10 and half from 11-30. MultiDoc2Dial also gives, when the
  question is grounded in its answer, the answer page's own section BM25 ranks highest as a
  negative at SIB: the passage that shares the question's words and topic and does not answer it.
- swe: a SWE-bench train issue and the code its fix changed (benchmarks/swe_train.py); none of
  the 35 repositories is among SWE-bench Lite's 12.
- category: a passage alone and the choice question over CATEGORIES, the label as a soft target.
- title (`scifact:title`, `zalo-legal:title`, not in the released mix): a document's own title as
  the query and its first chunk as the answer. Kept for experiments only: it teaches that the
  passage repeating the query's words is the answer, which BM25 already knows, and a model
  trained on many of them ranks a section restating a question above the one that answers it.

Targets are 0.95 / 0.05, not 1 / 0: qrels are incomplete and a BM25 negative can be an unmarked
answer. On half of the groups the path is dropped from the passage (`[heading]` + text), so the
model cannot lean on file names or document ids.

Kept out of training: every test query of the benchmarks (and any train query with the same
text), MultiDoc2Dial's validation split and its whole studentaid domain (new pages of the same
genre, the check that the model learnt the genre and not the pages). 10% of the other train
queries (sha1) are held out; for them all 30 BM25 candidates are scored, in their natural
proportion of answers, and split in two: one half fits the temperature, the other is the report
(AUC, Brier, and nDCG@10 of the 30 candidates reordered, against BM25's own order), for the
starting model and the tuned one.

The loop is the one in the Laya author's fine-tuning notebook
(notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb in NandhaKishorM/laya): policy
gradient on a proper scoring rule plus soft cross-entropy. Ported to one GPU with bf16, with a 5%
linear warm-up before the cosine decay; the token-embedding matrix (197M of the 322M parameters)
is frozen so the optimiser fits in 8 GB, which also keeps every language's token vectors as the
base model learned them.
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from beir_bench import load_beir, safe_name  # noqa: E402
from data import data_dir  # noqa: E402
from inventio.facts import CATEGORIES, category_question  # noqa: E402
from inventio.rankers import CRITERIA, INSTRUCTIONS, cap_query, passage_room, windows  # noqa: E402
from inventio.search import bm25  # noqa: E402
from inventio.store import connect  # noqa: E402

DATASETS = ("scifact", "coir-stackoverflow-qa", "zalo-legal", "multidoc2dial")
TITLES = ("scifact", "zalo-legal")  # corpora whose documents carry a real title
SOURCES = (*DATASETS, *(f"{t}:title" for t in TITLES), "swe", "category")
MIX = "multidoc2dial=18000,zalo-legal=10000,coir-stackoverflow-qa=12000,scifact=3700,swe=10000,category=4400"
HOLDOUT = 0.10
POOL = 30
POS, NEG = 0.95, 0.05
SIB = 0.25  # another section of the answer's own page: the same topic and words, most likely not the answer
CAT_ON, CAT_OFF = 0.94, 0.01  # a category's soft target: the label is a small model's, not a gold one
SEED = 20260923
# MultiDoc2Dial: the page a section belongs to (dmv-3-1 is section 1 of page dmv-3), and one other
# section of it as a negative, when the question is tightly grounded in its answer.
SIBLINGS = {"multidoc2dial": (lambda doc: doc.rsplit("-", 1)[0], 1)}
BM25_NEGS = {"multidoc2dial": 2}  # BM25 negatives per group from other documents (default 4)
HELD_DOMAIN = {"multidoc2dial": "studentaid"}  # held out whole: new pages of the same genre


def sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def held(key: str) -> bool:
    return int(sha(key)[:8], 16) / 0xFFFFFFFF < HOLDOUT


def render(h, *, heading: bool, path: bool) -> dict:
    """Hit.passage() as its `[path > heading]` line and its text, optionally without the heading
    or the path; `at` carries the text lines a fix changed, when the group knows them."""
    head = " > ".join(x for x in (h.path if path else "", h.heading_path if heading else "") if x)
    return {"head": f"[{head}]\n" if head else "", "body": h.text, "at": getattr(h, "at", None)}


def doc_of(h) -> str:
    return Path(h.path).stem


# ------------------------------------------------------------------------------ groups

def relevance_groups(name: str, ds: Path, con, rng):
    """One group per train query: (held out?, a function building the group). BM25 runs only
    when the group is built, so a source whose share is full skips its remaining queries cheaply.

    Gold is per document; every chunk of an answer document in the pool is an answer, and one
    BM25 missed is added only when the document is a single chunk (which chunk of a longer one
    answers is not known). MultiDoc2Dial trains on its topic groups (data.py `train_groups.jsonl`)
    instead of its first turns: the sections the topic went on to use (`near`) are `skip`, other
    sections of the answer's page are never negatives from BM25, and when the question was
    grounded in the answer itself (`tight`) the page's section BM25 ranks highest is a negative at
    SIB: a passage that shares the question's words and topic and does not answer it."""
    _, test_q, _ = load_beir(ds, "test")
    tests = {" ".join(q.lower().split()) for q in test_q.values()}  # MultiDoc2Dial repeats some openings
    if (ds / "train_groups.jsonl").exists():
        rows = [json.loads(l) for l in (ds / "train_groups.jsonl").open(encoding="utf-8")]
        entries = [(r["query"], set(r["gold"]), set(r["near"]), r["tight"], r["domain"] == HELD_DOMAIN.get(name))
                   for r in rows]
    else:
        _, train_q, train_rel = load_beir(ds, "train")
        entries = [(train_q[q], {safe_name(d) for d, s in train_rel[q].items() if s > 0}, set(), False,
                    held(f"{name}\t{train_q[q]}")) for q in train_rel if q in train_q and q not in test_q]
    entries = [e for e in entries if " ".join(e[0].lower().split()) not in tests]
    rng.shuffle(entries)
    files = con.execute("SELECT f.id, f.path, count(c.id) n FROM files f JOIN sources s ON s.id = f.source_id "
                        "JOIN chunks c ON c.file_id = f.id WHERE s.name = ? GROUP BY f.id", (name,)).fetchall()
    chunks = {Path(r["path"]).stem: r["n"] for r in files}
    family, n_sib = SIBLINGS.get(name, (None, 0))
    members: dict[str, list[tuple[int, str]]] = {}
    for r in files if family else ():
        members.setdefault(family(Path(r["path"]).stem), []).append((r["id"], Path(r["path"]).stem))
    for q, gold, near, tight, is_held in entries:

        def make(q=q, gold=gold, near=near, tight=tight):
            hits = bm25(con, q, POOL, [name])
            got = {doc_of(h) for h in hits}
            extra = [c for c in (gold_chunk(con, name, d) for d in sorted(gold - got) if chunks.get(d) == 1) if c][:1]
            fams = {family(d) for d in gold} if family else set()
            sibling = lambda d: d not in gold and d not in near and family(d) in fams  # noqa: E731
            sib_files = [fid for f in fams for fid, d in members.get(f, ()) if sibling(d)] if tight else []
            return {"src": f"{name}:relevance", "query": q, "heading": True, "hits": hits,
                    "gold": [i for i, h in enumerate(hits) if doc_of(h) in gold],
                    "skip": [i for i, h in enumerate(hits) if doc_of(h) in near],
                    "sib_idx": [i for i, h in enumerate(hits) if family and sibling(doc_of(h))],
                    "sib": bm25(con, q, n_sib, [name], files=sib_files) if sib_files else [],
                    "n_neg": BM25_NEGS.get(name, 4), "extra": extra}

        yield is_held, make


def title_groups(name: str, ds: Path, con, rng):
    """One group per unique title: the title as query, its document's first chunk as the answer."""
    _, _, test_rel = load_beir(ds, "test")
    banned = {safe_name(d) for rel in test_rel.values() for d in rel}
    rows = con.execute(
        "SELECT f.path, c.heading_path, c.text, min(c.start_line) FROM chunks c JOIN files f ON f.id = c.file_id "
        "JOIN sources s ON s.id = f.source_id WHERE s.name = ? GROUP BY c.file_id", (name,)).fetchall()
    count: dict[str, int] = {}
    for r in rows:
        count[r["heading_path"]] = count.get(r["heading_path"], 0) + 1
    rows = [r for r in rows if count[r["heading_path"]] == 1 and Path(r["path"]).stem not in banned]
    rng.shuffle(rows)
    for r in rows:
        title = r["heading_path"].split(" > ")[-1]
        q = title.split(". ", 1)[1] if title.startswith("Điều ") and ". " in title else title  # "Điều 5. ..."
        if len(q.split()) < 4:
            continue
        own = SimpleNamespace(path=r["path"], heading_path=r["heading_path"], text=r["text"])

        def make(q=q, own=own):
            hits = [h for h in bm25(con, q, POOL + 5, [name]) if h.path != own.path][:POOL - 1]
            return {"src": f"{name}:title", "query": q, "heading": False, "hits": [own] + hits, "gold": [0], "extra": []}

        yield held(f"{name}\t{r['path']}"), make


def swe_groups(rng):
    """One group per issue built by swe_train.py: the chunks the fix changed are the answers.
    A patch says which code was changed, not that anything else is a wrong answer, so negatives
    are only other source code: the untouched chunks of the files it touched, and every test,
    doc or config chunk, are `skip`, neither answer nor negative. With docs as negatives the model
    learns that an issue-shaped question wants code, and ranks prose answers down on
    StackOverflow QA (0.698 to 0.612 nDCG@10)."""
    from inventio.ingest import LANGS, doc_type

    code = lambda path: doc_type(path, LANGS.get(Path(path).suffix.lower(), "code")) == "SoftwareSourceCode"  # noqa: E731
    rows = [json.loads(l) for p in sorted((data_dir(None) / "swe-train" / "groups").glob("*.jsonl"))
            for l in p.open(encoding="utf-8")]
    rng.shuffle(rows)
    ns = lambda d: SimpleNamespace(**d)  # noqa: E731
    for r in rows:
        def make(r=r):
            skip = set(r["skip"]) | {i for i, h in enumerate(r["hits"]) if i not in r["gold"] and not code(h["path"])}
            return {"src": "swe:relevance", "query": r["query"], "heading": True, "hits": [ns(h) for h in r["hits"]],
                    "gold": r["gold"], "skip": sorted(skip), "extra": [ns(h) for h in r["extra"]]}

        yield held(f"swe\t{r['iid']}"), make


def gold_chunk(con, name: str, doc: str):
    """The first chunk of an answer document BM25 did not put among its 30."""
    r = con.execute(
        "SELECT f.path, c.heading_path, c.text FROM chunks c JOIN files f ON f.id = c.file_id "
        "JOIN sources s ON s.id = f.source_id WHERE s.name = ? AND f.path = ? ORDER BY c.start_line LIMIT 1",
        (name, f"docs/{doc}.md")).fetchone()
    return SimpleNamespace(path=r["path"], heading_path=r["heading_path"], text=r["text"]) if r else None


def train_items(g, rng) -> list[dict]:
    """Answers, BM25 hard negatives from other documents (half from ranks 1-10, half from 11-30)
    and the group's same-document negatives, in one rendering (path dropped half the time)."""
    path = rng.random() < 0.5
    rend = lambda h: render(h, heading=g["heading"], path=path)  # noqa: E731
    pos = [g["hits"][i] for i in g["gold"][:2]] or g["extra"]
    out = {*g["gold"], *g.get("skip", ()), *g.get("sib_idx", ())}
    neg_idx = [i for i in range(len(g["hits"])) if i not in out]
    top, rest = [i for i in neg_idx if i < 10], [i for i in neg_idx if i >= 10]
    k = g.get("n_neg", 4) // 2
    negs = rng.sample(top, min(k, len(top))) + rng.sample(rest, min(k, len(rest)))
    return ([{"src": g["src"], "query": g["query"], **rend(h), "p": POS} for h in pos]
            + [{"src": g["src"], "query": g["query"], **rend(g["hits"][i]), "p": NEG} for i in negs]
            + [{"src": g["src"], "query": g["query"], **rend(h), "p": SIB} for h in g.get("sib", ())])


def eval_items(g) -> list[dict]:
    """All candidates of a held-out group, as the ranker sees them at query time (path kept)."""
    gold, skip = set(g["gold"]), set(g.get("skip", ()))
    return [{"src": g["src"], "query": g["query"], **render(h, heading=g["heading"], path=True),
             "p": 1.0 if i in gold else 0.0, "rank": i} for i, h in enumerate(g["hits"]) if i not in skip]


def category_rows(rng) -> tuple[list[dict], list[dict]]:
    """Passages labelled by benchmarks/category_data.py, rendered as `facts.categorize` renders them;
    training drops the path half the time, so a file name is not the cue."""
    d = data_dir(None) / "categories"

    def item(r, src, path):
        head = " > ".join(x for x in (r["path"] if path else "", r["heading_path"]) if x)
        return {"src": src, "kind": "category", "query": "", "p": 1.0, "rank": 0, "label": r["label"],
                "head": f"[{head}]\n" if head else "", "body": r["text"]}

    train = [item(json.loads(l), "category", rng.random() < 0.5) for l in (d / "train.jsonl").open(encoding="utf-8")]
    rng.shuffle(train)
    test = [item(r, f"category {r['src'].split(':')[0]}", True)
            for r in map(json.loads, (d / "test.jsonl").open(encoding="utf-8"))]
    return train, test


def build(mix: dict[str, int], eval_groups: int, rng) -> tuple[list[dict], list[dict], dict]:
    """Up to `mix[source]` training items from each source (a source short of its quota just gives
    what it has). Held-out groups are scored whole and never trained on."""
    sources = []
    for name, quota in mix.items():
        if name == "swe":
            sources.append(("swe:relevance", swe_groups(rng), quota))
            continue
        if name == "category":
            sources.append(("category", category_rows(rng), quota))
            continue
        corpus, _, kind = name.partition(":")
        ds = data_dir(None) / "beir" / corpus
        db = ds / "inventio.db"
        if not db.exists():
            print(f"skip {name}: no map at {db} (run benchmarks/beir_bench.py {corpus} first)", flush=True)
            continue
        con = connect(db)
        groups = title_groups(corpus, ds, con, rng) if kind == "title" else relevance_groups(corpus, ds, con, rng)
        sources.append((f"{corpus}:{kind or 'relevance'}", groups, quota))
    train, held_out, report = [], [], {}
    for src, groups, share in sources:
        n_train, n_eval, n_groups = 0, 0, 0
        if src == "category":
            rows, test = groups
            train += rows[:share]
            held_out += test
            report[src] = {"train_items": min(share, len(rows)), "eval_items": len(test)}
            print(src, report[src], flush=True)
            continue
        for is_held, make in groups:
            if n_train >= share and n_eval >= eval_groups:
                break
            if is_held:
                if n_eval < eval_groups:
                    g = make()
                    if g["gold"]:  # an answer among the 30, or nothing to reorder
                        held_out += eval_items(g)
                        n_eval += 1
            elif n_train < share:
                items = train_items(make(), rng)
                if any(it["p"] == POS for it in items):
                    train += items
                    n_train += len(items)
                    n_groups += 1
        report[src] = {"train_items": n_train, "train_groups": n_groups, "eval_groups": n_eval,
                       "same_document_negatives": sum(it["p"] == SIB for it in train if it["src"] == src)}
        print(src, report[src], flush=True)
    rng.shuffle(train)
    return train, held_out, report


# ------------------------------------------------------------------------------ training

def load_base():
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config

    root = snapshot_download("convaiinnovations/laya")
    d = os.path.join(root, "multilingual")
    _fix_tokenizer_config(d)
    return d


def tokenize(rows, tok, cfg, *, whole=False):
    """Items as LayaRanker reads them: the query cut to its first QUERY_TOKENS tokens, and a passage
    longer than what is left read in windows of whole lines (rankers.windows).
    Training keeps one window per passage. Every window of a negative is a negative, so one is
    drawn; an answer keeps the window holding the lines its fix changed (`at`), and an answer
    without that mark is left out, since which of its windows answers is unknown.
    Held out (`whole`), every window is kept under its passage's `key` and the passage is scored
    by its best window, as at query time.
    A category item is the passage alone and the choice question facts.categorize asks, cut from
    the right as Laya cuts it there: the opening of a passage says what it does."""
    from laya.common import QTYPES, build_sequence, render_options

    q = {"t": "noul", "ins": INSTRUCTIONS, "crit": CRITERIA}
    k = len(render_options(q))
    cq = {"t": "choice", "ins": category_question()["instructions"], "crit": CATEGORIES}
    labels = list(CATEGORIES)
    max_len, head_len = cfg["max_len"], cfg["head_max_len"]
    items, cut, rooms = [], 0, {}
    for n, r in enumerate(rows):
        if r.get("kind") == "category":
            text = r["head"] + r["body"]
            seq, markers = build_sequence(tok, {"passage": text}, cq, max_len, head_len)
            if len(markers) != len(labels):
                cut += 1
                continue
            items.append({**r, "passage": text, "key": n, "windows": 1, "ids": seq, "markers": markers,
                          "qtype": QTYPES["choice"], "target": [CAT_ON if c == r["label"] else CAT_OFF for c in labels]})
            continue
        if r["query"] not in rooms:
            capped = cap_query(tok, r["query"])
            rooms[r["query"]] = capped, passage_room(tok, capped, q, max_len, head_len)
        query, room = rooms[r["query"]]
        wins, p = windows(tok, r["head"], r["body"], room), float(r["p"])
        if whole or len(wins) == 1:
            keep = wins
        elif p >= 0.5:
            keep = [w for w in wins if any(w[1] <= x <= w[2] for x in r.get("at") or ())][:1]
        else:
            keep = [wins[zlib.crc32(r["body"].encode()) % len(wins)]]
        if not keep:
            cut += 1
        for text, _, _ in keep:
            seq, markers = build_sequence(tok, {"query": query, "passage": text}, q, max_len, head_len)
            if len(markers) != k or (not whole and len(seq) >= max_len):
                cut += 1  # one line longer than the window: the label would be about text Laya never reads
                continue
            items.append({**r, "passage": text, "key": n, "windows": len(keep), "ids": seq, "markers": markers,
                          "qtype": QTYPES["noul"], "target": [1 - p, p]})
    return items, cut


def collate(items, pad_id):
    import torch

    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"])
    return ids, att, mpos, mmask, target, torch.tensor([it["qtype"] for it in items])


def logits_of(model, items, tok, dev) -> list:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(items), 16):
            ch = items[i:i + 16]
            ids, att, mpos, mmask, _, qtype = (x.to(dev) for x in collate(ch, tok.pad_token_id))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, _ = model(ids, att, mpos, mmask, qtype)
            lg = lg.float().cpu()
            out += [lg[j, :len(ch[j]["markers"])].tolist() for j in range(len(ch))]
    return out


def fit_temp(logits, items) -> float:
    """On passages read in one window: a long answer's other windows carry no label of their own."""
    import torch

    pairs = [(lg, it) for lg, it in zip(logits, items) if it["windows"] == 1]
    if len(pairs) < 10:
        return 1.0
    Z = torch.tensor([lg for lg, _ in pairs])
    T = torch.tensor([it["target"] for _, it in pairs], dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def category_report(logits, items) -> dict:
    """Per test stratum: accuracy and macro-F1 of the likeliest category against the labels.
    `in domain` pools the corpora whose kind was trained on, `out of domain` the held-out
    repositories and StackOverflow; the gap between them is the overfitting."""
    labels = list(CATEGORIES)
    pred = [(labels[max(range(len(lg)), key=lg.__getitem__)], it) for lg, it in zip(logits, items)]

    def score(sel):
        f1s = []
        for c in {it["label"] for _, it in sel}:
            tp = sum(p == c == it["label"] for p, it in sel)
            fp = sum(p == c != it["label"] for p, it in sel)
            fn = sum(it["label"] == c != p for p, it in sel)
            f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
        return {"n": len(sel), "accuracy": round(sum(p == it["label"] for p, it in sel) / len(sel), 3),
                "macro_f1": round(sum(f1s) / len(f1s), 3)}

    ood = ("category held-repo", "category coir-stackoverflow-qa")
    out = {src: score([x for x in pred if x[1]["src"] == src]) for src in sorted({it["src"] for it in items})}
    out["category in domain"] = score([x for x in pred if not x[1]["src"].startswith(ood)])
    out["category out of domain"] = score([x for x in pred if x[1]["src"].startswith(ood)])
    return out


def report(logits, items, temp: float) -> dict:
    """Per source: AUC and Brier on the natural mix, nDCG@10 of each group's candidates reordered.
    A passage read in windows counts once, at its best window. Category items: category_report."""
    cat = [(lg, it) for lg, it in zip(logits, items) if it.get("kind") == "category"]
    best: dict[int, tuple] = {}
    for lg, it in zip(logits, items):
        if it.get("kind") == "category":
            continue
        pi = 1 / (1 + math.exp(-(lg[1] - lg[0]) / temp))
        if it["key"] not in best or pi > best[it["key"]][0]:
            best[it["key"]] = pi, it
    out = category_report(*zip(*cat)) if cat else {}
    for src in sorted({it["src"] for _, it in best.values()}):
        sel = [(pi, it) for pi, it in best.values() if it["src"] == src]
        p = [pi for pi, _ in sel]
        y = [it["p"] for _, it in sel]
        pos = [a for a, b in zip(p, y) if b]
        neg = [a for a, b in zip(p, y) if not b]
        auc = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / max(1, len(pos) * len(neg))
        groups: dict[str, list] = {}
        for pi, (_, it) in zip(p, sel):
            groups.setdefault(it["query"], []).append((pi, it["rank"], it["p"]))

        def ndcg(order):
            dcg = sum(g / math.log2(i + 2) for i, (_, _, g) in enumerate(order[:10]))
            ideal = sum(1 / math.log2(i + 2) for i in range(min(10, int(sum(g for *_, g in order)))))
            return dcg / ideal if ideal else 0.0

        out[src] = {"items": len(sel), "positives": len(pos), "auc": round(auc, 4),
                    "brier": round(sum((a - b) ** 2 for a, b in zip(p, y)) / len(y), 4),
                    "ndcg10_ranked": round(sum(ndcg(sorted(g, key=lambda x: -x[0])) for g in groups.values()) / len(groups), 4),
                    "ndcg10_bm25": round(sum(ndcg(sorted(g, key=lambda x: x[1])) for g in groups.values()) / len(groups), 4)}
    return out


def train(items, calib, test, base_dir, out_dir: Path, *, name, epochs, micro, accum, time_steps=0):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from laya.common import build_model, proper_reward

    cfg = json.load(open(os.path.join(base_dir, "rl_agent_config.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(base_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(base_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(base_dir, "model.safetensors")), strict=True)
    dev = torch.device("cuda")
    model.to(dev)
    base_temp = float(cfg.get("temperature", [1.0, 1.0, 1.0])[2])
    before = None
    if not time_steps:
        before = report(logits_of(model, test, tok, dev), test, base_temp)
        print(f"starting model ({base_dir}) on held-out:", json.dumps(before), flush=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    for p in model.encoder.embeddings.tok_embeddings.parameters():
        p.requires_grad = False
    model.train()

    # batches of similar length waste less padding; shuffled again per epoch below
    items = sorted(items, key=lambda it: len(it["ids"]))
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc, "lr": 2.5e-5}, {"params": head, "lr": 1.0e-4}], weight_decay=0.01)
    batches = [items[i:i + micro] for i in range(0, len(items), micro)]
    total = max(1, math.ceil(len(batches) / accum) * epochs)
    warm = max(1, total // 20)
    floor = 1e-6 / 2.5e-5
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    G, S0, S1 = 4, 0.4, 0.1
    t0, step = time.time(), 0
    print(f"{len(items)} train items, {len(calib)} calibration, {len(test)} report, "
          f"{len(batches)} batches x {epochs} epochs, {total} updates ({warm} warm-up)", flush=True)
    for epoch in range(epochs):
        random.Random(42 + epoch).shuffle(batches)
        sigma = S0 + (S1 - S0) * epoch / max(1, epochs - 1)
        opt.zero_grad(set_to_none=True)
        run_loss = 0.0
        for b, chunk in enumerate(batches, 1):
            ids, att, mpos, mmask, target, qtype = (x.to(dev) for x in collate(chunk, tok.pad_token_id))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, act = model(ids, att, mpos, mmask, qtype)
            logits = logits.float()
            k = mmask.sum(-1, keepdim=True).float()
            eps = torch.randn((G,) + logits.shape, device=dev) * sigma * mmask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mmask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mmask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), qtype, mmask, w_sph=0.75, w_rps=1.0)
                adv = (r - r.mean(0, keepdim=True)) / ((r - r.mean(0, keepdim=True)).std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mmask).sum(-1) / (2 * sigma ** 2)
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mmask, -1e4), -1)).sum(-1).mean()
            loss = (-(adv * logp).mean() + loss_ce) / accum + 0.0 * act.sum()
            loss.backward()
            run_loss += loss.item() * accum
            if b % accum == 0 or b == len(batches):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            if time_steps and b == time_steps:
                torch.cuda.synchronize()
                per = (time.time() - t0) / b
                print(json.dumps({"sec_per_batch": round(per, 3), "micro_batch": micro, "batches_per_epoch": len(batches),
                                  "epochs": epochs, "projected_hours": round(per * len(batches) * epochs / 3600, 2),
                                  "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}), flush=True)
                return None
            if b % 200 == 0:
                el = time.time() - t0
                done = epoch * len(batches) + b
                print(f"epoch {epoch + 1}/{epochs} batch {b}/{len(batches)} loss {run_loss / 200:.4f} "
                      f"{el / 60:.0f} min, ~{el / done * (len(batches) * epochs - done) / 60:.0f} min left", flush=True)
                run_loss = 0.0
        save(model, tok, cfg, out_dir / "checkpoint_latest", None, name)
        print(f"epoch {epoch + 1} done in {(time.time() - t0) / 60:.0f} min; checkpoint saved", flush=True)

    cal_logits = logits_of(model, calib, tok, dev)
    rel = [(lg, it) for lg, it in zip(cal_logits, calib) if it.get("kind") != "category"]
    cat = [(lg, it) for lg, it in zip(cal_logits, calib) if it.get("kind") == "category"]
    temp = fit_temp(*zip(*rel)) if rel else float(cfg.get("temperature", [1.0, 1.0, 1.0])[2])
    temps = list(cfg.get("temperature", [1.0, 1.0, 1.0]))
    temps[2] = temp
    if cat:
        temps[0] = fit_temp(*zip(*cat))  # the choice question's
    save(model, tok, cfg, out_dir, temps, name)
    after = report(logits_of(model, test, tok, dev), test, temp)
    print("tuned model on held-out:", json.dumps(after), flush=True)
    return {"minutes": round((time.time() - t0) / 60, 1), "temperatures": temps, "updates": step,
            "start": str(base_dir), "before": before, "after": after}


def save(model, tok, cfg, d: Path, temps, name: str):
    from safetensors.torch import save_file

    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}, str(d / "model.safetensors"))
    model.encoder.config.save_pretrained(str(d / "encoder"))
    tok.save_pretrained(str(d / "tokenizer"))
    c = dict(cfg, fine_tuned=True, model_name=name)
    if temps is not None:
        c["temperature"] = temps
        c.pop("temperature_by_options", None)
    (d / "rl_agent_config.json").write_text(json.dumps(c, indent=2))


def main() -> int:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="dispositio", help="model name, and the checkpoint directory's")
    ap.add_argument("--out", help="checkpoint directory (default: <user cache>/inventio/<name>)")
    ap.add_argument("--init", help="start from this checkpoint directory instead of the published Laya")
    ap.add_argument("--mix", default=MIX, help=f"training items per source, source=N,... from {', '.join(SOURCES)}")
    ap.add_argument("--eval-groups", type=int, default=60, help="held-out groups scored per source (30 candidates each)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8, help="micro-batches per optimiser step (effective batch 64)")
    ap.add_argument("--time-steps", type=int, default=0, help="time this many micro-batches, print the projection, exit")
    args = ap.parse_args()

    rng = random.Random(SEED)
    mix = {k: int(v) for k, v in (p.split("=") for p in args.mix.split(","))}
    unknown = set(mix) - set(SOURCES)
    if unknown:
        ap.error(f"unknown sources {sorted(unknown)}; choose from {', '.join(SOURCES)}")
    rows, held_out, per_source = build(mix, args.eval_groups, rng)
    base = args.init or load_base()
    cfg = json.load(open(os.path.join(base, "rl_agent_config.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(base, "tokenizer"))
    items, cut = tokenize(rows, tok, cfg)
    evals, cut_eval = tokenize(held_out, tok, cfg, whole=True)
    split = lambda it: it["query"] or it["passage"]  # noqa: E731  (a category item has no query)
    by_query = sorted({split(it) for it in evals})
    cal_q = set(by_query[::2])  # half the held-out queries fit the temperature, the other half is the report
    calib = [it for it in evals if split(it) in cal_q]
    test = [it for it in evals if split(it) not in cal_q]
    out = Path(args.out or Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "inventio" / args.name)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "holdout.jsonl").open("w", encoding="utf-8") as f:
        for it in evals:
            f.write(json.dumps({k: it[k] for k in ("src", "query", "key", "passage", "p", "rank")}, ensure_ascii=False) + "\n")
    meta = {"name": args.name, "start": str(base), "mix": mix, "per_source": per_source,
            "train_items": len(items), "cut": cut, "held_out_items": len(evals),
            "cut_held_out": cut_eval, "epochs": args.epochs, "micro_batch": args.micro_batch, "accum": args.accum}
    print(json.dumps(meta, indent=1), flush=True)
    res = train(items, calib, test, base, out, name=args.name, epochs=args.epochs, micro=args.micro_batch,
                accum=args.accum, time_steps=args.time_steps)
    if res is not None:
        meta["result"] = res
        (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
        print(f"{args.name} in {out}; set INVENTIO_DISPOSITIO_MODEL={out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
