"""System One over BM25's pool: read the query and the passages once, answer `where_line` (which line
holds the answer) and `exists` (does any passage answer at all), and measure that against dispositio.

    python benchmarks/systemone.py data                       # Kev records -> <data>/s1/data/*.jsonl
    python benchmarks/systemone.py train --bal 4000 --out runs/s1-v1   # fine-tune Kev (recipe pinned here)
    python benchmarks/systemone.py spike <url> <tag> [--sets md2d,webshop,techqa] [--limit N]
    python benchmarks/systemone.py disp  <tag> [--sets ...]   # dispositio on the same pools
    python benchmarks/systemone.py gate  <kev tag> <v3 tag> <set>
    python benchmarks/systemone.py lines <tag>                # exact-line top-1 on MultiDoc2Dial test

The pool is BM25's `K` hits for the query, rendered as `P01 [path > heading]` then one `L000| line` per
non-blank line, so passage ids and line ids are what the model reads and answers with. `exists`
negatives are the same query with every gold (and, for MultiDoc2Dial, near) chunk removed and the pool
refilled from BM25's next hits. Rows go to `results/s1/<tag>-<set>.jsonl`, summaries are merged into
`results/s1/summary.json` — per-query rows are the receipt, the summary is for reading.

A served model answers `/v1/systemone`; Kev and dispositio are both driven through `spike` and `disp`,
so the two are always compared on identical pools and identical question strings.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
import time
import zlib
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from beir_bench import load_beir, safe_name  # noqa: E402
from data import data_dir  # noqa: E402
from inventio.scope import Scope  # noqa: E402
from inventio.search import bm25  # noqa: E402
from inventio.store import connect  # noqa: E402
from inventio.systemone import (EXISTS_ASKS, MAX_OPTIONS, MAX_QUERY_CHARS, MAX_STATE_CHARS, NOUL,  # noqa: E402
                                PASSAGE_ASKS, WHERE_ASKS, ask, exists, questions, render, where_line)

REPO = Path(__file__).resolve().parent.parent
ROWS = Path(__file__).resolve().parent / "results" / "s1"
DATA = data_dir(None) / "s1" / "data"
K = 15
# the Kev checkpoint we fine-tune from, and the base it was trained on (its own training_config.json)
INIT_FROM = "jaredpalmer/kev-0.8b"
BASE = "Qwen/Qwen3.5-0.8B-Base"
BASE_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
norm = lambda t: " ".join(t.split())  # noqa: E731
WORD = re.compile(r"\w+", re.UNICODE)   # the lexical probe's tokens (cmd_leak)


def pools(con, q, scope, is_gold, k=K, drop=()):
    """The natural pool and the negative pool: the same query with every gold (and `drop`) chunk gone,
    refilled from BM25's next hits."""
    hits = bm25(con, q, k + 40, scope)
    pos = hits[:k]
    neg = [h for h in hits if not is_gold(h) and Path(h.path).stem not in set(drop)][:k]
    return pos, neg


def auc(pos, neg):
    s = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return s / max(1, len(pos) * len(neg))


def asker(url="", run="", device=None, model="jev-latest"):
    """Where a state goes: the served server, or the checkpoint read in this process
    (`inventio.systemone.Model`) — the same state and the same questions either way."""
    if url:
        return lambda state, qs: ask(url, state, qs, model)
    from inventio.systemone import Model
    loaded = Model(run or None, device)
    return lambda state, qs: loaded.ask(state, qs)


def run_query(ask_, q, pos, neg, is_gold, passage_asks=False):
    """One query through the model twice: the natural pool, and the pool without the answer.

    The served shape is asked always (the choice heads and `exists`); `passage_asks` adds the
    per-passage questions, whose probabilities are recorded so the two ways of reading the same state
    can be compared — the choice head against argmax over the per-passage head, and `exists` against
    the max of those probabilities."""
    out = {}
    for arm, hits in (("pos", pos), ("neg", neg)):
        state, pids, lids, owner = render(hits)
        qs = questions(q, pids, lids, with_passage_asks=passage_asks)
        res = ask_(state, qs)
        a = res["answers"]
        pp = a["where_passage"]["probabilities"]
        top_p = max(pp, key=pp.get)
        row = {"exists": a["exists"]["noul"], "latency_ms": res.get("latency_ms"), "wall_ms": res["wall_ms"],
               "tokens": res.get("usage", {}).get("input_tokens"), "lines": len(lids),
               "where_line": "where_line" in qs and "where_line" in a}
        if passage_asks:   # the study shape: the per-passage head, recorded beside the served one
            pv = [a[p]["noul"] for p in pids]
            row.update(p_passage=[round(v, 4) for v in pv], exists_p=max(pv),
                       top_passage_p=max(range(len(pv)), key=pv.__getitem__))
        if arm == "pos":
            gold = [i for i, h in enumerate(hits) if is_gold(h)]
            row.update(gold=gold, top_passage=int(top_p[1:]) - 1, bm25_top_gold=0 in gold)
            if row["where_line"]:
                lp = a["where_line"]["probabilities"]
                li = int(max(lp, key=lp.get)[1:])
                row["top_line_owner"] = owner[li]
                row["top_line_text"] = state.split(f"{lids[li]}| ", 1)[1].split("\n", 1)[0]
        out[arm] = row
    return out


def spike_summary(rows, tag, name):
    has = [r for r in rows if r["pos"]["gold"]]
    absent = [r for r in rows if not r["pos"]["gold"]]
    n = len(has)
    p1 = sum(r["pos"]["top_passage"] in r["pos"]["gold"] for r in has)
    b1 = sum(r["pos"]["bm25_top_gold"] for r in has)
    lr = [r for r in has if "top_line_owner" in r["pos"]]
    l1 = sum(r["pos"]["top_line_owner"] in r["pos"]["gold"] for r in lr)
    ex = auc([r["pos"]["exists"] for r in has], [r["neg"]["exists"] for r in rows])
    # The deployment question is whether the answer is inside the state at all, and the queries whose
    # answer BM25 never put in the pool are exactly that question, answered for free. The `neg` arm
    # above is the wrong negative for it: it is built by the same procedure the model trains against, and
    # for a pool-miss query its 15 passages are the very ones the positive arm shows (160 of 613 rows on
    # md2d), so those rows enter the AUC as ties at 0.5 (measurement review, 2026-09-26).
    nat = auc([r["pos"]["exists"] for r in has], [r["pos"]["exists"] for r in absent]) if absent else None
    # The same two questions asked of the per-passage head: is argmax of its 15 probabilities the passage
    # that holds the answer, and is their max the honesty reading. Rows written before the harness asked
    # the per-passage questions have no vector, and are left unmeasured rather than counted as misses.
    per = [r for r in has if "p_passage" in r["pos"]]
    p1p = sum(r["pos"]["top_passage_p"] in r["pos"]["gold"] for r in per)
    exp = auc([r["pos"]["exists_p"] for r in has], [r["neg"]["exists_p"] for r in rows]) if per else None
    natp = auc([r["pos"]["exists_p"] for r in has], [r["pos"]["exists_p"] for r in absent]) if per and absent else None
    lat = sorted(x["latency_ms"] for r in rows for x in (r["pos"], r["neg"]) if x["latency_ms"] is not None)
    tok = sorted(x["tokens"] for r in rows for x in (r["pos"], r["neg"]) if x["tokens"])
    med = lambda v: v[len(v) // 2] if v else None  # noqa: E731
    return {"tag": tag, "set": name, "queries": len(rows), "gold_in_pool": n,
            "top_line_owner_passage@1": f"{l1}/{len(lr)}" if lr else "-",
            "passage@1": round(p1 / max(1, n), 3), "bm25@1": round(b1 / max(1, n), 3),
            "exists_auc": round(ex, 3),
            "answer_absent_queries": len(absent),
            "exists_auc_answer_absent": round(nat, 3) if nat is not None else None,
            "passage@1_pmax": round(p1p / max(1, len(per)), 3) if per else None,
            "exists_auc_pmax": round(exp, 3) if exp is not None else None,
            "exists_auc_answer_absent_pmax": round(natp, 3) if natp is not None else None,
            "exists_pos_median": med(sorted(r["pos"]["exists"] for r in has)),
            "exists_neg_median": med(sorted(r["neg"]["exists"] for r in rows)),
            "latency_ms_median": med(lat), "latency_ms_p90": lat[int(len(lat) * 0.9)] if lat else None,
            "tokens_median": med(tok), "tokens_max": tok[-1] if tok else None}


def record(kind, tag, name, s):
    ROWS.mkdir(parents=True, exist_ok=True)
    path = ROWS / "summary.json"
    all_ = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    all_.setdefault(kind, {})[f"{tag}/{name}"] = s
    path.write_text(json.dumps(all_, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(s), flush=True)


# --- the sets ------------------------------------------------------------------------------------

SCRATCH = {}


def set_md2d(limit):
    d = data_dir(None) / "beir" / "multidoc2dial"
    _, queries, qrels = load_beir(d)
    con = connect(d / "inventio.db")
    scope = Scope.only(["multidoc2dial"])
    for qid, q in list(queries.items())[: limit or None]:
        gold = {safe_name(c) for c in qrels[qid]}
        yield qid, q, con, scope, (lambda h, g=gold: Path(h.path).stem in g)


def set_techqa(limit):
    d = data_dir(None) / "beir" / "techqa"
    _, queries, qrels = load_beir(d)
    con = connect(d / "inventio.db")
    scope = Scope.only(["techqa"])
    for qid, q in list(queries.items())[: limit or None]:
        gold = {safe_name(c) for c in qrels[qid]}
        yield qid, q, con, scope, (lambda h, g=gold: Path(h.path).stem in g)


def set_webshop(limit):
    """The 13 hand-written on-call questions in examples/webshop, answered by a line of one wiki page."""
    from inventio.ingest import ingest_source
    root = REPO / "examples" / "webshop"
    con = connect(Path(tempfile.mkdtemp()) / "example.db")
    for src in sorted(p for p in root.iterdir() if p.is_dir()):
        ingest_source(con, src.name, src, True, [])
    for i, line in enumerate((root / "questions.jsonl").open(encoding="utf-8")):
        r = json.loads(line)
        src, path, ln = r["answer"].split(":")
        yield i, r["query"], con, None, (lambda h, s=src, p=path, n=int(ln):
                                         h.source == s and h.path == p and h.start_line <= n <= h.end_line)


SETS = {"md2d": set_md2d, "techqa": set_techqa, "webshop": set_webshop}
MD2D_TEST_SPANS = None


def md2d_test_spans():
    """dial_id -> the span texts the agent's first reply was grounded in (exact-line gold)."""
    global MD2D_TEST_SPANS
    if MD2D_TEST_SPANS is None:
        raw = data_dir(None) / "raw" / "multidoc2dial" / "multidoc2dial"
        pages = json.load((raw / "multidoc2dial_doc.json").open(encoding="utf-8"))["doc_data"]
        text = {(pid, sid): norm(sp["text_sp"]) for dom in pages.values() for pid, pg in dom.items()
                for sid, sp in pg["spans"].items()}
        MD2D_TEST_SPANS = {}
        for dom in json.load((raw / "multidoc2dial_dial_test.json").open(encoding="utf-8"))["dial_data"].values():
            for d in dom:
                t = d["turns"]
                if len(t) > 1:
                    MD2D_TEST_SPANS[d["dial_id"]] = [text[r["doc_id"], r["id_sp"]] for r in t[1]["references"]
                                                     if (r["doc_id"], r["id_sp"]) in text]
    return MD2D_TEST_SPANS


# --- training records ----------------------------------------------------------------------------

def slot_swap(hits, is_gold, k=K):
    """The same k slots with every gold slot filled by the best non-gold hit that BM25 ranked below the
    pool: no compaction, no shift, and the filler brings nothing about itself (page, domain, length) into
    the state. The two arms of a query then differ by exactly the swapped slots, which is what removes the
    corpus-composition cue a bag of words could ride (data review, 2026-09-26: a bigram TF-IDF reached AUC
    0.63 in-domain, 0.53-0.57 with a domain held out)."""
    spare = [h for h in hits[k:] if not is_gold(h)]
    out = []
    for h in hits[:k]:
        if is_gold(h):
            if spare:
                out.append(spare.pop(0))
        else:
            out.append(h)
    return out


def pick(wordings, key, salt):
    return wordings[int(hashlib.sha1(f"{key}\t{salt}".encode()).hexdigest()[:8], 16) % len(wordings)]


def records(q, pos, neg, lines_of, src, key, domain=None):
    """The questions one query asks, on both arms of the same query:

    - `where_line` (when the pool fits the option cap) and `where_passage`: which line, which passage;
    - `exists`: does any passage answer;
    - one question per passage, which is the decomposition `exists` needs. Without it the head is asked a
      single judgement over 3.3-6.5k tokens and never learns to compare slots, while dispositio, which
      beats it on this, is exactly a max over per-passage scores.
    """
    q = q[:MAX_QUERY_CHARS]
    out = []
    state, pids, lids, owner = render(pos)
    gold = []
    for i, h in enumerate(pos):
        mine = [j for j, o in enumerate(owner) if o == i]
        gold += [mine[k] for k in lines_of(i, h) if k < len(mine)]
    gold = sorted(set(gold))
    if state and (gold or neg):
        qs = {p: {"type": "noul", "instructions": pick(PASSAGE_ASKS, key, p).format(p=p, q=q),
                  "criteria": NOUL, "label": any(owner[j] == i for j in gold)}
              for i, p in enumerate(pids)}
        qs["exists"] = {"type": "noul", "instructions": pick(EXISTS_ASKS, key, "exists").format(q=q),
                        "criteria": NOUL, "label": bool(gold)}
        if gold:
            passages = sorted({owner[j] for j in gold})
            qs["where_passage"] = {"type": "choice", "label": pids[passages[0]],
                                   "instructions": pick(WHERE_ASKS, key, "wp").format(q=q),
                                   "criteria": {p: None for p in pids},
                                   "target": {pids[i]: 1 / len(passages) for i in passages}}
            if len(lids) <= MAX_OPTIONS:
                gl = [lids[j] for j in gold]
                qs["where_line"] = {"type": "choice", "label": gl[0],
                                    "instructions": f'Which line contains the answer to: "{q}"?',
                                    "criteria": {l: None for l in lids},
                                    "target": {l: 1 / len(gl) for l in gl}}
        out.append({"state": state, "questions": qs,
                    "_meta": {"source": src, "group_id": key, "domain": domain,
                              "arm": "pos" if gold else "neg"}})
    if neg:
        state, pids, _, _ = render(neg)
        qs = {p: {"type": "noul", "instructions": pick(PASSAGE_ASKS, key, p).format(p=p, q=q),
                  "criteria": NOUL, "label": False} for p in pids}
        qs["exists"] = {"type": "noul", "instructions": pick(EXISTS_ASKS, key, "exists").format(q=q),
                        "criteria": NOUL, "label": False}
        out.append({"state": state, "questions": qs,
                    "_meta": {"source": src, "group_id": key, "domain": domain, "arm": "neg"}})
    return out


def md2d_records(split):
    """MultiDoc2Dial topics (studentaid held out whole, like dispositio's recipe). Gold lines are the
    paragraphs holding the spans the agent's reply was grounded in; a heading span is skipped, its
    section's paragraphs are where the answer is read."""
    d = data_dir(None)
    raw = d / "raw" / "multidoc2dial" / "multidoc2dial"
    pages = json.load((raw / "multidoc2dial_doc.json").open(encoding="utf-8"))["doc_data"]
    clean = lambda t: re.sub(r"(#\d+(_\d+)?|\[\d+\])$", "", t.split(" | ")[0]).strip()  # noqa: E731
    section, span_text = {}, {}
    for dname, domain in pages.items():     # (page id, span id) -> section id, exactly as data.py
        for pi, (pid, page) in enumerate(domain.items()):
            title, parts = clean(page["title"]), {}
            for sid, sp in page["spans"].items():
                heads = [clean(t) for t in [*(p["text"] for p in sp["parent_titles"]), sp["title"]]]
                path = tuple(dict.fromkeys(h for h in heads if h and h != title))
                section[pid, sid] = path
                span_text[pid, sid] = norm(sp["text_sp"])
                paras = parts.setdefault(path, {})
                if sp["tag"] == "u":
                    paras.setdefault(sp["id_sec"], sp["text_sec"].strip())
            parts = {path: "\n\n".join(p for p in paras.values() if p) for path, paras in parts.items()}
            parts = {path: text for path, text in parts.items() if text}
            ids = {path: f"{dname}-{pi}-{si}" for si, path in enumerate(parts)}
            section.update({k: ids.get(v) for k, v in section.items() if k[0] == pid})
    con = connect(d / "beir" / "multidoc2dial" / "inventio.db")
    scope = Scope.only(["multidoc2dial"])
    dials = json.load((raw / f"multidoc2dial_dial_{split}.json").open(encoding="utf-8"))["dial_data"]
    vague = re.compile(r"\b(i have (a )?questions?|can you help|tell me about)\b", re.I)
    for dname, domain in dials.items():
        if dname == "studentaid":
            continue
        for dial in domain:
            turns, starts, prev = dial["turns"], [], None
            for i, t in enumerate(turns):
                if t["role"] == "user" and t["da"].startswith("query") and t["references"]:
                    pg = {r["doc_id"] for r in t["references"]}
                    if pg != prev:
                        starts.append(i)
                    prev = pg
            for n, i in enumerate(starts):
                u, a = turns[i], turns[i + 1] if i + 1 < len(turns) else None
                q = u["utterance"].strip()
                if a is None or a["role"] != "agent" or not a["da"].startswith("respond_solution"):
                    continue
                if vague.search(q) or ("?" not in q and len(q.split()) < 8):
                    continue
                sol = [r for r in a["references"] if r["label"] == "solution"]
                gold = {section.get((r["doc_id"], r["id_sp"])) for r in sol} - {None}
                spans = [span_text[r["doc_id"], r["id_sp"]] for r in sol if (r["doc_id"], r["id_sp"]) in span_text]
                if not gold or not spans:
                    continue
                end = starts[n + 1] if n + 1 < len(starts) else len(turns)
                is_gold = lambda h: Path(h.path).stem in gold  # noqa: E731
                hits = bm25(con, q, K + 40, scope)
                pos = hits[:K]
                neg = slot_swap(hits, is_gold)

                def lines_of(i, h):
                    if not is_gold(h):
                        return []
                    ls = [norm(x) for x in h.text.split("\n") if x.strip()]
                    out = []
                    for s in spans:
                        cand = [k for k, x in enumerate(ls) if s and s in x]
                        if not cand:
                            continue
                        k = max(cand, key=lambda k: len(s) / len(ls[k]))   # the most specific line, not the first
                        if s == ls[k] and len(ls[k]) < 60 and not ls[k].endswith("."):
                            continue   # a section heading the reply was grounded in, not the answering line
                        out.append(k)
                    return out
                yield from records(q, pos, neg, lines_of, f"md2d_{split}", f"{dial['dial_id']}:{i}",
                                    domain=dname)


def swe_records():
    """SWE-bench train issues (benchmarks/swe_train.py groups): gold lines are the lines the fix changed.
    Only the issues whose fix is inside BM25's 15 and whose pool fits the line question are kept."""
    for f in sorted(glob.glob(str(data_dir(None) / "swe-train" / "groups" / "*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            g = json.loads(line)
            hits = [SimpleNamespace(path=h["path"], heading_path=h["heading_path"], text=h["text"],
                                    at=h.get("at"), idx=i)
                    for i, h in enumerate(g["hits"])]
            gold, skip = set(g["gold"]), set(g["skip"])
            out_gold = lambda h: h.idx in gold | skip        # noqa: E731  (skipped hits are not answers)
            pos = hits[:K] if gold & set(range(K)) else []
            neg = slot_swap(hits, out_gold)

            def lines_of(i, h):
                if not h.at:
                    return []
                keep = [k for k, x in enumerate(h.text.split("\n")) if x.strip()]   # rendered lines skip blanks
                out = set()
                for a in h.at:
                    nxt = [j for j, k in enumerate(keep) if k >= a]
                    if nxt:
                        out.add(nxt[0])
                return sorted(out)
            yield from records(g["query"], pos, neg, lines_of, "swe_train", g["iid"],
                                domain=g["iid"].split("__")[0])


CAT_ON, CAT_OFF = 0.94, 0.01   # v3's soft target: the label is a small model's (Gemini Flash), not a gold one


def category_rows(split):
    """benchmarks/category_data.py's labelled passages: train, or test with its strata (`src`)."""
    return [json.loads(l) for l in (data_dir(None) / "categories" / f"{split}.jsonl").open(encoding="utf-8")]


def category_state(r, with_path=True):
    """The state `facts.categorize` sends a judge: `{"passage": "[path > heading]\ntext"}`."""
    from inventio.facts import passage
    if with_path:
        return {"passage": passage(r["path"], r["heading_path"], r["text"])}
    # the path dropped (training only): the heading alone in brackets, or the bare text, as v3's trainer rendered it
    return {"passage": (f"[{r['heading_path']}]\n" if r["heading_path"] else "") + r["text"]}


def judge_records():
    """The index-time questions the map is built with, in the exact request `facts` sends, labelled. Only the
    category question has labels (4,364 passages, benchmarks/category_data.py); the path is dropped half the
    time, as in dispositio's recipe, so a file name is not the cue."""
    from inventio.facts import CATEGORIES, category_question
    rng = random.Random(20260927)
    for r in category_rows("train"):
        q = {**category_question(), "label": r["label"],
             "target": {c: CAT_ON if c == r["label"] else CAT_OFF for c in CATEGORIES}}
        yield {"state": category_state(r, rng.random() < 0.5), "questions": {"category": q},
               "_meta": {"source": "category", "group_id": hashlib.sha1(r["text"].encode()).hexdigest()[:16],
                         "domain": r["src"].split(":")[0], "arm": "judge"}}


def cmd_judge(a):
    """The category question on its held-out passages (benchmarks/category_data.py test split): accuracy and
    macro-F1 per stratum, for a System One run read in this process. The out-of-domain strata are a held-out
    repository's prose and StackOverflow answers. v3's row on the same split is recorded under `judge/`."""
    from inventio.facts import CATEGORIES, SystemOneJudge, category_question
    judge = SystemOneJudge(a.run)
    rows = category_rows("test")[: a.limit or None]
    jobs = ((category_state(r), {"category": category_question()}) for r in rows)
    pred, t0 = [None] * len(rows), time.time()
    for i, ans in judge.batch(jobs):
        probs = ans["category"]
        pred[i] = max(probs, key=probs.get)
    secs = time.time() - t0

    def score(sel):
        if not sel:
            return None
        f1 = []
        for c in CATEGORIES:
            tp = sum(p == c == r["label"] for p, r in sel)
            fp = sum(p == c != r["label"] for p, r in sel)
            fn = sum(r["label"] == c != p for p, r in sel)
            if tp + fp + fn:
                f1.append(2 * tp / (2 * tp + fp + fn))
        return {"n": len(sel), "accuracy": round(sum(p == r["label"] for p, r in sel) / len(sel), 3),
                "macro_f1": round(sum(f1) / len(f1), 3)}

    both = list(zip(pred, rows))
    ood = ("held-repo", "coir-stackoverflow-qa")
    out = {"judge": judge.name, "ms_per_passage": round(secs * 1000 / max(len(rows), 1), 1),
           "all": score(both),
           "in_domain": score([x for x in both if not x[1]["src"].startswith(ood)]),
           "out_of_domain": score([x for x in both if x[1]["src"].startswith(ood)]),
           **{src: score([x for x in both if x[1]["src"].split(":")[0] == src])
              for src in sorted({r["src"].split(":")[0] for r in rows})}}
    record("judge", a.tag, "category", out)
    print(json.dumps(out), flush=True)


def call(cmd, **kw) -> int:
    """subprocess.call with this process's stdout/stderr handed over explicitly. A detached run (log in a file, no
    console) otherwise loses every child's output on Windows: close_fds keeps the file handle from being inherited,
    and a detached process has no console handles to fall back on (measured: an hour of a training run, no line)."""
    import subprocess
    sys.stdout.flush(); sys.stderr.flush()
    return subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr, **kw)


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt",
                   "added_tokens.json", "special_tokens_map.json")
RELEASE_FILES = ("config.json", "head.pt", "recipe.json", "training_metrics.json", "README.md", *TOKENIZER_FILES)


def cmd_export(a):
    """A trained run (LoRA adapter on the base) as a full-weight checkpoint: the adapter merged into the bf16 backbone
    (Kev's merge: W += delta in fp32, one rounding), the head, the temperature and the tokenizer beside it. What a user
    downloads is then one repository that loads with no base fetch. Proof, not assumption: `--check` asks both layouts
    the same states and prints the largest |dp| and the argmax flips."""
    import shutil
    import torch
    from inventio._systemone.checkpoint import Checkpoint, LoadOptions, read_meta, write_meta
    src, out = Path(a.run), Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    ck = Checkpoint(str(src))
    if ck.full:
        raise SystemExit(f"{src} is already full-weight")
    _, m = ck.load("cpu", LoadOptions(dtype=torch.bfloat16, merge=True))
    m.lm.save_pretrained(out, safe_serialization=True)
    meta = read_meta(src)
    meta.weights, meta.weights_dtype = "full", "bf16"
    meta.extra = {**meta.extra, "exported_from": str(src)}
    write_meta(out, meta)
    for f in (*TOKENIZER_FILES, "recipe.json", "training_metrics.json"):
        if (src / f).exists():
            shutil.copy(src / f, out / f)
    print(f"exported {src} -> {out}: " + ", ".join(f"{p.name} {p.stat().st_size / 2**20:.1f} MiB" for p in sorted(out.iterdir())),
          flush=True)


def cmd_check_export(a):
    """The adapter run and its export on the same md2d states, on this machine's accelerator: largest |dp| per kind
    and argmax flips. bf16 reassociation moves the third decimal; a flip on more than a few pools is a broken export."""
    from inventio.systemone import Model, questions, render
    runs = [Model(a.run, a.device), Model(a.export, a.device)]
    worst, flips, n = {"choice": 0.0, "noul": 0.0}, 0, 0
    for qid, q, con, scope, is_gold in SETS["md2d"](a.limit):
        pos, _ = pools(con, q, scope, is_gold)
        if not pos:
            continue
        state, pids, lids, _ = render(pos)
        qs = questions(q, pids, lids)
        x, y = (r.ask(state, qs)["answers"] for r in runs)
        for k in x:
            kind = "choice" if qs[k]["type"] == "choice" else "noul"
            read = (lambda r: r["probabilities"]) if kind == "choice" else (lambda r: {"true": r["noul"]})  # noqa: E731
            px, py = read(x[k]), read(y[k])
            worst[kind] = max(worst[kind], max(abs(px[o] - py[o]) for o in px))
            flips += max(px, key=px.get) != max(py, key=py.get)
        n += 1
    out = {"run": a.run, "export": a.export, "pools": n, "worst_dp": {k: round(v, 4) for k, v in worst.items()},
           "argmax_flips": flips}
    print(json.dumps(out), flush=True)
    if flips > max(1, n // 10):
        raise SystemExit("the export does not answer like the run")


def cmd_publish(a):
    """Upload an exported checkpoint as a tagged release, never onto `main` (it keeps v3, which older inventio loads):
    the files go to branch `<tag>` and the commit is tagged `<tag>`. `--dry-run` lists every file with its size and
    sha256 and stops. After the upload the tag is downloaded fresh into a throwaway cache and loaded, so a release is
    only reported once a stranger's machine could read it."""
    import hashlib
    import tempfile as tf
    from huggingface_hub import HfApi, snapshot_download
    src = Path(a.export)
    files = [src / f for f in RELEASE_FILES if (src / f).exists()] + sorted(src.glob("model*.safetensors"))
    for f in files:
        print(f"  {f.name:28s} {f.stat().st_size:>13,d}  {hashlib.sha256(f.read_bytes()).hexdigest()}", flush=True)
    if a.dry_run:
        print(f"dry run: would upload {len(files)} files to {a.repo} branch {a.tag} and tag {a.tag}", flush=True)
        return
    api = HfApi()
    if a.tag in {t.name for t in api.list_repo_refs(a.repo).tags}:
        raise SystemExit(f"{a.repo} already has tag {a.tag}")
    api.create_branch(a.repo, branch=a.tag, exist_ok=True)
    with tf.TemporaryDirectory() as tmp:
        for f in files:
            (Path(tmp) / f.name).write_bytes(f.read_bytes())
        info = api.upload_folder(folder_path=tmp, repo_id=a.repo, revision=a.tag, commit_message=a.message)
    api.create_tag(a.repo, tag=a.tag, revision=a.tag, tag_message=a.message)
    print(f"uploaded {info.oid if hasattr(info, 'oid') else info} and tagged {a.repo}@{a.tag}", flush=True)
    # local_dir, not a fresh cache_dir: a new HF cache on Windows needs symlink privilege (WinError 1314)
    with tf.TemporaryDirectory(dir=a.scratch or None) as cache:
        path = snapshot_download(a.repo, revision=a.tag, local_dir=cache,
                                 allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"])
        from inventio.systemone import Model, questions, render
        from inventio.search import Hit
        m = Model(path, a.device)
        hit = Hit(id=1, source="docs", public=True, root="/", path="limits.md", start_line=1, end_line=3,
                  heading_path="Rate limits", text="The API allows 60 requests a minute per key.\nRetries back off.")
        state, pids, lids, _ = render([hit])
        res = m.ask(state, questions("how many requests a minute?", pids, lids))
        print(f"fresh download of {a.repo}@{a.tag} loads: run {m.info()}, line "
              f"{res['answers']['where_line']['choice']}, exists {res['answers']['exists']['noul']:.3f}", flush=True)


# --- commands ------------------------------------------------------------------------------------

def cmd_spike(a):
    for name in a.sets.split(","):
        suffix = ("" if a.order == "bm25" else f"-{a.order}") + ("" if a.pool == K else f"-pool{a.pool}")
        rows, path = [], ROWS / f"{a.tag}-{name}{suffix}.jsonl"
        ROWS.mkdir(parents=True, exist_ok=True)

        def arrange(hits):
            """Same passages, same questions, different order in the state."""
            h = list(hits)
            if a.order == "reverse":
                return h[::-1]
            if a.order == "shuffle":
                random.Random(a.seed).shuffle(h)
            return h

        with path.open("w", encoding="utf-8") as f:
            for qid, q, con, scope, is_gold in SETS[name](a.limit):
                pos, neg = pools(con, q, scope, is_gold, k=a.pool)
                if not pos:
                    continue
                try:
                    r = {"qid": qid, "query": q,
                         **run_query(asker(a.url, model=a.model), q, arrange(pos), arrange(neg), is_gold,
                                     passage_asks=a.with_passage_asks)}
                except urllib.error.HTTPError as e:
                    print(f"skip {qid}: {e.code} {e.read()[:200]!r}", flush=True)
                    continue
                rows.append(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
        record("spike", a.tag, f"{name}{suffix}", spike_summary(rows, a.tag, name))


def cmd_parity(a):
    """The vendored reader against the served server, on the same states and the same questions.

    `--ranker systemone` reads the checkpoint in the query's own process; every number this repository
    reports came from the served one. This measures the distance: the largest |Δp| for each question
    kind, and whether any argmax moved. bf16 kernels differ by reassociation, so a small Δp is
    expected; a moved argmax, or a Δp past a few percent, is a reader that is not the same reader.
    """
    from inventio.systemone import Model, questions, render, served
    local = Model(a.run, a.device)
    print(f"run {local.run}  device {local.info()['device']}  dtype {local.info()['dtype']}  "
          f"temperature {local.info()['temperature']}", flush=True)
    if a.url:
        info = served(a.url)
        there = str(info.get("run") or "").replace("\\", "/").rstrip("/")
        here = str(local.run).replace("\\", "/").rstrip("/")
        if not (there.endswith(here) or here.endswith(there)):   # same weights, or the comparison is a lie
            raise SystemExit(f"the server serves {there!r} and --run is {here!r}: load the same checkpoint")
        print(f"served {info.get('run')}  temperature {info.get('temperature')}", flush=True)
    fixed = asker(a.url, model=a.model) if a.url else None
    worst, flips, rows = {}, 0, 0
    for qid, q, con, scope, is_gold in SETS[a.set](a.limit):
        pos, neg = pools(con, q, scope, is_gold, k=a.pool)
        if not pos:
            continue
        for arm, hits in (("pos", pos), ("neg", neg)):
            state, pids, lids, owner = render(hits)
            qs = questions(q, pids, lids)
            mine = local.ask(state, qs)["answers"]
            if not a.url:
                continue
            theirs = fixed(state, qs)["answers"]
            for key in qs:
                kind = qs[key]["type"]
                if kind == "noul":
                    d = abs(mine[key]["noul"] - theirs[key]["noul"])
                    worst[kind] = max(worst.get(kind, 0.0), d)
                    flips += (mine[key]["noul"] > 0.5) != (theirs[key]["noul"] > 0.5)
                elif kind == "choice":
                    a1, a2 = mine[key]["probabilities"], theirs[key]["probabilities"]
                    d = max(abs(a1[k] - a2[k]) for k in a1)
                    worst[kind] = max(worst.get(kind, 0.0), d)
                    flips += max(a1, key=a1.get) != max(a2, key=a2.get)
        rows += 1
        print(f"{qid} {arm}: worst {', '.join(f'{k} {v:.4f}' for k, v in sorted(worst.items()))}"
              f"{'' if a.url else '  (only the local reader asked)'}", flush=True)
    if a.url:
        print(f"{rows} pools: worst |Δp| " + ", ".join(f"{k} {v:.4f}" for k, v in sorted(worst.items()))
              + f"; argmax flips {flips}")
    return 0


def load_run(tag, set):
    """Per-query top-1 correctness and `exists` scores, from a spike run or a dispositio run.
    `top1` is the passage holding the top-ranked line when the run asked the line question, else the
    top-ranked passage, and None when the query's answer was not in the pool."""
    disp = ROWS / f"{tag}-{set}-disp.jsonl"
    if disp.exists():
        return {r["qid"]: {"top1": r["top1"], "pos": r["exists_pos"], "neg": r["exists_neg"], "line": False}
                for r in map(json.loads, disp.open(encoding="utf-8"))}
    out = {}
    for r in map(json.loads, (ROWS / f"{tag}-{set}.jsonl").open(encoding="utf-8")):
        p = r["pos"]
        line = "top_line_owner" in p
        top1 = None if not p["gold"] else ((p["top_line_owner"] if line else p["top_passage"]) in p["gold"])
        out[r["qid"]] = {"top1": top1, "pos": p["exists"], "neg": r["neg"]["exists"], "line": line}
    return out


def cmd_gate(a):
    A, B = load_run(a.kev, a.set), load_run(a.v3, a.set)
    has_line = any(v["line"] for v in A.values())
    # one metric per comparison: `top1` from both runs, on the queries that answered the line question
    both = [q for q in A if q in B and A[q]["top1"] is not None and B[q]["top1"] is not None
            and (not has_line or A[q]["line"])]
    dropped = [q for q in A if q in B and A[q]["top1"] is not None and has_line and not A[q]["line"]]
    x = [A[q]["top1"] for q in both]
    y = [B[q]["top1"] for q in both]
    d = [i - j for i, j in zip(x, y)]
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000)) if d else [0.0]
    s = {"set": a.set, "n": len(both), "dropped_no_line_question": len(dropped),
         f"top1_{a.kev}": round(sum(x) / len(x), 3) if x else None,
         f"top1_{a.v3}": round(sum(y) / len(y), 3) if y else None,
         "diff": round(sum(d) / len(d), 3) if d else None,
         "diff_lo": round(boots[50], 3), "diff_hi": round(boots[1949], 3),
         "wins": sum(i > 0 for i in d), "losses": sum(i < 0 for i in d),
         f"exists_auc_{a.kev}": round(auc([v["pos"] for v in A.values() if v["top1"] is not None],
                                          [v["neg"] for v in A.values()]), 3),
         f"exists_auc_{a.v3}": round(auc([v["pos"] for v in B.values() if v["top1"] is not None],
                                         [v["neg"] for v in B.values()]), 3)}
    record("gate", f"{a.kev}-vs-{a.v3}", a.set, s)


def cmd_leak(a):
    """Can a bag of words tell a record's positive arm from its negative arm? The acceptance test for any
    change to how negatives are built: in-domain (grouped by query) it should sit near the cross-domain
    level, because a bag sees no query. What it can see is corpus composition — which passages the two
    arms hold, how long the state is — and a model trained on that learns the composition, not
    answerability (measured 2026-09-26: the compacted negative arm scored 0.732 in-domain while the
    one-swap arm scores 0.623; per source: md2d 0.634 -> 0.598, code 0.927 -> 0.811, where for code the
    state's *length alone* separates the arms at 0.77 with the filler chunks systematically longer)."""
    import numpy as np
    from scipy.sparse import csr_matrix
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    def featurize(texts, dim=1 << 18):
        rows, cols, vals = [], [], []
        for i, t in enumerate(texts):
            ws = WORD.findall(t.casefold())
            c = Counter(ws + [f"{x}_{y}" for x, y in zip(ws, ws[1:])])
            for g, n in c.items():
                rows.append(i); cols.append(zlib.crc32(g.encode()) % dim); vals.append(1 + math.log(n))
        X = csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(len(texts), dim))
        return X

    def fit(X, y):
        return LogisticRegression(max_iter=200, C=1.0, solver="liblinear").fit(X, y)

    def grouped(X, y, groups, folds=5):
        key = sorted(set(groups))
        np.random.RandomState(0).shuffle(key)
        fold = {q: i % folds for i, q in enumerate(key)}
        out = []
        for f in range(folds):
            te = np.array([i for i, q in enumerate(groups) if fold[q] == f])
            tr = np.array([i for i, q in enumerate(groups) if fold[q] != f])
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                continue
            out.append(roc_auc_score(y[te], fit(X[tr], y[tr]).decision_function(X[te])))
        return out

    rows = [json.loads(l) for l in Path(a.records).open(encoding="utf-8")]
    rows = [r for r in rows if r["state"].strip()]
    X = featurize([r["state"] for r in rows])
    y = np.array([1 if r["_meta"]["arm"] == "pos" else 0 for r in rows])
    g = [r["_meta"]["group_id"] for r in rows]
    dom = [r["_meta"].get("domain") or r["_meta"]["source"] for r in rows]
    src = [r["_meta"]["source"] for r in rows]
    print(f"{Path(a.records).name}: {len(rows)} records, arms {dict(Counter(y))}, {X.nnz} nonzero features")
    all_ = grouped(X, y, g)
    print(f"  in-domain grouped 5-fold AUC: mean {statistics.mean(all_):.3f} {[round(x, 3) for x in all_]}")
    for s in sorted(set(src)):
        i = [k for k, v in enumerate(src) if v == s]
        if len(i) < 50 or len({y[k] for k in i}) < 2:
            continue
        a_ = grouped(X[i], y[i], [g[k] for k in i])
        print(f"  {s:12s} n={len(i):5d} in-domain mean {statistics.mean(a_):.3f} {[round(x, 3) for x in a_]}")
        pos = [len(rows[k]["state"]) for k in i if y[k] == 1]
        neg = [len(rows[k]["state"]) for k in i if y[k] == 0]
        print(f"  {s:12s} state length alone: AUC {roc_auc_score([y[k] for k in i], [len(rows[k]['state']) for k in i]):.3f} "
              f"(pos median {statistics.median(pos):.0f} vs neg {statistics.median(neg):.0f} chars)")
    doms = [d for d in sorted(set(dom)) if dom.count(d) >= 150]
    if len(doms) > 1:
        out = {}
        for d in doms:
            te = [k for k in range(len(rows)) if dom[k] == d]
            tr = [k for k in range(len(rows)) if dom[k] != d]
            if len({y[k] for k in tr}) < 2 or len({y[k] for k in te}) < 2:
                continue
            out[d] = round(roc_auc_score([y[k] for k in te],
                                         fit(X[tr], [y[k] for k in tr]).decision_function(X[te])), 3)
        print(f"  leave-one-domain-out: {out}")
    record("leak", Path(a.records).stem, "records",
           {"file": Path(a.records).name, "records": len(rows), "auc_in_domain": round(statistics.mean(all_), 3)})


def mix_questions(rows, passage_asks, exists_asks):
    """Rewrite the question list of every record: at most `passage_asks` per-passage questions (every
    gold passage kept, the rest drawn by the query's own key so a rebuild asks the same ones) and
    `exists_asks` wordings of `exists`.

    The per-record loss is a **sum over questions** (kev/train.py batch_loss), so the question list *is*
    the weight each head gets: a record with 15 passage questions and one `exists` hands the honesty
    head 6% of the gradient. This is the knob for that, and it changes questions only — the state, the
    passages and the labels stay the ones the record was built with.
    """
    out = []
    for r in rows:
        qs = r["questions"]
        asks = {k: q for k, q in qs.items() if k.startswith("P")}
        keep = set()
        if asks:
            gold = [k for k, q in asks.items() if q.get("label")]
            rest = sorted(k for k in asks if k not in gold)
            rng = random.Random(hashlib.sha1(f"{r['_meta']['group_id']}\tpassages".encode()).hexdigest()[:8])
            rng.shuffle(rest)
            keep = set(gold) | set(rest[: max(0, passage_asks - len(gold))])
        new_qs = {k: v for k, v in qs.items() if not k.startswith("P") or k in keep}
        if "exists" in qs:
            q = qs["exists"]
            asked = q["instructions"].split('"')
            for i in range(1, exists_asks):
                new_qs[f"exists_{i}"] = {**q, "instructions": EXISTS_ASKS[i % len(EXISTS_ASKS)].format(
                    q=asked[1] if len(asked) > 1 else "")}
        out.append({**r, "questions": new_qs})
    return out


def wait_ready(port, tries=120):
    """Block until a served model answers, so a chain never scores a server that is not up."""
    import urllib.request
    for _ in range(tries):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3).read()
            return True
        except Exception:
            time.sleep(2)
    return False


def cmd_score(a):
    """Serve a run and measure it: spike the sets, pair it against each baseline, exact line, and the
    acceptance test on the records it was trained on. One command, so the chain is not typed by hand and
    a lost session does not lose the numbers."""
    import os
    import subprocess
    kev = Path(os.environ.get("KEV_DIR", "C:/Users/LEGION/kev")).resolve()
    python = kev / ".venv" / "Scripts" / "python.exe"
    run = Path(a.run) if a.run else kev / "runs" / a.tag
    log = Path(tempfile.gettempdir()) / f"{a.tag}-serve.log"
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen([str(python), str(Path(__file__).resolve().parent / "kev_win.py"), "serve",
                                 "--run", str(run), "--port", str(a.port)], cwd=REPO, stdout=f,
                                stderr=subprocess.STDOUT, creationflags=0x00000008 | 0x00000200)
    print(f"serve pid={proc.pid} -> {log}", flush=True)
    if not wait_ready(a.port):
        raise SystemExit(f"no server on :{a.port}; see {log}")
    url = f"http://127.0.0.1:{a.port}"
    call([sys.executable, str(Path(__file__)), "spike", url, a.tag, "--sets", a.sets])
    for base in [b for b in a.against.split(",") if b]:
        for s in a.sets.split(","):
            call([sys.executable, str(Path(__file__)), "gate", a.tag, base, s])
    call([sys.executable, str(Path(__file__)), "lines", a.tag])
    if a.leak:
        call([sys.executable, str(Path(__file__)), "leak", a.leak])
    summary = json.loads((ROWS / "summary.json").read_text(encoding="utf-8"))
    keys = ("passage@1", "top_line_owner_passage@1", "exists_auc", "exists_auc_answer_absent",
            "latency_ms_median", "tokens_median")
    print("\n=== readings", flush=True)
    for s in a.sets.split(","):
        tags = [a.tag, *[b for b in a.against.split(",") if b]]
        print(f"-- {s}", flush=True)
        for k in keys:
            print(f"   {k:32s} " + "  ".join(f"{t} {summary['spike'].get(f'{t}/{s}', {}).get(k)}" for t in tags),
                  flush=True)
    proc.kill()   # the numbers are read: a served checkpoint left on the card only takes memory from the next step
    print(f"serve pid={proc.pid} stopped", flush=True)


def cmd_run(a):
    """`train` then `score`, for one tag: the whole stretch in one command (each half is defined above
    and both are recorded — the recipe by `train`, the numbers by `score`)."""
    import os
    import subprocess
    cmd = [sys.executable, str(Path(__file__)), "train", "--records", a.records, "--bal", str(a.bal),
           "--bal-swe", str(a.bal_swe), "--max-state", str(a.max_state), "--epochs", str(a.epochs),
           f"--bal-tag={a.bal_tag}", "--out", f"runs/{a.tag}"]
    if a.no_code:
        cmd.append("--no-code")
    if a.judge_records:
        cmd += ["--judge-records", a.judge_records]
    print("$ " + " ".join(cmd), flush=True)
    if call(cmd, cwd=REPO):
        raise SystemExit("training failed")
    metrics = json.loads(Path(a.kev_dir if hasattr(a, "kev_dir") else
                              os.environ.get("KEV_DIR", "C:/Users/LEGION/kev")).joinpath("runs", a.tag,
                                                                                          "training_metrics.json").read_text())
    print(f"trained: {metrics['records_seen']} records, {metrics['optimizer_steps']} steps, "
          f"{metrics['wall_seconds'] / 3600:.2f} h, peak {metrics['peak_device_bytes'] / 2**30:.2f} GiB",
          flush=True)
    call([sys.executable, str(Path(__file__)), "score", a.tag, "--sets", a.sets,
                     "--against", a.against, "--port", str(a.port), "--leak", a.leak], cwd=REPO)
    if a.judge_records:   # the index-time question it was also trained on, read in this process
        run = Path(os.environ.get("KEV_DIR", "C:/Users/LEGION/kev")) / "runs" / a.tag
        call([str(Path(os.environ.get("KEV_DIR", "C:/Users/LEGION/kev")) / ".venv" / "Scripts" / "python.exe"),
                         str(Path(__file__)), "judge", a.tag, "--run", str(run)], cwd=REPO)


def cmd_domains(a):
    """The answer-absent reading split by the domain a query's gold comes from.

    md2d's aggregate hides four stories, and one of them is a control: `studentaid` was held out of the
    recipe whole, so a lead that holds there is not a corpus cue (the bag-of-words probe that separated
    v1's two arms in-domain sits at chance across domains). Read with `threshold`, which cuts the same
    reading for a caveat line.
    """
    from beir_bench import load_beir
    d = data_dir(None) / "beir" / "multidoc2dial"
    _, _, qrels = load_beir(d)
    dom = {}
    for qid, gold in qrels.items():
        pref = {str(c).split("_")[0].split("-")[0] for c in gold}
        dom[qid] = pref.pop() if len(pref) == 1 else "mixed"
    rows = [json.loads(l) for l in (ROWS / f"{a.tag}-md2d.jsonl").open(encoding="utf-8")]
    has = [r for r in rows if r["pos"]["gold"]]
    absent = [r for r in rows if not r["pos"]["gold"]]
    auc = lambda p, n: (round(sum((x > y) + 0.5 * (x == y) for x in p for y in n) / (len(p) * len(n)), 3)
                        if p and n else None)  # noqa: E731
    out = {"tag": a.tag, "answer_absent": len(absent),
           "all": auc([r["pos"]["exists"] for r in has], [r["pos"]["exists"] for r in absent])}
    for dm in sorted({dom.get(r["qid"], "?") for r in absent}):
        n = [r["pos"]["exists"] for r in absent if dom.get(r["qid"]) == dm]
        p = [r["pos"]["exists"] for r in has if dom.get(r["qid"]) == dm]
        out[dm] = {"absent": len(n), "answerable": len(p), "auc": auc(p, n)}
    record("domains", a.tag, "md2d", out)


def cmd_fitcheck(a):
    """Does the `exists` head fit the data it was trained on? In-sample AUC against the held-out one.

    Equal numbers mean the head is underfit and more weight on it can move it; an in-sample number far
    above the test number means it fits and the lever is elsewhere (the negatives, the labels, the
    state). Asks each record in its own wording and in the wording the harness serves with, because the
    wordings are picked per record (`pick`) and a head can fit one and not the other.
    """
    import urllib.request
    rows = [json.loads(l) for l in open(a.records, encoding="utf-8")]
    rng = random.Random(a.seed)
    pos = [r for r in rows if r["_meta"]["arm"] == "pos"]
    neg = [r for r in rows if r["_meta"]["arm"] == "neg"]
    rng.shuffle(pos); rng.shuffle(neg)
    sample = pos[:a.n] + neg[:a.n]
    rng.shuffle(sample)
    out = {"own": ([], []), "served": ([], [])}
    for r in sample:
        q = r["questions"]["exists"]
        served_q = {"type": "noul", "instructions": EXISTS_ASKS[0].format(q=q["instructions"].split('"')[1]),
                    "criteria": NOUL}
        res = ask(a.url, r["state"], {"own": q, "served": served_q}, a.model)
        y = 1 if q["label"] else 0
        out["own"][y].append(res["answers"]["own"]["noul"])
        out["served"][y].append(res["answers"]["served"]["noul"])
    s = {"tag": Path(a.records).stem, "n_pos": len(out["own"][1]), "n_neg": len(out["own"][0])}
    for w in ("own", "served"):
        p_, n_ = out[w][1], out[w][0]
        s[f"auc_{w}"] = round(sum((x > y) + 0.5 * (x == y) for x in p_ for y in n_) / (len(p_) * len(n_)), 3)
        s[f"median_pos_{w}"] = round(statistics.median(p_), 3)
        s[f"median_neg_{w}"] = round(statistics.median(n_), 3)
    record("fitcheck", Path(a.records).stem, "records", s)


def cmd_threshold(a):
    """Where the tool says "this map may not answer": the whole curve, and the two defensible ways to cut
    it, so the number in `inventio/systemone.py` is read off a run instead of chosen by hand.

    The two classes come from the spike: the pools whose answer BM25 put in (answerable) and the pools
    whose answer is not in the map at all (absent). A caveat costs a reader a glance; a missing caveat
    is the failure the reading exists for — so the balanced point is the default here and the
    90%-keep point is printed beside it for a quieter tool.
    """
    rows = [json.loads(l) for l in (ROWS / f"{a.tag}-{a.set}.jsonl").open(encoding="utf-8")]
    has = [r["pos"]["exists"] for r in rows if r["pos"]["gold"]]
    absent = [r["pos"]["exists"] for r in rows if not r["pos"]["gold"]]
    if not has or not absent:
        raise SystemExit(f"{a.set} needs both classes (answerable {len(has)}, absent {len(absent)})")
    curve = []
    for thr in [round(x / 100, 2) for x in range(5, 96, 5)]:
        kept = sum(p >= thr for p in has) / len(has)
        flag = sum(p < thr for p in absent) / len(absent)
        curve.append({"thr": thr, "keeps_answerable": round(kept, 3), "flags_absent": round(flag, 3),
                      "balanced": round((kept + flag) / 2, 3)})
    bal = max(curve, key=lambda c: c["balanced"])
    quiet = max((c for c in curve if c["keeps_answerable"] >= 0.9), key=lambda c: c["thr"], default=curve[0])
    record("threshold", a.tag, a.set, {"tag": a.tag, "set": a.set, "answerable": len(has), "absent": len(absent),
                                       "balanced_point": bal, "quiet_point": quiet, "curve": curve})


def cmd_mix(a):
    """Change the question mix of a record file without rebuilding it: the state, the pools and the
    labels stay byte-identical, so two runs differ in one thing. Writes `<out>.mix.json` beside the
    result, which `train` copies into the run's recipe."""
    import hashlib
    src = Path(a.records)
    rows = [json.loads(l) for l in src.open(encoding="utf-8")]
    mixed = mix_questions(rows, a.passage_asks, a.exists_asks)
    out = Path(a.out)
    with out.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in mixed)
    side = out.with_suffix(".mix.json")
    side.write_text(json.dumps(
        {"built_from": str(src), "built_from_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
         "out_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
         "passage_asks": a.passage_asks, "exists_asks": a.exists_asks,
         "rule": "every gold passage kept, the rest drawn by the query key; `exists` asked in this many wordings"},
        indent=1) + "\n", encoding="utf-8")
    kind = Counter(("exists" if k.startswith("exists") else "P" if k.startswith("P") else k)
                   for r in mixed for k in r["questions"])
    total = sum(kind.values())
    print(json.dumps({"out": str(out), "records": len(mixed), "kinds": dict(kind),
                      "questions_per_record": round(total / len(mixed), 2),
                      "exists_share": round(kind["exists"] / total, 3), "mix": str(side)}), flush=True)


def cmd_lines(a):
    spans = md2d_test_spans()
    rows = [json.loads(l) for l in open(ROWS / f"{a.tag}-md2d.jsonl", encoding="utf-8")]
    has = [r for r in rows if r["pos"]["gold"] and "top_line_text" in r["pos"]]
    hit = sum(any(s and s in norm(r["pos"]["top_line_text"]) for s in spans.get(r["qid"], [])) for r in has)
    record("lines", a.tag, "md2d", {"tag": a.tag, "set": "md2d", "n": len(has),
                                    "exact_line@1": round(hit / max(1, len(has)), 3)})


def cmd_data(a):
    from collections import Counter
    DATA.mkdir(parents=True, exist_ok=True)
    if a.judge:
        rows = list(judge_records())
        with (DATA / "judge.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print("judge", len(rows), Counter(r["_meta"]["domain"] for r in rows), "->", DATA / "judge.jsonl", flush=True)
        return
    for name, gens in (("train", [md2d_records("train"), swe_records()]), ("dev", [md2d_records("validation")])):
        rows = [r for g in gens for r in g]
        random.Random(0).shuffle(rows)
        with (DATA / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print(name, len(rows), Counter((r["_meta"]["source"], r["_meta"]["arm"]) for r in rows), flush=True)


def cmd_scale(a):
    """How request time grows with the number of questions and with the state's length. A fresh nonce
    line per request, so every number is a new state (no cache hit), and one warm-up dropped per cell."""
    import uuid
    states, qs_text = [], []
    for qid, q, con, scope, g in set_md2d(a.pools):
        pos, _ = pools(con, q, scope, g)
        if pos:
            states.append(render(pos)[0])
            qs_text.append(q)
    print(f"{len(states)} states of ~{len(states[0].split())} words", flush=True)

    def noul(q, i):
        return {"type": "noul", "instructions": f'Does passage P{i % 15 + 1:02d} answer: "{q}"?'}

    def run(state, qs, n=5):
        lat, tok = [], None
        for _ in range(n + 1):
            res = ask(a.url, f"request {uuid.uuid4().hex[:8]}\n" + state, qs)
            lat.append(res["latency_ms"])
            tok = res["usage"]["input_tokens"]
        return statistics.median(lat[1:]), tok

    rows = []
    if "q" in a.axis:
        for nq in (1, 2, 4, 16, 64, 128, 255):
            ms, tok = run(states[0], {f"q{i}": noul(qs_text[i % len(qs_text)], i) for i in range(nq)})
            rows.append({"axis": "questions", "n": nq, "tokens": tok, "ms": ms})
            print(f"  {nq:4d} questions {tok:6d} tokens {ms:7.1f} ms", flush=True)
    if "len" in a.axis:
        for n_pools in (1, 2, 3, 4, 6, 8, 12, 16):
            qs = {"where": {"type": "choice",
                            "instructions": f'Which passage contains the answer to: "{qs_text[0]}"?',
                            "criteria": {f"P{i + 1:02d}": None for i in range(15)}},
                  "exists": {"type": "noul", "instructions": f'Does any passage answer: "{qs_text[0]}"?'},
                  "n": noul(qs_text[0], 0)}
            try:
                ms, tok = run("\n\n".join(states[:n_pools]), qs, n=3)
            except urllib.error.HTTPError as e:
                print(f"  {n_pools:2d} pools failed: {e.code}", flush=True)
                break
            rows.append({"axis": "state", "n": n_pools, "tokens": tok, "ms": ms})
            print(f"  {n_pools:2d} pools {tok:6d} tokens {ms:7.1f} ms", flush=True)
    s = {"tag": a.tag, "rows": rows}
    record("scale", a.tag, a.axis, s)


def cmd_train(a):
    """The run's recipe, in the repo: the balanced record file, then Kev's trainer through
    `kev_win.py` with every knob pinned, and a `recipe.json` recording the Kev commit, the base
    revision and the data's hash so the run is reproducible from this repo alone."""
    import hashlib
    import os
    import subprocess
    kev = Path(os.environ.get("KEV_DIR", "C:/Users/LEGION/kev")).resolve()
    python = kev / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        raise SystemExit(f"no Kev environment at {kev}; set KEV_DIR")
    records = Path(a.records) if a.records else DATA / "train.jsonl"
    mix_side = records.with_suffix(".mix.json")  # written by `mix`: which question mix these records carry
    fit = None
    if not records.exists():
        raise SystemExit(f"{records} not built; run `systemone.py data` first")
    if a.bal or a.bal_swe:
        rows = [json.loads(l) for l in records.open(encoding="utf-8")]
        if a.fit_max_state:
            # The draw has to see only what the trainer admits. At `--max_state 6656` the trainer drops 4,879
            # of 14,038 records (every long code pool: 3,550 negatives and 1,329 positives), so drawing from
            # the unfiltered file would print a composition the run never sees and count code positives that
            # do not exist. `kev_win.py fits` runs the trainer's own predicate.
            unfit = Path(a.records_fit) if a.records_fit else records.parent / f"{records.stem}.fit{a.fit_max_state}.jsonl"
            if not unfit.exists() or a.refit:
                cmd = [str(python), str(Path(__file__).resolve().parent / "kev_win.py"),
                       "fits", "--data", str(records), "--base", BASE, "--base_revision", BASE_REVISION,
                       "--max_state", str(a.fit_max_state), "--out", str(unfit)]
                print(" ".join(cmd), flush=True)
                if call(cmd):
                    raise SystemExit("kev_win.py fits failed")
            else:
                print(f"reusing the filtered file {unfit}", flush=True)
            rows = [json.loads(l) for l in unfit.open(encoding="utf-8")]
            fit = unfit
        rng = random.Random(a.seed)

        def draw(subset, target):
            """Whole queries, not rows. Sampling the two arms independently left 1,101 + 1,101 of
            3,101 md2d queries with only one of them, so `exists` was never trained on the contrast
            between a pool that answers and the same pool without the answer (data review, 2026-09-26).
            A target of 0 keeps the source as it is."""
            by_query = {}
            for r in subset:
                by_query.setdefault(r["_meta"]["group_id"], []).append(r)
            if not target:
                return list(subset)
            queries = list(by_query.values())
            rng.shuffle(queries)
            half, kept = target // 2, []
            npos = nneg = 0
            for g in queries:
                p = sum(r["_meta"]["arm"] == "pos" for r in g)
                if npos + p > half or nneg + (len(g) - p) > half:
                    continue
                kept += g
                npos, nneg = npos + p, nneg + len(g) - p
                if npos >= half and nneg >= half:
                    break
            return kept

        md2d = [r for r in rows if r["_meta"]["source"] == "md2d_train"]
        code = [] if a.no_code else [r for r in rows if r["_meta"]["source"] != "md2d_train"]
        kept = draw(md2d, a.bal) + draw(code, a.bal_swe)
        rng.shuffle(kept)
        mix = Counter((r["_meta"]["source"], r["_meta"]["arm"]) for r in kept)
        records = DATA / f"train_bal{a.bal}-{a.bal_swe}{a.bal_tag}.jsonl"
        records.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
        print(f"balanced to {len(kept)} records from {records if fit is None else fit.name}: "
              + " ".join(f"{s}/{arm} {n}" for (s, arm), n in sorted(mix.items()))
              + f" -> {records}", flush=True)
    if a.judge_records:
        judge = [json.loads(l) for l in open(a.judge_records, encoding="utf-8")]
        rows = [json.loads(l) for l in records.open(encoding="utf-8")] + judge
        random.Random(a.seed).shuffle(rows)
        records = DATA / f"{records.stem}+judge.jsonl"
        records.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"+{len(judge)} judge records -> {records}", flush=True)
    if fit is not None:
        recipe_fit = {"fit_max_state": a.fit_max_state, "fit_records": str(fit),
                      "fit_records_sha256": hashlib.sha256(Path(fit).read_bytes()).hexdigest()}
    else:
        recipe_fit = {}
    # 8 GB on this card, and the new question shape fattens the pass: 18 branches, one of them a choice
    # over up to 512 line ids. Measured 2026-09-26 at --max_state 6656: the worst records by *packed*
    # length (state + branches, up to 10.4k tokens by tokenizer proxy) OOM the card, while the worst by
    # *state* length (a proxy that picks the wrong records) peak at 7,763/7,932 MiB. `--row_budget` splits
    # a record whose row does not fit one pass into consecutive question groups, each part carrying its
    # share of the record's questions, so the loss is a sum over variants and stays exact
    # (kev.train.batch_loss). `expandable_segments` is not supported on Windows (torch warns), so the
    # budget is the lever that keeps this run on the card, not the allocator flag.
    recipe = {**recipe_fit, "kev_dir": str(kev), "kev_commit": subprocess.run(
        ["git", "-C", str(kev), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None,
        "init_from": INIT_FROM, "base": BASE, "base_revision": BASE_REVISION,
        "records": str(records), "records_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
        "judge_records": a.judge_records or None,
        "mix": json.loads(mix_side.read_text(encoding="utf-8")) if mix_side.exists() else "as built by `data`",
        "max_state": a.max_state, "epochs": a.epochs, "lr": a.lr, "seed": a.seed,
        "augmentations": "none (no none-option, no distractor: the options are exhaustive line ids)",
        "memory": {"row_budget": a.row_budget,
                   "why": "the pass is bounded because the card is 8 GB and the worst packed record "
                          "OOMs without it; the split is loss-exact (share-weighted) as long as "
                          "--perm_kl and --anchor_w stay 0"}}
    # The trainer refuses to run into an existing --out, so a non-dry run must not create it: the receipt
    # is written next to the trainer's own files once it has made the directory. A dry run is the one that
    # creates it (that is what a dry run is for: the recipe before the run).
    out = kev / a.out     # the trainer runs with cwd=<kev checkout>, so --out is relative to that
    if a.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        (out / "recipe.json").write_text(json.dumps(recipe, indent=1) + "\n", encoding="utf-8")
    elif out.exists():
        raise SystemExit(f"{out} exists and Kev refuses to overwrite a run; move it aside or pick --out")
    cmd = [str(python), str(Path(__file__).resolve().parent / "kev_win.py"), "train",
           "--data", str(records), "--init_from", INIT_FROM, "--base", BASE, "--base_revision", BASE_REVISION,
           "--max_state", str(a.max_state), "--device", "cuda", "--dtype", "bf16", "--weights_dtype", "bf16",
           "--checkpointing", "1", "--shared_prefix", "1", "--batch", "1", "--accum", "8",
           "--lr", str(a.lr), "--epochs", str(a.epochs), "--seed", str(a.seed),
           "--p_none", "0", "--p_none_distract", "0", "--p_distract", "0",
           "--row_budget", str(a.row_budget), "--out", a.out]
    print(" ".join(cmd), flush=True)
    if a.dry_run:
        return 0
    rc = call(cmd, cwd=kev)
    if rc == 0:
        (out / "recipe.json").write_text(json.dumps(recipe, indent=1) + "\n", encoding="utf-8")
        print(f"recipe -> {out / 'recipe.json'}", flush=True)
    raise SystemExit(rc)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("data", help="build the Kev training records"); s.set_defaults(fn=cmd_data)
    s.add_argument("--judge", action="store_true",
                   help="build only the index-time judge's records (<data>/s1/data/judge.jsonl)")
    s = sub.add_parser("export", help="merge a run's adapter into a full-weight checkpoint")
    s.add_argument("run"); s.add_argument("out"); s.set_defaults(fn=cmd_export)
    s = sub.add_parser("check-export", help="the run and its export on the same states: |dp| and argmax flips")
    s.add_argument("run"); s.add_argument("export"); s.add_argument("--limit", type=int, default=30)
    s.add_argument("--device", default=None); s.set_defaults(fn=cmd_check_export)
    s = sub.add_parser("publish", help="upload an export as a tagged release (branch and tag, never main)")
    s.add_argument("export"); s.add_argument("--repo", default="minhquan2310/dispositio")
    s.add_argument("--tag", default="v4"); s.add_argument("--message", default="dispositio v4: the System One model")
    s.add_argument("--device", default=None); s.add_argument("--dry-run", action="store_true")
    s.add_argument("--scratch", default="", help="where the fresh-download check writes (default: the temp dir)")
    s.set_defaults(fn=cmd_publish)
    s = sub.add_parser("judge", help="the category question on its held-out passages")
    s.add_argument("tag"); s.add_argument("--run", default=None, help="System One run (default: the recorded one)")
    s.add_argument("--limit", type=int, default=0)
    s.set_defaults(fn=cmd_judge)
    s = sub.add_parser("train", help="fine-tune Kev on the records (recipe pinned here)")
    s.add_argument("--records", default="", help="record file (default <data>/s1/data/train.jsonl)")
    s.add_argument("--out", default="runs/s1", help="run directory, inside the Kev checkout")
    s.add_argument("--bal", type=int, default=0,
                   help="balance the MultiDoc2Dial part to about this many records; 0 keeps all. Queries "
                        "are drawn whole, so both arms of a query stay together")
    s.add_argument("--bal-swe", type=int, default=0,
                   help="balance the code part to about this many records; 0 keeps all. The code arm is "
                        "lopsided on its own (107 positives against 3,922 negatives), so a quota here is "
                        "about the positive/negative prior the exists head sees on code-like states")
    s.add_argument("--max-state", type=int, default=6656)
    s.add_argument("--fit-max-state", type=int, default=0,
                   help="drop the records the trainer's admission rule drops at this state cap before the "
                        "balance draw, so the printed composition is the run's (0: draw from everything)")
    s.add_argument("--records-fit", default="", help="the filtered file to read/write with --fit-max-state")
    s.add_argument("--refit", action="store_true", help="recompute the filtered file even if it exists")
    s.add_argument("--no-code", action="store_true",
                   help="train on prose only. The code arm's records do not fit the card: measured with the "
                        "encoder, packed (state + branches) is p50 7,074 and max 8,310 for code against p50 "
                        "4,063 and max 7,571 for md2d, and a ~8.2k packed row OOMs 8 GB. The code arm needs "
                        "a smaller state budget (fewer passages, or one passage per state), not a bigger cap")
    s.add_argument("--row-budget", type=int, default=0,
                   help="padded row tokens per forward/backward pass (0: the whole micro-batch at once, "
                        "which is what v1 used and what OOMs an 8 GB card on this shape's long records)")
    s.add_argument("--bal-tag", default="",
                   help="a suffix on the balanced file's name, so a new record shape never overwrites the "
                        "previous shape's draw (the file a run's recipe names is its data)")
    s.add_argument("--epochs", type=int, default=1); s.add_argument("--lr", type=float, default=2e-5)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--judge-records", default="",
                   help="append these records (systemone.py data --judge) after the balance draw")
    s.add_argument("--dry-run", action="store_true", help="write the recipe and print the command, then stop")
    s.set_defaults(fn=cmd_train)
    s = sub.add_parser("scale", help="request time against questions and state length")
    s.add_argument("url"); s.add_argument("tag")
    s.add_argument("--axis", default="q,len"); s.add_argument("--pools", type=int, default=60)
    s.set_defaults(fn=cmd_scale)
    s = sub.add_parser("spike", help="ask a served System One model over the pools")
    s.add_argument("url"); s.add_argument("tag"); s.add_argument("--model", default="jev-latest")
    s.add_argument("--sets", default="md2d"); s.add_argument("--limit", type=int, default=0)
    s.add_argument("--with-passage-asks", action="store_true",
                   help="also ask the per-passage questions (a study; the served shape is the choice heads)")
    s.add_argument("--order", choices=("bm25", "reverse", "shuffle"), default="bm25",
                   help="control: permute the passages inside the state, content unchanged. A model "
                        "that reads content keeps its accuracy; one that follows BM25's order collapses")
    s.add_argument("--seed", type=int, default=0, help="--order shuffle")
    s.add_argument("--pool", type=int, default=K,
                   help=f"passages rendered into the state (default {K}); what BM25 finds beyond this is "
                        "invisible to the model, so this is the recall/quality trade in the product")
    s.set_defaults(fn=cmd_spike)
    s = sub.add_parser("gate", help="a System One run against another run, or against the v3 rows recorded "
                                    "under results/s1/rows (<tag>-<set>-disp.jsonl)")
    s.add_argument("kev"); s.add_argument("v3"); s.add_argument("set"); s.set_defaults(fn=cmd_gate)
    s = sub.add_parser("parity", help="the in-process reader against the served server, same states")
    s.add_argument("--run", required=True, help="the run directory the server is serving")
    s.add_argument("--url", default="", help="the served server to compare against (empty: local only)")
    s.add_argument("--model", default="jev-latest")
    s.add_argument("--set", default="md2d"); s.add_argument("--limit", type=int, default=5)
    s.add_argument("--pool", type=int, default=K)
    s.add_argument("--device", default=None, help="cpu|cuda|mps; default: this machine's accelerator")
    s.set_defaults(fn=cmd_parity)
    s = sub.add_parser("leak", help="can a bag of words tell a positive record from its negative arm?")
    s.add_argument("records", help="a records file to test (the run's own data file)")
    s.set_defaults(fn=cmd_leak)
    s = sub.add_parser("lines", help="exact-line top-1 on MultiDoc2Dial test"); s.add_argument("tag")
    s.set_defaults(fn=cmd_lines)
    s = sub.add_parser("mix", help="rewrite a record file's question mix (which head gets how much loss)")
    s.add_argument("records"); s.add_argument("--out", required=True)
    s.add_argument("--passage-asks", type=int, default=6, help="per-passage questions kept per record")
    s.add_argument("--exists-asks", type=int, default=3, help="wordings of `exists` asked per record")
    s.set_defaults(fn=cmd_mix)
    s = sub.add_parser("domains", help="the answer-absent reading split by md2d domain (studentaid is the control)")
    s.add_argument("tag"); s.set_defaults(fn=cmd_domains)
    s = sub.add_parser("fitcheck", help="does the `exists` head fit its own training data?")
    s.add_argument("records"); s.add_argument("--url", default="http://127.0.0.1:8009")
    s.add_argument("--model", default="jev-latest"); s.add_argument("--n", type=int, default=300)
    s.add_argument("--seed", type=int, default=0); s.set_defaults(fn=cmd_fitcheck)
    s = sub.add_parser("threshold", help="where the tool says the map may not answer (the curve, and two cuts)")
    s.add_argument("tag"); s.add_argument("--set", default="md2d"); s.set_defaults(fn=cmd_threshold)
    s = sub.add_parser("score", help="serve a run and measure it: spike, gate, lines, leak — one command")
    s.add_argument("tag"); s.add_argument("--run", default="", help="run directory (default <kev>/runs/<tag>)")
    s.add_argument("--sets", default="md2d,techqa,webshop"); s.add_argument("--port", type=int, default=8011)
    s.add_argument("--against", default="", help="comma-separated tags to pair against (gate)")
    s.add_argument("--leak", default="", help="the records file to run the acceptance test on")
    s.set_defaults(fn=cmd_score)
    s = sub.add_parser("run", help="train then score, for one tag")
    s.add_argument("tag"); s.add_argument("--records", required=True); s.add_argument("--bal", type=int, default=4000)
    s.add_argument("--bal-swe", type=int, default=0); s.add_argument("--bal-tag", default="")
    s.add_argument("--max-state", type=int, default=6656); s.add_argument("--epochs", type=int, default=1)
    s.add_argument("--no-code", action="store_true"); s.add_argument("--sets", default="md2d,techqa,webshop")
    s.add_argument("--against", default=""); s.add_argument("--leak", default="")
    s.add_argument("--port", type=int, default=8011)
    s.add_argument("--judge-records", default="", help="passed to train; the run is then also measured by `judge`")
    s.set_defaults(fn=cmd_run)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
