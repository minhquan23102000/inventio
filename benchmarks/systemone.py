"""System One over BM25's pool: read the query and the passages once, answer `where_line` (which line
holds the answer) and `exists` (does any passage answer at all), and measure that against dispositio.

    python benchmarks/systemone.py data                       # Kev records -> <data>/s1/data/*.jsonl
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
import json
import os
import random
import re
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from beir_bench import load_beir, safe_name  # noqa: E402
from data import data_dir  # noqa: E402
from inventio.scope import Scope  # noqa: E402
from inventio.search import bm25  # noqa: E402
from inventio.store import connect  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ROWS = Path(__file__).resolve().parent / "results" / "s1"
DATA = data_dir(None) / "s1" / "data"
K = 15
MAX_OPTIONS = 255
MAX_QUERY_CHARS = 1500
norm = lambda t: " ".join(t.split())  # noqa: E731


# --- the state the model reads, and the questions it answers -------------------------------------

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


def questions(q, pids, lids, with_line=True):
    """The questions one request asks. `where_line` needs a line id per option (Kev: 255 options)."""
    qs = {"where_passage": {"type": "choice", "instructions": f'Which passage contains the answer to: "{q}"?',
                            "criteria": {p: None for p in pids}},
          "exists": {"type": "noul", "instructions": f'Does any passage answer: "{q}"?',
                     "criteria": {"true": "At least one passage states or directly implies the answer",
                                  "false": "No passage addresses this"}}}
    if with_line and len(lids) <= MAX_OPTIONS:
        qs["where_line"] = where_line(q, lids)
    return qs


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


def ask(url, state, qs, model="jev-latest", timeout=600):
    body = json.dumps({"state": state, "model": model, "questions": qs}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", body, {"content-type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = json.load(r)
    res["wall_ms"] = round((time.time() - t) * 1000, 1)
    return res


def run_query(url, q, pos, neg, is_gold, model):
    """One query through the model twice: the natural pool, and the pool without the answer."""
    out = {}
    for arm, hits in (("pos", pos), ("neg", neg)):
        state, pids, lids, owner = render(hits)
        qs = questions(q, pids, lids)
        res = ask(url, state, qs, model)
        a = res["answers"]
        pp = a["where_passage"]["probabilities"]
        top_p = max(pp, key=pp.get)
        row = {"exists": a["exists"]["noul"], "latency_ms": res.get("latency_ms"), "wall_ms": res["wall_ms"],
               "tokens": res.get("usage", {}).get("input_tokens"), "lines": len(lids),
               "where_line": "where_line" in qs and "where_line" in a}
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
    n = len(has)
    p1 = sum(r["pos"]["top_passage"] in r["pos"]["gold"] for r in has)
    b1 = sum(r["pos"]["bm25_top_gold"] for r in has)
    lr = [r for r in has if "top_line_owner" in r["pos"]]
    l1 = sum(r["pos"]["top_line_owner"] in r["pos"]["gold"] for r in lr)
    ex = auc([r["pos"]["exists"] for r in has], [r["neg"]["exists"] for r in rows])
    lat = sorted(x["latency_ms"] for r in rows for x in (r["pos"], r["neg"]) if x["latency_ms"] is not None)
    tok = sorted(x["tokens"] for r in rows for x in (r["pos"], r["neg"]) if x["tokens"])
    med = lambda v: v[len(v) // 2] if v else None  # noqa: E731
    return {"tag": tag, "set": name, "queries": len(rows), "gold_in_pool": n,
            "top_line_owner_passage@1": f"{l1}/{len(lr)}" if lr else "-",
            "passage@1": round(p1 / max(1, n), 3), "bm25@1": round(b1 / max(1, n), 3),
            "exists_auc": round(ex, 3),
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

def records(q, pos, neg, lines_of, src, key):
    """One positive record (where_line + exists=true) and one negative (exists=false) for a query."""
    q = q[:MAX_QUERY_CHARS]
    out = []
    state, pids, lids, owner = render(pos)
    gold = []
    for i, h in enumerate(pos):
        mine = [j for j, o in enumerate(owner) if o == i]
        gold += [mine[k] for k in lines_of(i, h) if k < len(mine)]
    if gold and len(lids) <= MAX_OPTIONS:
        w = where_line(q, lids)
        gl = [lids[j] for j in sorted(set(gold))]
        w.update(label=gl[0], target={l: 1 / len(gl) for l in gl})
        out.append({"state": state, "questions": {"where_line": w, "exists": exists(q, True)},
                    "_meta": {"source": src, "group_id": key, "arm": "pos"}})
    if neg:
        state, _, lids, _ = render(neg)
        out.append({"state": state, "questions": {"exists": exists(q, False)},
                    "_meta": {"source": src, "group_id": key, "arm": "neg"}})
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
                near = {section.get((r["doc_id"], r["id_sp"])) for t in turns[i:end] for r in t["references"]} - {None}
                is_gold = lambda h: Path(h.path).stem in gold  # noqa: E731
                hits = bm25(con, q, K + 40, scope)
                pos = hits[:K]
                neg = [h for h in hits if Path(h.path).stem not in gold | near][:K]

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
                yield from records(q, pos, neg, lines_of, f"md2d_{split}", f"{dial['dial_id']}:{i}")


def swe_records():
    """SWE-bench train issues (benchmarks/swe_train.py groups): gold lines are the lines the fix changed.
    Only the issues whose fix is inside BM25's 15 and whose pool fits the line question are kept."""
    for f in sorted(glob.glob(str(data_dir(None) / "swe-train" / "groups" / "*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            g = json.loads(line)
            hits = [SimpleNamespace(path=h["path"], heading_path=h["heading_path"], text=h["text"], at=h.get("at"))
                    for h in g["hits"]]
            gold, skip = set(g["gold"]), set(g["skip"])
            pos = hits[:K] if gold & set(range(K)) else []
            neg = [h for i, h in enumerate(hits) if i not in gold | skip][:K]

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
            yield from records(g["query"], pos, neg, lines_of, "swe_train", g["iid"])


# --- commands ------------------------------------------------------------------------------------

def cmd_spike(a):
    for name in a.sets.split(","):
        rows, path = [], ROWS / f"{a.tag}-{name}.jsonl"
        ROWS.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for qid, q, con, scope, is_gold in SETS[name](a.limit):
                pos, neg = pools(con, q, scope, is_gold)
                if not pos:
                    continue
                try:
                    r = {"qid": qid, "query": q, **run_query(a.url, q, pos, neg, is_gold, a.model)}
                except urllib.error.HTTPError as e:
                    print(f"skip {qid}: {e.code} {e.read()[:200]!r}", flush=True)
                    continue
                rows.append(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
        record("spike", a.tag, name, spike_summary(rows, a.tag, name))


def cmd_disp(a):
    import torch
    from inventio.rankers import make_ranker
    ranker = make_ranker("dispositio")
    for name in a.sets.split(","):
        top, n, pos_ex, neg_ex, ms, rows = 0, 0, [], [], [], []
        ROWS.mkdir(parents=True, exist_ok=True)
        for qid, q, con, scope, is_gold in SETS[name](a.limit):
            pos, neg = pools(con, q, scope, is_gold)
            if not pos or not neg:
                continue
            torch.cuda.synchronize()
            t = time.time()
            ps = ranker.score(q, pos)
            torch.cuda.synchronize()
            ms.append((time.time() - t) * 1000)
            pn = ranker.score(q, neg)
            neg_ex.append(max(pn))
            row = {"qid": qid, "top1": None, "exists_pos": max(ps), "exists_neg": max(pn)}
            if any(is_gold(h) for h in pos):
                n += 1
                pos_ex.append(max(ps))
                row["top1"] = bool(is_gold(pos[max(range(len(ps)), key=ps.__getitem__)]))
                top += row["top1"]
            rows.append(row)
        (ROWS / f"{a.tag}-{name}-disp.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        ms.sort()
        record("disp", a.tag, name, {
            "tag": a.tag, "set": name, "queries": len(rows), "gold_in_pool": n,
            "passage@1": round(top / max(1, n), 3), "exists_auc": round(auc(pos_ex, neg_ex), 3),
            "ms_per_pool_median": round(statistics.median(ms[1:]), 1) if len(ms) > 1 else None,
            "ms_p90": round(ms[int(len(ms) * 0.9)], 1)})


def cmd_gate(a):
    kev = {r["qid"]: r for r in map(json.loads, open(ROWS / f"{a.kev}-{a.set}.jsonl", encoding="utf-8"))}
    v3 = {r["qid"]: r for r in map(json.loads, open(ROWS / f"{a.v3}-{a.set}-disp.jsonl", encoding="utf-8"))}

    def kev_top1(r):
        p = r["pos"]
        return (p["top_line_owner"] if "top_line_owner" in p else p["top_passage"]) in p["gold"]

    both = [q for q in kev if q in v3 and kev[q]["pos"]["gold"] and v3[q]["top1"] is not None]
    x = [kev_top1(kev[q]) for q in both]
    y = [v3[q]["top1"] for q in both]
    d = [i - j for i, j in zip(x, y)]
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    s = {"set": a.set, "n": len(both), f"top1_{a.kev}": round(sum(x) / len(x), 3), f"top1_{a.v3}": round(sum(y) / len(y), 3),
         "diff": round(sum(d) / len(d), 3), "diff_lo": round(boots[50], 3), "diff_hi": round(boots[1949], 3),
         "wins": sum(i > 0 for i in d), "losses": sum(i < 0 for i in d),
         "exists_auc_kev": round(auc([kev[q]["pos"]["exists"] for q in kev if kev[q]["pos"]["gold"]],
                                     [kev[q]["neg"]["exists"] for q in kev]), 3),
         "exists_auc_v3": round(auc([r["exists_pos"] for r in v3.values() if r["top1"] is not None],
                                    [r["exists_neg"] for r in v3.values()]), 3)}
    record("gate", f"{a.kev}-vs-{a.v3}", a.set, s)


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("data", help="build the Kev training records"); s.set_defaults(fn=cmd_data)
    s = sub.add_parser("scale", help="request time against questions and state length")
    s.add_argument("url"); s.add_argument("tag")
    s.add_argument("--axis", default="q,len"); s.add_argument("--pools", type=int, default=60)
    s.set_defaults(fn=cmd_scale)
    s = sub.add_parser("spike", help="ask a served System One model over the pools")
    s.add_argument("url"); s.add_argument("tag"); s.add_argument("--model", default="jev-latest")
    s.add_argument("--sets", default="md2d"); s.add_argument("--limit", type=int, default=0)
    s.set_defaults(fn=cmd_spike)
    s = sub.add_parser("disp", help="dispositio on the same pools")
    s.add_argument("tag"); s.add_argument("--sets", default="md2d"); s.add_argument("--limit", type=int, default=0)
    s.set_defaults(fn=cmd_disp)
    s = sub.add_parser("gate", help="a System One run against dispositio")
    s.add_argument("kev"); s.add_argument("v3"); s.add_argument("set"); s.set_defaults(fn=cmd_gate)
    s = sub.add_parser("lines", help="exact-line top-1 on MultiDoc2Dial test"); s.add_argument("tag")
    s.set_defaults(fn=cmd_lines)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
