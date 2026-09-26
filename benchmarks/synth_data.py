"""Counterfactual passages and link questions for finetune_laya.py's `synth` source.

    %run benchmarks/synth_data.py        # inside the omp eval kernel, whose `completion` calls the small model

Written by a small general model (Gemini Flash, the harness's `smol` role), in batches of 20
calls, from the train split only: every query is a train query not in the 10% finetune_laya.py
holds out, and link pairs never use a document that answers a test query. About 1,000 calls:

- answer_units.jsonl (100 per corpus): a train query and its answer split into numbered units;
  which units state the answer, and whether the rest would still answer. The answer with those
  units deleted is a negative that keeps the answer's topic and words.
- lexical_mask.jsonl (75 per corpus, half answers, half BM25's best non-answer): the passage
  rewritten to share as few words with the query as it can, facts unchanged. Same label.
- link_questions.jsonl (75 per corpus): a chunk and its BM25 neighbour in another document; a
  question only both together answer, or `possible: false`.
- asks.json (10 calls): 24 wordings, each with its own criteria, of every question finetune_laya's
  `instr` asks (relevance, passage language, query language, code, digits) and of its opposite.

Gemini API terms forbid using the service "to develop models that compete with" it; a 140M
passage reranker is not a general model, which is the reading category_data.py's labels rest on.
"""

import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent)]
from beir_bench import load_beir, safe_name  # noqa: E402
from data import data_dir  # noqa: E402
from finetune_laya import held  # noqa: E402
from inventio.scope import Scope  # noqa: E402
from inventio.search import bm25  # noqa: E402
from inventio.store import connect  # noqa: E402

DS = ("scifact", "coir-stackoverflow-qa", "zalo-legal", "multidoc2dial")
SEED = 20260926
BATCH = 20
WORD = re.compile(r"\w+", re.UNICODE)

T_UNITS = """A search question and a passage split into numbered units. Which units state what the question asks for (the answer, the rule, the fix, the evidence)? List only those units; a unit that merely shares the topic or words is not one.
Then imagine the passage with those units deleted: would the rest still answer the question? And could the full passage answer it at all?

Question: {q}

Passage:
{p}"""
S_UNITS = {"type": "object", "properties": {"answer_units": {"type": "array", "items": {"type": "integer"}},
           "rest_still_answers": {"type": "boolean"}, "answerable": {"type": "boolean"}},
           "required": ["answer_units", "rest_still_answers", "answerable"]}

T_MASK = """Rewrite the passage so that it shares as few words with the question as possible, while saying exactly the same things: replace every word it has in common with the question by a synonym or a paraphrase, and restructure sentences if needed. Keep the passage's language. Do not add or drop any fact, number, name of a law, or step. Leave code, identifiers, file names and quoted error messages unchanged. Do not answer the question; only rewrite.

Question: {q}

Passage:
{p}"""
S_MASK = {"type": "object", "properties": {"rewrite": {"type": "string"}, "facts_unchanged": {"type": "boolean"}},
          "required": ["rewrite", "facts_unchanged"]}

T_LINK = """Two passages from the same document collection. Write one question a real reader of this collection could ask whose full answer needs BOTH passages: part of the answer is only in A and part only in B, or A gives a rule and B the case it applies to. Write it in the passages' language, in the reader's own words, not copying phrases from either passage. If no such natural question exists (the passages only share words or a broad topic), set possible to false.

Passage A:
{a}

Passage B:
{b}"""
S_LINK = {"type": "object", "properties": {"possible": {"type": "boolean"}, "question": {"type": "string"},
          "from_a": {"type": "string"}, "from_b": {"type": "string"}}, "required": ["possible", "question", "from_a", "from_b"]}


def train_pairs(name: str, n: int, rng) -> list[dict]:
    """`n` (train query, answer document) pairs, one per query, none held out or equal to a test query."""
    ds = data_dir(None) / "beir" / name
    corpus, test_q, _ = load_beir(ds, "test")
    tests = {" ".join(q.lower().split()) for q in test_q.values()}
    if name == "multidoc2dial":
        rows = [json.loads(l) for l in open(ds / "train_groups.jsonl", encoding="utf-8")]
        by_safe = {safe_name(d): d for d in corpus}
        ents = [(r["query"], by_safe.get(r["gold"][0], r["gold"][0])) for r in rows if r["gold"] and r["domain"] != "studentaid"]
    else:
        _, tq, trel = load_beir(ds, "train")
        ents = [(tq[q], d) for q in trel if q in tq for d, s in trel[q].items() if s > 0]
    ents = [(q, d) for q, d in ents if not held(f"{name}\t{q}") and " ".join(q.lower().split()) not in tests
            and d in corpus and 200 < len(corpus[d][1]) < 3500]
    rng.shuffle(ents)
    seen, out = set(), []
    for q, d in ents:
        if q not in seen:
            seen.add(q)
            out.append({"src": name, "query": q, "doc": d, "title": corpus[d][0], "text": corpus[d][1]})
        if len(out) >= n:
            break
    return out


def units(text: str) -> list[str]:
    out = []
    for block in re.split(r"\n\s*\n|\n(?=\s*(?:\d+\.|[-*•]|[a-zđ]\)))", text):
        block = block.strip()
        if not block:
            continue
        if "{" in block or ";" in block and block.count("\n") > 1:
            out.append(block)  # code: one unit
        else:
            out += [s.strip() for s in re.split(r"(?<=[.!?;:])\s+(?=[A-ZÀ-Ỹ0-9\"(])", block) if s.strip()]
    return out


def jid(*parts: str) -> str:
    return hashlib.sha1("".join(parts).encode()).hexdigest()[:12]


def jobs(rng) -> dict[str, list[dict]]:
    beir = data_dir(None) / "beir"
    cands = {d: train_pairs(d, 175, rng) for d in DS}
    unit_jobs = [{**c, "id": jid(c["query"], c["doc"]), "units": u}
                 for d in DS for c in cands[d][:100] if 2 <= len(u := units(c["text"])) <= 40]
    mask_jobs, link_jobs = [], []
    for d in DS:
        con = connect(beir / d / "inventio.db")
        corpus, _, test_rel = load_beir(beir / d, "test")
        safe = {safe_name(k): k for k in corpus}
        for n, c in enumerate(cands[d][100:175]):
            if n % 2 == 0:
                text, lab = c["text"], "pos"
            else:
                hs = [h for h in bm25(con, c["query"], 10, Scope.only([d]))
                      if safe.get(Path(h.path).stem) != c["doc"] and 200 < len(h.text) < 3500]
                if not hs:
                    continue
                text, lab = hs[0].text, "neg"
            mask_jobs.append({"src": d, "query": c["query"], "doc": c["doc"], "label": lab, "text": text, "id": jid(c["query"], lab)})
        banned = {safe_name(x) for rel in test_rel.values() for x in rel}
        rows = con.execute("SELECT c.id, f.path, c.heading_path, c.text FROM chunks c JOIN files f ON f.id = c.file_id "
                           "WHERE length(c.text) BETWEEN 300 AND 2500 ORDER BY random() LIMIT 400").fetchall()
        n = 0
        for r in rows:
            if Path(r["path"]).stem in banned:
                continue
            words = [w for w in WORD.findall(r["text"]) if len(w) > 3][:40]
            hs = [h for h in bm25(con, " ".join(words), 8, Scope.only([d]))
                  if h.path != r["path"] and Path(h.path).stem not in banned and 300 <= len(h.text) <= 2500]
            if hs:
                b = hs[0]
                link_jobs.append({"src": d, "id": f"{d}:{r['id']}:{b.id}", "a_path": r["path"], "a_head": r["heading_path"],
                                  "a": r["text"], "b_path": b.path, "b_head": b.heading_path, "b": b.text})
                n += 1
            if n >= 75:
                break
    return {"answer_units": unit_jobs, "lexical_mask": mask_jobs, "link_questions": link_jobs}


PROMPTS = {
    "answer_units": (lambda j: T_UNITS.format(q=j["query"], p="\n".join(f"[{i}] {x}" for i, x in enumerate(j["units"]))), S_UNITS),
    "lexical_mask": (lambda j: T_MASK.format(q=j["query"], p=j["text"]), S_MASK),
    "link_questions": (lambda j: T_LINK.format(a=j["a"], b=j["b"]), S_LINK),
}


def run(complete, out_dir: Path) -> None:
    """`complete(prompt, schema)` returns a handle with `.wait()`; appends one line per answered job,
    skips jobs already in the file, BATCH calls at a time."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, todo in jobs(random.Random(SEED)).items():
        build, schema = PROMPTS[name]
        path = out_dir / f"{name}.jsonl"
        done = {json.loads(l)["id"] for l in path.open(encoding="utf-8")} if path.exists() else set()
        todo = [j for j in todo if j["id"] not in done]
        with path.open("a", encoding="utf-8") as f:
            for s in range(0, len(todo), BATCH):
                hs = [(j, complete(build(j), schema)) for j in todo[s:s + BATCH]]
                for j, h in hs:
                    try:
                        f.write(json.dumps({**j, "out": h.wait()}, ensure_ascii=False) + "\n")
                    except Exception as e:  # a refused or failed call leaves the job for the next run
                        print("failed", name, j["id"], str(e)[:120])
                f.flush()
                time.sleep(1)
        print(name, sum(1 for _ in path.open(encoding="utf-8")), "answered", flush=True)


ASK_SPEC = {
    ("rel", "pos"): "the passage answers the query / gives what the query asks for",
    ("rel", "neg"): "the passage does NOT answer the query: it fails to, leaves it unanswered, is only on the same topic",
    ("lang", "pos"): "the passage is written in Vietnamese",
    ("lang", "neg"): "the passage is written in English (or: in a language other than Vietnamese)",
    ("qlang", "pos"): "the query is written in Vietnamese",
    ("qlang", "neg"): "the query is written in English (or: not in Vietnamese)",
    ("code", "pos"): "the passage is source code / a piece of a program",
    ("code", "neg"): "the passage is prose (natural-language text, not code)",
    ("digits", "pos"): "the passage contains at least one number written in digits",
    ("digits", "neg"): "the passage contains no digits at all",
}
T_ASKS = """Write 24 different yes/no questions that a classifier reads to decide, about a pair (`query`, `passage`), whether: {spec}.
Refer to the two texts exactly as `query` and `passage` with backticks. Vary wording, sentence shape, length and register a lot; about a quarter in Vietnamese, the rest in English. Every question must be answered "true" exactly when {spec}. For each, give short criteria saying when the answer is true and when false."""
S_ASKS = {"type": "object", "properties": {"questions": {"type": "array", "items": {"type": "object", "properties": {
    "instructions": {"type": "string"}, "true": {"type": "string"}, "false": {"type": "string"}},
    "required": ["instructions", "true", "false"]}}}, "required": ["questions"]}


def run_asks(complete, out_dir: Path) -> None:
    path = out_dir / "asks.json"
    if path.exists():
        return
    hs = {k: complete(T_ASKS.format(spec=v), S_ASKS) for k, v in ASK_SPEC.items()}
    bank: dict = {}
    for (kind, pol), h in hs.items():
        bank.setdefault(kind, {})[pol] = [q for q in h.wait()["questions"] if "`" in q["instructions"]]
    path.write_text(json.dumps(bank, ensure_ascii=False, indent=1), encoding="utf-8")



if __name__ == "__main__" and "completion" in globals():
    smol = lambda prompt, schema: completion(prompt, model="smol", schema=schema)  # noqa: E731, F821
    run(smol, data_dir(None) / "synth")
    run_asks(smol, data_dir(None) / "synth")
