"""Passages for the content categories (inventio/facts.py CATEGORIES), to be labelled.

    python benchmarks/category_data.py

No human labels exist for "what does this passage do for its reader", so both sets are labelled
by a small general model (Gemini Flash through the omp harness, prompt in PROMPT); relevance
labels stay human. The aim is a model that carries to documents it has never seen, and the split
is built to measure exactly that:

- One repository in five (`test_repo`, by name hash) is held out whole. Its prose, at most
  PER_REPO_TEST passages per repository so one project's template cannot fill the stratum, and
  StackOverflow answers, a genre never trained on, are the out-of-domain test.
- The other repositories give the training passages, at most PER_REPO_TRAIN each, with SciFact
  abstracts, Zalo's articles of Vietnamese law and SWE-bench issues (train split), a few hundred
  each; those three also give a small in-domain test stratum. The gap between the two strata is
  the overfitting.

Passages are rendered as facts.categorize renders them, `[path > heading]` + text, and are drawn
from Markdown, reStructuredText and plain-text chunks of the SWE-bench train repositories.

Output, under `<data>/categories/`: `train_unlabelled.jsonl`, `test_unlabelled.jsonl`. The labelled
`train.jsonl` and `test.jsonl` (the same rows with `label`, one call per passage with PROMPT and a
one-of-CATEGORIES schema) are what finetune_laya.py reads.
"""

import hashlib
import json
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import data_dir  # noqa: E402

SEED = 20260924
PER_REPO_TRAIN = 150
PER_REPO_TEST = 20
PER_CORPUS_TRAIN = 250
PER_CORPUS_TEST = 40
SO_TEST = 60
MIN_CHARS = 120

PROMPT = """Classify the passage by what it does for its reader. Choose exactly one category.

{defs}

Judge the passage's main purpose, not its topic: a passage about a law that tells you the steps to
file a claim is a Procedure; a passage in a changelog that lists what changed is a Record.

Passage:
{passage}"""


def test_repo(repo: str) -> bool:
    return int(hashlib.sha1(repo.encode()).hexdigest(), 16) % 5 == 0


def digest(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


def rows_of(db: Path, sql: str) -> list[tuple]:
    con = sqlite3.connect(db)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def main() -> int:
    rng = random.Random(SEED)
    root = data_dir(None)
    out = root / "categories"
    out.mkdir(parents=True, exist_ok=True)
    train, test, seen = [], [], set()

    def add(to, src, path, heading_path, text):
        if len(text.strip()) >= MIN_CHARS and digest(text) not in seen:
            seen.add(digest(text))  # every repository's copy of one licence counts once
            to.append({"src": src, "path": path, "heading_path": heading_path, "text": text})
            return True
        return False

    for db in sorted((root / "swe-train" / "maps").glob("*.db")):
        rows = rows_of(db, "SELECT f.path, c.heading_path, c.text FROM chunks c JOIN files f ON f.id = c.file_id "
                           "WHERE f.lang IN ('markdown', 'text')")
        rng.shuffle(rows)
        held = test_repo(db.stem)
        cap, got = (PER_REPO_TEST if held else PER_REPO_TRAIN), 0
        for p, hp, t in rows:
            if got == cap:
                break
            got += add(test if held else train, f"held-repo:{db.stem}" if held else f"repo:{db.stem}", p, hp, t)

    for name in ("scifact", "zalo-legal"):
        rows = rows_of(root / "beir" / name / "inventio.db",
                       "SELECT f.path, c.heading_path, c.text FROM chunks c JOIN files f ON f.id = c.file_id")
        rng.shuffle(rows)
        n = 0
        for p, hp, t in rows:
            if n == PER_CORPUS_TEST + PER_CORPUS_TRAIN:
                break
            n += add(test if n < PER_CORPUS_TEST else train, name, p, hp, t)
    lite = {json.loads(l)["instance_id"] for l in (root / "swe-lite" / "lite.jsonl").open(encoding="utf-8")}
    issues = [json.loads(l) for l in (root / "swe-train" / "train.jsonl").open(encoding="utf-8")]
    rng.shuffle(issues)
    n = 0
    for r in issues:
        if n == PER_CORPUS_TEST + PER_CORPUS_TRAIN:
            break
        if r["instance_id"] not in lite:
            n += add(test if n < PER_CORPUS_TEST else train, "swe-issue", f"{r['repo']}/issues", "", r["problem_statement"])
    rows = rows_of(root / "beir" / "coir-stackoverflow-qa" / "inventio.db",
                   "SELECT f.path, c.heading_path, c.text FROM chunks c JOIN files f ON f.id = c.file_id")
    rng.shuffle(rows)
    n = 0
    for p, hp, t in rows:
        if n == SO_TEST:
            break
        n += add(test, "coir-stackoverflow-qa", p, hp, t)

    for name, rows in (("train", train), ("test", test)):
        rng.shuffle(rows)
        with (out / f"{name}_unlabelled.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    group = lambda s: s.split(":")[0]  # noqa: E731
    print(json.dumps({"train": dict(Counter(group(r["src"]) for r in train)),
                      "test": dict(Counter(group(r["src"]) for r in test)),
                      "held-out repositories": sorted({r["src"][10:] for r in test if r["src"].startswith("held-repo:")})},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
