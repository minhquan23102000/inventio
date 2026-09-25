"""Fine-tuning groups from the SWE-bench train split: a GitHub issue, and the code its fix changed.

    python benchmarks/swe_train.py [--per-repo 50] [--workers 8]

SWE-bench's train split (princeton-nlp/SWE-bench, 19,008 issues from 35 repositories, none of
them among SWE-bench Lite's 12) pairs each issue with the pull request that fixed it. The label
is the patch, written by the project's developers: a chunk is an answer when the fix changes one
of its lines. Other chunks of a file the fix touches are left out of both sides (the fix did not
change them, yet they may well be what an engineer would read); chunks of every other file are
the negatives. No label comes from a model.

Up to `--per-repo` issues are taken per repository, spread over its history (pandas alone has
5,049). Each repository is cloned once without file contents (`--filter=blob:none`); its issues
are visited in time order with `git checkout`, which fetches and rewrites only the files that
changed, and the map follows through Inventio's incremental ingest. The whole repository is
indexed, code, tests and docs, as a user would point Inventio at it. Repositories run in
parallel, one process each.

Output, under `<data>/swe-train/`: `groups/<owner>__<repo>.jsonl`, one line per issue with the
query, BM25's 30 candidates and which of them the fix touches. A rerun resumes.
"""

import argparse
import json
import re
import subprocess
import time
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import data_dir  # noqa: E402

POOL = 30
LITE_REPOS = {"astropy/astropy", "django/django", "matplotlib/matplotlib", "mwaskom/seaborn", "pallets/flask",
              "psf/requests", "pydata/xarray", "pylint-dev/pylint", "pytest-dev/pytest",
              "scikit-learn/scikit-learn", "sphinx-doc/sphinx", "sympy/sympy"}
GIT = ["git", "-c", "core.longpaths=true", "-c", "core.autocrlf=false", "-c", "advice.detachedHead=false"]


def touched(patch: str) -> dict[str, set[int]]:
    """Per file, the pre-fix line numbers the patch removes or inserts next to."""
    out: dict[str, set[int]] = {}
    path, old = None, 0
    for line in patch.splitlines():
        if line.startswith("--- "):
            path = line[6:] if line.startswith("--- a/") else None
        elif line.startswith("+++ "):
            continue
        elif line.startswith("@@"):
            old = int(re.match(r"@@ -(\d+)", line).group(1))
        elif path is None:
            continue
        elif line.startswith("-"):
            out.setdefault(path, set()).add(old)
            old += 1
        elif line.startswith("+"):
            out.setdefault(path, set()).update({max(old - 1, 1), old})  # an insertion sits between two lines
        elif line.startswith(" "):
            old += 1
    return out


def sample(rows: list[dict], per_repo: int) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = {}
    for r in rows:
        if r["repo"] not in LITE_REPOS:
            by.setdefault(r["repo"], []).append(r)
    out = {}
    for repo, rs in by.items():
        rs.sort(key=lambda r: r["created_at"])
        step = max(1, len(rs) / per_repo)
        out[repo] = [rs[int(i * step)] for i in range(min(per_repo, len(rs)))]
    return out


def build_repo(job) -> tuple[str, int, int]:
    from inventio.ingest import ingest_source
    from inventio.scope import Scope
    from inventio.search import bm25
    from inventio.store import connect

    repo, issues, root = job
    name = repo.replace("/", "__")
    out_path = root / "groups" / f"{name}.jsonl"
    done = {json.loads(l)["iid"] for l in out_path.open(encoding="utf-8")} if out_path.exists() else set()
    todo = [r for r in issues if r["instance_id"] not in done]
    if not todo:
        return repo, 0, 0
    clone = root / "clones" / name
    for attempt in range(3):
        if (clone / ".git").exists():
            break
        if subprocess.run([*GIT, "clone", "-q", "--filter=blob:none", "--no-checkout",
                           f"https://github.com/{repo}.git", str(clone)]).returncode:
            subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", str(clone)], capture_output=True)
            time.sleep(30 * (attempt + 1))
    else:
        return repo, 0, len(todo)  # the network kept failing; a rerun picks the repository up
    con = connect(root / "maps" / f"{name}.db")
    kept = skipped = 0
    with out_path.open("a", encoding="utf-8") as out:
        for r in todo:
            if subprocess.run([*GIT, "-C", str(clone), "checkout", "-q", "-f", r["base_commit"]],
                              capture_output=True).returncode:
                skipped += 1  # a commit the clone lacks, or a file name Windows cannot hold
                continue
            ingest_source(con, name, clone, True, [])
            con.commit()  # a rerun after a crash starts from this map instead of re-reading the repository
            lines = touched(r["patch"])
            hits = bm25(con, r["problem_statement"], POOL, Scope.only([name]))

            def at(path, a, b):
                return sorted(n - a for n in lines.get(path, ()) if a <= n <= b)

            gold = [i for i, h in enumerate(hits) if at(h.path, h.start_line, h.end_line)]
            skip = [i for i, h in enumerate(hits) if h.path in lines and i not in gold]
            extra = []
            if not gold:  # the changed code, when BM25 did not bring it
                for row in con.execute(
                        "SELECT f.path, c.heading_path, c.text, c.start_line, c.end_line FROM chunks c "
                        "JOIN files f ON f.id = c.file_id JOIN sources s ON s.id = f.source_id WHERE s.name = ? "
                        f"AND f.path IN ({','.join('?' * len(lines))}) ORDER BY f.path, c.start_line",
                        (name, *lines)).fetchall():
                    if at(row["path"], row["start_line"], row["end_line"]):
                        extra.append({"path": row["path"], "heading_path": row["heading_path"], "text": row["text"],
                                      "at": at(row["path"], row["start_line"], row["end_line"])})
                        if len(extra) == 2:
                            break
            if not gold and not extra:
                skipped += 1  # the fix changes only files the map does not hold
                continue
            out.write(json.dumps({
                "iid": r["instance_id"], "repo": repo, "query": r["problem_statement"],
                "hits": [{"path": h.path, "heading_path": h.heading_path, "text": h.text,
                          **({"at": at(h.path, h.start_line, h.end_line)} if i in gold else {})}
                         for i, h in enumerate(hits)],
                "gold": gold, "skip": skip, "extra": extra}, ensure_ascii=False) + "\n")
            out.flush()
            kept += 1
    con.close()
    return repo, kept, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", help="data directory (default: user cache)")
    ap.add_argument("--per-repo", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    root = data_dir(args.data) / "swe-train"
    for d in ("groups", "clones", "maps"):
        (root / d).mkdir(parents=True, exist_ok=True)
    rows_path = root / "train.jsonl"
    if not rows_path.exists():
        from datasets import load_dataset

        keep = ("instance_id", "repo", "base_commit", "patch", "problem_statement", "created_at")
        with rows_path.open("w", encoding="utf-8") as f:
            for r in load_dataset("princeton-nlp/SWE-bench", split="train"):
                f.write(json.dumps({k: r[k] for k in keep}) + "\n")
    rows = [json.loads(l) for l in rows_path.open(encoding="utf-8")]
    picked = sample(rows, args.per_repo)
    print(f"{sum(map(len, picked.values()))} issues from {len(picked)} repositories", flush=True)
    jobs = sorted(((repo, rs, root) for repo, rs in picked.items()), key=lambda j: -len(j[1]))
    total = 0
    with Pool(args.workers) as pool:
        for repo, kept, skipped in pool.imap_unordered(build_repo, jobs):
            total += kept
            print(f"{repo}: {kept} groups, {skipped} skipped; {total} so far", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
