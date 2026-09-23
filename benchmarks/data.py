"""Download public retrieval benchmarks into the layout the runners read.

    python benchmarks/data.py beir scifact           # BEIR zip from UKP
    python benchmarks/data.py coir stackoverflow-qa  # CoIR, from the Hugging Face hub
    python benchmarks/data.py swe-lite               # SWE-bench Lite + one clone per repo

Everything lands under the data directory (default `<user cache>/inventio/bench`, override with
`--data` or INVENTIO_BENCH_DATA), never inside this repository: the SWE-bench clones alone are
about 2 GB. `coir` and `swe-lite` need the `datasets` package (`pip install -e ".[bench]"`).

Layouts:
  beir/<name>/            corpus.jsonl, queries.jsonl, qrels/test.tsv   (BEIR format)
  beir/coir-<name>/       the same format, converted from CoIR's parquet splits
  swe-lite/lite.jsonl     instance_id, repo, base_commit, patch, problem_statement
  swe-lite/<owner>__<repo>/  bare-enough clone of the github.com/swe-bench mirror
"""

import argparse
import io
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inventio.store import cache_dir  # noqa: E402

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


def data_dir(arg: str | None) -> Path:
    return Path(arg or os.environ.get("INVENTIO_BENCH_DATA") or cache_dir() / "bench")


def fetch_beir(name: str, root: Path) -> Path:
    dest = root / "beir"
    if (dest / name / "corpus.jsonl").exists():
        return dest / name
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BEIR_URL.format(name), timeout=600) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(dest)
    return dest / name


def fetch_coir(name: str, root: Path) -> Path:
    from datasets import load_dataset

    dest = root / "beir" / f"coir-{name}"
    if (dest / "corpus.jsonl").exists():
        return dest
    (dest / "qrels").mkdir(parents=True, exist_ok=True)
    qc = load_dataset(f"CoIR-Retrieval/{name}-queries-corpus")
    qrels = load_dataset(f"CoIR-Retrieval/{name}-qrels")["test"]
    with (dest / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for r in qc["corpus"]:
            f.write(json.dumps({"_id": r["_id"], "title": r["title"] or "", "text": r["text"]}) + "\n")
    with (dest / "queries.jsonl").open("w", encoding="utf-8") as f:
        for r in qc["queries"]:
            f.write(json.dumps({"_id": r["_id"], "text": r["text"]}) + "\n")
    with (dest / "qrels" / "test.tsv").open("w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for r in qrels:
            f.write(f"{r['query_id']}\t{r['corpus_id']}\t{r['score']}\n")
    return dest


def fetch_swe_lite(root: Path) -> Path:
    dest = root / "swe-lite"
    dest.mkdir(parents=True, exist_ok=True)
    lite = dest / "lite.jsonl"
    if not lite.exists():
        from datasets import load_dataset

        rows = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        with lite.open("w", encoding="utf-8") as f:
            for r in rows:
                keep = ("instance_id", "repo", "base_commit", "patch", "problem_statement")
                f.write(json.dumps({k: r[k] for k in keep}) + "\n")
    repos = sorted({json.loads(l)["repo"].replace("/", "__") for l in lite.open(encoding="utf-8")})
    todo = [r for r in repos if not (dest / r).exists()]
    procs = [subprocess.Popen(["git", "clone", "-q", "--no-checkout", f"https://github.com/swe-bench/{r}.git",
                               str(dest / r)]) for r in todo]
    if any(p.wait() for p in procs):
        raise SystemExit("a clone failed; rerun to retry the missing repositories")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["beir", "coir", "swe-lite"])
    ap.add_argument("name", nargs="?", help="dataset name for beir / coir")
    ap.add_argument("--data", help="data directory (default: user cache)")
    args = ap.parse_args()
    root = data_dir(args.data)
    if args.kind in ("beir", "coir") and not args.name:
        ap.error(f"{args.kind} needs a dataset name")
    out = {"beir": lambda: fetch_beir(args.name, root), "coir": lambda: fetch_coir(args.name, root),
           "swe-lite": lambda: fetch_swe_lite(root)}[args.kind]()
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
