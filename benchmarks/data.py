"""Download public retrieval benchmarks into the layout the runners read.

    python benchmarks/data.py beir scifact           # BEIR zip from UKP
    python benchmarks/data.py coir stackoverflow-qa  # CoIR, from the Hugging Face hub
    python benchmarks/data.py zalo                   # Zalo AI legal text retrieval (Vietnamese)
    python benchmarks/data.py swe-lite               # SWE-bench Lite + one clone per repo

Everything lands under the data directory (default `<user cache>/inventio/bench`, override with
`--data` or INVENTIO_BENCH_DATA), never inside this repository: the SWE-bench clones alone are
about 2 GB. `coir` and `swe-lite` need the `datasets` package (`pip install -e ".[bench]"`).

Layouts:
  beir/<name>/            corpus.jsonl, queries.jsonl, qrels/{train,test}.tsv   (BEIR format)
  beir/coir-<name>/       the same format, converted from CoIR's parquet splits
  beir/zalo-legal/        the same format, from GreenNode/zalo-ai-legal-text-retrieval-vn
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


def write_qrels(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for q, d, s in rows:
            f.write(f"{q}\t{d}\t{int(s)}\n")


def fetch_coir(name: str, root: Path) -> Path:
    from datasets import load_dataset

    dest = root / "beir" / f"coir-{name}"
    (dest / "qrels").mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):  # a copy fetched before train qrels were kept gains them here
        if not (dest / "qrels" / f"{split}.tsv").exists():
            qrels = load_dataset(f"CoIR-Retrieval/{name}-qrels")[split]
            write_qrels(dest / "qrels" / f"{split}.tsv", ((r["query_id"], r["corpus_id"], r["score"]) for r in qrels))
    if (dest / "corpus.jsonl").exists():
        return dest
    qc = load_dataset(f"CoIR-Retrieval/{name}-queries-corpus")
    with (dest / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for r in qc["corpus"]:
            f.write(json.dumps({"_id": r["_id"], "title": r["title"] or "", "text": r["text"]}) + "\n")
    with (dest / "queries.jsonl").open("w", encoding="utf-8") as f:
        for r in qc["queries"]:
            f.write(json.dumps({"_id": r["_id"], "text": r["text"]}) + "\n")
    return dest


def fetch_zalo(root: Path) -> Path:
    """Zalo AI 2021 legal text retrieval: Vietnamese questions over articles of Vietnamese law,
    human-annotated (MIT on the dataset card). Train and test qrels share 24 queries; the
    fine-tuning script drops every test query from training, so the overlap never trains."""
    from huggingface_hub import hf_hub_download

    dest = root / "beir" / "zalo-legal"
    if (dest / "qrels" / "test.tsv").exists():
        return dest
    (dest / "qrels").mkdir(parents=True, exist_ok=True)
    get = lambda f: hf_hub_download("GreenNode/zalo-ai-legal-text-retrieval-vn", f, repo_type="dataset")  # noqa: E731
    with open(get("corpus.jsonl"), encoding="utf-8") as src, (dest / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for line in src:
            r = json.loads(line)
            f.write(json.dumps({"_id": r["_id"], "title": r.get("title") or "", "text": r.get("text") or ""},
                               ensure_ascii=False) + "\n")
    with open(get("queries.jsonl"), encoding="utf-8") as src, (dest / "queries.jsonl").open("w", encoding="utf-8") as f:
        for line in src:
            r = json.loads(line)
            f.write(json.dumps({"_id": r["_id"], "text": r["text"]}, ensure_ascii=False) + "\n")
    for split in ("train", "test"):
        rows = [json.loads(line) for line in open(get(f"qrels/{split}.jsonl"), encoding="utf-8")]
        write_qrels(dest / "qrels" / f"{split}.tsv", ((r["query-id"], r["corpus-id"], r["score"]) for r in rows))
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
    ap.add_argument("kind", choices=["beir", "coir", "zalo", "swe-lite"])
    ap.add_argument("name", nargs="?", help="dataset name for beir / coir")
    ap.add_argument("--data", help="data directory (default: user cache)")
    args = ap.parse_args()
    root = data_dir(args.data)
    if args.kind in ("beir", "coir") and not args.name:
        ap.error(f"{args.kind} needs a dataset name")
    out = {"beir": lambda: fetch_beir(args.name, root), "coir": lambda: fetch_coir(args.name, root),
           "zalo": lambda: fetch_zalo(root), "swe-lite": lambda: fetch_swe_lite(root)}[args.kind]()
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
