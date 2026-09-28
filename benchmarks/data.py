"""Download public retrieval benchmarks into the layout the runners read.

    python benchmarks/data.py beir scifact           # BEIR zip from UKP
    python benchmarks/data.py coir stackoverflow-qa  # CoIR, from the Hugging Face hub
    python benchmarks/data.py zalo                   # Zalo AI legal text retrieval (Vietnamese)
    python benchmarks/data.py multidoc2dial          # US public-service pages: rules and procedures
    python benchmarks/data.py techqa                 # IBM technotes: support questions, fixes and procedures
    python benchmarks/data.py swe-lite               # SWE-bench Lite + one clone per repo
    python benchmarks/data.py hotpotqa [train]       # HotpotQA bridge questions over a linked mini-wiki

Everything lands under the data directory (default `<user cache>/inventio/bench`, override with
`--data` or INVENTIO_BENCH_DATA), never inside this repository: the SWE-bench clones alone are
about 2 GB. `coir` and `swe-lite` need the `datasets` package (`pip install -e ".[bench]"`).

Layouts:
  beir/<name>/            corpus.jsonl, queries.jsonl, qrels/{train,test}.tsv   (BEIR format)
  beir/coir-<name>/       the same format, converted from CoIR's parquet splits
  beir/zalo-legal/        the same format, from GreenNode/zalo-ai-legal-text-retrieval-vn
  beir/multidoc2dial/     the same format, one document per section of a MultiDoc2Dial page
  beir/techqa/            the same format, from illuin-conteb/tech-qa (test only)
  beir/hotpotqa-links/    the same format, dev bridge questions; text carries Markdown links
  swe-lite/lite.jsonl     instance_id, repo, base_commit, patch, problem_statement
  swe-lite/<owner>__<repo>/  bare-enough clone of the github.com/swe-bench mirror
"""

import argparse
import io
import re
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


MULTIDOC2DIAL_URL = "https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip"


def fetch_multidoc2dial(root: Path) -> Path:
    """MultiDoc2Dial (IBM, Apache-2.0): people asking about the pages of four US public services
    (Social Security, Veterans Affairs, DMV, student aid), which state rules, eligibility and
    procedures. The query is the first user turn of a dialogue, a question that stands on its
    own ("What can I do if I forgot to update my address?"); the answers are the page sections
    the agent's reply was grounded in, marked by the annotators. A document is one section,
    titled by its page and headings, so the benchmark asks for the part of a page that answers.

    `dialogues.jsonl` gives each query's split and domain. `train_groups.jsonl` is what the
    fine-tune (benchmarks/finetune_laya.py) learns from, cleaner than the first turns: every user
    turn of a train dialogue that opens a topic (its first, or one grounded in another page than
    the turn before), kept when it is a question that stands on its own and the agent's reply
    gives a solution; the gold is that reply's `solution` sections, `near` every other section
    the topic went on to use (they may answer part of it), and `tight` says the user's own turn
    was grounded in the gold section itself. The validation split is kept for choosing between
    fine-tunes, never for training or the reported test."""
    dest = root / "beir" / "multidoc2dial"
    if (dest / "qrels" / "test.tsv").exists():
        return dest
    raw = root / "raw" / "multidoc2dial"
    if not (raw / "multidoc2dial" / "multidoc2dial_doc.json").exists():
        raw.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(MULTIDOC2DIAL_URL, timeout=600) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(raw)
    src = raw / "multidoc2dial"
    (dest / "qrels").mkdir(parents=True, exist_ok=True)
    pages = json.load((src / "multidoc2dial_doc.json").open(encoding="utf-8"))["doc_data"]
    section = {}  # (page id, span id) -> section document id
    clean = lambda t: re.sub(r"(#\d+(_\d+)?|\[\d+\])$", "", t.split(" | ")[0]).strip()  # noqa: E731
    with (dest / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for dname, domain in pages.items():
            for pi, (pid, page) in enumerate(domain.items()):
                title, parts = clean(page["title"]), {}  # heading path -> its paragraphs, in page order
                for sid, sp in page["spans"].items():
                    heads = [clean(t) for t in [*(p["text"] for p in sp["parent_titles"]), sp["title"]]]
                    path = tuple(dict.fromkeys(h for h in heads if h and h != title))
                    section[pid, sid] = path
                    paras = parts.setdefault(path, {})
                    if sp["tag"] == "u":  # h1..h6 spans are the headings, already in the title
                        paras.setdefault(sp["id_sec"], sp["text_sec"].strip())
                parts = {path: "\n\n".join(p for p in paras.values() if p) for path, paras in parts.items()}
                parts = {path: text for path, text in parts.items() if text}
                ids = {path: f"{dname}-{pi}-{si}" for si, path in enumerate(parts)}  # safe as file names
                section.update({k: ids.get(v) for k, v in section.items() if k[0] == pid})
                for path, text in parts.items():
                    f.write(json.dumps({"_id": ids[path], "title": " > ".join([title, *path]), "text": text},
                                       ensure_ascii=False) + "\n")
    vague = re.compile(r"\b(i have (a )?questions?|can you help|tell me about)\b", re.I)
    with (dest / "queries.jsonl").open("w", encoding="utf-8") as fq, \
            (dest / "dialogues.jsonl").open("w", encoding="utf-8") as fd, \
            (dest / "train_groups.jsonl").open("w", encoding="utf-8") as fg:
        for split in ("train", "validation", "test"):
            dials = json.load((src / f"multidoc2dial_dial_{split}.json").open(encoding="utf-8"))["dial_data"]
            rows = []
            for dname, domain in dials.items():
                for d in domain:
                    turns = d["turns"]
                    secs = lambda t, label=None: {section.get((r["doc_id"], r["id_sp"])) for r in t["references"]  # noqa: E731
                                                  if label in (None, r["label"])} - {None}
                    if split == "train":
                        fg.writelines(json.dumps(g) + "\n" for g in topic_groups(d, dname, secs, vague))
                    if len(turns) < 2 or turns[0]["role"] != "user" or turns[1]["role"] != "agent":
                        continue
                    gold = secs(turns[1])
                    if not gold:
                        continue
                    fq.write(json.dumps({"_id": d["dial_id"], "text": turns[0]["utterance"].strip()}) + "\n")
                    fd.write(json.dumps({"_id": d["dial_id"], "split": split, "domain": dname}) + "\n")
                    rows += [(d["dial_id"], g, 1) for g in sorted(gold)]
            write_qrels(dest / "qrels" / f"{split}.tsv", rows)
    return dest


def topic_groups(dial: dict, domain: str, secs, vague):
    """The training groups of one MultiDoc2Dial dialogue (see fetch_multidoc2dial)."""
    turns, starts, prev = dial["turns"], [], None
    for i, t in enumerate(turns):
        if t["role"] == "user" and t["da"].startswith("query") and t["references"]:
            pages = {r["doc_id"] for r in t["references"]}
            if pages != prev:
                starts.append(i)
            prev = pages
    for n, i in enumerate(starts):
        u, a = turns[i], turns[i + 1] if i + 1 < len(turns) else None
        text = u["utterance"].strip()
        if a is None or a["role"] != "agent" or not a["da"].startswith("respond_solution"):
            continue
        if vague.search(text) or ("?" not in text and len(text.split()) < 8):
            continue
        gold = sorted(secs(a, "solution"))[:2]
        if not gold:
            continue
        end = starts[n + 1] if n + 1 < len(starts) else len(turns)
        near = set().union(*(secs(t) for t in turns[i:end])) - set(gold)
        yield {"_id": f"{dial['dial_id']}:{i}", "domain": domain, "query": text, "gold": gold,
               "near": sorted(near), "tight": bool(secs(u)) and secs(u) <= set(gold)}


def fetch_techqa(root: Path) -> Path:
    """TechQA (IBM, Apache-2.0) as cut by ConTEB (illuin-conteb/tech-qa): questions people posted
    on IBM support forums, each answered by one passage of a technote (a support article of
    question, cause, answer and steps). 119 questions over 2,309 passages of 360 technotes, test
    only: no split of it is trained on. ConTEB leaves every passage but a technote's first without
    its title; here each carries it as its heading, as a section of a real document does in
    Inventio (the title is what the first passage says before " - United States")."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    dest = root / "beir" / "techqa"
    if (dest / "qrels" / "test.tsv").exists():
        return dest
    (dest / "qrels").mkdir(parents=True, exist_ok=True)
    get = lambda f: pd.read_parquet(hf_hub_download("illuin-conteb/tech-qa", f, repo_type="dataset"))  # noqa: E731
    docs = get("documents/train-00000-of-00001.parquet")
    doc = lambda cid: cid.rsplit("_", 1)[0]  # noqa: E731
    title = lambda t: " ".join(t.split(" - United States", 1)[0].split()) if " - United States" in t[:400] else ""  # noqa: E731
    titles = {doc(r.chunk_id): title(r.chunk) for r in docs.itertuples() if r.chunk_id.endswith("_0")}
    with (dest / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for r in docs.itertuples():
            f.write(json.dumps({"_id": r.chunk_id, "title": titles.get(doc(r.chunk_id), ""), "text": r.chunk},
                               ensure_ascii=False) + "\n")
    qs = get("queries/train-00000-of-00001.parquet")
    with (dest / "queries.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(qs.itertuples()):
            f.write(json.dumps({"_id": f"q{i}", "text": r.query.strip()}, ensure_ascii=False) + "\n")
    write_qrels(dest / "qrels" / "test.tsv", ((f"q{i}", r.chunk_id, 1) for i, r in enumerate(qs.itertuples())))
    return dest


HOTPOT_TRAIN_QUESTIONS = 4000   # bridge questions drawn from train (seed 0); their contexts are the train map


def fetch_hotpotqa(root: Path, split: str = "dev") -> Path:
    """HotpotQA (Yang et al. 2018, CC BY-SA 4.0), distractor dev set from hotpotqa/hotpot_qa: the one
    public set where answering needs a link. A bridge question names page A; its answer is on page B,
    which A mentions and the question does not (Asai et al. 2020: that page "often has little lexical or
    semantic relationship to the question"). The corpus is every paragraph of the 7,405 dev questions'
    contexts (66,581 pages, gold and distractors alike), one Markdown file per page, and the queries are
    the 5,918 bridge questions, both supporting pages relevant as in BEIR's HotpotQA.

    The Hugging Face copy has no hyperlinks, so the links are drawn back the way a wiki writes them: the
    first mention of another page's title (without its parenthesis, "Ed Wood (film)" -> "Ed Wood") becomes
    a Markdown link to that page. A title named by more than LINK_MAX_PAGES pages is vocabulary, not a
    link, and a surface shared by several titles links only the page titled exactly that. It is a proxy
    for Wikipedia's own links, drawn by code as inventio draws `citation`; `queries.jsonl` records for
    each question which page links to which (`bridge`) and whether that link was drawn.

    `split="train"` builds the same shape from HOTPOT_TRAIN_QUESTIONS bridge questions of the train set
    (qrels/train.tsv, beir/hotpotqa-train-links), for training; no dev page or question is in it."""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    import random

    train = split == "train"
    dest = root / "beir" / ("hotpotqa-train-links" if train else "hotpotqa-links")
    qrels_file = dest / "qrels" / ("train.tsv" if train else "test.tsv")
    if qrels_file.exists():
        return dest
    (dest / "qrels").mkdir(parents=True, exist_ok=True)
    files = ([f"distractor/train-0000{i}-of-00002.parquet" for i in (0, 1)] if train
             else ["distractor/validation-00000-of-00001.parquet"])
    rows = [r for f in files for r in pd.read_parquet(hf_hub_download("hotpotqa/hotpot_qa", f, repo_type="dataset")).to_dict("records")]
    if train:
        bridge = [r for r in rows if r["type"] == "bridge" and len(set(r["supporting_facts"]["title"])) == 2]
        rows = random.Random(0).sample(bridge, HOTPOT_TRAIN_QUESTIONS)
    text, pid = {}, {}
    for r in rows:
        for t, sents in zip(r["context"]["title"], r["context"]["sentences"]):
            if t not in pid:
                pid[t] = f"p{len(pid)}"
                text[t] = "".join(sents).strip()
    links = link_titles(text, pid)
    with (dest / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for t, p in pid.items():
            f.write(json.dumps({"_id": p, "title": t, "text": links[t][0]}, ensure_ascii=False) + "\n")
    qrels = []
    with (dest / "queries.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            gold = list(dict.fromkeys(r["supporting_facts"]["title"]))
            if r["type"] != "bridge" or len(gold) != 2:
                continue
            a, b = gold
            src, dst = (a, b) if b in links[a][1] else (b, a) if a in links[b][1] else (a, b)
            f.write(json.dumps({"_id": r["id"], "text": r["question"], "answer": r["answer"],
                                "bridge": {"from": pid[src], "to": pid[dst], "linked": dst in links[src][1]}},
                               ensure_ascii=False) + "\n")
            qrels += [(r["id"], pid[a], 1), (r["id"], pid[b], 1)]
    write_qrels(qrels_file, qrels)
    return dest


LINK_MAX_PAGES = 50   # a title mentioned by more pages than this is vocabulary ("United States"), not a link


def link_titles(text: dict[str, str], pid: dict[str, str]) -> dict[str, tuple[str, set[str]]]:
    """Each page's text with its first mention of every other page's title made a Markdown link, and the
    titles it links (see fetch_hotpotqa)."""
    word = re.compile(r"\w+")
    surface = lambda t: re.sub(r"\s*\([^)]*\)$", "", t).strip()  # noqa: E731
    by_surface: dict[tuple, list[str]] = {}
    for t in text:
        key = tuple(word.findall(surface(t)))
        if key and max(map(len, key)) >= 3:
            by_surface.setdefault(key, []).append(t)
    target = {}
    for key, ts in by_surface.items():
        exact = [t for t in ts if t == " ".join(key) or t == surface(t) and len(ts) == 1]
        if len(ts) == 1 or exact:
            target[key] = (exact or ts)[0]
    longest = max(map(len, target))

    def mentions(t):
        toks = list(word.finditer(text[t]))
        out, i = [], 0
        while i < len(toks):
            for n in range(min(longest, len(toks) - i), 0, -1):
                key = tuple(m.group() for m in toks[i:i + n])
                if key in target and target[key] != t:
                    out.append((toks[i].start(), toks[i + n - 1].end(), target[key]))
                    i += n
                    break
            else:
                i += 1
        return out

    found = {t: mentions(t) for t in text}
    pages = {}
    for ms in found.values():
        for dst in {m[2] for m in ms}:
            pages[dst] = pages.get(dst, 0) + 1
    out = {}
    for t, ms in found.items():
        s, seen, cut, parts = text[t], set(), 0, []
        for a, b, dst in ms:
            if dst in seen or pages[dst] > LINK_MAX_PAGES:
                continue
            seen.add(dst)
            parts += [s[cut:a], f"[{s[a:b]}]({pid[dst]}.md)"]
            cut = b
        out[t] = ("".join(parts) + s[cut:], seen)
    return out


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
    ap.add_argument("kind", choices=["beir", "coir", "zalo", "multidoc2dial", "techqa", "swe-lite", "hotpotqa"])
    ap.add_argument("name", nargs="?", help="dataset name for beir / coir")
    ap.add_argument("--data", help="data directory (default: user cache)")
    args = ap.parse_args()
    root = data_dir(args.data)
    if args.kind in ("beir", "coir") and not args.name:
        ap.error(f"{args.kind} needs a dataset name")
    out = {"beir": lambda: fetch_beir(args.name, root), "coir": lambda: fetch_coir(args.name, root),
           "zalo": lambda: fetch_zalo(root), "multidoc2dial": lambda: fetch_multidoc2dial(root),
           "techqa": lambda: fetch_techqa(root), "swe-lite": lambda: fetch_swe_lite(root),
           "hotpotqa": lambda: fetch_hotpotqa(root, args.name or "dev")}[args.kind]()
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
