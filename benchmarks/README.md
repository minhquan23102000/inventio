# Benchmarks

Public retrieval benchmarks run through Inventio's real path: the same ingest (chunking by
heading and by function, path and heading indexed next to the text), the same BM25 query, the
same rankers. Nothing here is a re-implementation of BM25 for the occasion.

| Benchmark | What it asks | Corpus | Queries | Published BM25 nDCG@10 |
|---|---|---|---|---|
| [BEIR](https://arxiv.org/abs/2104.08663) SciFact | text: find the abstract that supports or refutes a claim | 5,183 abstracts | 300 | 0.665 |
| [CoIR](https://arxiv.org/abs/2407.02883) StackOverflow QA | mixed prose and code: find the accepted answer to a question | 19,931 answers | 1,994 | 0.568 |
| SWE-bench Lite, as in [CodeRAG-Bench](https://arxiv.org/abs/2406.14497) | code: from a GitHub issue, find the files the fix touches | the repository at the issue's commit | 300 | 0.430 |

SWE-bench Lite is run twice. `code` is the CodeRAG-Bench corpus (non-test `.py` files only), so
its BM25 number is comparable to the paper's. `mixed` indexes the whole repository the way you
would point Inventio at it: code, tests, `.rst`/`.md`/`.txt` docs, configs. The gold files are
the same; everything else is noise a real user would have.

## Reproduce

```sh
pip install -e ".[bench,laya,typesafe]"

python benchmarks/data.py beir scifact
python benchmarks/data.py coir stackoverflow-qa
python benchmarks/data.py swe-lite                 # about 2 GB of clones

python benchmarks/beir_bench.py scifact --rankers none,laya,typesafe
python benchmarks/beir_bench.py coir-stackoverflow-qa --rankers none
python benchmarks/beir_bench.py coir-stackoverflow-qa --rankers none,laya,typesafe --limit 300
python benchmarks/swe_bench.py --rankers none,laya,typesafe
```

Data goes to `<user cache>/inventio/bench` (`--data` or `INVENTIO_BENCH_DATA` to move it), never
into this repository. Results go to `benchmarks/results/`: a `summary.json` per benchmark, and
for SWE-bench one line per (instance, variant, ranker) with the top-10 files it returned.
Ranker scores are cached (ignored by git), so an interrupted run resumes and a rerun is free.

Every ranker reorders the same 30 BM25 candidates (chunks); chunks are then collapsed to
documents or files, first occurrence wins. `recall@30chunks` is therefore the ceiling for
every ranker. `laya` is `convaiinnovations/laya` multilingual, zero-shot. `typesafe` is Jev
through the TypeSafe API, asked the same yes/no question with the same criteria
(`inventio/rankers.py`).

## Results

nDCG@10 (higher is better, 1.0 = every gold document at the top). Run on 2026-09-23.

| Benchmark | Published BM25 | Inventio BM25 | + Laya zero-shot | + TypeSafe Jev | ceiling (gold in the 30 candidates) |
|---|---|---|---|---|---|
| SciFact (300 queries) | 0.665 | 0.670 | 0.302 | **0.765** | 0.849 |
| StackOverflow QA (1,994 queries) | 0.568 | 0.670 | 0.193 | **0.791** | 0.805 |
| SWE-bench Lite `code` (300 issues) | 0.430 | 0.540 | 0.391 | **0.696** | 0.823 |
| SWE-bench Lite `mixed` (300 issues) | | 0.400 | 0.264 | **0.515** | 0.633 |

For SWE-bench, the share of issues where a file the fix touches is the first result, or in the
first five:

| Variant | BM25 top 1 | top 5 | + TypeSafe top 1 | top 5 |
|---|---|---|---|---|
| `code` | 38% | 64% | 56% | 78% |
| `mixed` | 24% | 48% | 40% | 58% |

Cost per query: BM25 35-140 ms; Laya 0.6-0.9 s on the laptop GPU; TypeSafe 1.0-1.3 s. Indexing
a repository at one commit takes 6 s (`code`) to 14 s (`mixed`) on average.

Published retrievers on the same test sets (nDCG@10, single-stage, whole corpus):

| Benchmark | Figures | Source |
|---|---|---|
| SciFact | e5-mistral-7b-instruct 0.764, bge-large-en-v1.5 0.746, bge-base-en-v1.5 0.740, e5-large-v2 0.722, ColBERT 0.671; cross-encoder reranking BM25's top 100: 0.688 | MTEB results in each model's Hugging Face card; BEIR paper, Table 2 |
| StackOverflow QA | E5-Mistral 7B 0.915, Voyage-Code-002 0.877, E5-base 0.869, BGE-base 0.736, OpenAI Ada-002 0.724, Contriever 0.661, BGE-M3 0.610 | CoIR paper, Table 3 |
| SWE-bench Lite | SFR-Mistral 7B 0.627, Jina-v2-code 0.583, GIST-large 0.478, BGE-base 0.449, OpenAI embedding-3-small 0.433, Voyage-code 0.291 | CodeRAG-Bench paper, retrieval table |

The same model across papers: CoIR's E5-Mistral, BGE-Base and E5-base are
`e5-mistral-7b-instruct`, `bge-base-en-v1.5` and `e5-base-v2` (its model list); Voyage-Code-002
in CoIR and voyage-code-2 in CodeRAG-Bench are one model. CodeRAG-Bench does not state which
BGE-base version it ran, so the main README's BGE-base row joins two papers on that name.

What the numbers say:

- The BM25 baseline is sound: on SciFact it lands on the published figure (0.670 against
  0.665), so the measurement path is not flattering itself.
- Structure pays where the corpus has structure. On repositories Inventio's BM25 beats the
  published file-level BM25 by 0.11 (0.540 against 0.430), on the same issues, files and gold.
  The likely reason (not isolated by an ablation) is chunking by function and indexing each
  chunk's file path and function name next to its text. In CodeRAG-Bench's table this is above
  most embedders (BGE-base 0.449, GIST-large 0.478, OpenAI embedding-3-small 0.433, Voyage-code
  0.291) and below the two strongest (Jina-v2-code 0.583, the 7B SFR-Mistral 0.627). With
  TypeSafe on top (0.696) it is above every retriever in that table.
- On text the picture is different. SciFact: TypeSafe on top ties the best embedder listed
  (0.765 against e5-mistral-7b's 0.764). StackOverflow QA: it stays below the strongest
  embedders (0.791 against 0.869-0.915), because only 80% of questions have their answer among
  the 30 candidates; there the pool is the limit, as on whole repositories.
- TypeSafe Jev reordering the same 30 candidates adds 0.10 to 0.16 everywhere, and on SWE-bench
  `code` turns 38% first-hit into 56%.
- Zero-shot Laya makes every ranking worse, by a lot. It is not usable as a ranker until it is
  fine-tuned on this question (the Jev labels are the intended teacher).
- A whole repository is harder than its code: tests and docs push the fix's files out of the 30
  candidates (ceiling 0.823 falls to 0.633). The ranker cannot recover what BM25 did not hand
  it, so on mixed repositories the candidate pool, not the ranker, is the limit to work on.
- TypeSafe's firewall refused 409 of 59,820 pairs on StackOverflow QA and 113 on SWE-bench;
  they kept their BM25 place.

## Caveats

- Every configuration was run on every test query (SciFact 300, StackOverflow QA 1,994,
  SWE-bench Lite 300). An earlier StackOverflow QA run on the first 300 queries gave higher
  numbers (0.713 / 0.837); those 300 are easier than the full set.
- Laya reads at most 512 tokens of query plus passage. SWE-bench issues are often longer than
  that on their own, so Laya sees a truncated question there.
- TypeSafe's edge firewall refuses some texts outright (HTTP 403 "Attention Required"; it
  happens on Django source). Such a pair stays unscored and keeps its BM25 place after the
  scored ones; the count is reported with the results.
- Time per query is measured only over queries scored fresh, on one RTX 5070 laptop GPU for
  Laya and over the network for TypeSafe (12 parallel requests).
