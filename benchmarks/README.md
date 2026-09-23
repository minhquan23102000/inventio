# Benchmarks

Public retrieval benchmarks run through Inventio's real path: the same ingest (chunking by
heading and by function, path and heading indexed next to the text), the same BM25 query, the
same rankers. Nothing here is a re-implementation of BM25 for the occasion.

| Benchmark | What it asks | Corpus | Queries | Published BM25 nDCG@10 |
|---|---|---|---|---|
| [BEIR](https://arxiv.org/abs/2104.08663) SciFact | text: find the abstract that supports or refutes a claim | 5,183 abstracts | 300 | 0.665 |
| [CoIR](https://arxiv.org/abs/2407.02883) StackOverflow QA | mixed prose and code: find the accepted answer to a question | 19,931 answers | 1,994 | 0.568 |
| SWE-bench Lite, as in [CodeRAG-Bench](https://arxiv.org/abs/2406.14497) | code: from a GitHub issue, find the files the fix touches | the repository at the issue's commit | 300 | 0.430 |
| [Zalo legal text retrieval](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn) | Vietnamese: find the law article that answers a question | 61,425 articles | 788 | – |

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
python benchmarks/swe_bench.py --types --variants mixed --rankers none,typesafe

# categories and fact links: Jev judges every chunk and candidate pair, then the arms
python benchmarks/beir_bench.py scifact --arms --mlt --rankers none,typesafe,laya
python benchmarks/beir_bench.py coir-stackoverflow-qa --arms --mlt --rankers none,typesafe,laya
python benchmarks/swe_bench.py --strat --variants mixed --rankers none,laya   # names and code-decided widening
python benchmarks/data.py zalo && python benchmarks/beir_bench.py zalo-legal --rankers none,laya
python benchmarks/pack_check.py                    # packed neighbour call vs one call per pair

# fine-tuned Laya: results are tagged laya-tuned, beside the published model's
python benchmarks/finetune_laya.py
INVENTIO_LAYA_MODEL=<user cache>/inventio/laya-tuned python benchmarks/beir_bench.py scifact --rankers laya
```

Data goes to `<user cache>/inventio/bench` (`--data` or `INVENTIO_BENCH_DATA` to move it), never
into this repository. Results go to `benchmarks/results/`: a `summary.json` per benchmark, and
for SWE-bench one line per (instance, variant, ranker) with the top-10 files it returned.
Ranker scores are cached (ignored by git), so an interrupted run resumes and a rerun is free.

Every ranker reorders the same 30 BM25 candidates (chunks); chunks are then collapsed to
documents or files, first occurrence wins. `recall@30chunks` is therefore the ceiling for
every ranker. `laya` is `convaiinnovations/laya` multilingual, zero-shot; `laya-tuned` is the
same model after `finetune_laya.py` (see the [main README](../README.md#fine-tuning-laya)).
`typesafe` is Jev through the TypeSafe API, asked the same yes/no question with the same
criteria (`inventio/rankers.py`).

## Results

nDCG@10 (higher is better, 1.0 = every gold document at the top). Run on 2026-09-23.

| Benchmark | Published BM25 | Inventio BM25 | + Laya zero-shot | + Laya fine-tuned | + TypeSafe Jev | ceiling (gold in the 30 candidates) |
|---|---|---|---|---|---|---|
| SciFact (300 queries) | 0.665 | 0.670 | 0.302 | 0.684 | **0.765** | 0.849 |
| StackOverflow QA (1,994 queries) | 0.568 | 0.670 | 0.193 | 0.698 | **0.791** | 0.805 |
| SWE-bench Lite `code` (300 issues) | 0.430 | 0.540 | 0.391 | 0.602 | **0.696** | 0.823 |
| SWE-bench Lite `mixed` (300 issues) | | 0.400 | 0.264 | 0.426 | **0.515** | 0.633 |
| Zalo legal (788 queries) | | 0.756 | 0.512 | **0.817** | not run | 0.945 |

Zalo's BM25 searches adjacent syllables as phrases, which Inventio does for every Vietnamese
question; over single syllables it is 0.543, ceiling 0.815. Jev was not run on Zalo.

For SWE-bench, the share of issues where a file the fix touches is the first result, or in the
first five:

| Variant | BM25 top 1 | top 5 | + Laya fine-tuned top 1 | top 5 | + TypeSafe top 1 | top 5 |
|---|---|---|---|---|---|---|
| `code` | 38% | 64% | 44% | 68% | 56% | 78% |
| `mixed` | 24% | 48% | 28% | 50% | 40% | 58% |

Cost per query: BM25 35-140 ms; Laya 0.4-0.9 s on the laptop GPU; TypeSafe 1.0-1.3 s. Indexing
a repository at one commit takes 6 s (`code`) to 14 s (`mixed`) on average.

### Categories and fact links (`--arms`)

The pool of each query is built three ways: `base` (BM25 30), `facts` (base plus BM25's best
chunks of the query's predicted categories and the chunks linked to the top hits by judged
`about` links) and `control` (BM25 with as many candidates as `facts`). With `--mlt`, two more:
`about` (base plus the judged links only) and `mlt` (base plus the same link candidates,
unjudged: `search.widen_by_neighbours`, the code behind `--neighbours`). Each distinct chunk
is scored once per query, so the arms differ only in their pools. `facts_vs_control` and
`about_vs_mlt` are paired per-query comparisons with a bootstrap 95% interval. Results and
what they say are in the [main README](../README.md#do-content-categories-and-fact-links-help);
per-query rows are in `results/beir-<name>/arms.jsonl`, totals in its `summary.json` under
`arms`.

Jev calls: one per chunk for the eight categories, one per chunk for up to ten neighbours.
SciFact took 5,183 + 5,180 calls (51,792 pairs) in 335 s. StackOverflow QA took 26,941 + 26,517
calls (264,864 pairs); 77 chunks were refused by the API's edge firewall and keep only their
BM25 place. A run stopped halfway (for example HTTP 402, no credits left, which stops a run at
once) resumes from the stored judgments.

### Widening the pool by document type (SWE-bench Lite `mixed`)

Every file carries a document type decided by code from its path (`SoftwareSourceCode`,
`Test`, `Configuration`, `Article`). With `--types`, Jev is asked once per query which types
would hold the answer, and BM25's best 30 chunks inside each likely type join the pool. Three
arms on the same 300 issues, same index, same ranker:

| Arm | Candidates (mean) | Gold in the pool | BM25 nDCG@10 | + TypeSafe nDCG@10 | + TypeSafe top 1 | top 5 |
|---|---|---|---|---|---|---|
| `base`: BM25's 30 | 30 | 63% | 0.401 | 0.511 | 40% | 58% |
| `control`: BM25's top N, N as large as the widened pool | 56.5 | 72% | 0.402 | 0.560 | 43% | 64% |
| `types`: BM25's 30 + the predicted types' best | 56.5 | **81%** | 0.402 | **0.627** | **47%** | **71%** |

- The control is what makes this a finding: `types` and `control` hand the ranker the same
  number of candidates, and the typed pool holds the gold file for 30 issues the larger BM25
  pool misses (2 the other way). With TypeSafe, `types` beats `control` on 45 issues and
  loses on 17 (238 tie); mean nDCG@10 gain 0.067, 95% bootstrap interval 0.042 to 0.094.
- Without a ranker the widened chunks sit after BM25's, so only the pool ceiling moves; the
  gain arrives when a ranker can reorder them.
- The types that widened the pool: `SoftwareSourceCode` on 268 of 300 issues, `Article` on
  190. On a repository the typed pass lets code in past the docs and tests that crowd BM25's 30.
- `base` gives 0.511 here against 0.515 in the table above: this index was built with the
  `code` extra, so the repositories' non-Python files are cut by tree-sitter.
- Cost: one type question per query (about 0.8 s over the network) plus scoring 26 more
  candidates (2.2 s per query in total against 1.0 s for `control`).
- The type question went to Jev in every arm, including the BM25 rows. Laya as the type
  predictor was not measured.

Results: `results/swe-lite-types/` (one line per issue, ranker and arm, with Jev's type
probabilities and the types that widened the pool).

### Names in the question, and widening decided by code (SWE-bench Lite `mixed`, `--strat`)

No model decides these pools. `sym` adds the files whose path the issue names and the chunks
that define an identifier it names (a name defined in more than three places is skipped);
`code` adds BM25's best chunks among source files; `all` and `under` do the same for every
type, or for the types rarer in BM25's 30 than in the repository. Each arm has a BM25 control
with as many candidates. Ranked by the fine-tuned Laya, 300 issues:

| Arm | Candidates (mean) | Gold in the pool | control: gold in the pool | nDCG@10 | vs control (95% CI) |
|---|---|---|---|---|---|
| `base`: BM25's 30 | 30 | 63% | | 0.426 | |
| `sym`: + named files and definitions | 33.1 | 73% | 64% | 0.495 | +0.065 (+0.037, +0.097) |
| `code`: + best source chunks | 48.2 | 83% | 70% | 0.496 | +0.053 (+0.033, +0.074) |
| `code+sym` | 50.9 | **86%** | 71% | **0.532** | +0.088 (+0.059, +0.119) |
| `under`: + rarer types | 90.3 | 82% | 78% | 0.437 | −0.017 (−0.039, +0.004) |
| `all`: + every type | 109.4 | 83% | 81% | 0.437 | −0.025 (−0.050, −0.001) |

- Names are the cheapest gain: three more candidates, ten more issues in a hundred with the file
  to fix in the pool. That is why the lookup is on by default in `inventio query`.
- With no ranker, putting the named files ahead of BM25's order (`symfirst`) gives 0.456
  against 0.401 (+0.056, 95% CI +0.015 to +0.098), top 5 from 48% to 58%. `inventio query`
  without a ranker does this.
- `code` wins because an issue tracker's answer is always source code; it is what `--types`
  would do if it always predicted that type, a property of this benchmark rather than a
  default. Widening by every type drowns the ranker.

Results: `results/swe-lite-strat/`.

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
- Zero-shot Laya makes every ranking worse, by a lot. Fine-tuned on the benchmarks'
  human-labelled train splits it is ahead of BM25 on every test set: clearly on Zalo (+0.061,
  95% CI +0.042 to +0.082), StackOverflow QA (+0.028, +0.015 to +0.042) and SWE-bench `code`
  (+0.062, +0.025 to +0.098), which it never trained on; within the noise on SciFact and
  SWE-bench `mixed`.
- A whole repository is harder than its code: tests and docs push the fix's files out of the 30
  candidates (ceiling 0.823 falls to 0.633). The ranker cannot recover what BM25 did not hand
  it, so on mixed repositories the candidate pool, not the ranker, is the limit to work on.
- TypeSafe's firewall refused 409 of 59,820 pairs on StackOverflow QA and 113 on SWE-bench;
  they kept their BM25 place.

## Caveats

- Every configuration was run on every test query (SciFact 300, StackOverflow QA 1,994,
  SWE-bench Lite 300). An earlier StackOverflow QA run on the first 300 queries gave higher
  numbers (0.713 / 0.837); those 300 are easier than the full set.
- Laya reads at most 1,024 tokens of query plus passage. SWE-bench issues are often longer than
  that on their own, so Laya sees a truncated question there.
- TypeSafe's edge firewall refuses some texts outright (HTTP 403 "Attention Required"; it
  happens on Django source). Such a pair stays unscored and keeps its BM25 place after the
  scored ones; the count is reported with the results.
- Time per query is measured only over queries scored fresh, on one RTX 5070 laptop GPU for
  Laya and over the network for TypeSafe (12 parallel requests).
