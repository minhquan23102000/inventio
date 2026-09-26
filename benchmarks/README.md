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

# dispositio is the released model (INVENTIO_DISPOSITIO_MODEL for another checkpoint; results go
# under its last path part), laya is Laya multilingual as published
python benchmarks/beir_bench.py scifact --rankers none,laya,dispositio,typesafe
python benchmarks/beir_bench.py coir-stackoverflow-qa --rankers none,laya,dispositio,typesafe
python benchmarks/swe_bench.py --rankers none,laya,dispositio,typesafe
python benchmarks/data.py zalo && python benchmarks/beir_bench.py zalo-legal --rankers none,laya,dispositio
python benchmarks/data.py multidoc2dial && python benchmarks/beir_bench.py multidoc2dial --rankers none,laya,dispositio,typesafe
python benchmarks/data.py techqa && python benchmarks/beir_bench.py techqa --rankers none,laya,dispositio,typesafe
python benchmarks/example_bench.py none laya dispositio typesafe    # examples/webshop, 13 on-call questions
python benchmarks/swe_bench.py --types --variants mixed --rankers none,typesafe

# categories and fact links: Jev judges every chunk and candidate pair, then the arms
python benchmarks/beir_bench.py scifact --arms --mlt --rankers none,typesafe,dispositio
python benchmarks/beir_bench.py coir-stackoverflow-qa --arms --mlt --rankers none,typesafe,dispositio
python benchmarks/swe_bench.py --strat --variants mixed --rankers none,dispositio   # names and code-decided widening
python benchmarks/pack_check.py                    # packed neighbour call vs one call per pair

# dispositio: SWE-bench train groups, category passages (labelled by a small LLM), two stages
python benchmarks/swe_train.py --per-repo 150 --workers 12
python benchmarks/category_data.py
python benchmarks/finetune_laya.py --out <stage 1>
python benchmarks/finetune_laya.py --init <stage 1> --out <stage 2> \
    --mix coir-stackoverflow-qa=16000,multidoc2dial=4000,swe=3000,scifact=2000,zalo-legal=2000,category=2000
```

Data goes to `<user cache>/inventio/bench` (`--data` or `INVENTIO_BENCH_DATA` to move it), never
into this repository. Results go to `benchmarks/results/`: a `summary.json` per benchmark, and
for SWE-bench one line per (instance, variant, ranker) with the top-10 files it returned.
Ranker scores are cached (ignored by git), so an interrupted run resumes and a rerun is free.

Every ranker reorders the same 30 BM25 candidates (chunks); chunks are then collapsed to
documents or files, first occurrence wins. `recall@30chunks` is therefore the ceiling for
every ranker. `dispositio` is [dispositio](https://huggingface.co/minhquan2310/dispositio), Laya
fine-tuned for Inventio; `laya` is Laya multilingual as published.
`typesafe` is Jev through the TypeSafe API, asked the same yes/no question with the same
criteria (`inventio/rankers.py`).

## Results

nDCG@10 (higher is better, 1.0 = every gold document at the top). Run on 2026-09-24.

| Benchmark | Published BM25 | Inventio BM25 | + Laya as published | + dispositio | + TypeSafe Jev | ceiling (gold in the 30 candidates) |
|---|---|---|---|---|---|---|
| SciFact (300 queries) | 0.665 | 0.670 | 0.302 | 0.728 | **0.765** | 0.849 |
| StackOverflow QA (1,994 queries) | 0.568 | 0.670 | 0.193 | 0.590 | **0.791** | 0.805 |
| SWE-bench Lite `code` (300 issues) | 0.430 | 0.540 | 0.391 | 0.661 | **0.696** | 0.823 |
| SWE-bench Lite `mixed` (300 issues) | | 0.400 | 0.264 | 0.486 | **0.515** | 0.633 |
| Zalo legal (788 queries) | | 0.756 | 0.512 | **0.830** | not run | 0.945 |
| MultiDoc2Dial (615 queries) | | 0.470 | 0.389 | **0.643** | 0.486 | 0.811 |
| TechQA (119 queries) | | 0.370 | 0.171 | 0.444 | **0.655** | 0.815 |

Zalo's BM25 searches adjacent syllables as phrases, which Inventio does for every Vietnamese
question; over single syllables it is 0.543, ceiling 0.815. Jev was not run on Zalo.
MultiDoc2Dial and TechQA are built by `data.py`: one document per section of a page (titled by
the page and its headings) and one per passage of a technote (titled by the technote), so both
ask for the part of a document that answers, not the document.

dispositio against BM25, per query, paired bootstrap 95% interval: MultiDoc2Dial +0.172
(+0.146, +0.199), SWE-bench `code` +0.121 (+0.080, +0.159), `mixed` +0.086 (+0.055, +0.118),
Zalo +0.074 (+0.053, +0.094), TechQA +0.074 (+0.024, +0.124), SciFact +0.058 (+0.026, +0.089),
StackOverflow QA −0.079 (−0.095, −0.064). Jev against dispositio: +0.035 (+0.006, +0.064) on
`code`, +0.029 (+0.001, +0.056) on `mixed`, +0.037 on SciFact, +0.201 on StackOverflow QA,
+0.211 on TechQA, −0.157 on MultiDoc2Dial.

StackOverflow QA's train and test answers share one corpus, and 70% of the wrong candidates a
test question sees are answers to train questions, which dispositio learned as answers. It
ranks them too high: p > 0.5 for 3.1% of them against 0.6% of other documents, and 90% of its
wrong first results are one of them (77% for the first release, the base rate being 70%). With
the train answers removed from every question's candidates (the gold kept), BM25 is 0.729,
dispositio 0.752 (+0.024, +0.014 to +0.033), the first release 0.765, Jev 0.799. The model card
reports both.

For SWE-bench, the share of issues where a file the fix touches is the first result, or in the
first five:

| Variant | BM25 top 1 | top 5 | + dispositio top 1 | top 5 | + TypeSafe top 1 | top 5 |
|---|---|---|---|---|---|---|
| `code` | 38% | 64% | 51% | 75% | 56% | 78% |
| `mixed` | 24% | 48% | 34% | 58% | 40% | 58% |

Cost per query: BM25 35-140 ms; dispositio 0.4-1.2 s on the laptop GPU (a long SWE-bench issue
has more windows to read); TypeSafe 1.0-1.3 s. Indexing a repository at one commit takes 6 s
(`code`) to 14 s (`mixed`) on average.

### Training dispositio

Relevance labels come from people: the MultiDoc2Dial, StackOverflow QA, SciFact and Zalo train
splits, and 3,923 SWE-bench train issues (35 repositories, none in Lite) with the chunks their
merged fix changed as answers. Category labels come from a small LLM, since no human set
exists. MultiDoc2Dial's student aid domain is held out of training whole. The released model is
two stages from Laya as published: stage 1 (55,786 items, MultiDoc2Dial first) and stage 2
(28,513 items, StackOverflow QA first). The runs that did not ship say what the data teaches:

| Run | MultiDoc2Dial, student aid (held out) | StackOverflow QA | SWE `code` | webshop questions first |
|---|---|---|---|---|
| first release: relevance fine-tune, then SWE-bench and categories | 0.569 | **0.699** | 0.655 | 8 / 13 |
| + MultiDoc2Dial, continued from the first release | 0.653 | | | |
| stage 1 only: from Laya, no title items | 0.707 | 0.649 | 0.646 | |
| stage 1 + stage 2 (released) | **0.718** | 0.590 | **0.661** | **9 / 13** |

- **Title items** (a document's title as the query, its first chunk as the answer) were 65% of
  the first release's lineage. They teach that the passage repeating the question's words is
  the answer: the first release put a section restating the question above the section that
  answered it. Continuing from it kept the habit (the second row, trained without titles,
  still fell on TechQA from 0.439 to 0.405), so stage 1 starts again from Laya without them.
- **Same-page negatives.** A MultiDoc2Dial question grounded in one section also gets that
  page's other section BM25 ranks highest as a negative, at 0.25: the passage that shares the
  topic and the words and does not answer. On the held-out domain the answer then beats every
  other section of its page for 80% of questions, against 57%.
- **SWE-bench negatives are source code only.** With docs as negatives the model learned that
  an issue-shaped question wants code, and StackOverflow QA fell to 0.612. A patch says which
  code was changed, not that a doc is a wrong answer.
- **Stage 2** is there because after stage 1 StackOverflow QA was below BM25. On the held-out
  StackOverflow questions it rose from 0.873 to 0.931; on the test it fell further, the
  memorisation described under Results.

Results, data and terms: the [model card](https://huggingface.co/minhquan2310/dispositio). The
first release stays on Hugging Face as the revision `v1`.

### Categories and fact links (`--arms`)

The pool of each query is built three ways: `base` (BM25 30), `facts` (base plus BM25's best
chunks of the query's predicted categories and the chunks linked to the top hits by judged
`about` links) and `control` (BM25 with as many candidates as `facts`). With `--mlt`, two more:
`about` (base plus the judged links only) and `mlt` (base plus the same link candidates,
unjudged: `search.widen_by_neighbours`, on by default with a ranker, `--no-neighbours`). Each distinct chunk
is scored once per query, so the arms differ only in their pools. `facts_vs_control` and
`about_vs_mlt` are paired per-query comparisons with a bootstrap 95% interval. Per-query rows
are in `results/beir-<name>/arms.jsonl`, totals in its `summary.json` under `arms`.

Measured with the earlier category set (eight schema.org types, one yes/no question each,
judged by Jev), before the categories were redefined by what a passage does for its reader;
"earlier dispositio" is the relevance-only checkpoint dispositio was trained from.

| Benchmark | Ranker | BM25 30 | + facts | BM25, same size | facts vs same size (95% CI) |
|---|---|---|---|---|---|
| SciFact | Jev | 0.765 | **0.771** | 0.769 | +0.002 (−0.003, +0.007) |
| SciFact | earlier dispositio | **0.684** | 0.677 | 0.675 | +0.001 (−0.003, +0.006) |
| SciFact | Laya as published | **0.302** | 0.249 | 0.252 | −0.004 (−0.012, +0.005) |
| StackOverflow QA | Jev | 0.791 | **0.811** | 0.806 | +0.005 (−0.001, +0.011) |
| StackOverflow QA | earlier dispositio | **0.698** | 0.676 | 0.697 | −0.021 (−0.028, −0.015) |
| StackOverflow QA | Laya as published | **0.193** | 0.157 | 0.156 | +0.001 (−0.002, +0.004) |

Answer among the candidates: SciFact 84.9% (BM25 30), **87.6%** (+ facts), 86.3% (same size);
StackOverflow QA 80.5%, **83.1%**, 82.4%. Pools grow from 30 to about 47 candidates.

- Categories and links find answers BM25 missed, more than as many extra BM25 candidates do.
- Ranked by Jev the gain over a same-size pool is within the noise: a bigger pool, not a
  better one. That set gave almost every chunk the same types (`Intangible` on 90-99% of
  chunks), so its category pool was close to more BM25; the new set partitions (the largest
  category holds 31-79% on four unseen collections) and has yet to be measured here.

Judged links against the same candidates unjudged (the default neighbours):

| Benchmark | Ranker | about links (judged) | neighbours (unjudged) | neighbours − about (95% CI) |
|---|---|---|---|---|
| SciFact | Jev | 0.769 | **0.785** | +0.015 (+0.004, +0.030) |
| SciFact | earlier dispositio | 0.681 | **0.690** | +0.009 (−0.001, +0.023) |
| StackOverflow QA | Jev | **0.803** | 0.796 | −0.007 (−0.012, −0.002) |
| StackOverflow QA | earlier dispositio | 0.677 | **0.686** | +0.009 (+0.004, +0.015) |

In three of the four cells the ranker does better among the raw neighbours than among the ones
Jev kept, so the judgment is not paying for itself yet.

Jev calls then: one per chunk for the eight types, one per chunk for up to ten neighbours.
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
with as many candidates. Ranked by the earlier dispositio checkpoint, 300 issues:

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
- TypeSafe Jev reordering the same 30 candidates adds 0.10 to 0.16 on SciFact, StackOverflow QA
  and SWE-bench, and on SWE-bench `code` turns 38% first-hit into 56%. On MultiDoc2Dial it adds
  only 0.016: telling the section that answers from its page's other sections is not what a
  general judge does best.
- Laya as published makes every ranking worse, by a lot. dispositio, fine-tuned on human
  relevance labels, is ahead of BM25 with the interval above zero on every test set but
  StackOverflow QA as published (see Results for why), including SWE-bench Lite, whose 12
  repositories are not among the 35 it trained on.
- A whole repository is harder than its code: tests and docs push the fix's files out of the 30
  candidates (ceiling 0.823 falls to 0.633). The ranker cannot recover what BM25 did not hand
  it, so on mixed repositories the candidate pool, not the ranker, is the limit to work on.
- TypeSafe's firewall refused 409 of 59,820 pairs on StackOverflow QA and 113 on SWE-bench;
  they kept their BM25 place.

## Caveats

- Every configuration was run on every test query (SciFact 300, StackOverflow QA 1,994,
  SWE-bench Lite 300). An earlier StackOverflow QA run on the first 300 queries gave higher
  numbers (0.713 / 0.837); those 300 are easier than the full set.
- Laya reads at most 1,024 tokens of query plus passage. The query is cut to its first 384
  tokens (15% of SWE-bench Lite issues are longer); a longer passage is read in windows of
  whole lines, each with its `[path > heading]` line, and scored by its best window.
- TypeSafe's edge firewall refuses some texts outright (HTTP 403 "Attention Required"; it
  happens on Django source). Such a pair stays unscored and keeps its BM25 place after the
  scored ones; the count is reported with the results.
- Time per query is measured only over queries scored fresh, on one RTX 5070 laptop GPU for
  Laya and over the network for TypeSafe (12 parallel requests).
