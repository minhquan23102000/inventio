# Inventio

> *Inventio*, from *invenire*: to come upon. The first of the five canons of rhetoric was not
> invention in our sense. The orator did not make his material up; he went looking through
> the *loci*, the places where it already lay, and found it.

That is the rule this tool is built on: the structure of a body of knowledge is found in the
sources (their folders, headings, definitions, the names they share), not made up by a model.

Local retrieval without embeddings. `inventio` indexes your repositories and documents into
one SQLite file, finds candidates with BM25, and lets a decision model reorder the short list:
[Laya](https://github.com/NandhaKishorM/laya) on your own GPU, or
[TypeSafe Jev](https://docs.typesafe.ai) in the cloud for sources you have marked public.
Every answer comes back as the original passage with its coordinates (`path:start-end`), so a
person or an agent can open the exact lines.

There is no vector store and no generation step. The "R" of RAG lives here; the "G" is whoever
calls it.

## Install

```sh
pip install -e .                 # core: standard library only (sqlite3 with FTS5)
pip install -e ".[laya]"         # local ranker (torch + transformers)
pip install -e ".[typesafe]"     # cloud ranker; needs TYPESAFE_API_KEY
```

## Use

```sh
inventio init ~/code/fraud-rules --name rules
inventio init ~/notes/wiki --name wiki --public --exclude "drafts/*"
inventio sources

inventio query "which job recomputes customer risk overnight?"
inventio query "..." --ranker laya              # local GPU/CPU
inventio query "..." --ranker typesafe --source wiki
inventio query "..." --json                     # for agents
```

Each result shows where it lives and where it leads:

```
1. wiki:runbook.md:1-2  Daily scoring
   The nightly job `fraud_score_daily` recomputes customer risk before the morning review.
   -> mentions rules:jobs/score.py:1-2  (fraud_score_daily)
```

Re-running `init` on a source updates it in place: files whose size and modification time
match the map are skipped without being read, files with the same content hash are only
re-stamped, and only new or edited files are chunked again (deleted ones leave the map). On
astropy (1,260 files, 22,327 chunks) the first index takes 12.8 s and a re-run after editing
one file 1.4 s, most of it rebuilding links across the whole map. `--full` rebuilds from scratch.

The map lives in your user cache (`%LOCALAPPDATA%\inventio\map.db`, or `$XDG_CACHE_HOME`,
or `~/.cache`), never inside an indexed repository. Override it with `--db` or `INVENTIO_DB`.
Set a default ranker with `INVENTIO_RANKER`.

## How the map is built

Everything the source already knows is read by code, not guessed by a model.

1. **Structure.** Markdown is cut at headings (fenced code is respected), Python at top-level
   functions and classes (large classes at their methods), anything else at blank-line blocks
   of about 1,500 characters. The file path and heading path of each chunk are indexed next
   to its text, so BM25 can match on where a passage sits as well as on what it says.
2. **Links.** Drawn by code, named with schema.org's `CreativeWork` vocabulary:
   - `citation`: a Markdown link to a file or a heading, resolved to the chunk it points at.
   - `mentions`: a chunk names an identifier another chunk defines (a Python function or
     class), or chunks in different files share a rare identifier-shaped token
     (`fraud_score_daily`, `risk.daily.score`, `FRAML-123`). This is the bridge between a
     repository and the prose written about it, and it works across sources.
3. **Ranking.** BM25 (SQLite FTS5, diacritics folded) picks 30 candidates; the ranker asks one
   yes/no question per candidate with explicit criteria ("states the specific answer" versus
   "only on a related topic") and sorts by the probability.

## Benchmarks

Three public retrieval benchmarks, run through Inventio's real ingest and query path, on every
query of each test set. nDCG@10: 1.0 means every right answer sits at the top.

| Benchmark | Inventio, no model | + TypeSafe Jev | + Laya, not yet fine-tuned | Published retrievers, same test set |
|---|---|---|---|---|
| [SWE-bench Lite](https://arxiv.org/abs/2406.14497): GitHub issue → code files to fix (300) | 0.540 | **0.696** | 0.391 | SFR-Mistral 7B 0.627 · Jina-v2-code 0.583 · GIST-large 0.478 · OpenAI embedding-3-small 0.433 |
| [SciFact](https://arxiv.org/abs/2104.08663): claim → scientific abstract (300) | 0.670 | **0.765** | 0.302 | e5-mistral-7b 0.764 · bge-large-v1.5 0.746 · ColBERT 0.671 |
| [StackOverflow QA](https://arxiv.org/abs/2407.02883): question → answer, prose + code (1,994) | 0.670 | 0.791 | 0.193 | E5-Mistral 7B 0.915 · Voyage-Code-002 0.877 · E5-base 0.869 · OpenAI Ada-002 0.724 |

Per query: 35-140 ms with no model (CPU, standard library), about 1.2 s with TypeSafe,
0.6-0.9 s with Laya on a laptop RTX 5070.

- **Inventio alone**, with no model and no embeddings, is above every embedder but the two
  strongest on SWE-bench Lite. The likely reason, not yet isolated by an ablation: chunks
  follow functions and carry their file path and name. On text it is level with ColBERT on
  SciFact and stays below the modern embedders, on SciFact and StackOverflow QA alike.
- **With TypeSafe Jev** reordering Inventio's 30 candidates, it has the best score in the
  SWE-bench Lite comparison (the file to fix comes first for 56% of issues, in the top 5 for
  78%), matches a 7B embedder on SciFact, and gains 0.12 on StackOverflow QA but stays under
  the strongest embedders there: the answer is among the 30 candidates for only 80% of those
  questions, so no reordering of them can pass 0.805. The candidate pool is the next limit.
- **Laya** is a small decision model that runs on your own GPU, so private sources never
  leave the machine. Out of the box it ranks worse than no model at all; its author calls it
  "a fast base to specialise". It is meant to learn from Jev: every TypeSafe ranking on a
  public source is stored as a (query, passage, probability) label, and those labels are the
  teacher signal for fine-tuning Laya ([below](#fine-tuning-laya-from-jev)). The fine-tuned
  student has not been measured yet; this column is its starting point.
- To read the comparison fairly: the published figures are single-stage embedders over the
  whole corpus (from the BEIR, CoIR and CodeRAG-Bench papers and the models' MTEB cards),
  while Inventio + TypeSafe is two-stage. The two-stage figure in the BEIR paper, a
  cross-encoder reranking the top 100, is 0.688 on SciFact.

Method, the whole-repository variant of SWE-bench, caveats and one-command reproduction:
[benchmarks/README.md](benchmarks/README.md).

## Measured on an own corpus

40 questions over 2,132 chunks of Markdown (126 files: design maps, agent skills, logs, in
Vietnamese and English). Each question was written by a small LLM from one passage, picked at
random from 40 different files; the answer is that passage. Questions and raw results:
[docs/design/evidence/](docs/design/evidence/). The bench file format is below.

| Configuration | answer in top 1 | top 5 | top 10 | time per query |
|---|---|---|---|---|
| BM25 on text only | 23/40 | 30/40 | 32/40 | 5 ms |
| BM25 + file and heading path (default) | 24/40 | 32/40 | 35/40 | 5 ms |
| + Laya multilingual, zero-shot | 12/40 | 28/40 | 30/40 | 0.6 s (RTX 5070 laptop) |
| + TypeSafe Jev | **32/40** | **36/40** | **37/40** | 1.1 s |
| + TypeSafe Jev + link expansion | 30/40 | 36/40 | 37/40 | 1.1 s |

What this does and does not say:

- BM25 put the answer among its 30 candidates for 38 of 40 questions. That is the ceiling for
  every ranker; the two misses need a different wording or a path BM25 does not see.
- Zero-shot Laya orders the short list worse than BM25 alone. Its author describes it as "a
  fast base to specialise, not a zero-shot decision engine", and this agrees. It needs
  fine-tuning (below) before it earns its place.
- Widening the candidate pool through links did not bring in a single missing answer here and
  cost two top-1 hits, so it is off by default (`--links` turns it on). Links still appear in
  every result, for navigation. A corpus where code and prose name the same identifiers, such
  as a repository next to its wiki, is where this should be measured again.
- The questions were generated from the answer passages, which favours word overlap; 40 is a
  small sample. Treat these as directions, not as a benchmark.

Reproduce with your own questions:

```sh
inventio bench my-questions.jsonl --ranker none
inventio bench my-questions.jsonl --ranker typesafe --links --json
```

One question per line; a result counts when it overlaps the lines you name:

```json
{"question": "which job recomputes customer risk overnight?", "source": "wiki", "path": "runbook.md", "start_line": 1, "end_line": 2}
```

## Privacy

- Every source is private unless you pass `--public` to `init`.
- `--ranker typesafe` refuses (exit code 3) when any candidate comes from a private source, and
  sends nothing. Narrow the query with `--source`, or rank locally with `--ranker laya`.
- The map file contains source text. It is ignored by this repository's `.gitignore` and is
  written outside the indexed trees; keep it that way.

## Fine-tuning Laya from Jev

Jev is the teacher and Laya the student. Jev judges well but runs in the cloud, so it may only
see public sources; Laya runs on your machine but has to learn the judgement first. Every
TypeSafe ranking stores its (query, passage, probability) triples in the map. Export them:

```sh
inventio labels jev-labels.jsonl
```

These come only from public sources, by construction. Fine-tune Laya on them (see the Laya
README's fine-tuning section). The student then ranks private sources, which the teacher never
sees. Not wired yet: `--ranker laya` always loads the published checkpoint, so a tuned one
cannot be benchmarked through Inventio until it can be selected.

## Not yet

- Connectors beyond the local file system (Confluence, Slack, mail).
- Semantic categories generated at index time. Small local generators and zero-shot
  classifiers were measured and none beat the structure the source already has.

## License

Apache-2.0.
