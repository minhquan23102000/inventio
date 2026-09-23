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

Public retrieval benchmarks, run through Inventio's real ingest and query path. nDCG@10, higher
is better; every ranker reorders the same 30 BM25 candidates.

| Benchmark | Published BM25 | Inventio BM25 | + Laya zero-shot | + TypeSafe Jev |
|---|---|---|---|---|
| [SciFact](https://arxiv.org/abs/2104.08663) (text, 300 queries) | 0.665 | 0.670 | 0.302 | **0.765** |
| [StackOverflow QA](https://arxiv.org/abs/2407.02883) (prose + code, 300 queries) | 0.568¹ | 0.713 | 0.203 | **0.837** |
| [SWE-bench Lite](https://arxiv.org/abs/2406.14497), issue → files to fix, code only | 0.430 | 0.540 | 0.391 | **0.696** |
| SWE-bench Lite, whole repository (code, tests, docs) | | 0.400 | 0.264 | **0.515** |

¹ On all 1,994 queries; Inventio BM25 scores 0.670 there.

- The BM25 baseline matches the published one on SciFact, so the measurement is not flattering
  itself. On repositories Inventio's BM25 is 0.11 above the published one, and above most
  embedding models in the CodeRAG-Bench table; the likely reason, not yet isolated by an
  ablation, is chunking by function and indexing each chunk's file path and function name.
- TypeSafe reranking adds 0.10 to 0.16 everywhere. On SWE-bench Lite (code only) the file to fix
  is the first result for 56% of issues (38% with BM25 alone) and in the top 5 for 78% (64%).
- Zero-shot Laya lowers every score; it needs fine-tuning first.
- On a whole repository, tests and docs push the right files out of the 30 candidates (82% of
  issues keep one in the pool with code only, 63% with everything). That pool is the next limit.

Method, caveats, and one-command reproduction: [benchmarks/README.md](benchmarks/README.md).

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

Every TypeSafe ranking stores its (query, passage, probability) triples in the map. Export them:

```sh
inventio labels jev-labels.jsonl
```

These come only from public sources, by construction. They are teacher labels for fine-tuning
Laya on your own retrieval task (see the Laya README's fine-tuning section); `bench` then
measures the tuned model against the same questions.

## Not yet

- Connectors beyond the local file system (Confluence, Slack, mail).
- Semantic categories generated at index time. Small local generators and zero-shot
  classifiers were measured and none beat the structure the source already has.

## License

Apache-2.0.
