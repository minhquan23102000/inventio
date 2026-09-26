# Status

Where the work stands, for picking it up on another machine. Last updated 2026-09-26, on branch
`dispositio-small` (not pushed, nothing published).

## Branch `dispositio-small`

| Commit | Change |
|---|---|
| `f0e1d71` | `inventio update`; a query says once a day when a newer dispositio is released (model name only; `INVENTIO_OFFLINE=1` or `HF_HUB_OFFLINE=1` turns it off) |
| `744ad77` | Skill: one more try with `--pool 30` when rephrasing finds nothing; relay the update notice |
| (next) | `finetune_laya.py --student/--teacher` (distillation), `instr` and `synth` sources; `probe_model.py`; `synth_data.py` |

Checkpoints in `%LOCALAPPDATA%\inventio\`: `dispositio-small` (step 1), `dispositio-small-mt2` (step 3).

### Step 1: dispositio v2 distilled into mmBERT-small

Teacher dispositio v2; targets 0.5 teacher + 0.5 written label; 3 epochs, 56 min on an RTX 5070.
nDCG@10 on every BEIR test query, pool 30, same path as the README (`beir_bench.py`):

| Set | BM25 | v2 | small | small / v2 |
|---|---|---|---|---|
| SciFact | 0.670 | 0.737 | 0.733 | 0.99 |
| StackOverflow QA | 0.670 | 0.676 | 0.691 | 1.02 |
| Zalo legal | 0.756 | 0.820 | 0.838 | 1.02 |
| MultiDoc2Dial | 0.470 | 0.638 | 0.622 | 0.98 |
| TechQA | 0.370 | 0.405 | 0.416 | 1.03 |

- Ranker only, 15 candidates, 100 queries on the 5070: 293 ms (v2) against 154 ms (small).
  Weights 615 MB against 276 MB.
- Not measured: the M3, and the 40 private questions (both on the Mac).
- The cached `scores-dispositio.jsonl` is not v2: v2 scored fresh gives 0.737 on SciFact against
  the cache's 0.728, 0.676 on StackOverflow QA against 0.590.

### Step 3: reading the question, counterfactual passages

`benchmarks/probe_model.py`, on pairs and question wordings no model trained on:

| Probe (pass mark) | v2 | small | small-mt2 |
|---|---|---|---|
| A relevance AUC (>= 0.88) | 0.882 | 0.911 | 0.914 |
| corr(A, "does it fail to answer?") (< 0) | +0.99 | +1.00 | -0.72 |
| "is it in Vietnamese?" AUC (>= 0.9) | 0.45 | 0.41 | 0.94 |
| "is it in English?" AUC (>= 0.9) | 0.59 | 0.60 | 0.97 |
| answer units deleted: falls below 0.5 (>= 0.7) | 0.25 | 0.28 | 0.68 |
| query words paraphrased away: mean change (<= 0.15) | 0.28 | 0.35 | 0.05 |
| paraphrased answers vs non-answers AUC (>= 0.85) | 0.74 | 0.68 | 0.99 |

- A first run (`-mt`) with two to four fixed wordings per question passed the probes in those
  wordings and failed new ones ("is it in English?" AUC 0.03). `-mt2` trains on 24 generated
  wordings per question and its opposite, one in four held out.
- Still leans on the criteria text: "in English? / false: another language" gives 0.28;
  "false: in Vietnamese" gives 0.98.
- Relevance against step 1: SciFact 0.730 (-0.4%), StackOverflow 0.690, Zalo 0.831 (-0.8%),
  MultiDoc2Dial 0.609 (-2.2%), TechQA 0.402 (-3.3%, 119 queries).
- memoria's 100 labelled messages (local, never trained on), hit@2 of 36: BM25 15, v2 17-18,
  small 14-15, mt2 12-13.

Smol (Gemini Flash) calls: about 1,000, in batches of 20.

## Recent changes on `main`

| Commit | Change |
|---|---|
| `b55f28c` | Confluence connector: `_Page.list` shadowed the builtin `list` and crashed on Python below 3.14; renamed `list_items` |
| `06f00fc` | `rankers.py`: load the Hugging Face snapshot with `local_files_only` (skips about 0.44 s of network per query); `MPS_BATCH = 4` on Apple GPUs (3.5 s against 4.9 s for 16 on an M3); results unchanged |
| `cc7662d` | `serve.py`: background server that keeps dispositio loaded, started by the first query, 127.0.0.1 only, token in `serve.json`, exits after `INVENTIO_SERVE_IDLE` (900 s); `INVENTIO_SERVE=0` turns it off; `inventio serve --stop` |
| `727cda6` | Skill: when question and documents are in different languages, translate the whole question and ask again |
| `b75f2b5` | `query --pool` defaults to 15 instead of 30 |

## Measured on a private map (Mac M3, 16 GB)

A wiki space (1,180 pages), a ticket project (1,396 tickets) and two code repositories (1,232
files). 40 questions: 12 written by hand, 28 taken from real links (a ticket's title as the
question, the page or file it links to as the answer). Question set and scripts stay off this
repository because they name internal documents.

| Configuration | Top 1 | Top 10 | MRR@10 | Median per query |
|---|---|---|---|---|
| BM25 | 16/40 | 21/40 | 0.435 | 0.08 s |
| dispositio, pool 30 | 18/40 | 23/40 | 0.483 | 2.70 s |
| dispositio, pool 15 (default now) | 19/40 | 24/40 | 0.508 | 1.47 s |
| dispositio, pool 10 | 16/40 | 20/40 | 0.429 | 0.88 s |

- Served query: first one about 12 s (starts the server), then 1-4 s. Without the server, 9-10 s.
- Shorter reading windows (512 tokens instead of 1024) saved almost nothing: most chunks are
  shorter than 512 tokens. Time follows the number of candidates.
- Tried and slower on MPS: autocast fp16, bf16 (also changes the top 5), CPU. `.half()` crashes.
- Of 322M parameters, 197M are the embedding table; the rest runs at about 1.5 TFLOPS on the M3
  GPU. Core ML on the Neural Engine is the remaining hardware path; not tried.

## Open

- **Public benchmarks not rerun with pool 15.** The README numbers are at `--pool 30`. Risk is
  highest where BM25 often ranks the answer 16-30 (StackOverflow QA, SWE-bench `mixed`).
- **Cross-language questions** are handled only in the skill. A person typing a question in
  one language about documents in another still gets nothing from BM25.
- **Right file, wrong section**: on some questions dispositio picks the section next to the
  answer.
- **No link from a page to code when a table name appears only inside SQL strings**: `mentions`
  links need a definition in the map. Schema cards might fill this; untested.
- **`mirror_dir()` ignores `--db`** and writes into the real data directory, so
  `tests/test_connectors.py::test_where_scopes_every_term_before_bm25` fails from the second run
  on. Undecided: mirrors follow `--db`, or isolate them in tests only. Until then, delete
  `<data dir>/mirrors/notes` after a test run.
- Smaller: printed ranks can appear out of order (results are grouped by document type); the
  first query is slow without warning; `--db` after the subcommand gives a generic argparse error.
- Untested: the server on Windows, and two first queries starting it at once.

## Next

- Decide: release small (step 1) as v3 after the M3 and 40-question check; step 3 misses two marks
  (answer deletion 0.68 < 0.7; MultiDoc2Dial and TechQA below step 1).
- memoria: no model beats v2 there; reading instructions is not yet judging what to recall.
- Try other checkpoints: `INVENTIO_DISPOSITIO_MODEL=<dir or hf id> inventio query ...`; compare on
  the same question set.
- Decide the mirror isolation fix.

## Commands

```sh
uv tool install -q --reinstall-package inventio "$PWD[laya]"     # install from this checkout
uv run -q --python 3.12 --with pytest --with-editable ".[laya]" pytest -q tests
inventio serve --stop                                          # after changing the model
```
