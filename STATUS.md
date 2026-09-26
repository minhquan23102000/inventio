# Status

Where the work stands, for picking it up on another machine. Last updated 2026-09-26, on `main`
(the former `dispositio-small` branch merged at `8b44f14`; nothing pushed, nothing published).

## Training the small ranker

| Commit | Change |
|---|---|
| `2dacd47` | Cached build/teacher scores, `torch.compile`, the link probe; the ablation below |
| `f0e1d71` | `inventio update`; a query says once a day when a newer dispositio is released (model name only; `INVENTIO_OFFLINE=1` or `HF_HUB_OFFLINE=1` turns it off) |
| `744ad77` | Skill: one more try with `--pool 30` when rephrasing finds nothing; relay the update notice |
| `4b4b7c7` | `finetune_laya.py --student/--teacher` (distillation), `instr` and `synth` sources; `probe_model.py`; `synth_data.py` |

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

### A run's speed, and the ablation behind step 3's numbers

| Change | Effect on an RTX 5070 laptop |
|---|---|
| `torch.compile` on the encoder's layers (triton-windows) | 53 -> 67 items/s; a 2-epoch arm trains in 29 min instead of 56 |
| Data and teacher scores cached under `%LOCALAPPDATA%\inventio\train-cache` (1.4 GB) | a second run of the same mix reads them back: 61 s -> 30 s, identical `mean_shift` |
| Gradient checkpointing off | 33 items/s: the memory it saves is what keeps a 1024-token batch from paging to system memory. It stays on |

Flash-attention is not the lever: no Windows wheel, and the batches are sorted by length, so there
is almost no padding. `head_checkpointing` is dead in Laya's `DecisionModel.forward`.

Four arms from `dispositio-small`, teacher v2, 2 epochs, 16/4, differing only in the data added to
the released MIX. nDCG@10, pool 30, on the sets step 3 moved:

| Arm | MultiDoc2Dial | TechQA | SciFact |
|---|---|---|---|
| `dispositio-small` (start) | 0.6224 | 0.4155 | 0.7326 |
| MIX only | 0.6271 | 0.4185 | 0.7391 |
| + `instr` | 0.6183 | 0.4077 | 0.7257 |
| + `synth` (6 repeats) | 0.6096 | **0.4353** | 0.7369 |
| + `synth` (1 repeat) | 0.6187 | 0.4010 | 0.7310 |
| + both (step 3) | 0.6086 | 0.4019 | 0.7295 |

- Two more epochs on the same mix do not cost anything: the drop is not "more training".
- `instr` costs 0.9-1.3 points on all three sets by itself. `synth` at 6 repeats costs 1.8 on
  MultiDoc2Dial and *gains* 1.7 on TechQA. Together they lose more than the sum of their parts, so
  the ingredients interact, and TechQA's gain disappears.
- Each ingredient buys only its own competence and nothing else. `instr` fixes the question
  (corr(A, "does it fail to answer?") -0.70, "in Vietnamese?" 0.93) and leaves masking at step 1's
  level (0.66). `synth` fixes masking (0.996, and 0.973 at one repeat) and leaves the language
  questions at chance (0.51, 0.42).
- Link questions, held-out seventh, both halves in the top 5: BM25 11/23, v2 10/23, small 9/23,
  step 3 **11/23**, while on the 132 trained ones BM25 81, step 3 117. Step 3 memorised the 155
  questions it saw (6 repeats each); nothing generalised to the 23 it did not.

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
- **Right file, wrong section**: measured now on MultiDoc2Dial's held-out queries whose page has
  another section in the pool (n=46): the answer section is ranked first for **0.696** (BM25 0.304),
  `benchmarks/probe_model.py --same_file`. v4's target is 0.80 (`V4PLAN.md` §1).
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

- **v3 (step 4) is training now**: `dispositio-small-mask`, the MIX plus the masking rows, 2 epochs
  from `dispositio-small` — step 3's ability to read the question without the attribute questions
  that cost it 0.9-1.3 points. Then BEIR (~8 min) and the probes. Publishing it still needs
  SWE-bench Lite `mixed` (~70 min) or a card that says that row was not re-run, and tag `v2` before
  the upload.
- **v4's data plan is `V4PLAN.md`.** It replaces the ad-hoc list that stood here: failure targets,
  the row-by-row data set, the evaluation and the gates are fixed before any training runs.
- **Correction to what stood here.** The list said to mine negatives from the teacher's top ranks
  instead of BM25's 1-30, "which contain false negatives", citing Rank1. Rank1's ~80% is mT5-13B
  negatives, not BM25. Measured on our own cache (teacher v2 over the written negatives of the v3
  data): above 0.1, SciFact 4.7%, Zalo 4.1%, StackOverflow QA 4.5%, MultiDoc2Dial 7.6%, **SWE-bench
  22.2%** (above 0.5: 2-3% on the text sets, 10.3% on SWE-bench). Denoising stays in the plan,
  aimed at SWE-bench.
- **Instruction negatives are out**, and with them the "instruction data up to 1:1" that stood
  here: they need instance instructions, and ours is one fixed sentence. The 15%/two-thirds defect
  rates Promptriever reports are still the priors for anything LLM-made.
- memoria: no model beats v2 there; reading instructions is not yet judging what to recall.
- Release step: `local_files_only` keeps a user on whatever they downloaded, so a v3 needs
  `inventio update` (already in) plus a note in README/model card; `main` is not pushed yet.
- M3: the 40 private questions and the MLX prototype are still unmeasured (both need the Mac).
- Quantising the small ranker (int8/ONNX) is untried; the M3 measurement says the cost is candidates
  per query, not weights, so the first question is whether it helps at all.

## Commands

```sh
uv tool install -q --reinstall-package inventio "$PWD[laya]"     # install from this checkout
uv run -q --python 3.12 --with pytest --with-editable ".[laya]" pytest -q tests
inventio serve --stop                                          # after changing the model
```
