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
- **Links and neighbours are on by default with a ranker** (`--no-links`, `--no-neighbours`; skipped
  with `--ranker none`, where they would only sit below BM25). Answer brought into a pool that had
  none (BM25 30 + up to 10 each, `%TEMP%/probe_widen.py`): SciFact 9 (neighbours) / 4 (links) of
  300, StackOverflow QA 14 / 31 of 1,994, MultiDoc2Dial 6 / 0 of 613 (BEIR maps have only judged
  `about` links; a real map's cites/mentions links are unmeasured here). Cost per query, CPU,
  before the ranker: links about 0 ms; neighbours 23 ms (TechQA, 2k chunks), 65 ms (SciFact, 5k),
  201 ms (StackOverflow QA, 27k), **1.7 s (Zalo, 67k)** — five 24-word BM25 queries. On Vietnamese
  maps that bag of words is also searched as adjacent pairs (`đ` survives the diacritic fold, so
  the phrase rule fires on a list with no adjacency): off, 1.45 s and 27% different neighbours;
  not changed until measured. The ranker reads up to 20 more passages.

## Next

- **v4 is not a ranker.** Zero's direction (locked D1, D2): a System One model for RAG, like Jev,
  that reads the query and BM25's pool once with passage and line ids and answers typed questions
  together: `where` (a Choice over line ids) and `exists` (a Noul) first, `next` and `conflict`
  later. The selling point is inference speed with many questions over many tokens, not file size.
  Laya (questions before the state, one pass per question) is the wrong shape; the Kev shape
  (Qwen3.5 decoder + LoRA + pointer head, state computed once, one row per question on its cache;
  github.com/jaredpalmer/kev, Apache-2.0, "No Jev outputs were used for training") is the candidate.
  `V4PLAN.md` (reranker data plan) is superseded.
- **Zero-shot spike, RTX 5070 laptop** (`benchmarks/systemone.py`, subcommands `data`, `spike`,
  `disp`, `gate`, `lines`, `scale`; Kev run with `benchmarks/kev_win.py`; pools are
  BM25's 15 rendered as `P01 [path > heading]` / `L000| line`; `exists` negatives = the same query
  with every gold chunk removed and the pool refilled from BM25). Top-1 = the answer passage, or the
  top line inside it; MultiDoc2Dial counts the 453 of 613 test queries whose answer is in the pool.

  | | webshop (13) top passage / top line | exists AUC | MD2D top passage / top line | exists AUC | time per pool |
  |---|---|---|---|---|---|
  | BM25 order | 6/13 | – | 0.375 | – | – |
  | dispositio v3 (one question, trained on MD2D) | 4/13 | 0.56 | **0.614** | **0.744** | 27 ms / 65 ms |
  | Kev-0.8B zero-shot | 7/13 / **12/13** | **0.76** | 0.393 / 0.358 | 0.569 | 67 ms / 168 ms |
  | TinyJev-0.6B zero-shot (first 200 MD2D) | 6/13 / 9/13 | 0.72 | 0.291 (Kev 0.369 on the same) | 0.505 | 181 ms / ~1 s |

  Kev-0.8B, time against questions on one 3.2k-token state: 1-4 questions ~150 ms, 16: 202 ms,
  64: 434 ms, 255: 1.33 s. Against state length (3 questions): 3.3k 168 ms, 9k 581 ms, 18k 1.31 s,
  24k 1.98 s; ~36k fails on 8 GB. On Windows only with `%TEMP%/kev_serve_win.py`: torch's Windows
  wheels have no flash kernel, and SDPA's GQA path then takes the math kernel (O(L²) memory, OOM
  past ~8k tokens, twice the time at 3k). Zero-shot, `where` passage is BM25-level on MD2D and
  `exists` is weak; training on inventio's questions is the open step. Mac speed unmeasured.
  Envs: `C:/Users/LEGION/kev/.venv` (torch cu128, fla 0.5.2, triton-windows), `C:/Users/LEGION/tinyjev-env`.
- **D4, gates fixed before the first training run (2026-09-26).** Kev-0.8B, LoRA warm-started
  from `jaredpalmer/kev-0.8b`, questions `where_line` (Choice over line ids, soft target spread
  over the gold lines) and `exists` (Noul), rendered exactly as `spike` renders them. Data
  `python benchmarks/systemone.py data` -> `<data>/s1/data/{train,dev}.jsonl`, rows
  `benchmarks/results/s1/`, people's labels only: MultiDoc2Dial train topics (studentaid held out, as
  for dispositio; gold lines = the paragraphs holding the reply's grounding spans) and SWE-bench
  train issues (gold lines = lines the fix changed); `exists=false` = the same query with gold and
  near chunks removed. v1 file: MD2D 3,157 positive + 4,500 negative, SWE 107 + 107. Only 107 of
  the 1,386 SWE issues with the fix in BM25's 15 fit Kev's 255-option limit (code pools: median
  7.6k tokens, often >255 lines), so v1 learns almost only from prose. Gates, on the spike's pools:
  (1) MultiDoc2Dial test, 453 queries with the answer in the pool: top line's passage above v3's
  0.614, paired bootstrap interval above zero; (2) `exists` AUC above v3's 0.744; (3) webshop top
  line inside the answer passage at least 11/13 (zero-shot 12/13). Reported, no gate: exact-line
  top-1 on MD2D test (gold = paragraphs holding the grounding spans) against zero-shot Kev; TechQA
  (never trained on) against v3.
- **Two defects found in v1's own labels, and the fix (v1 was already training on the old file).**
  (a) 33 of 1,893 MultiDoc2Dial positives (1.7%) had their gold line on a bare section heading
  ("Citizenship"), because a short annotator span matched the first line containing it; the rule now
  takes the most specific matching line and skips a span that *is* a heading line, its section's
  paragraphs being where the answer is read. (b) The builder was missing `data.py`'s filter for
  vague queries ("I have a question"), so ~100 more topics were in it. Both fixed in
  `benchmarks/systemone.py`: the file is now MD2D 2,955 positive + 4,365 negative, SWE 107 + 107.
  v1 trained on the earlier file, kept beside the run as `runs/s1-v1/records.jsonl`
  (sha256 `4202a8f6…`, 58.7 MB); a v1.1 would train on the fixed one.
- **D4 result: the first fine-tune (`runs/s1-v1`) — 1 of 3 gates met.** Kev-0.8B + LoRA (11.3M
  trainable), 3,963 records, 1 epoch, 98 min on the 5070 (14.0M forward tokens, 3.6 GB peak). Its
  exact flags are `runs/s1-v1/training_config.json` (written by the trainer; `recipe.json` next to a
  run is new and also pins the Kev commit); the data was `runs/s1-v1/records.jsonl`, the pre-label-fix
  file (sha256 `4202a8f6…`). Same pools and question strings as the spike.
  `top line's passage@1` = the passage holding the top-ranked line answers; `exact line@1` needs the
  annotator's span, which only MultiDoc2Dial has (MultiDoc2Dial n=453, TechQA n=85 with 2 queries
  whose pool exceeds the line question's 255 options, webshop n=13):

  | set | metric | fine-tuned | zero-shot | dispositio v3 | BM25 |
  |---|---|---|---|---|---|
  | MultiDoc2Dial | top line's passage@1 | 0.642 | 0.358 | 0.614 | 0.375 |
  | MultiDoc2Dial | `where_passage`@1 | 0.547 | 0.393 | 0.614 | 0.375 |
  | MultiDoc2Dial | exact line@1 | 0.603 | 0.252 | – | – |
  | MultiDoc2Dial | `exists` AUC | 0.714 | 0.569 | 0.744 | – |
  | TechQA | top line's passage@1 | 0.235 | 0.282 | 0.200 | 0.149 |
  | TechQA | `exists` AUC | 0.645 | 0.609 | 0.576 | – |
  | webshop | line in the answer passage | 11/13 | 12/13 | 4/13 | 6/13 |
  | webshop | `exists` AUC | 0.769 | 0.763 | 0.559 | – |

  Gate 3 met (11/13). Gate 1 not met: paired against v3, +0.029 (-0.020, +0.075) — inside the noise.
  Gate 2 not met: 0.714 against 0.744. What the fine-tune did buy is the shape itself: exact line
  0.252 -> 0.603 and `exists` AUC 0.569 -> 0.714 on MultiDoc2Dial, and on the 13 real on-call
  questions it points inside the right section 11/13 where BM25 gets 6 and v3 gets 4. Time per pool:
  167 ms (MultiDoc2Dial, 3.3k tokens), 68 ms (webshop), 373 ms (TechQA, 6.5k tokens) — flat to 16
  questions on one state (`scale`).
- **D5, the second run's records and gates, fixed before it trains (2026-09-26).** One change to
  v1: the questions one record asks. Every record now asks `exists` plus **one question per
  passage** (`P01..P15`, "does passage P04 answer Q"), the positive arm also asks `where_passage`
  (Choice over the 15) and `where_line` as before. Reason: `exists` was trained as one judgement over
  3.3-6.5k tokens with no supervision on comparing slots, and dispositio — the thing that beats it —
  is exactly a max over per-passage scores; the wrong-neighbour errors (Q1 rank 4 while BM25 ranks it
  1; Q10 picks the passage before the answer) are that missing comparison. Wordings: three per
  question kind, picked by the record's key (dispositio's lesson: one wording passes every probe and
  fails the same questions worded anew), the first being the exact string `spike` asks. The negative
  arm is now the **same 15 slots with the gold swapped in place** (`slot_swap`), 14 of 15 passages
  identical to the positive arm, where it used to be a compacted pool that also dropped the
  `near` pages: the arms differed by corpus composition, and a bigram TF-IDF could tell them apart
  in-domain (AUC 0.63, 0.53-0.57 with a domain held out). `_meta` now carries `domain`, so the
  per-domain reading (studentaid held out whole: +0.067, while ssa -0.058 and va -0.034 cancel it)
  and the leave-one-domain-out probe are computable.
  The option cap 255 -> 512 needed a change **outside** this repo: `materialize` admits a record
  through `kev.api`'s request schema, so `kev/api.py`'s `Choice` cap binds both training and serving.
  `benchmarks/kev_win.py` raises it (and carries the reason). It does **not** admit the code arm:
  the binding cap is the state. Measured with the trainer's own `fits()` (`kev_win.py fits`):
  at `--max_state 6656` the code arm keeps **373 negatives and 56 positives of 5,308** records
  (872+128 at 8192, 1,271+204 at 9,216, and 8 GB OOMs on a ~8.2k-token state), so the code arm is
  short-file-biased whatever the cap; DataReview's "raise the cap, rebuild the code arm" is not
  reachable this way. What the cap does *not* say is the real bound, which is the **packed** row
  (state + every branch), measured with the encoder on the drawn set: MD2D p50 4,063 / max 7,571,
  code p50 7,074 / max 8,310, and a ~8.2k packed row OOMs this card at 7,779 MiB of 7,932. The new
  shape is what fattens the row (18 branches where v1 asked 1-2, one of them a choice over up to 512
  line ids), so `--row_budget 5120` was tried first: it holds memory at 4,337 MiB but re-encodes the
  state per part and cost ~6x per record (measured; a 17-minute run had not finished 10 of 427
  steps). v1.1 therefore trains **prose only** (`--no-code`), the arm the shape change is about and
  the one that fits; the code arm is a design fork, not a cap (fewer passages, or one passage per
  state). Data: `train_bal4000-0-q2-6656-md2d.jsonl`, drawn from `train.fit6656.jsonl` (what the cap
  admits) so the printed composition is the run's. v1's own data is kept beside it
  (`train_bal4000-2000.v1shape.jsonl`, sha256 `8251ea1f…`).
  **What the record change did and did not fix, measured** (`systemone.py leak`, a bigram TF-IDF
  that sees the state and never the query, grouped 5-fold by query): the compacted negative arm that
  v1 trained on was separable at **0.732** in-domain and at chance across domains (md2d 0.512, code
  0.394), which is a corpus-composition cue, not answerability. The one-swap arm is **0.623**
  (md2d 0.634 -> 0.598, code 0.927 -> 0.811): better, **not** at the floor. For code the state's
  length alone separates the arms at 0.77 (the swap-in filler is systematically longer), so the next
  data change is length-matched fillers and hard negatives mined from BM25's 16-30 band — a passage
  that shares the query's words and does not answer.
  Gates, all on the spike's pools, all above the untrained floor: (1) MultiDoc2Dial, paired against
  v1 on the same 453 queries, top line's passage@1 interval above zero (v1 0.642; a win over v3's
  0.614 is the reported ideal, not the gate); (2) `exists` on the answer-absent reading >= **0.839**
  (v1), with the gold-removed reading >= 0.714 reported beside it — the two differ by half a point of
  AUC (0.714 vs 0.839 on md2d, 0.645 vs 0.869 on techqa) because 160/160 and 32/32 of those rows
  have pos and neg states that are *the same state*; (3) TechQA (never trained on) top line's
  passage@1 above zero-shot's **0.282** (v1 sits below it at 0.235 — the gate is that the shape
  change not deepen that); (4) webshop line in the answer passage >= **12/13**, the untrained floor
  (v1 met a gate set at 11/13 while scoring 11/13 — a gate under the floor cannot fire). Reported,
  no gate: pool 30 (0.419 top1, 433 ms against 0.547/167 ms at pool 15 — raising the pool costs the
  answer, so it is not a lever), and the reverse/shuffle order probes (v1 0.336/0.371 against 0.547
  normal), which must stay clearly below the normal reading or the model has stopped reading order.
- **Is v1's `exists` lead real, or was it a corpus cue? (2026-09-27, M1).** `systemone.py domains <tag>`
  splits the answer-absent reading by the domain a query's gold comes from, with `studentaid` as the
  control (held out of every recipe). v1 leads v1.1 in **every** domain — dmv 0.838/0.753, ssa
  0.804/0.698, va 0.882/0.833, and **studentaid 0.825/0.737** — so the lead is not a cue v1 saw and
  v1.1 did not: it holds on the domain neither trained on. The earlier worry that part of v1's 0.839 was
  the in-domain bag-of-words separation (0.732 in-domain, chance across domains) is answered.
- **Is the `exists` head underfit? (2026-09-27, M2).** `systemone.py fitcheck` asks a served run the
  `exists` question on 300 positive and 300 negative records of its own training file: v1.1 answers its
  own data at AUC **0.800** (record's own wording) and **0.797** (the wording the harness serves with),
  medians 0.418 positive against 0.148 negative. The held-out readings are 0.758 (answer-absent pools,
  a harder population) and 0.666 (the same-negative construction). So the head is not saturated on
  what it was trained on and it loses ~0.13 to held-out pools on that construction: both more weight
  and better negatives have something to move. That is what `s1-v1.2` tests on the weight axis alone.
- **Corrections (2026-09-27, from the oracle read of this file and the code).** Exact line did **not**
  rise in v1.1: v1 is 0.603 and v1.1 is 0.585 (an earlier line in this file said 0.520 for v1 — wrong).
  The choice head is order-sensitive here (v1 `where_passage` 0.547 bm25 / 0.371 shuffled / 0.336
  reversed): this recipe has neither `--option_isolation` nor `--perm_kl`, and the hybrid backbone
  refuses isolation. The per-passage branches are **not** independent (query, passage) scorers like
  dispositio's: each one reads the whole 15-passage state. "6%" is the share of the **loss weight**, not
  of the gradient, which was never measured. And the study re-spike reused the tag `s1-v1.1`, so
  `cmd_spike`'s `"w"` rewrote that tag's latency and token fields (the gated `top1`/`exists` fields are
  untouched; the original summary is in `%TEMP%/s1-after.log`). Studies need their own tag.
- **The caveat line is off by default (2026-09-27).** `NOT_IN_MAP = 0.30` had no operating point behind
  it, and `exists` is calibrated per corpus (medians on answerable pools: md2d 0.447, techqa 0.232,
  webshop 0.377), so one number cannot hold a false-alarm rate across the three. The reading is kept in
  `SystemOneRanker.last`; the message prints only when `INVENTIO_SYSTEMONE_CAVEAT` sets a threshold,
  and it says what was measured — the passages read do not seem to answer — because the pools this was
  validated on have the answer in the map, just not in BM25's fifteen.
- **How the served model is read (2026-09-27, measured).** The harness can ask the same state two ways:
  the choice heads (`where_passage`, `where_line`, `exists` — three branches) and the per-passage head
  (one question per passage, the training device). On md2d's 453 answerable pools the choice head reads
  the passage better: 0.653 against 0.620, paired +0.033 [-0.007, +0.071], 51 wins / 36 losses, at
  ~100 ms less per state (247.6 ms with the per-passage asks, 92.7-150.9 ms without). The two honesty
  readings are the same number: `exists` head 0.758 against the max of the per-passage probabilities
  0.755 on the answer-absent pools. So the per-passage questions stay a **training** device and the
  served shape is the three branches; `spike --with-passage-asks` is the study that showed it.
- **The mix is the weight (2026-09-27).** `kev/train.py` sums the question losses per record, so a
  record's question list *is* the share each head gets: v1.1 asked one `exists` in 16.66 questions
  (6%) and v1 asked it in 1.5 (66%). `systemone.py mix <records> --out <records> --passage-asks 6
  --exists-asks 3` rewrites a record file's questions only — states, pools and labels verified
  byte-identical on all 9,154 records — and writes `<out>.mix.json`, which `train` copies into the
  run's recipe. On the prose arm that is 9.66 questions per record and a 31.1% share for `exists`, with
  the per-passage head still seeing 6 questions a state and the choice heads two epochs of positives.
- **`--ranker systemone` reads a checkpoint itself now (2026-09-27).** The serving path is vendored
  inside the package (`inventio/_systemone`: Kev's `model.py`, `checkpoint.py`, `api.py`, `device.py`
  at f153596, Apache-2.0, with the trainer-only and other-backend parts cut, the option cap at 512 and
  the CUDA-graph path dropped — the module docstring lists every change). `inventio/systemone.py`'s
  `Model` loads it in the process that answers the query, so `pip install 'inventio[systemone]'` plus
  `inventio query --ranker systemone` is the whole setup: no server to start, no port, no env var.
  `inventio systemone` says which checkpoint answers, `--use <dir|owner/name>` records one (in the data
  home, one file), `--load` prints the card; `INVENTIO_SYSTEMONE_URL` still reads a model on another
  machine over HTTP, which is the only shape the public-sources rule applies to. `SystemOneRanker`
  reads the state once, returns those per-passage probabilities as the score, keeps `last` (the passage,
  the line and both honesty readings), and refuses in silence never: a checkpoint that cannot be read
  raises with the README appended, exit code 3. Proven so far: the load path on the real run (CLI
  `systemone --load`: temperature 1.0, bf16, base Qwen3.5-0.8B-Base) and a full forward on a fabricated
  two-layer checkpoint built inside the test (`tests/test_systemone.py`, 3 tests, no GPU, no network).
  **Not yet proven: one forward pass of the real hybrid checkpoint in this process, and that its
  numbers equal the served server's** — the 5070 is busy with the v1.2 run until ~15:10, then the gate
  scores until ~15:40, so both wait for the card. The research harness (`benchmarks/systemone.py`
  spike/gate/lines) still talks HTTP to `benchmarks/kev_win.py serve` — unchanged on purpose, so the
  numbers being compared do not move while v1.2 is judged; switching it to the in-process reader comes
  after the golden check. The CLI prints one line when the reading says the map may not answer
  (`systemone.honesty`), and the default ranker is still dispositio v3 — v1.2 has to pass the gates
  first.
- **The default ranker is the System One reader, not Laya (2026-09-27, Zero's call).** `default_ranker()`
  = `INVENTIO_RANKER`, else `systemone` when the runtime is installed (a distribution lookup, never an
  import: building the command line must not cost a torch import), else BM25 order. The default
  **falls back with one line** when no checkpoint can be read — no weights recorded, the published model
  not there yet — and answers in BM25 order, because a default must not fail a query to reach for a model
  it cannot load; an explicit `--ranker systemone` still gets the whole message and exit 3. `--ranker
  dispositio`/`laya` still read v3 where the `laya` extra is installed. **Correction (same day):** an
  earlier line here credited Laya's type head with "63% -> 73% in pool, nDCG@10 0.430 -> 0.495" — those
  are the **symbols** widening's numbers (`search.py`). Type widening was only ever measured with
  **TypeSafe** predicting the types (SWE-bench Lite `mixed`, pool 30, `results/swe-lite-types`): answer
  file in pool 0.63 -> 0.813 against a same-size BM25 control 0.72; nDCG@10 ranked by TypeSafe 0.511 ->
  0.627 (control 0.560). Laya's type head was never measured. Laya's judge (categories, `about` links) was
  trained on categories only (Gemini-labelled, `category_data.py`); its links are zero-shot, and on scifact
  the facts arm (Jev judge) did not beat its control (0.769 vs 0.785).
- **Zero's call (2026-09-27): the System One model takes the judge and the type head too, end to end, and
  v4 ships as tag `v4` on `minhquan2310/dispositio`.** Wired: `facts.SystemOneJudge` (packs a chunk and
  its ten neighbours into one state, caches judgments under `systemone:<run>`), `default_judge()` =
  systemone when installed, `SystemOneRanker.types()` (the same choice question over the query alone).
  Measurement: `systemone.py judge` (category test split, per stratum, Kev run or `--laya`),
  `swe_bench.py --types --type-predictor`, `beir_bench.py --arms --judge` (on the judge's own map copy).
  Training: `systemone.py data --judge` (4,364 category records, soft target 0.94/0.01 as dispositio's
  recipe, path dropped half the time) appended with `train --judge-records`. Laya is removed once the
  System One model meets or beats it on the category split and the type/facts arms.
- **v3 against the Kev model: what the gates actually say** (same pools, paired, from
  `benchmarks/results/s1/summary.json`). md2d top-1 **v3 0.614** (v1 0.642: +0.029 [-0.020, +0.075] —
  noise), techqa **v3 0.200** (v1 +0.035 [-0.071, +0.141] — noise), webshop **v3 0.308 against 0.846**
  (+0.538 [+0.231, +0.769], 7W/0L). Latency on the 5070: v3 65 ms per pool per question, the Kev model
  ~150 ms flat for 1-4 questions; on an Apple GPU v3 is the slow side (3.5 s for 30 chunks, `MPS_BATCH`).
  So the decisive difference is the *shape* — webshop, the exact-line head, the `exists` head — not that
  v3 loses everywhere; the one reading where v3 leads is `exists` AUC on md2d's negatives (0.744).
- **Two things the in-process reader got wrong, found by running it (2026-09-27).** (1) It encoded with
  the *training* limits, so a state the server had answered was refused: Kev's `ContextOverflow`
  ("branch too long: 1078 tokens with a 384-token state, row limit 1024") — a line question offers one
  option per line, so a few hundred lines is past the training row budget. `kev.serve` passes
  `SERVE_MAX_STATE`/`SERVE_MAX_BRANCH` (65,536 / 73,728) and so does `Model.ask` now, with a regression
  test that builds a 300-line state (`tests/test_systemone.py`). (2) A query could end in silence: the
  client waited on the background worker with no deadline. Both waits are bounded now
  (`INVENTIO_SERVE_START` 900 s, `INVENTIO_SERVE_TIMEOUT` 3600 s) and end in "running here instead" with
  a line naming the log. Recorded limit, not a bug: the hybrid base on **CPU** dies inside a Triton
  kernel (`fla` is installed and transformers falls back to its GPU kernels; `USE_HUB_KERNELS=NO` does
  not reach that fallback), so this reader is verified on the accelerator.
- **D5 verdict (2026-09-27).** v1.1 against v1, paired on the same pools (`benchmarks/results/s1/`):
  md2d top1 0.658 vs 0.642, diff **+0.015 [-0.015, +0.046]**, 29W/22L -> **noise, the gate's md2d arm
  fails**; exists on the answer-absent reading **0.757 vs 0.839** and exists_auc 0.666 vs 0.714 -> the
  **exists gate fails**; techqa (the out-of-domain prose set) **+0.103 [+0.034, +0.184]**, 10W/1L ->
  passes; webshop **12/13** vs 11/13 -> passes; exact line **0.585 vs 0.520**. Latency 150.9 ms
  against v1's 167 on the same pools. So the shape bought what it aimed at (line, per-passage,
  webshop) and paid with `exists`.
- **Why exists fell, measured on the run's own data.** v1.1's records carry **16.7 questions per
  state** (v1: 1.5): `where_passage` is asked once per pool passage while `exists` is asked once, so
  the exists head's share of the gradient fell about an order of magnitude (and the run also took
  fewer steps: 374 against 496). Fix is a question mix: sample a few passages per state, give
  `exists` and `where_line` a fixed share.
- **The acceptance test on the fixed file: 0.598 in-domain**, length alone 0.492, cross-domain
  0.504-0.539 -> at chance. The arms still differ by corpus composition, not by the answer.
- **D5 finished (2026-09-27 01:38).** `kev/runs/s1-v1.1`: 2,987 records, 374 optimizer steps, 2.49 h
  wall, peak 5.91 GiB of the card's 7.73, **0 truncated and 0 rejected** (the prose-only file fits
  6,656 whole), loss 1.544 at step 10 -> 0.114 at 370, 12.71 M forward tokens, LoRA 11.3 M weights.
  The recipe is next to it (`recipe.json`, records sha256 `7a2c23de...`) and the run directory is
  inside the Kev checkout. Scoring (spike / paired gate against v1 / exact line / leak) runs as one
  detached chain so a closed session cannot lose it.
- **D5 launched (2026-09-26 23:05).** `runs/s1-v1.1` (inside the Kev checkout: `--out` is relative to
  it, which is what the trainer's cwd does), prose only, one epoch, and the pace is the run's own:
  `ep0 step 10/374 loss 1.544 2.605s/rec` -> 2,987 records, ~2h10m. Completion is `saved
  <out>/checkpoint` in the log, then `recipe.json` (written after the trainer makes the directory)
  and `training_metrics.json` (forward tokens, peak memory). Two Windows facts the run needed: the
  run directory must not exist when the trainer starts (it refuses), and `--max_state 6656` is
  recyclable only because the *prose* arm's packed row fits 8 GB.
- **Two forks the result opens.** (1) Wire it into `inventio` as a ranker now (it is at parity with
  v3 on passages within the noise, and it is the only thing that can name a line or say "not in the
  map"), or train again first. (2) `exists` is where it loses, and its negatives are all
  gold-removed (the easy kind); the hard kind is a passage that shares words and does not answer —
  mining those from BM25's 16-30 is the untried lever. The 255-option limit for code pools is a
  second, separate one (see the SWE count above).
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
- `main` (README for v3) is not pushed to GitHub; Hugging Face `main` is v3, `v2` tag keeps v2.
- M3: the 40 private questions and the MLX prototype are still unmeasured (both need the Mac).
- Quantising the small ranker (int8/ONNX) is untried; the M3 measurement says the cost is candidates
  per query, not weights, so the first question is whether it helps at all.

## Commands

```sh
uv tool install -q --reinstall-package inventio "$PWD[laya]"     # install from this checkout
uv run -q --python 3.12 --with pytest --with-editable ".[laya]" pytest -q tests
inventio serve --stop                                          # after changing the model
python benchmarks/systemone.py data                            # Kev records from MD2D + SWE-bench
python benchmarks/systemone.py train --bal 4000 --out runs/s1-v1   # fine-tune (Kev f153596, checked out at KEV_DIR)
python benchmarks/systemone.py spike http://127.0.0.1:8008 s1-v1 --sets md2d,webshop,techqa
python benchmarks/systemone.py gate s1-v1 v3 md2d              # top-1 paired against dispositio
```
