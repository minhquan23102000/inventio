# v4 plan: the training data, decided before anything is trained

Written 2026-09-26, after the v3 ablation (`STATUS.md`) and two research passes
(`local://research-v4-data.md`, `local://research-paraphrase-eval.md`). Every design decision below
names the measurement or the paper it comes from. Where nothing settles a question, it says so and
turns it into an arm to measure rather than a choice to believe.

## 1. Why v4

v3's data had three parts. Two of them did not survive their own ablation:

| Part | What it cost (nDCG@10, 2 epochs from `dispositio-small`, one part per arm) | Verdict |
|---|---|---|
| `instr`: attribute questions ("is this passage in Vietnamese?") | −0.9 to −1.3 on MultiDoc2Dial, TechQA and SciFact, alone | dropped |
| `synth`: counterfactual passages | MultiDoc2Dial −1.8, TechQA +1.7 alone; link rows memorised 117/132 trained vs 11/23 held out | rebuilt (see §4) |
| relevance rows (human labels) | two more epochs on them alone *gained* 0.5-0.7 on all three | unchanged |

And two failures are still open, which v4 is aimed at:

| Failure | Measured now | Target |
|---|---|---|
| Right file, wrong section (the ranker picks the section next to the answer) | `probe_model.py --same_file`: on MultiDoc2Dial's held-out queries whose page has another section in the pool (n=46), the answer section is ranked first for **0.696** (BM25 0.304) | ≥ 0.80, the number the v2 card reports for the student-aid slice |
| The user's wording differs from the document's | not measured for any dispositio yet; the paraphrase axis is built in §6 | mean ΔnDCG@10 ≤ 0.03 (see §7) |

## 2. The rule every training row must pass

The model reads **one** (instruction, query, passage) and returns p("this passage answers"). It never
sees two passages together. So a row may only assert something about *that one pair*:

- **Legal:** "this passage answers, though it shares no word with the query"; "this passage does not
  answer, though it shares the query's words and topic".
- **Illegal:** "these two passages answer *together*" — the v3 link rows. 0.75 on each half asserts
  something about a set; the model has nowhere to put it, and the loss can only be lowered by
  recognising the question (memorised) or by raising topical non-answers (which is what the
  MultiDoc2Dial negative is designed to punish). Both happened. Promptriever's instruction negatives
  are the same failure mode in another setting: the model learned the surface marker that flips the
  label, not the underlying reason (2409.11136).

## 3. What the literature settles, and what it does not

| Question | What is established | Source |
|---|---|---|
| Are top-ranked passages safe negatives? | No: adding un-denoised top-retrieved hard negatives cost **7.3 MRR**; ~70% of unlabelled top-retrieved passages on 100 MS MARCO questions were actually relevant; denoise with a cross-encoder at <0.1 negative / >0.9 positive | RocketQA 2010.08191 |
| Same problem for BM25 ranks 1-30? | **Nobody has measured it** — Rank1's ~80% is mT5-13B negatives, not BM25, and overstates the case (§8) | 2502.18418 |
| How deep should negatives go? | Effectiveness peaks at **50 per query**, slightly worse at 100; 15 candidates is inside the safe zone, but more is not automatically better | Rank-DistiLLM 2405.07920 |
| Does LLM-made counterfactual data help? | **Contested.** Kaushik (1909.12434) reports gains without size control; Huang (2010.04762) finds no gain over size-matched data, worse on negation, and names the mechanism: minimal edits keep ~70% of the 2/3/4-grams, so the model can learn the edit's fingerprint | both read |
| Are automatic unanswerables easy to spot? | Yes, measured: a model trained on TF-IDF negatives scores 83.0 on automatic negatives and 67.6 on human ones; 93% of human ones are genuinely unanswerable | SQuAD 2.0 1806.03822 |
| Same-meaning query rewrites as augmentation? | **No recipe exists**, and no mixing ratio. The adjacent line generates queries *from documents* (InPars, Promptagator, Dragon/GenQ); InPars' own negative result — synthetic rows added on top of a large in-domain set *decreased* in-domain scores — is the same shape as our ablation | 2202.05144, 2209.11755, 2302.07452 |
| Are rerankers brittle to paraphrase? | Yes: Penha et al. (2111.13057) measure −0.08 mean ΔnDCG@10 for paraphrased queries (TREC-DL19), −0.03 (ANTIQUE); the whole pipeline loses ~20%, the reranker alone ~9%; their generator's validity as a *paraphrase* is only 37-63%. Hagen et al. (2024.findings-emnlp.248) extend it: neither cross-encoders nor dual encoders are more robust, paraphrase is the second-worst variation after typos, and training on variations improves robustness but **not** effectiveness | both read |
| A Vietnamese paraphrase benchmark? | Does not exist; MIRACL/Mr.TyDi have no Vietnamese, mMARCO has Vietnamese but translated queries with one phrasing | 2501.19264, 2402.14334 |

## 4. The rows of v4

Share is of ~56,000 relevance rows. Every LLM-made row is paired with the original it was made from
(Promptriever, Kaushik) and kept only if a filter passes (§5).

| Row | What it teaches | Label | Share | Evidence |
|---|---|---|---|---|
| relevance (unchanged) | the backbone: human labels, BM25 negatives 1-30, half from 1-10 | 0.95 / 0.05 | ~56,000 | v3's ablation: two more epochs on these gained everywhere |
| negative denoising (**new**) | removes the false negatives among BM25's 1-30 | teacher's score decides keep/relabel | all | RocketQA: −7.3 MRR without it. Ours measured in §8: 4-8% of text negatives above 0.1, **22% on SWE-bench** |
| same-page sibling sections (keep) | "same page, same words, not the answer" | 0.25 | 3,431 | already in MIX; targets `same_file` |
| deletion rows (rebuild) | answer-absence is not shared words | 0.05 (was 0.1) | 1x, ~1,100 | SQuAD 2.0's three controls: stay on topic; keep a *plausible* answer-shaped span; look like the positives. Huang: 1x, not 6x — repetition strengthens the fingerprint |
| rewrite rows (keep smaller) | the answer does not depend on shared words | 0.95 / 0.05 (was mixed) | 1x, ~600 | measured on held-out rows: 0.996 mask AUC at 6x, 0.973 at 1x; the labels must not carry a second signal (§5) |
| **query rewrites (new)** | the answer does not depend on the *query's* wording | the original row's label | K=3 on ~2,000 queries (~6,000 rows, ~10%) | no recipe exists; InPars' in-domain negative result and Penha's brittleness argue for a small, filtered, paired addition — and for measuring it as its own arm |
| attribute questions | — | — | — | dropped: unconcerned with whether the passage answers |
| link halves | — | — | — | dropped: §2 |

## 5. Labels

- **One scale.** v3 used 0.1 and 0.75 for machine-made rows while human rows used 0.95/0.05; a value
  that appears only in machine-made rows is a marker the model can key on. v4 uses 0.95/0.05
  everywhere, and 0.25 only for the same-page siblings where a human set it.
- **The teacher as a filter, not a label source.** v3 set every relevance target to
  0.5·teacher + 0.5·written. Measured now (teacher = v2, temperature 0.65, over the cached items):
  its mean score on *human-labelled positives* is 0.761 on StackOverflow QA but **0.442 on
  MultiDoc2Dial** — the mixture therefore coaches the student to hedge on the corpus where the
  teacher is unsure. v4: keep the written label for positives, and use the teacher where the
  literature says it pays — gating negatives (RocketQA). A per-corpus weight, taken from the
  teacher's measured AUC on that corpus (already printed by every training run), is the fallback if
  the ablation says the mixture still helps somewhere.
- **Filters before a row is kept:** 2 of 3 samples agree (System 2 → System 1, 2407.06023) *and* the
  teacher agrees in sign; for query rewrites also a round-trip check (the rewritten query must
  retrieve its source passage) — Promptagator's filter was worth +2.5 nDCG@10 where it helped, while
  it *hurt* the two smallest sets (NFCorpus, SciFact), which is why it is measured here, not assumed.
  Expect to reject a lot: Promptriever reports ~15% of generated instructions and ~⅓ of generated
  negatives were wrong; Penha's paraphrase generator was valid 37-63% of the time.

## 6. Evaluation, decided now

Every model is scored on all of it, in one run, on data never trained on:

| Axis | Metric | Source |
|---|---|---|
| Public retrieval | nDCG@10, pool 30, on SciFact / StackOverflow QA / Zalo legal / MultiDoc2Dial / TechQA, plus SWE-bench Lite `mixed` | `benchmarks/beir_bench.py`, `swe_bench.py` |
| Right file, wrong section | the `same_file` block of `probe_model.py` (new) | this repo, defined in §1 |
| Reading the question, not the words | the A/B/C/E probes, cut, mask, and the held-out seventh of the link questions | `probe_model.py` |
| Paraphrase robustness (new) | ΔnDCG@10 (mean over variants), Robustness@10 (min over variants), RBO@0.95 between the ranking for the original query and for each paraphrase — **on the original query's fixed BM25 pool**, so only the ranker is measured | Penha 2111.13057; Robustness@k: InstructIR 2402.14334; RBO: Webber 2010 / Moffat 2015 |
| The real task | memoria's 100 labelled messages, hit@2 of 36 (local only, never trained on) | `%TEMP%/memoria_rank.py` |

Paraphrases are generated for the held-out test queries of all five sets, **including Vietnamese
(Zalo)**, K=5 each, filtered for validity, and the resulting set is kept as a file so every model
after this one is scored on the same queries. p-MRR is *not* used: it is defined over documents
whose relevance flips, so it is undefined for a paraphrase whose answer is unchanged.

## 7. Gates, decided now

v4 ships only if, against v3 (`dispositio-small`, and the v3 arm with link):

1. no public set loses more than **0.005** nDCG@10 (v3's own noise, from the ablation's arms);
2. `same_file` on MultiDoc2Dial improves by at least **0.05** (0.696 now);
3. on the paraphrase axis: mean ΔnDCG@10 ≤ **0.03** and Robustness@10 ≥ **90%** of the original-query
   nDCG@10 — a "robust" reading by the published band (Penha's paraphrase band is −0.08; brittle is
   Δ > 0.08 or Robustness@10 < 80%);
4. memoria hit@2 does not fall below v3's;
5. every probe keeps its pass mark, and the link probe is reported, not gated: a pairwise model
   cannot learn it (§2).

If an arm fails (1) or (2), that factor is dropped, not tuned until it passes.

## 8. Corrections and measurements this plan rests on

- **Correction.** I said Rank1 showed ~80% of BM25 ranks 1-30 negatives were really positives. It
  does not: the 80% is mT5-13B ranks 1-5 and 5-10, a stronger ranker than BM25, so it is an upper
  bound, not a measurement. No paper measures BM25 1-30.
- **Measured here** (teacher v2 over the written negatives of the v3 data, 56,251 cached items):

  | Source | negatives | mean teacher score | > 0.1 | > 0.5 | > 0.9 |
  |---|---|---|---|---|---|
  | SciFact | 2,896 | 0.042 | 4.7% | 2.3% | 1.7% |
  | Zalo legal | 7,800 | 0.039 | 4.1% | 1.9% | 1.2% |
  | StackOverflow QA | 9,330 | 0.037 | 4.5% | 2.4% | 1.5% |
  | MultiDoc2Dial | 8,368 | 0.062 | 7.6% | 3.1% | 1.7% |
  | MultiDoc2Dial siblings (0.25) | 3,431 | 0.077 | 10.7% | 3.7% | 1.7% |
  | SWE-bench | 6,578 | **0.131** | **22.2%** | **10.3%** | 3.9% |

  So BM25 1-30 is far cleaner than the dense retriever ranks the papers warn about: the denoising
  fix is worth least where I expected most, and it is concentrated on SWE-bench, where a fifth of
  the "negatives" are code the teacher considers plausible. That is a correction to my own plan as
  much as to my earlier claim.
- **Measured here** (teacher v2 on human-labelled positives): mean 0.761 (StackOverflow QA), 0.712
  (SciFact), 0.700 (Zalo), 0.598 (SWE-bench), **0.442** (MultiDoc2Dial) — the corpus-dependence §5 uses.

## 9. Arms

One factor per arm, same seed, same initial checkpoint (`dispositio-small`), same 2 epochs, 16/4 —
the design that made v3's failure legible. Control = `MIX` alone, then:

| Arm | Adds over the control |
|---|---|
| A | negative denoising (teacher gate) |
| B | rebuilt deletion rows (1x, length-matched, one label scale) |
| C | query rewrites (K=3, filtered, paired) |
| D | A + B + C |

Cost: ~30 min of GPU per arm with the caches warm (measured: 53 → 67 items/s with `torch.compile`),
plus ~8 min of scoring per arm, plus one paraphrase-set generation pass (~1-2k local-LLM calls).

## 10. Still unknown, and turned into measurements rather than assumptions

- Whether a *fingerprint* protocol separates answer-absence from edit style: none exists in the
  literature. Built here: score the trained model and the teacher on machine-made negatives and on
  human negatives at the same label; a systematic gap the teacher does not have means the student
  learned the edit, not the absence (§4's deletion rows).
- Whether length matching removes the short-passage cue: no paper has a procedure. If the deletion
  arm's gain disappears when lengths are matched, the cue was real.
- Whether the teacher-as-filter is better than the v3 mixture for *positives* too — arm A decides.
- Vietnamese rewrite validity, and whether rewrites transfer between languages: no evidence exists,
  so the generated set is reported with its validity rate per language.
