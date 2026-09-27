---
license: apache-2.0
base_model: jaredpalmer/kev-0.8b
language:
- en
- vi
- multilingual
tags:
- reranker
- retrieval
- rag
- decision-model
- inventio
datasets:
- IBM/multidoc2dial
- princeton-nlp/SWE-bench
- BeIR/scifact
- GreenNode/zalo-ai-legal-text-retrieval-vn
---

# dispositio v4

*Dispositio* is the second canon of classical rhetoric: after *inventio* finds the material,
*dispositio* puts it in order. This model is the local ranker, category judge and type head of
[Inventio](https://github.com/minhquan23102000/inventio), a retrieval tool for code and documents
that uses BM25 and structure instead of embeddings.

v4 is a **System One** decision model: [Kev 0.8B](https://huggingface.co/jaredpalmer/kev-0.8b)
(Qwen3.5-0.8B-Base with a decision head) fine-tuned on Inventio's data, weights merged, 1.4 GB in
bf16. It reads one *state* — a question and up to 15 candidate passages whose lines are numbered —
and answers every question asked about it in one pass:

- **where**: which passage holds the answer, and which line;
- **exists**: does any passage answer at all;
- **category** (one passage in the state): Rule, Procedure, Reference, Explanation, Finding,
  Record or Other;
- **type** (the question alone): which kind of document would hold the answer.

v3, the per-passage Laya model (144M), is revision `v3`; its files also stay on `main`, so an Inventio that
reads `main` still loads what it can read. v4 lives on the tag `v4`.

## Use

```sh
pip install "inventio[dispositio] @ git+https://github.com/minhquan23102000/inventio"
inventio init ~/code/webshop --name app
inventio query "the nightly backup has not finished, what do I do?"   # ranked by this model
inventio facts --source app                                            # categories, judged by it
```

The state format, the question wordings and the reader are Inventio's (`inventio/systemone.py`,
`inventio/_systemone`, Kev's serving path vendored under Apache-2.0); other wordings were not trained.
A state longer than 6,656 tokens was never seen in training.

## Results

All numbers are from this checkpoint (`s1-v1.3`), on an RTX 5070 laptop GPU unless marked. Pools are
BM25's best 15 chunks through Inventio's real ingest; "first" means the passage that holds the answer
is ranked first, among questions whose answer BM25 put in the pool.

| Set | Questions | BM25 first | v3 first | **v4 first** | v4 exists AUC, answer absent |
|---|---|---|---|---|---|
| MultiDoc2Dial (US public-service pages; student aid domain held out whole) | 453 | 0.375 | 0.614 | **0.638** | 0.805 (160 pools) |
| TechQA (IBM technotes, never trained on) | 87 | 0.149 | 0.200 | **0.322** | 0.895 (32 pools) |
| examples/webshop (13 on-call questions, written after training) | 13 | 6/13 | 4/13 | **9/13** | — |

- The right line: on MultiDoc2Dial the top line is inside the answer for 0.561 of the questions.
- Against the previous training run (v1.2) paired per query: MultiDoc2Dial −0.007 (−0.033, +0.018),
  TechQA +0.011 (−0.069, +0.080), webshop 2 questions lost (below).
- Time to read one pool: median 168 ms on MultiDoc2Dial (3,270 tokens), 368 ms on TechQA (6,480).
  On a CPU the same model reads 15 passages (4,200 tokens) in about 15 s.

**Categories**, against the labelling model's choice on 380 held-out passages:

| | v3 | **v4** |
|---|---|---|
| All: accuracy / macro-F1 | 0.663 / 0.642 | **0.811 / 0.769** |
| Repositories held out whole (200) | 0.615 | **0.780** |
| StackOverflow answers, never trained on (60) | 0.333 | **0.617** |
| SciFact (40) / Zalo (40) / SWE-bench issues (40) | 0.825 / 0.900 / 1.000 | 0.975 / 0.900 / 1.000 |
| ms per passage | 53 | 96 |

**Types** (SWE-bench Lite, 300 issues over whole repositories, BM25's 30 plus the best chunks of the
types this model predicts): the file the fix changes is in the pool for **0.793** of the issues, against
0.63 for BM25's 30 and 0.73 for BM25 grown to the same size (57.6 chunks). With TypeSafe predicting the
types the same pool reached 0.813 (control 0.72). Measured on Modal (type head on an L4).

**Facts on SciFact** (categories and `about` links drawn by this model, then the pool widened by them):
recall of the answer in the pool 0.849, the same as BM25's 30, while a same-size BM25 pool reaches
0.854. The model linked 2 of 30,517 pairs it was asked: links do not work yet (Limitations).


## Training

Two epochs, 1,838 optimizer steps over 7,346 records, 6.2 hours on the laptop GPU (peak 4.6 GiB),
LoRA on Kev 0.8B then merged. `recipe.json` pins the Kev commit, the base revision and the sha256 of
every data file.

| Source | Records | Label |
|---|---|---|
| MultiDoc2Dial train dialogues, three domains (student aid held out) | 2,987 states | human: the grounding section; each state asks where, the line, `exists` in three wordings, and one question per passage in six |
| Category passages: prose of 25 SWE-bench train repositories, SciFact, Zalo, SWE-bench issues | 4,364 | a small general LLM (Gemini Flash), soft target 0.94 / 0.01 |

- Some states have their gold passages removed from the pool, so `exists` is also taught "no".
- A bag-of-bigrams probe tells answerable states from unanswerable ones at AUC 0.598 in domain
  and 0.50–0.54 on a held-out domain, so the `exists` answer is not read off surface words.

Reproduce with `benchmarks/systemone.py data`, `data --judge`, then `run` in the Inventio repository.

## Limitations

- Asked "which date does a backup taken the next morning get?" (examples/webshop), it points at
  `from datetime import timedelta` instead of the policy; the previous run got it. Asked how long
  backups are kept, it points at `BACKUP_RETENTION_DAYS = 35` in code, which answers, while the
  labelled answer is the policy section. Thirteen questions are few.
- `exists` is calibrated per corpus (answerable medians 0.447, 0.232 and 0.377 on three sets): do not
  read 0.5 as a threshold. It was validated on pools whose answer is in the map but not in BM25's 15.
- It was not trained to judge whether two passages are about the same thing (Inventio's `about`
  links) and links almost none.
- Code: this fine-tune has no code retrieval states (their packed length does not fit the 8 GB card);
  what it knows of ranking code is Kev's.
- Category labels are one model's reading, not a gold set.

## Terms

Apache-2.0, like Kev and Qwen3.5. The training data carries its own terms: MultiDoc2Dial
Apache-2.0; SciFact claims CC BY 4.0 and abstracts ODC-By 1.0; Zalo legal (card: MIT); SWE-bench MIT,
with each repository's text under its own licence; category labels generated with Gemini.
