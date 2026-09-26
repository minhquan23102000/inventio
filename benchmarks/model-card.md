---
license: apache-2.0
base_model: convaiinnovations/laya
language:
- en
- vi
- multilingual
pipeline_tag: text-classification
tags:
- reranker
- retrieval
- rag
- code-search
- laya
- inventio
datasets:
- IBM/multidoc2dial
- BeIR/scifact
- CoIR-Retrieval/stackoverflow-qa
- GreenNode/zalo-ai-legal-text-retrieval-vn
- princeton-nlp/SWE-bench
---

# dispositio

*Dispositio* is the second canon of classical rhetoric: after *inventio* finds the material,
*dispositio* puts it in order. This model is the local ranker and category judge of
[Inventio](https://github.com/minhquan23102000/inventio), a retrieval tool for code and
documents that uses BM25 and structure instead of embeddings.

It is [Laya](https://huggingface.co/convaiinnovations/laya) multilingual (a 322M typed-decision
model on [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base)) fine-tuned for the two
questions Inventio asks:

- **Relevance**, a yes/no question: does this `passage` answer the `query`? Used to reorder
  BM25's 30 candidates.
- **Category**, a choice question: what does this `passage` do for its reader? One of Rule,
  Procedure, Reference, Explanation, Finding, Record, Other.

It runs on a laptop GPU (0.4-0.9 s to rank 30 candidates) or a CPU, so private documents never
leave the machine.

## Use

With Inventio:

```sh
pip install "inventio[laya] @ git+https://github.com/minhquan23102000/inventio"
inventio init ~/code/webshop --name app
inventio query "the nightly backup has not finished, what do I do?"   # ranked by dispositio by default
```

Directly, with the `laya` package:

```python
import laya

agent = laya.load("minhquan2310/dispositio")
q = {"type": "noul",
     "instructions": "Does the `passage` answer the `query`?",
     "criteria": {"true": "The passage states the specific answer, rule, or instruction the query asks for.",
                  "false": "The passage is only on a related topic, or uses the same words without answering the query."}}
state = {"query": "how do I rotate the API key?",
         "passage": "[docs/ops.md > Keys]\nRun `ops keys rotate --service api`, then restart the workers."}
print(agent.predict(state, {"rel": q})["answers"]["rel"]["noul"])   # probability of "true"
```

The model was trained on exactly these instructions and criteria and on passages rendered as
`[path > heading]` followed by the text; other wordings work less well. The category question
and its criteria are `category_question()` in
[inventio/facts.py](https://github.com/minhquan23102000/inventio/blob/main/inventio/facts.py).
A query is cut to its first 384 tokens, and a passage longer than the rest of Laya's 1,024 is
read in windows of whole lines, scored by its best window.

## Results

nDCG@10 on every test query, the ranker reordering the same 30 BM25 candidates inside Inventio.
dispositio − BM25 is a paired bootstrap over queries.

| Benchmark | BM25 | Laya, not tuned | **dispositio** | dispositio − BM25 (95% CI) | TypeSafe Jev (cloud) |
|---|---|---|---|---|---|
| MultiDoc2Dial, student aid pages, held out whole (123 questions) | 0.575 | 0.469 | **0.718** | +0.143 (+0.086, +0.204) | 0.634 |
| MultiDoc2Dial, all four domains (615) | 0.470 | 0.389 | **0.643** | +0.172 (+0.146, +0.199) | 0.486 |
| TechQA, IBM support notes, never trained on (119) | 0.370 | 0.171 | **0.444** | +0.074 (+0.024, +0.124) | 0.655 |
| SWE-bench Lite, code only (300 issues) | 0.540 | 0.391 | **0.661** | +0.121 (+0.080, +0.159) | 0.696 |
| SWE-bench Lite, whole repository (300) | 0.400 | 0.264 | **0.486** | +0.086 (+0.055, +0.118) | 0.515 |
| SciFact (300 claims) | 0.670 | 0.302 | **0.728** | +0.058 (+0.026, +0.089) | 0.765 |
| Zalo legal, Vietnamese (788 questions) | 0.756 | 0.512 | **0.830** | +0.074 (+0.053, +0.094) | not run |
| StackOverflow QA, without the train split's answers among the candidates (1,994) | 0.729 | not run | **0.752** | +0.024 (+0.014, +0.033) | 0.799 |
| StackOverflow QA as published (1,994) | 0.670 | 0.193 | 0.590 | −0.079 (−0.095, −0.064) | 0.791 |

- **Procedures and rules.** MultiDoc2Dial asks about the pages of four US public services
  (rules, eligibility, procedures). The student aid domain was kept out of training whole: new
  pages, same genre. On it, the section that answers is scored above every other section of the
  same page for 80% of questions (57% for the previous release and for Jev), the case where
  BM25 cannot help: every section shares the question's words.
- **StackOverflow QA as published is below BM25, and the reason is the benchmark's corpus.** Its
  train and test answers share one corpus: 70% of the wrong candidates a test question sees are
  answers to train questions, which this model saw as answers. It scores them too high (p > 0.5
  for 3.1% of them, 0.6% of other documents), and 90% of its wrong first results are one of them.
  With those documents removed from the candidates it is ahead of BM25. A team's own documents
  were never in training, so the first row is the one that describes them; the second is kept
  because it is the published benchmark.
- SWE-bench Lite's 12 repositories are not among the 35 it trained on. The file to fix is first
  for 51% of issues (BM25 38%, Jev 56%).
- Laya as published ranks worse than BM25 alone on every public set; its author presents it as a
  base to specialise.
- On [examples/webshop](https://github.com/minhquan23102000/inventio/tree/main/examples/webshop),
  13 on-call questions over a runbook, a policy, an incident report and code, written after
  training: the answer is first for 9 (BM25 6, Laya 9, Jev 13).
- Categories, against the labelling model's choice: 66% agreement on repositories held out whole
  from training, 53% on StackOverflow answers (a genre never labelled for training), 63% together;
  17% for Laya as published. They partition: on four unseen
  collections the largest category holds 36% to 80% of the chunks (80% on docker-compose, mostly
  changelog).
- 0.4-0.9 s to rank 30 candidates on an RTX 5070 laptop GPU.

## Training

Two stages, one epoch each, on an RTX 5070 laptop GPU, with the token embeddings frozen. Stage 1
starts from Laya multilingual as published (55,786 items, 40 minutes); stage 2 continues from it
with programming Q&A as the main source (28,513 items, 24 minutes), because after stage 1
StackOverflow QA was below BM25.

| Source | Stage 1 | Stage 2 | Label |
|---|---|---|---|
| MultiDoc2Dial train dialogues (US public-service pages), without the student aid domain | 16,304 | 4,004 | human: the section the agent's answer was grounded in |
| StackOverflow QA train split | 12,004 | 16,000 | human: the accepted answer |
| Zalo legal train split | 10,002 | 2,002 | human |
| SWE-bench train split: 3,923 issues from 35 repositories, none in SWE-bench Lite | 10,001 | 3,000 | human: the code the fix changed |
| SciFact train split | 3,683 | 2,001 | human |
| Category passages: prose of 25 of those repositories, SciFact, Zalo, SWE-bench issues | 4,364 | 2,000 | a small general LLM |

- **Negatives.** For every relevance question, BM25 candidates from other documents that are not
  answers, half from ranks 1-10 and half from 11-30. For a MultiDoc2Dial question grounded in one
  section, also the answer page's own section BM25 ranks highest, at a soft target of 0.25: the
  passage that shares the topic and the words and does not answer.
- **Not used.** A document's title as the query and its first chunk as the answer, which an
  earlier release trained on: it teaches that the passage repeating the question is the answer,
  which BM25 already knows. MultiDoc2Dial questions whose grounding is vague, and Zalo articles
  of the same law as negatives (several articles of one law often answer together).
- **SWE-bench.** Each issue is the query; the chunks whose lines the merged fix changed are the
  answers; other source-code chunks among BM25's 30 are the negatives. Tests, docs, config and
  untouched chunks of the touched files are neither: the patch does not say they are wrong.
- **Categories.** No human-labelled set exists for this question. Passages were labelled by
  Gemini Flash with one prompt and a one-of-seven schema; one repository in five is held out
  whole for the test. No relevance label comes from a model, and the model was not trained on
  the output of any retrieval or judging model.
- Relevance targets are 0.95 / 0.05 (qrels are incomplete); category targets 0.94 / 0.01. Half
  of the passages lose their path so file names are not a cue. Every test query of every
  benchmark, and any train query with the same text, is kept out.

Reproduce with `benchmarks/data.py`, `benchmarks/swe_train.py`, `benchmarks/category_data.py`
and `benchmarks/finetune_laya.py` in the Inventio repository.

## Limitations

- A paraphrase that needs an inference is missed: asked "how do I take a new backup without
  loading the main database?", it does not see that "take it from the replica, not the primary"
  answers, and scores every passage below 0.1. An answer that is one step of a numbered list is
  found less often than a section that answers as a whole.
- On IBM support notes (TechQA) Jev is far ahead (0.655): the answer is one section of a long
  technote, and dispositio picks the right section over its neighbours for 35% of questions,
  Jev for 67%.
- It over-ranks documents it was trained on as answers (StackOverflow train answers); do not
  evaluate it on a corpus that contains them without removing them.
- Category labels are one model's reading, not a gold set.
- Laya reads 1,024 tokens: a query is cut at 384, a long passage is scored by its best window.
- It was not trained to judge whether two passages are about the same thing (Inventio's `about`
  links); use Jev for those.
- The previous release is kept as the revision `v1`
  (`huggingface_hub.snapshot_download("minhquan2310/dispositio", revision="v1")`, then point
  `INVENTIO_DISPOSITIO_MODEL` at the folder). It is 0.012 ahead on StackOverflow QA without train
  answers, and behind by 0.12-0.15 on MultiDoc2Dial.

## Terms

The weights are released under Apache-2.0, like Laya (mmBERT-base is MIT). The training data
carries its own terms, which you should consider for your use: MultiDoc2Dial Apache-2.0;
SciFact claims CC BY 4.0 and abstracts ODC-By 1.0; StackOverflow content CC BY-SA 4.0; Zalo
legal (card: MIT); SWE-bench MIT, with each repository's code under its own licence; category
labels generated with Gemini.
