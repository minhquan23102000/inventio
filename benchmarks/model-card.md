---
license: apache-2.0
base_model: jhu-clsp/mmBERT-small
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

This release (v3) is the previous dispositio (v2, a 322M fine-tune of
[Laya](https://huggingface.co/convaiinnovations/laya) on mmBERT-base) distilled into a
[Laya](https://github.com/NandhaKishorM/laya) decision model on
[mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small): 144M parameters, 98M of them the
token embeddings. It answers the two questions Inventio asks:

- **Relevance**, a yes/no question: does this `passage` answer the `query`? Used to reorder
  BM25's candidates.
- **Category**, a choice question: what does this `passage` do for its reader? One of Rule,
  Procedure, Reference, Explanation, Finding, Record, Other.

It ranks 15 candidates in 0.15 s on an RTX 5070 laptop GPU (v2: 0.29 s), weighs 289 MB (v2:
644 MB), and runs on a CPU too, so private documents never leave the machine.

**What changed for a user of v2.** Public benchmarks are level with v2 or better; on the 13
on-call questions of Inventio's own example, the kind of question Inventio is built for, it
puts the answer first less often than v2 and than BM25 alone (below). The previous release
stays fetchable as revision `v2`.

## Use

With Inventio:

```sh
pip install "inventio[laya] @ git+https://github.com/minhquan23102000/inventio"
inventio init ~/code/webshop --name app
inventio query "the nightly backup has not finished, what do I do?"   # ranked by dispositio by default
inventio update                                                      # an install that has v2 moves to this one
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
print(agent.predict(state, {"rel": q})["answers"]["rel"]["noul"])   # probability of "true": 0.095
```

On this two-line passage v3 says 0.095 and 0.033 for "The office is closed on public holidays."
(v2: 0.58 and 0.02). The order is right, and the order is all Inventio uses, but v3's
probabilities are lower than v2's on short passages; on the real runbook section of Inventio's
example it gives 0.97. Do not read 0.5 as a threshold.

The model was trained on exactly these instructions and criteria and on passages rendered as
`[path > heading]` followed by the text; other wordings work less well. The category question
and its criteria are `category_question()` in
[inventio/facts.py](https://github.com/minhquan23102000/inventio/blob/main/inventio/facts.py).
A query is cut to its first 384 tokens, and a passage longer than the rest of the model's 1,024
is read in windows of whole lines, scored by its best window.

The previous release: `huggingface_hub.snapshot_download("minhquan2310/dispositio",
revision="v2")`, then point `INVENTIO_DISPOSITIO_MODEL` at the folder. `v1` is kept the same way.

## Results

nDCG@10 on every test query, the ranker reordering the same 30 BM25 candidates inside Inventio
(`benchmarks/beir_bench.py`). v3 − v2 is a paired bootstrap over queries.

| Benchmark | BM25 | v2 | **v3 (this)** | v3 − v2 (95% CI) |
|---|---|---|---|---|
| SciFact (300 claims) | 0.670 | 0.737 | 0.733 | −0.006 (−0.028, +0.014) |
| StackOverflow QA as published (1,994) | 0.670 | 0.676 | **0.691** | +0.015 (+0.006, +0.024) |
| Zalo legal, Vietnamese (788) | 0.756 | 0.820 | **0.838** | +0.017 (+0.003, +0.032) |
| MultiDoc2Dial, all four domains (615) | 0.470 | **0.638** | 0.622 | −0.015 (−0.032, +0.000) |
| TechQA, IBM support notes, never trained on (119) | 0.370 | 0.405 | 0.416 | +0.010 (−0.035, +0.051) |

- **Inventio's own example.**
  [examples/webshop](https://github.com/minhquan23102000/inventio/tree/main/examples/webshop):
  13 on-call questions over a runbook, a policy, an incident report and code, written after
  training, most without the words of the section that answers. The answer is first for **4**
  (MRR@10 0.58), against 7 for v2 (0.69) and 6 for BM25 alone (0.67). Thirteen questions are
  few, but a user's own questions look more like these than like the benchmarks.
- **Categories** fell in distillation, against the labelling model's choice on data held out
  from training: 59% agreement out of domain (v2 64%), 64% on repositories held out whole (v2
  68%), 40% on StackOverflow answers (v2 50%).
- **StackOverflow QA** puts train and test answers in one corpus, and this model was trained on
  the train answers as answers; earlier releases scored some of them too high. Here it is ahead
  of BM25 as published. The variant without train answers among the candidates was not re-run.
- **Not re-run for this release**: SWE-bench Lite (the `v2` revision's card reports 0.661 code only, 0.486 whole
  repository), the student aid slice of MultiDoc2Dial held out whole, and speed on Apple GPUs.
  See the `v2` revision's card for those.

## Training

Distilled from v2 in one stage: 3 epochs over 55,786 items, 57 minutes on an RTX 5070 laptop
GPU, token embeddings frozen. Each relevance target is half the written label and half v2's
probability (51,422 items; v2 moved them by 0.079 on average). Category targets are the written
labels.

| Source | Items | Label |
|---|---|---|
| MultiDoc2Dial train dialogues (US public-service pages), without the student aid domain | 16,304 | human: the section the agent's answer was grounded in |
| StackOverflow QA train split | 12,004 | human: the accepted answer |
| Zalo legal train split | 10,002 | human |
| SWE-bench train split: 3,923 issues from 35 repositories, none in SWE-bench Lite | 10,001 | human: the code the fix changed |
| SciFact train split | 3,683 | human |
| Category passages: prose of 25 of those repositories, SciFact, Zalo, SWE-bench issues | 4,364 | a small general LLM |

- **Negatives.** For every relevance question, BM25 candidates from other documents that are not
  answers, half from ranks 1-10 and half from 11-30. For a MultiDoc2Dial question grounded in one
  section, also the answer page's own section BM25 ranks highest, at a soft target of 0.25.
- **SWE-bench.** Each issue is the query; the chunks whose lines the merged fix changed are the
  answers; other source-code chunks among BM25's 30 are the negatives.
- **Categories.** No human-labelled set exists for this question. Passages were labelled by
  Gemini Flash with one prompt and a one-of-seven schema; one repository in five is held out
  whole for the test.
- Relevance targets are 0.95 / 0.05 before the teacher's half; category targets 0.94 / 0.01.
  Half of the passages lose their path so file names are not a cue. Every test query of every
  benchmark, and any train query with the same text, is kept out.

Reproduce with `benchmarks/data.py`, `benchmarks/swe_train.py`, `benchmarks/category_data.py`
and `benchmarks/finetune_laya.py --student jhu-clsp/mmBERT-small --teacher <v2>` in the Inventio
repository.

## Limitations

- On Inventio's own example it ranks below v2 and below BM25 (above). Use revision `v2` if your
  questions are worded unlike your documents and the time per query matters less.
- It leans on the words the query and the passage share: a passage rewritten without the query's
  words loses 0.35 of its score on average, and asking "does the passage *fail* to answer?"
  returns the same order as asking whether it answers. It was not trained to read other questions.
- A paraphrase that needs an inference is missed: asked "how do I take a new backup without
  loading the main database?", it does not see that "take it from the replica" answers.
- Category labels are one model's reading, not a gold set.
- It was not trained to judge whether two passages are about the same thing (Inventio's `about`
  links).

## Terms

The weights are released under Apache-2.0, like Laya (mmBERT is MIT). The training data carries
its own terms, which you should consider for your use: MultiDoc2Dial Apache-2.0; SciFact claims
CC BY 4.0 and abstracts ODC-By 1.0; StackOverflow content CC BY-SA 4.0; Zalo legal (card: MIT);
SWE-bench MIT, with each repository's code under its own licence; category labels generated
with Gemini.
