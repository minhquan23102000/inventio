# dispositio

[dispositio](https://huggingface.co/minhquan2310/dispositio), the second canon of rhetoric
after *inventio*, is the local model. Since v4 it is a **System One** decision model: a 0.8B
decoder ([Kev](https://huggingface.co/jaredpalmer/kev-0.8b) on Qwen3.5-0.8B-Base, fine-tuned on
Inventio's data) that reads one state — the question and the candidate passages — and answers every
question about it in a single pass: which passage holds the answer, which line, and whether any
passage answers at all. The same checkpoint judges `inventio facts` (the category of every prose
chunk, and the `about` links) and predicts the document types `--types` widens by. No query or
document leaves the machine.

```sh
inventio query "..."                      # ranked by dispositio once [dispositio] is installed
inventio facts --source wiki              # categories and links, judged by the same model
inventio model                            # which checkpoint answers, and where it came from
inventio model --use <run dir or owner/name[@rev]>   # a checkpoint of your own
inventio update                           # fetch a newer release of the published model
```

The checkpoint is read in the process that answers the query: nothing to start, no port. The first
query with it starts a background server that keeps it loaded (127.0.0.1 only, a token in
`serve.json`, exits after 15 minutes idle: `INVENTIO_SERVE_IDLE`; `INVENTIO_SERVE=0` runs every query
in its own process; `inventio serve --stop` stops it). The reader is part of this package
(`inventio/_systemone`, Kev's serving path under Apache-2.0, vendored so one install carries it).
Without the `[dispositio]` extra, or with no GPU memory to spare, a query answers in BM25 order with
one line saying why; an explicit `--ranker dispositio` or `--facts` fails instead (exit 3).

Once a day a query asks Hugging Face for the release's latest revision, sending the model's name
and nothing else, and prints one line when a newer one is out; `inventio update` fetches it.
`INVENTIO_OFFLINE=1` (or `HF_HUB_OFFLINE=1`) turns the check off. `--ranker none` is BM25 order,
`--ranker typesafe` the cloud; `INVENTIO_RANKER` fixes one for every query, `INVENTIO_DEVICE`
picks the device. `INVENTIO_DISPOSITIO_URL` reads a model served on another machine (the only shape
that sends the map anywhere, so non-public sources are refused).

The model also says when the passages it read do not seem to answer. That caveat is off by default:
`exists` is calibrated per corpus (medians on answerable pools 0.447, 0.232 and 0.377 across three
sets), so one threshold cannot hold a stated false-alarm rate across them. It also leans on BM25's
order: on MultiDoc2Dial its AUC is 0.805 as BM25 ranks the passages, 0.793 shuffled, 0.688 reversed.
`INVENTIO_DISPOSITIO_CAVEAT=0.15` turns it on at a threshold you choose:

```
# the passages read do not seem to answer this (p=0.12, below the 0.15 set here); they are the
# closest the map has; the model's best line was '    if len(crit) < 2:'
```

The ranker reads BM25's best 15 chunks (`--pool`) in one pass, then up to 7 of the chunks the question
names and the top hits link to, in a final beside the best 8 of the first pass (`--neighbours` adds the
chunks that share their most distinctive words). A pool longer than 26,000 characters, about the 6,656
tokens the model was trained on, drops passages from the tail. `--pool 30` reads 30 as two heats of 15 and a final over the best 8
and 7 (4 and 4 when widening takes 7 seats; a pass's probabilities share its pool, so two passes cannot be merged by score), about three
times as long; it reaches the answers BM25 puts at ranks 16-30 (MultiDoc2Dial 0.606 -> 0.640,
SWE-bench Lite code 0.599 -> 0.643, see [Benchmarks](../README.md#benchmarks)). On 40 questions over a private wiki, ticket tracker and two repositories, 15 found as many
answers as 30 and 10 lost some.

**v5 against v4** (v5 is v4 trained further on HotpotQA link-following states and Vietnamese law;
paired per question, bootstrap 95%; the model card has the rest):

| | v4 | **v5** |
|---|---|---|
| HotpotQA bridge questions (300), nDCG@10 with the linked pages read (BM25 alone 0.718) | 0.846 | **0.900** (+0.054 [+0.040, +0.069]) |
| Zalo legal (200), nDCG@10, BM25's 15 | 0.818 | 0.832 (+0.014 [−0.005, +0.033]) |
| MultiDoc2Dial: the passage that answers ranked first (453) | 0.638 | 0.664 (+0.027 [−0.004, +0.057]) |
| TechQA, never trained on (87) | 0.322 | 0.345 (+0.023 [−0.058, +0.103]) |
| examples/webshop, 13 on-call questions | 9/13 | **11/13** |
| Category judge, held-out passages (380): accuracy / macro-F1 | 0.811 / 0.769 | 0.811 / 0.774 |

**v4 against v3** (v3: the per-passage [Laya](https://github.com/NandhaKishorM/laya) model, 144M,
still fetchable as revision `v3` of the model repository; `main` carries v3's files). Same pools of 15, one pass per question:

| | v3 | **v4** |
|---|---|---|
| MultiDoc2Dial: the passage that answers ranked first (453 questions whose answer is in the pool) | 0.614 | **0.638** |
| TechQA, never trained on (87) | 0.200 | **0.322** (BM25 0.149) |
| examples/webshop, 13 on-call questions | 4/13 | **9/13** (BM25 6/13) |
| Category judge, held-out passages (380): accuracy / macro-F1 | 0.663 / 0.642 | **0.811 / 0.769** |
| SWE-bench Lite: answer file in pool after `--types` (300; BM25 30: 0.63, same-size BM25: 0.73) | not measured | **0.793** (TypeSafe as predictor: 0.813) |
| Time to read 15 candidates, RTX 5070 laptop | 0.15 s | 0.17 s (MultiDoc2Dial), 0.37 s (TechQA's long notes) |

Training data, the recipe and every reading are on the model card and in `benchmarks/systemone.py`;
`benchmarks/modal_bench.py` runs the judge, facts and type measurements on Modal.
