# Inventio

**inventio connects everything: lightweight, fast, simple, accurate.**

Ask once and get back the exact lines that answer, whether they sit in code, Confluence, Jira,
GitHub or a database schema, for a person or a coding agent.

```sh
inventio init ~/code/shop                                        # code, Markdown, text and PDF
inventio init https://acme.atlassian.net/wiki/spaces/OPS         # Confluence -> wiki-OPS
inventio init https://acme.atlassian.net/browse/SHOP             # Jira -> jira-SHOP
inventio init https://github.com/acme/shop                       # issues and pull requests -> gh-shop
inventio init postgresql://reader@db.internal/core --name core   # table schemas (the data extra)
inventio init s3://lake/warehouse/                               # dataset schemas
# Kafka topics too; Notion and Google Drive through plugins (examples/connectors)
```

*Illustrative: the shape of one result across three systems. The runnable example, with real
output, is in [Quick start](#quick-start).*

```
$ inventio query "why did checkout fail last Friday?" -k 1
== Article
1. wiki-OPS:Incidents/Checkout-failures-2026-09-26.md:3-9  Checkout failures on 26 September > Cause  p=0.94
   Orders with a discount code failed at payment from 17:40 to 18:25: apply_discount rounded the total before tax, and the pa…
   https://acme.atlassian.net/wiki/spaces/OPS/pages/48213#Cause
   -> bug · defines SHOP-812 · jira-SHOP:SHOP/SHOP-812.md:1-12  SHOP-812 Discount rounding breaks payment totals
   -> code · defines apply_discount · shop:checkout/discount.py:41-63  apply_discount
   -> table · defines shop.orders · core:shop/orders.md:1-24  shop.orders
```

The answer is on a Confluence page; under it are the Jira bug, the code and the table the page
names. Code draws these links from names the sources share, not a model.

- **Lightweight.** Everything lives in one SQLite file: no vector database, no index server, no
  embeddings. Without the model extra it installs in seconds. The local model (dispositio, 0.8B)
  is optional, 1.4 GB, downloaded once.
- **Fast.** BM25 answers in 35-140 ms per query on a CPU. With the model, reading 15 candidates
  takes about 0.16 s on a laptop GPU (RTX 5070); on a CPU alone, about 15 s. `inventio sync`
  fetches only what changed.
- **Simple.** One command per source, `inventio init <folder or URL>`. Sign in once; the login is
  kept in the operating system's keychain, and Confluence and Jira are read as the person who
  signed in.
- **Accurate.** Every answer is a coordinate, `source:path:start-end`, so you open the exact
  lines. On SWE-bench Lite (find the files a GitHub issue's fix touches, 300 issues, nDCG@10),
  inventio with dispositio reaches 0.643 against the 7B embedder SFR-Embedding-Mistral at 0.627;
  with no model, on a CPU, 0.540 against BGE-base at 0.449. Embedders still lead on
  StackOverflow QA (E5-Mistral 7B 0.915 against 0.711). The embedder figures are published,
  single-stage over the whole corpus; see [the full table](#benchmarks).

## Quick start

The repository carries a small example: the code of an online shop's nightly database backup,
and the wiki around it, a runbook, a retention policy and an incident report. Everything below
runs as shown:

![inventio answering an on-call question with wiki:runbook.md:8-13 and opening those lines](https://raw.githubusercontent.com/minhquan23102000/inventio/main/docs/assets/demo.gif)

```sh
uv tool install "inventio[dispositio]"
inventio skill                   # teaches your coding agents (~/.agents/skills) to install and use inventio
git clone https://github.com/minhquan23102000/inventio && cd inventio

inventio init examples/webshop/app --name app --public     # --public: may be sent to a cloud judge
inventio init examples/webshop/wiki --name wiki --public
inventio query "the nightly backup has not finished, what do I do?" -k 3
```

## Use it from your coding agent

```sh
uv tool install "inventio[dispositio]"
inventio skill                   # into ~/.agents/skills; --project for this repository's .agents/skills
```

Agents that read `.agents/skills` then know how to install inventio, add sources (`init`, `sync`,
`login`), find with `query`, open with `read`, follow links with `show`, and cite coordinates
instead of paraphrasing. Run `inventio skill` again after an upgrade to refresh it.

## Installing

`uv tool install` puts `inventio` on your PATH for every terminal; `uvx --from "inventio[dispositio]"
inventio ...` runs it once without installing. The unreleased `main` installs with
`"inventio[dispositio] @ git+https://github.com/minhquan23102000/inventio"`.
Without `[dispositio]` it installs in seconds and ranks by BM25 alone; `[dispositio]` adds PyTorch and
the model that ranks, judges `--facts` and predicts `--types` (1.4 GB, downloaded once from Hugging
Face on the first query). A GPU reads 15 candidates in about 0.16 s (RTX 5070 laptop); a CPU reads
them too, in about 15 s (fp32, 4,200 tokens of state). `--pool 30` reads twice as many in three passes,
about three times as long.

## Sources

| Source | Add it with | Details |
|---|---|---|
| Code, Markdown, text and PDF | `inventio init <dir>` | [Folders](https://github.com/minhquan23102000/inventio/blob/main/docs/sources.md#folders) |
| Confluence | `inventio init https://<site>.atlassian.net/wiki/spaces/OPS` | [Confluence and Jira](https://github.com/minhquan23102000/inventio/blob/main/docs/sources.md#confluence-and-jira) |
| Jira | `inventio init https://<site>.atlassian.net/browse/SHOP` | [Confluence and Jira](https://github.com/minhquan23102000/inventio/blob/main/docs/sources.md#confluence-and-jira) |
| GitHub issues and pull requests | `inventio init https://github.com/acme/shop` | [GitHub](https://github.com/minhquan23102000/inventio/blob/main/docs/sources.md#github-issues-and-pull-requests) |
| SQL, Kafka and S3 schemas | `inventio init postgresql://…`, `kafka://…`, `s3://…` (the `data` extra) | [Where the data lives](https://github.com/minhquan23102000/inventio/blob/main/docs/sources.md#where-the-data-lives) |
| Anything else (Notion, Google Drive, …) | a plugin package of your own | [Your own source](https://github.com/minhquan23102000/inventio/blob/main/docs/sources.md#your-own-source) |

Sign in once with `inventio login <url>`; the login is kept in the operating system's keychain
([Signing in once](https://github.com/minhquan23102000/inventio/blob/main/docs/sources.md#signing-in-once)). `inventio sync` fetches only what changed.

## More

- [Using inventio](https://github.com/minhquan23102000/inventio/blob/main/docs/usage.md): reading a result, what the map holds, opening, following and narrowing results.
- [How it works](https://github.com/minhquan23102000/inventio/blob/main/docs/how-it-works.md): ingest and query, step by step.
- [dispositio](https://github.com/minhquan23102000/inventio/blob/main/docs/dispositio.md): the local model that ranks and judges.
- [Benchmarks](https://github.com/minhquan23102000/inventio/blob/main/benchmarks/README.md): method, reproduction, and what the numbers say set by set.

## Benchmarks

nDCG@10 on every test query, through Inventio's real ingest and query path; the rankers reorder
the same 30 BM25 candidates (`--pool 30`; `query` hands the ranker 15 by default, see
[dispositio](https://github.com/minhquan23102000/inventio/blob/main/docs/dispositio.md)). dispositio reads at most 15 in one pass: its plain
row is the first 15 of the 30 (the rest keep BM25's order), the `--pool 30` row reads all 30 the way
`query --pool 30` does, as two heats of 15 and a final over the best 8 and 7.
Method and reproduction: [benchmarks/README.md](https://github.com/minhquan23102000/inventio/blob/main/benchmarks/README.md#reproduce); what the numbers say, set by set: [Reading the table](https://github.com/minhquan23102000/inventio/blob/main/benchmarks/README.md#reading-the-table).

| System | Runs on | SWE-bench Lite | SciFact | StackOverflow QA | Zalo legal | MultiDoc2Dial | TechQA |
|---|---|---|---|---|---|---|---|
| **Inventio + Jev** | TypeSafe cloud, ~1.2 s/query | **0.696** | **0.765** | 0.791 | not run | 0.486 | **0.655** |
| **Inventio + dispositio**, `--pool 30`&nbsp;\* | laptop GPU, 0.9-2.7 s/query | 0.643 | 0.716 | 0.711 | **0.805** | **0.640** | 0.489 |
| **Inventio + dispositio** | laptop GPU, 0.3-1.0 s/query | 0.599 | 0.718 | 0.702 | **0.805** | 0.606 | 0.467 |
| **Inventio**, no model | CPU, 35-140 ms/query | 0.540 | 0.670 | 0.670 | 0.756 | 0.470 | 0.370 |
| **Inventio + Laya**, not tuned | laptop GPU, 0.6-0.9 s/query | 0.391 | 0.302 | 0.193 | 0.512 | 0.389 | 0.171 |
| E5-Mistral 7B | 7B embedder | – | 0.764 | **0.915** | – | – | – |
| SFR-Embedding-Mistral 7B | 7B embedder | 0.627 | – | – | – | – | – |
| Voyage-Code-2 | Voyage cloud | 0.291 | – | 0.877 | – | – | – |
| Jina-v2-code | 161M embedder | 0.583 | – | – | – | – | – |
| BGE-base | 110M embedder | 0.449 | 0.740 | 0.736 | – | – | – |

SWE-bench Lite: a GitHub issue, find the code files its fix touches (300 issues). SciFact: a
claim, find the abstract that settles it (300). StackOverflow QA: a question, find its accepted
answer (1,994). Zalo: a Vietnamese question, find the law article (788). MultiDoc2Dial: a
question about a US public-service page, find the section that answers (615). TechQA: a question
from IBM's support forums, find the passage of the technote that answers (119). Published
figures come from CodeRAG-Bench, CoIR and the models' MTEB cards; they are single-stage
embedders over the whole corpus, while Inventio with a ranker is two-stage.

\* Not re-run through the benchmark scripts: the five text sets reuse the plain dispositio row's first pass, cached,
and add the second heat and the final over the same BM25 pools (`query --pool 30` gives the same top 10
on the 20 TechQA queries checked); SWE-bench Lite was run on Modal. The time is the plain row's plus the two
extra passes, measured per set on the laptop.

## Privacy

- Every source is private unless `init` gets `--public`.
- Jev refuses (exit code 3) and sends nothing when any candidate comes from a private source;
  `facts --judge typesafe` refuses a private source the same way.
- With dispositio as ranker and judge the whole path runs offline, apart from the daily release
  check ([dispositio](https://github.com/minhquan23102000/inventio/blob/main/docs/dispositio.md)), which carries no query or text; `python benchmarks/local_proof.py <dir>
  "<question>"` (Hugging Face offline) fails if any connection is attempted.
- The map holds source text. Keep it out of indexed trees and version control. Mirrors of
  Confluence, Jira and GitHub live beside it in the data directory; `drop` deletes a source's mirror.
- Logins live in the operating system's keychain (`inventio login`), never in the map or a mirror.

## Limitations

- Directories, Confluence Cloud, Jira Cloud, GitHub issues and pull requests, and the schemas of
  SQL databases, Kafka topics and S3 datasets; no Slack, mail, GitHub Discussions or wiki yet.
  Sync is on demand (`inventio sync`); nothing listens for changes. A dbt project is read as its
  SQL files, not yet its `manifest.json` descriptions.
- A GitHub sync lists the whole repository and asks two more requests per changed pull request
  (reviews, review comments) and one per issue with comments: a repository of a few thousand
  items may meet the API's hourly limit on its first sync, which stops there and resumes on the
  next. A bare `#12` in code or in another source names no repository and links nowhere.
- Filtering by a connector's fields needs mirrors written by this version: the first `sync`
  after upgrading fetches every Confluence page and Jira ticket once more.
- Text inside images, diagrams and attached files is not read.
- dispositio v5 puts the right passage first for 11 of the 13 questions of examples/webshop (v4: 9).
  How well it carries to a team's own documents is measured on those 13 and on 40 private questions
  only.
- `about` links need a judge of "are these two passages about the same thing". dispositio was
  not trained for it and links almost nothing (2 pairs of 30,517 on SciFact); the earlier Laya model
  linked nearly every pair. They are not printed under results; `show` lists them as judged.
- On SWE-bench Lite (300 issues, code and docs, pool 15) the files and definitions a question names
  add +0.074 nDCG@10 [+0.048, +0.102]; the links their top hits carry add +0.007 [−0.015, +0.030],
  and the shared-word neighbours −0.011 [−0.029, +0.007] for 0.5 s more a query. On HotpotQA bridge
  questions, where the answer page is one a first page links to, reading the linked pages adds
  +0.100 [+0.080, +0.121] with v5. What links add on a real wiki, tracker and repository is not
  measured yet.
- The categories `inventio facts` assigns ([What the map holds](https://github.com/minhquan23102000/inventio/blob/main/docs/usage.md#what-the-map-holds)) are new. Whether `--facts` with them finds answers a same-size BM25 pool
  does not is not measured yet; the earlier measurement, with schema.org types, is in
  [benchmarks/README.md](https://github.com/minhquan23102000/inventio/blob/main/benchmarks/README.md#categories-and-fact-links---arms).

## Why the name

> *Inventio*, from *invenire*: to come upon. In classical rhetoric the orator did not make his
> material up; he went looking through the *loci*, the places where it already lay.

## License

Apache-2.0. dispositio's weights carry its training data's terms; see its model card.
