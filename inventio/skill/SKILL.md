---
name: inventio
description: >-
  Finds where something is written across a team's code, Confluence pages, Jira tickets, GitHub
  issues and pull requests, and database, Kafka and S3 schemas, with the local `inventio` CLI
  (BM25 plus a small reranker; nothing leaves the machine). Use when a question has to be
  answered from internal docs, tickets or code ("why do we...", "where is X documented", "which
  ticket changed Y", "what table holds Z"), when a Confluence, Jira or GitHub URL needs reading,
  when every caller or every mention of a name must be found, or when inventio has to be
  installed, signed in, or given a new source (inventio init, sync, login, -w filters). Not for
  searching the public web.
---

# inventio

`inventio` keeps one map of every indexed source on this machine. Find with `query`, open with
`read`, follow with `show`, enumerate with `grep`. Every result carries a coordinate
`source:path:start-end` that `read` and `show` accept, so cite coordinates, not paraphrase.

## Check it is there

```sh
inventio sources
```

Lists each source with its file count, its login (`auth keyring|env|gh|driver|missing`) and when
it was synced. `command not found`: install it (below). `no sources`: ask the user which folders,
spaces, projects or repositories to index; do not index what they did not name, since a map
copies source text onto the machine.

## Install

```sh
uv tool install "inventio[laya] @ git+https://github.com/minhquan23102000/inventio"
```

`[laya]` adds the dispositio reranker (PyTorch, downloaded once); without it results come in BM25
order. Schema cards need `[data]` plus the database's driver in the same tool environment:
`uv tool install "inventio[laya,data] @ git+..." --with psycopg2-binary` (Postgres; `pymysql` for
MySQL). A `pip install` into another environment is invisible to a `uv tool` install.

## Sign in and add sources

```sh
inventio login https://<site>.atlassian.net          # asks email + API token, tries them, keeps them in the OS keychain
inventio login postgresql://reader@host/db           # asks the password, connects once, keeps it
gh auth login                                        # GitHub: inventio uses the GitHub CLI's login

inventio init ./repo --name app                      # a folder: code by function, docs by heading
inventio init https://<site>.atlassian.net/wiki/spaces/OPS
inventio init https://<site>.atlassian.net/browse/SHOP --jql "updated >= -365d"
inventio init https://github.com/acme/shop           # issues and pull requests; index the clone for code
inventio init postgresql://reader@host/db            # the catalog only, never a row
inventio sync                                        # later: fetch only what changed
```

`login` prompts, so it needs the user at a terminal: when `init` stops with "no login ...; run
`inventio login <url>`", hand that exact command to the user rather than asking them to paste a
token into the chat. Variables (`ATLASSIAN_EMAIL`, `ATLASSIAN_API_TOKEN`, `INVENTIO_SQL_PASSWORD`)
override the keychain; a stale one in your shell makes every call fail with 401 even after a
good `login`, so check `inventio sources` (it says `env`) before blaming the token.

Sources are private unless `init` gets `--public`. Never add `--public`, `--ranker typesafe` or
`--judge typesafe` on your own: they allow text to be sent to a cloud model.

## Answer a question

1. `inventio query "<the question in the user's words>" -k 5`. Ask it as a question; the reranker
   reads meaning, BM25 needs the words the documents use, so a second query with the likely
   terms (a table name, a job name, a ticket key) often finds what the first missed. BM25 does
   not cross languages: when the question is in one language (Vietnamese) and the documents may
   be in another (English), ask again in the documents' language before concluding.
2. `inventio read <coordinate>` for the lines behind each promising hit; the snippet in the
   query output is cut.
3. `inventio show <coordinate>` for where it leads: the code a runbook names, the ticket a page
   cites, the sections around it. Follow these links before concluding something is not
   written down.
4. `inventio grep "<exact name or regex>"` when the answer is "every": all callers, every page
   citing `SHOP-812`. `query` ranks a few; `grep` is exhaustive and prints the total.
5. `inventio ls <source>:<folder>/` to see what a source holds when the question is about
   structure.

Answer with the coordinate or URL of each passage you rely on. `read` also takes a Confluence,
Jira or GitHub URL someone pasted, from the mirror or fetched live.

If the reranker cannot load (MemoryError, no PyTorch, no network for its first download), add
`--ranker none`: the same pool in BM25 order, so read a few more hits before concluding.

## Narrow before searching: `-w`

On `query`, `grep` and `bench`, `-w` decides what may be searched before ranking, written like
GitHub's search box:

```sh
inventio query "why is the reconcile late" -w "kind:jira -status:Done updated:>=-30d"
inventio query "retention rule" -w "kind:confluence labels:policy"
inventio query "replica" -w "kind:github item:pull state:merged"
inventio grep "orders" -w "kind:sql"
```

Terms separated by spaces all hold; `a,b` is either value; `-key:v` negates; `>=` `>` `<=` `<`
compare (ISO dates, `-30d`, `-2w`); `*` wildcards; quotes for spaces (`assignee:"Nguyen An"`);
case does not matter. Keys every map has: `source kind type lang path category`. Connector
fields: Jira `status resolution issuetype priority assignee reporter labels project created
updated`; Confluence `space author editor labels created updated`; GitHub `repo item state author
assignee labels milestone created updated closed`. A file without the field does not match the
term and does match its negation. A misspelt key exits 2 and lists the keys the map has; read
that list instead of guessing. Values are the source's own words (a Jira status may be
`DONE/ClOSED`), so `grep` or `read` one item to see them before filtering on them.

## For scripts

`--json` on `query`, `read`, `show`, `grep`. Exit codes: 0 found, 1 no match (for `query`, "0
chunks in scope" means the filter is too tight, not that the answer is absent), 2 bad input or a
source refused, 3 a cloud ranker refused private text.
