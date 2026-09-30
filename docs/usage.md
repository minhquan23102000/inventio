# Using inventio

Reading what a query prints, what the map holds, and the commands that open, follow and narrow a result. Start from [Quick start](https://github.com/minhquan23102000/inventio/blob/main/README.md#quick-start).

## Reading a result

The query in [Quick start](https://github.com/minhquan23102000/inventio/blob/main/README.md#quick-start) prints:

```
== Article
1. wiki:runbook.md:8-13  Nightly backup runbook > When the nightly backup has not finished  p=0.96
   1. Check the scheduler at 07:00. If `nightly_backup` is still running or has failed, stop it. 2. Take a fresh backup from the replica, not the primary: `make ba…
   -> code · defines nightly_backup · app:jobs/backup.py:4-9  nightly_backup
   -> page · names nightly_backup too · wiki:incidents/2026-03-14.md:3-8  Incident 2026-03-14: no backup to restore > What happened
2. wiki:runbook.md:3-6  Nightly backup runbook > Nightly backup  p=0.03
   The job `nightly_backup` copies the orders database to off-site storage. It starts at 01:00 and must finish before the morning order peak at 08:00.
   -> code · defines nightly_backup · app:jobs/backup.py:4-9  nightly_backup
   -> page · names nightly_backup too · wiki:incidents/2026-03-14.md:3-8  Incident 2026-03-14: no backup to restore > What happened
3. wiki:incidents/2026-03-14.md:3-8  Incident 2026-03-14: no backup to restore > What happened  p=0.00
   The `nightly_backup` run filled the storage bucket at 03:40 and stopped without an error, and nobody checked the scheduler. At 11:00 a bad migration corrupted t…
   -> code · defines nightly_backup · app:jobs/backup.py:4-9  nightly_backup
   -> page · names nightly_backup too · wiki:runbook.md:3-6  Nightly backup runbook > Nightly backup
   -> page · names nightly_backup too · wiki:runbook.md:8-13  Nightly backup runbook > When the nightly backup has not finished
```

The runbook's steps come first; `p` is dispositio's probability that the passage answers, and
results are grouped by document type, numbered by rank. Each line under a result is where it
leads, and every word on it comes from the sources: what the target is (`code`, from its path),
the fact that makes the link (the runbook names `nightly_backup`, `jobs/backup.py` defines it;
the incident report names the same job), and the target's own title. No model draws or names
these links, so a link on screen is never a guess. Between tickets the verb is the ticket's
own (`is caused by SHOP-2`), and a Jira, Confluence or GitHub URL to a page or ticket the map
mirrors links to it.

A list of steps says what to do, not why. `inventio facts` has dispositio read every prose
chunk once and say what it does for its reader (its category). Links then say what waits at
the other end, `steps`, `record` or `finding`, where that judgment is confident enough (at least
0.95 agreement with the reference labels on 380 held-out passages); below that they stay `page`.

```sh
inventio facts
inventio query "why do we check the backup scheduler at 07:00?" -k 2
```

```
== Article
1. wiki:incidents/2026-03-14.md:10-13  Incident 2026-03-14: no backup to restore > What changed  p=0.92
   The 07:00 scheduler check was added to the runbook, and the job now pages the on-call engineer when it has not finished by 06:30 or when the bucket is more than…
2. wiki:incidents/2026-03-14.md:3-8  Incident 2026-03-14: no backup to restore > What happened  p=0.04
   The `nightly_backup` run filled the storage bucket at 03:40 and stopped without an error, and nobody checked the scheduler. At 11:00 a bad migration corrupted t…
   -> code · defines nightly_backup · app:jobs/backup.py:4-9  nightly_backup
   -> page · names nightly_backup too · wiki:runbook.md:3-6  Nightly backup runbook > Nightly backup
   -> steps · names nightly_backup too · wiki:runbook.md:8-13  Nightly backup runbook > When the nightly backup has not finished
```

The why is in the incident report: the check was added after the night a backup filled its
bucket at 03:40 and stopped without an error. From there the map reaches the code of the job
and the runbook step that now carries the check. `facts` also judges which chunks of different
categories speak about the same thing (`about` links); `inventio show` lists them, marked as
judged, but they are not printed under results, since a judged link can be wrong (see
[Limitations](https://github.com/minhquan23102000/inventio/blob/main/README.md#limitations)).

## What the map holds

- **Chunks** with their file path and heading path indexed next to the text, so BM25 matches
  where a passage sits as well as what it says.
- **Document types** from the path: `SoftwareSourceCode`, `Test`, `Configuration`, `Article`,
  and `Dataset` for the card of a table, topic or data file (see [Where the data lives](sources.md#where-the-data-lives)).
- **Links drawn by code.** `citation`: a Markdown link, resolved to the chunk it points at.
  `mentions`: a chunk names what another defines, or two files share a rare identifier
  (`nightly_backup`, `SHOP-812`). This connects code to the prose written about it.
- **Categories** (after `inventio facts`): what a prose chunk does for its reader.

  | Category | The passage | A question it answers |
  |---|---|---|
  | Rule | states what must, may or must not be done | "is it allowed to...", "what is the limit" |
  | Procedure | walks through how to do something | "how do I..." |
  | Reference | describes what something is or contains, for lookup; every `Dataset` card is filed here by code, not judged | "what are the fields of...", "which table holds..." |
  | Explanation | explains why or how something works | "why does..." |
  | Finding | reports what was measured or observed | "does it work", "how much faster" |
  | Record | tells what happened or was decided | "when did this change", "what broke last time" |
  | Other | none of these: a table of contents, credits, boilerplate | |

  Categories cut a corpus into regions instead of all covering it, so `--facts` widens toward
  the kind of passage the question wants. `about` links join passages of different categories
  that state something about the same thing: the rule, the runbook that carries it out, the
  incident that broke it.
- **Judgments.** Every model decision is stored with the text it read, so a rebuilt map pays
  nothing twice.

## Working with a result

```sh
inventio read app:jobs/backup.py:4-9          # the lines
inventio show wiki:runbook.md:8-13            # what the map knows: type, category, links, sections around
inventio grep "nightly_backup"                # every line that says it: code, pages, tickets
inventio ls wiki                              # a source like a folder; a file lists its sections
inventio sync                                 # folders re-read, pages and tickets: only what changed
```

```
$ inventio show wiki:runbook.md:8-13
wiki:runbook.md:8-13  Nightly backup runbook > When the nightly backup has not finished
  Article (by path) · markdown · public
  section 8-13 · Procedure p=1.00 (jev-latest) · mentions 1 name
links
  -> mentions app:jobs/backup.py:4-9  nightly_backup  · SoftwareSourceCode  (nightly_backup)
  ~  about    wiki:incidents/2026-03-14.md:10-13  Incident 2026-03-14: no backup to restore > What changed  · Article · Record p=1.00 (jev-latest)  p=0.93 (jev-latest)
  ~  about    wiki:policy.md:3-8  Data retention policy > Backup retention  · Article · Rule p=1.00 (jev-latest)  p=0.65 (jev-latest)
  ~  about    wiki:incidents/2026-03-14.md:3-8  Incident 2026-03-14: no backup to restore > What happened  · Article · Record p=1.00 (jev-latest)  p=0.84 (jev-latest)
structure
  < wiki:runbook.md:3-6  Nightly backup runbook > Nightly backup  · Article · Rule p=0.38 (jev-latest)
```

`query` is for a question, `show` for where a result leads, `grep` for an exact name (every
caller, every page citing a ticket), `ls` for seeing what is there. Each prints coordinates
that `read` and `show` open, and `read` also takes a page or ticket URL someone pasted in chat;
`inventio -h` shows the same walk. `show` labels every fact with what decided it: the path or a
link (code), or a model with its probability (category, `about`). `--json` gives the same for
agents.

All sources share one map, so a query and the links reach across them (the job in `app`, the
runbook in `wiki`, a ticket in Jira), and `--db` or `INVENTIO_DB` keeps a separate map. It lives
in your user data directory (`%LOCALAPPDATA%\inventio`, `~/Library/Application Support/inventio`,
or `~/.local/share/inventio`), never inside an indexed repository.

### Narrowing a search

```sh
inventio query "why no backup" -w "kind:jira -status:Done updated:>=-90d"
inventio query "retention" -w "path:Ops/* type:Article"
inventio grep "nightly_backup" -w "source:app,wiki"
```

`-w` (on `query`, `grep` and `bench`) decides what a search may look at before BM25 runs, the way
GitHub's search box reads: `key:value` terms separated by spaces all hold, `a,b` is either value,
`-key:value` negates, `>=` `>` `<=` `<` compare (dates are ISO; `-90d` and `-2w` count back from
today), `*` is a wildcard, quotes hold a value with spaces (`assignee:"Nguyen An"`), and case does
not matter. Every way into the pool (names the question uses, predicted types and categories,
neighbours, links) keeps to it, so the 15 candidates are all spent inside the scope.

Every map has `source`, `kind` (`dir`, `confluence`, `jira`, `github`, `sql`, `kafka`, `s3`),
`type`, `lang`, `path` and `category`. Connectors add their items' fields: Jira `status`,
`resolution`, `issuetype`, `priority`, `assignee`, `reporter`, `labels`, `project`, `created`,
`updated`; Confluence `space`, `author`, `editor`, `labels`, `created`, `updated`; GitHub `repo`,
`item` (`issue` or `pull`), `state` (`open`, `closed`, `merged`, `draft`), `author`, `assignee`,
`labels`, `milestone`, `created`, `updated`, `closed`. A chunk without the field does not match a
term on it and does match its negation, so `-status:Done` keeps the code. A key the map does not
hold is an error that lists the keys it does; a scope with nothing in it says so instead of "no
match". `--source NAME` is `-w source:NAME`.
