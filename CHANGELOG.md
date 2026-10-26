# Changelog

Notable changes to inventio, newest first. Versions are on [PyPI](https://pypi.org/project/inventio/);
the ranker models are on [Hugging Face](https://huggingface.co/minhquan2310/dispositio).

## Unreleased

### Added
- `inventio serve --http PORT`: a server that runs until stopped, loads the model at start and answers
  `POST /query` (`{"text", "where", "source", "k", "pool", "ranker", ...}`) with exactly the rows
  `query --json` prints, on 127.0.0.1 only, for a program that asks many questions (a portal, an agent
  harness). The URL and a bearer token are written to `serve-http.json` in the data directory (owner
  only); `GET /health` says whether the model is loaded. The map is opened per request, so a `sync` is
  seen by the next question. `serve --stop` and `update` stop it too.
- `serve --http --map NAME=PATH` (repeatable): one server answers from several maps; a request names
  its map with `"map"`, and naming none is refused when there are several. The paths are fixed when the
  server starts, so a caller chooses a map, never a file. `GET /health` lists the names.
- `query --json --locations-only` (`"locations_only": true` over HTTP): each hit's place (source, path,
  url, heading, lines, rank) without its text or links, for a caller that reads the content itself
  with its own credential.

### Changed
- BM25 order is the default ranker even when the dispositio extra is installed; `--ranker dispositio`
  or `INVENTIO_RANKER=dispositio` asks for the model. On a CPU the model costs about 15 s a query
  against 35-140 ms for BM25, and on 10 questions over an internal wiki with known answer pages it
  found the same number of them in its top 8 as BM25 alone (9 of 10; median 14.2 s against 0.20 s on
  an M3).

- A mirrored item's web location (the manifest's `url` and `anchors`) is written into the map at sync
  (`file_web`), and `query`, `read` and `serve --http` read it from there. A server that runs for days
  used to keep the manifest it read first, so a page renamed or moved after that came back without a
  URL; and a server on another disk than the mirror (a container beside the one that syncs) answered
  with no URL at all. A map synced by an earlier version reads the manifest until its next `sync`.
- A ranked query asks dispositio only `where_passage`, the answer it ranks by: `where_line` and
  `exists` (read only by the `INVENTIO_DISPOSITIO_CAVEAT` note) are asked when that variable is set.
  Every question is its own row over the passages, so the ranking is unchanged (the same top 8 pages
  on 16 questions over an internal wiki); on an M3 a default query with `-k 20` went from 13.0 s to
  9.3 s at the median (one pass ~6 s -> ~4 s).

## 0.5.6 (2026-09-29)

### Changed
- PDF files are read by default: `pypdf` is a dependency of inventio, the `pdf` extra is gone, and a
  PDF is no longer skipped with a note to install it. `inventio[pdf]` still installs; pip only
  says the extra does not exist.

## 0.5.5 (2026-09-29)

### Added
- PDF files, with the `pdf` extra (pypdf): the text of each page under a `Page N` heading, so a
  result says which page it came from and `read`/`grep` show its words. A scan (no text layer),
  a password-protected PDF or a damaged one is left out, and `init`/`sync` say which and why.
- Plugins can sign in: `inventio login`/`logout <url>` and `sources` hand a plugin's URL to its
  own `login`, `logout` and `credential`; `connectors.http.Client` gains `post` (a JSON body)
  and `raw` (a download or an export).
- Two connectors outside the package, in `examples/connectors/`: `inventio-notion` (every page
  an integration can see, or one page tree) and `inventio-gdrive` (a Drive folder or My Drive:
  Docs as Markdown, Sheets as a table, Slides as text, PDFs by page). Both are tested against
  local servers shaped like the documented APIs, not yet against real accounts.

### Fixed
- A plugin imported before inventio (a test, a script) no longer fails to register: it is
  registered by its entry point's name, which the interface now says must be its `KIND`.

## 0.5.4 (2026-09-29)

### Added
- Connectors from other packages: a package registers a connector module under the entry-point
  group `inventio.connectors`, and once installed beside inventio its URLs work with `init`,
  `sync`, `-w` filters and web links like a built-in source. A plugin that fails to import is
  reported and skipped; one that names a built-in kind is refused. The interface is the list at
  the top of `inventio/connectors/__init__.py` (now also naming `Remote.FORMAT`), versioned by
  `connectors.CONNECTOR_API = 1`. README: "Your own source".

## 0.5.3 (2026-09-29)

### Faster
- `grep` searches the text of each file kept in the map and opens only files whose size or
  modification time changed since the last sync, so the output is the same as before. On a map
  of 7,169 files on a MacBook (M3): 0.24-0.45 s, against 26-94 s for 0.5.2 when the files were
  not in the OS cache (about 1 s when they were). The map grows by the size of the text it
  indexes (80 to 96 MB there), and the first `sync` after upgrading reads every file once.

### Changed
- GitHub: with several accounts signed in to the GitHub CLI, each repository is read with the
  first account that can see it, the active one first, so a company repository syncs while a
  personal account is active, without `gh auth switch`. `sync` says which account it used when
  it is not the active one. `GH_TOKEN` or `GITHUB_TOKEN`, when set, is still used as given.
- `sync`: a source that fails (a login, the network) is reported and the other sources still
  sync; the exit code is then 2. Before, the first failure stopped the whole sync.

## 0.5.2 (2026-09-29)

### Faster
- dispositio on Apple GPUs: a rewritten gated delta rule for the model's DeltaNet layers (no
  compiled kernel exists for them on MPS) and SDPA attention. A warm query with links went from
  about 29 s to about 16 s on an M3, with the same top three results. CUDA and CPU are unchanged.

## 0.5.1 (2026-09-28)

### Fixed
- The background model server released no accelerator memory between queries (6 to 17 GB over
  five queries on a 16 GB Mac); it now frees it after each pass.
- A second server was started while the first was busy with a long query; the server now answers
  the client's ping while a query runs.
- Remote fetches retry timeouts and responses cut short mid-transfer, which ended large GitHub
  syncs around 3,000 items.

### Changed
- Agent skill: ask in the language the documents are written in, and use `-w` for time
  (`created:>=-30d`) and `grep` for counts.

## 0.5.0 (2026-09-28)

### Changed
- Default model dispositio v5, trained further to follow links and on Vietnamese law.
- Links are drawn and named by code from what the map knows (definitions, citations,
  mentions), and printed as kind, fact and title. Two prose passages naming the same defined
  identifier are linked; a Jira, Confluence or GitHub URL written in any source becomes a citation
  to the mirrored item.
- dispositio reads the chunks that links and named files add, in one extra pass.
- Prose sections are also cut before a list item or a legal clause, so long pages give shorter
  chunks.
- `--neighbours` is off by default (on SWE-bench Lite it cost 0.011 nDCG and 0.47 s a query).
- `bench --rows` writes each question's rank, how the answer entered, and the top 10.

## 0.4.0 (2026-09-27)

First release on PyPI.

- A local map of code, documents, Confluence spaces, Jira queries, GitHub issues and pull
  requests, database, Kafka and S3 schemas, in one SQLite file; BM25 plus structure and links,
  with the dispositio model (v4) reranking on the machine by default.
- `init`, `sync`, `query`, `read`, `show`, `grep`, `ls`, `facts`, `sources`, `login`, `bench`;
  `-w` filters on source fields (status, labels, dates, author).
- Logins kept in the OS keychain; GitHub through the GitHub CLI's own login.
- A background server keeps the model loaded between queries.
- An agent skill (`inventio skill`).
