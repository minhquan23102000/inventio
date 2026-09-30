# Sources

Each source is added with one `inventio init` and kept current with `inventio sync`.

## Folders

`inventio init <dir>` indexes a directory; code is cut at its functions and classes
(tree-sitter), Markdown and text at their headings. `sync` or another `init` re-reads only the
files whose size, time or content changed.

PDF files are read too: the text of each page is kept under a `Page N` heading, so a result says which page it came from and
`read` and `grep` show the page's words. The text is the PDF's own text layer: a scan has none,
and a PDF that needs a password cannot be opened; both are left out, and `init`/`sync` name them.
Two-column layouts and tables come out in the order the PDF stores its text.

## Signing in once

```sh
inventio login https://<site>.atlassian.net            # asks email and API token, tries them, keeps them
inventio login postgresql://reader@db.internal/core    # asks the password, connects once, keeps it
inventio sources                                       # each source's login: keyring, env, gh, driver or missing
inventio logout https://<site>.atlassian.net
```

A login is kept in the operating system's keychain (Windows Credential Manager, macOS Keychain,
Secret Service), never in the map, a mirror or a file beside your code, so `init` and `sync` work
from any directory. It is kept per place: every space and project of one Atlassian site shares
one login, and two sites keep two. `init` on a terminal asks for a login it does not have; run
by an agent or a script it stops and names the `inventio login` to run. Variables still come
first (`ATLASSIAN_EMAIL` and `ATLASSIAN_API_TOKEN`, `INVENTIO_SQL_PASSWORD`, the driver's own
`PGPASSWORD`), for CI and agents handed their credentials. GitHub signs in through the GitHub
CLI. A Linux machine without a keychain service has only the variables.

## Confluence and Jira

```sh
inventio init https://<site>.atlassian.net/wiki/spaces/OPS            # a Confluence space -> wiki-OPS
inventio init https://<site>.atlassian.net/browse/SHOP --jql "updated >= -365d"   # Jira -> jira-SHOP
```

Every request runs as the person who signed in, so a mirror holds only what they may read.

A remote source is mirrored as Markdown under the data directory, one file per page or ticket,
and indexed like a folder: the same chunks, links and judgments.

- **Pages** keep the page tree as folders. Links between pages of the space become `citation`
  links; `include` becomes a link to the page it pulls from; code stays code; tables stay
  tables, long ones repeating their header. Images and embedded diagrams (Lucidchart, Figma,
  draw.io) are not read but keep their place as `![Lucidchart diagram](url)`.
- **Tickets** are records: summary, status and people, links to other tickets, the
  description, then every comment under its own heading. A file named after its key defines
  it, so a page or a line of code that says `SHOP-812` links to the ticket.
- **Sync** lists each item's version (page version, ticket update time) without its body,
  fetches only what changed, moves renamed pages and deletes what is gone.
- **Results** carry the web address of their heading or comment beside the coordinate.
  `inventio read` takes either: a coordinate, or a page or ticket URL, read from the mirror
  when it is there and fetched live (not added to the map) when it is not. `inventio show`
  follows the links, so an agent can walk from a rule to its code to its incident.

Connectors live in `inventio/connectors/`: one module per kind lists items with a version and
turns one item into Markdown; mirroring, indexing and `read` are shared, so mail or chat is
one more module.

## GitHub issues and pull requests

```sh
gh auth login                                        # once, if the GitHub CLI is not signed in yet
inventio init https://github.com/acme/shop           # -> gh-shop
inventio query "why read from the replica" -w "item:pull state:merged"
```

Inventio keeps no GitHub token; each run asks the GitHub CLI for the one it holds (`gh auth
token`), for github.com and for a GitHub Enterprise host the CLI is signed in to (`GH_TOKEN`,
`GITHUB_TOKEN` or `GH_ENTERPRISE_TOKEN` without the CLI, or to force one). With several accounts
signed in (`gh auth login` again adds one), each repository is read with the first account that can
see it, the active one first, so a company repository syncs while a personal account is active;
sync prints which account it used when that is not the active one. One file per issue or pull request: the
description, the comments, and for a pull request its reviews and every review comment under a
heading with the file and line it is on and the last lines of its diff. `#12` and links to the
repository's own items become relative links, so they are `citation` links in the map; the file
defines `shop#12` and `acme/shop#12`, so a page or ticket that writes `acme/shop#12` links to it.
Sync lists every item with its update time and fetches only what changed. The code is not
fetched: index your clone with `inventio init <dir>`.

## Where the data lives

```sh
uv tool install "inventio[dispositio,data]" --with psycopg2-binary
#   DuckDB, SQLAlchemy, Kafka, boto3, and your database's driver in the same environment (pymysql for MySQL ...)
inventio init postgresql://reader@db.internal/core  # password from `inventio login`, INVENTIO_SQL_PASSWORD or PGPASSWORD
inventio init "kafka://broker:9092?registry=http://registry:8081"
inventio init s3://lake/warehouse/                  # AWS_* variables; AWS_ENDPOINT_URL for MinIO and the like
```

A runbook says "the job copies the orders database" without saying where. Each table, view, Kafka
topic, S3 dataset and local data file (`.parquet`, `.csv`, `.tsv`, `.jsonl`) becomes one
Markdown card: where to connect, the columns with their types and comments, keys and indexes,
a view's definition, a topic's partitions, retention and value schema. Only the catalog is read,
never a row; a topic without a registered schema has one message read for its field names and
types, and its values are not kept. A database password is never written to the map.

A card defines its table's name, so code, SQL, dbt models, runbooks and tickets that say
`shop.orders` link to it, and a foreign key is a link from one card to the other.
Cards are `Dataset` documents filed under `Reference` by code, never sent to a model. BM25
finds a card by its name and its column comments; a card without comments is best reached
through `inventio show` from the code or runbook that names it. S3 folders like `dt=2026-09-24/`
are partitions of one dataset, not datasets of their own.

## Your own source

A system inventio has no connector for (Notion, a ticket tool, an internal wiki) can be added
from a package of your own, without changing inventio. The package holds one module with the
names listed at the top of
[`inventio/connectors/__init__.py`](../inventio/connectors/__init__.py): `KIND`, `origin(url)` to
claim the URLs it syncs, `locate(url)`, `heading_url`, `fetch_url`, and a `Remote` whose
`listing()` says which items exist and their version, and whose `fetch(ids)` turns each one into
Markdown with the fields `-w` filters on. Register the module in the package's `pyproject.toml`:

```toml
[project.entry-points."inventio.connectors"]
notion = "inventio_notion"
```

Install it next to inventio (`uv tool install inventio --with inventio-notion`), and
`inventio init https://notion.so/...` mirrors it like any built-in source: only changed items
are fetched, `-w status:Open` filters, and each result links to its page on the web. A plugin
cannot replace a built-in kind, and one that fails to import is reported without stopping the
command. `connectors.CONNECTOR_API` (now 1) goes up when a name in that list changes meaning.
