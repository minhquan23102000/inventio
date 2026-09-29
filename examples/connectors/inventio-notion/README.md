# inventio-notion

Notion pages as an [inventio](../../../README.md) source, and a worked example of a connector
written outside inventio: one module, [`inventio_notion.py`](inventio_notion.py), registered
under the `inventio.connectors` entry point in [`pyproject.toml`](pyproject.toml).

**Status:** written against Notion's API reference and the official SDK's types (API version
2025-09-03), tested on a local server that answers in those shapes. The sign-in path has been
tried against the real API (a wrong token is refused with 401 and nothing is kept); a sync of a
real workspace has not been run yet.

## Set up (about 5 minutes, a free Notion account is enough)

1. At <https://www.notion.so/profile/integrations>, create an **internal** integration for your
   workspace and copy its token. Read content is all it needs.
2. In Notion, open each top page you want indexed, then `...` > **Connections** > add the
   integration. Its subpages come with it.
3. Install and sign in:

```sh
uv tool install inventio --with inventio-notion      # or, from this repo: --with ./examples/connectors/inventio-notion
inventio login https://www.notion.so                 # asks for the token, tries it, keeps it in the keychain
inventio init https://www.notion.so                  # every page the integration can see -> source "notion"
inventio init https://www.notion.so/Runbook-<id>     # one page and its subpages -> source "notion-<id>"
inventio query "how is the nightly backup restored" --source notion
inventio query "replica lag" -w 'status:"In progress" labels:db'
```

`NOTION_TOKEN` in the environment is used before the keychain (CI, agents).

## What a page becomes

- One Markdown file per page, laid out as the page tree (a database's rows under a folder named
  after the database). Only pages edited since the last sync are fetched again.
- Headings stay headings; each links back to its block on the web.
- A subpage, a link to a page, a relation and an `@`-mention of a page that is mirrored become
  relative links, so they are `citation` links in the map.
- A database row starts with its properties (`- **Status:** Done`). Status, select,
  multi-select, people, checkbox and date properties become `-w` fields named after the property
  in lower case (`status`, `priority`, `owner`); `Tags` also becomes `labels`; every page has
  `created` and `updated`.
- Images and files keep their place as `![caption](link)`. Files Notion hosts point at their
  block, because Notion's own file links expire within the hour.
- Tables of contents, breadcrumbs and blocks the API does not expose are left out.

## Limits

- One integration token per machine: two workspaces need two machines or `NOTION_TOKEN`.
- The page list comes from Notion's search. When search reports that it stopped early (its
  `request_status` says `incomplete`), sync fails instead of mirroring part of the workspace.
- Comments, page history and the files themselves (PDFs attached to a page) are not read.

## Test

```sh
uv pip install -e ../../.. -e . pytest
pytest tests
```
