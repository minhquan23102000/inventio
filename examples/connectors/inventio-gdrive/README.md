# inventio-gdrive

Google Drive documents as an [inventio](../../../README.md) source: a folder and everything under
it, or all of My Drive, one Markdown file per document, laid out as the folder tree. One module,
[`inventio_gdrive.py`](inventio_gdrive.py), registered under the `inventio.connectors` entry
point in [`pyproject.toml`](pyproject.toml).

**Status:** written against the Drive API v3 reference and Google's OAuth guide for desktop apps,
and tested against a local server that answers in those shapes. It has not yet been run against
a real Google account.

## Set up (about 10 minutes, once)

Google lets a program read your Drive only through an OAuth client that you create:

1. In the [Google Cloud console](https://console.cloud.google.com/), create a project (any name).
2. **APIs & Services > Library**: enable the **Google Drive API**.
3. **Google Auth Platform > Branding**: fill in an app name and your email. Under **Audience**,
   choose **Internal** if your account belongs to a Google Workspace organisation; otherwise
   **External**, and add your own address as a test user.
4. **Clients > Create client > Desktop app**, then download its JSON.
5. Install and sign in:

```sh
uv tool install "inventio[pdf]" --with inventio-gdrive   # or, from this repo: --with ./examples/connectors/inventio-gdrive
inventio login https://drive.google.com                  # asks for the JSON's path, then opens the browser
inventio init https://drive.google.com/drive/folders/<id>
inventio query "..." -w format:sheet
```

The browser asks you to allow read-only access to Drive (`drive.readonly`); the answer comes back
to inventio on `127.0.0.1`, and the client and its refresh token are kept in the OS keychain.
`inventio logout https://drive.google.com` removes them. With **External** in testing, Google
expires the refresh token after 7 days; `sync` then says to run `inventio login` again. For a
one-off run, `GOOGLE_DRIVE_TOKEN` can hold an access token instead.

## What is read

| In Drive | In the map |
|---|---|
| Google Docs | Drive's Markdown export: headings, lists, tables, links (inline images dropped) |
| Google Sheets | the first sheet as a table (Drive exports one sheet as CSV) |
| Google Slides | the text of the slides |
| PDFs | the text of each page under `Page N`, as inventio reads a PDF on disk (needs `[pdf]`) |
| Markdown and text files | as they are |

Other files (Word, images, video) are left out. `-w` filters on `format` (doc, sheet, slides,
pdf, markdown, text), `owner`, `editor`, `created` and `updated`. A result links to the file in
Drive, not to a heading inside it. `sync` downloads a file again only when Drive's version of it
changed. The listing asks for shared-drive items too, so a shared drive's folder is synced by its
URL like one of your own (untested on a real shared drive).

## Test

```sh
uv pip install -e ../../.. -e . pytest pypdf
pytest tests
```
