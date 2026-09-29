"""Google Drive for inventio: a folder (or all of My Drive) and everything under it, one Markdown
file per document, laid out as the folder tree.

    inventio login https://drive.google.com                       # once: a browser asks you to allow read access
    inventio init https://drive.google.com/drive/folders/<id>     # a folder, shared drives too -> gdrive-<id>
    inventio init https://drive.google.com/drive/my-drive         # all of My Drive -> gdrive

What is read, and how:
- Google Docs: Drive's own Markdown export, so headings, lists, tables and links stay.
- Google Sheets: the first sheet as a Markdown table (Drive exports one sheet as CSV).
- Google Slides: the slides' text (Drive's plain-text export).
- PDFs: the text of each page, as inventio reads a PDF on disk (the `pdf` extra).
- Markdown and plain-text files: as they are.
Other files (Word, images, video) are left out. A file is fetched again only when Drive's
version of it changed.

Written against the Drive API v3 and Google's OAuth guide for desktop apps, and tested on a
local server that answers in their documented shapes; not yet run against a real Drive.
"""

import base64
import hashlib
import http.server
import json
import os
import posixpath
import re
import secrets
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

from inventio import credentials
from inventio.connectors.http import Client, RemoteError
from inventio.connectors.markdown import segment, table, tidy
from inventio.connectors.mirror import Doc, Entry

KIND = "gdrive"
API = "https://www.googleapis.com/drive/v3"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPE = "https://www.googleapis.com/auth/drive.readonly"
WEB = "https://drive.google.com"
KEY = "gdrive"  # keychain key: one Google account per machine
MAX_BYTES = 50_000_000
FOLDER = "application/vnd.google-apps.folder"
EXPORT = {  # Google's own formats: what Drive exports them as
    "application/vnd.google-apps.document": ("text/markdown", "doc"),
    "application/vnd.google-apps.spreadsheet": ("text/csv", "sheet"),
    "application/vnd.google-apps.presentation": ("text/plain", "slides"),
}
DOWNLOAD = {"application/pdf": "pdf", "text/markdown": "markdown", "text/x-markdown": "markdown", "text/plain": "text"}
FIELDS = ("nextPageToken,incompleteSearch,files(id,name,mimeType,modifiedTime,createdTime,version,webViewLink,"
          "size,owners(displayName),lastModifyingUser(displayName))")
ID = r"([A-Za-z0-9_-]{15,})"
FOLDER_URL = re.compile(r"/folders/" + ID)
FILE_URL = re.compile(r"/(?:document|spreadsheets|presentation|file)/d/" + ID)
DATA_URI = re.compile(r"<?data:[a-z]+/[\w.+-]+;base64,[A-Za-z0-9+/=\s]+>?")


def _host(url: str) -> str | None:
    u = urllib.parse.urlparse(url)
    h = (u.hostname or "").lower()
    return h if u.scheme in ("http", "https") and h in ("drive.google.com", "docs.google.com") else None


def origin(url: str, query: str | None = None) -> str | None:
    if _host(url) != "drive.google.com":
        return None
    if query:
        raise ValueError("--jql applies to a Jira URL")
    m = FOLDER_URL.search(urllib.parse.urlparse(url).path)
    return f"{WEB}/drive/folders/{m.group(1)}" if m else f"{WEB}/drive/my-drive"


def locate(url: str) -> tuple[str, str] | None:
    """(file id, "") of a document or file URL."""
    if not _host(url):
        return None
    u = urllib.parse.urlparse(url)
    m = FILE_URL.search(u.path)
    if m:
        return m.group(1), ""
    ids = urllib.parse.parse_qs(u.query).get("id")
    return (ids[0], "") if ids and u.path.rstrip("/") in ("/open", "/uc") else None


def heading_url(item: dict, heads: list[str]) -> str:
    return item["url"]  # Drive has no stable link to a heading from outside the editor


def fetch_url(url: str) -> tuple[str, Doc] | None:
    hit = locate(url)
    if not hit:
        return None
    api = client()
    f = api.get(f"/files/{hit[0]}", {"fields": FIELDS[FIELDS.index("files(") + 6:-1], "supportsAllDrives": "true"})
    doc = document(api, f)
    if doc is None:
        raise RemoteError(f"{f['name']} is a {f['mimeType']} file, which this connector does not read")
    return f.get("webViewLink") or url, doc


# ------------------------------------------------------------------------------ signing in

_access: dict[str, tuple[str, float]] = {}  # refresh token -> (access token, expiry)


def _form(url: str, fields: dict) -> dict:
    """A form POST to Google's token endpoint; its errors say why in JSON."""
    req = urllib.request.Request(url, data=urllib.parse.urlencode(fields).encode(),
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        try:
            why = json.load(e).get("error_description") or e.reason
        except ValueError:
            why = e.reason
        raise RemoteError(f"{e.code} {why} from {url}") from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise RemoteError(f"cannot reach {url}: {getattr(e, 'reason', e)}") from None


def _token() -> str:
    env = os.environ.get("GOOGLE_DRIVE_TOKEN")
    if env:
        return env
    kept = credentials.get(KEY)
    if not kept:
        raise credentials.Missing(WEB, "Google Drive login")
    have = _access.get(kept["refresh_token"])
    if have and have[1] > time.time() + 60:
        return have[0]
    try:
        r = _form(TOKEN_URL, {"client_id": kept["client_id"], "client_secret": kept["client_secret"],
                              "refresh_token": kept["refresh_token"], "grant_type": "refresh_token"})
    except RemoteError as e:
        if str(e).startswith(("400", "401")):  # revoked, or expired (a consent screen still "in testing")
            raise RemoteError(f"Google no longer accepts the kept Drive login ({e}); "
                              f"run `inventio login {WEB}` again") from None
        raise
    _access[kept["refresh_token"]] = (r["access_token"], time.time() + float(r.get("expires_in", 3600)))
    return r["access_token"]


def client(token: str | None = None) -> Client:
    return Client(API, {"Authorization": f"Bearer {token or _token()}"})


def credential(origin_url: str) -> str:
    if os.environ.get("GOOGLE_DRIVE_TOKEN"):
        return "env"
    return "keyring" if credentials.get(KEY) else "missing"


def _client_file(path: str) -> tuple[str, str]:
    """(client id, secret) from the JSON the Google Cloud console downloads for a Desktop client."""
    try:
        data = json.loads(open(os.path.expanduser(path.strip().strip("'\"")), encoding="utf-8").read())
    except (OSError, ValueError) as e:
        raise RemoteError(f"cannot read {path}: {e}") from None
    c = data.get("installed") or data.get("web") or data
    if not c.get("client_id") or not c.get("client_secret"):
        raise RemoteError(f"{path} holds no client_id and client_secret (download the Desktop client's JSON)")
    return c["client_id"], c["client_secret"]


def _consent(client_id: str, timeout: float = 300) -> tuple[str, str, str]:
    """Open Google's consent page and wait on 127.0.0.1 for its answer: (code, verifier, redirect)."""
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state, got = secrets.token_urlsafe(16), {}

    class Answer(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if q.get("state", [""])[0] == state:
                got.update({k: v[0] for k, v in q.items()})
            ok = "code" in got
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(("inventio: signed in, you can close this tab" if ok
                              else f"inventio: not signed in ({got.get('error', 'no answer')})").encode())

        def log_message(self, format, *args):  # noqa: A002 - the base's name
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Answer)
    redirect = f"http://127.0.0.1:{srv.server_address[1]}"
    url = AUTH_URL + "?" + urllib.parse.urlencode({
        "client_id": client_id, "redirect_uri": redirect, "response_type": "code", "scope": SCOPE,
        "code_challenge": challenge, "code_challenge_method": "S256", "state": state})
    print(f"opening Google's consent page; if no browser opens, visit:\n{url}", file=sys.stderr)
    threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    srv.timeout, deadline = 5, time.time() + timeout
    while not got and time.time() < deadline:
        srv.handle_request()
    srv.server_close()
    if "code" not in got:
        why = got.get("error") or f"no answer within {timeout / 60:g} minutes"
        raise RemoteError(f"Google did not sign in ({why}); nothing was kept")


def login(url: str, ask, ask_secret) -> str:
    client_id, secret = _client_file(ask("path to the OAuth client JSON (Google Cloud console > "
                                         "Clients > Desktop app > download): "))
    code, verifier, redirect = _consent(client_id)
    r = _form(TOKEN_URL, {"client_id": client_id, "client_secret": secret, "code": code,
                          "code_verifier": verifier, "grant_type": "authorization_code", "redirect_uri": redirect})
    user = client(r["access_token"]).get("/about", {"fields": "user(displayName,emailAddress)"})["user"]
    credentials.put(KEY, {"client_id": client_id, "client_secret": secret, "refresh_token": r["refresh_token"]})
    return f"signed in to Google Drive as {user.get('emailAddress') or user.get('displayName')}; kept in the keychain"


def logout(url: str) -> str:
    return "removed the Google Drive login" if credentials.delete(KEY) else "no Google Drive login kept"


# ------------------------------------------------------------------------------ the mirror


def _readable(f: dict) -> bool:
    return f["mimeType"] in EXPORT or (f["mimeType"] in DOWNLOAD and int(f.get("size") or 0) <= MAX_BYTES)


class Remote:
    FORMAT = 1

    def __init__(self, origin_url: str):
        self.origin = origin_url
        m = FOLDER_URL.search(origin_url)
        self.root = m.group(1) if m else "root"
        self.name = f"gdrive-{self.root[:8]}" if m else "gdrive"
        self.api = client()
        self.files: dict[str, dict] = {}

    def _children(self, folder: str) -> list[dict]:
        params, out = {"q": f"'{folder}' in parents and trashed = false", "fields": FIELDS, "pageSize": 1000,
                       "supportsAllDrives": "true", "includeItemsFromAllDrives": "true"}, []
        while True:
            r = self.api.get("/files", params)
            if r.get("incompleteSearch"):
                raise RemoteError(f"Drive stopped listing folder {folder} early (incompleteSearch); nothing was synced")
            out += r.get("files", [])
            if not r.get("nextPageToken"):
                return out
            params["pageToken"] = r["nextPageToken"]

    def listing(self) -> dict[str, Entry]:
        entries, todo, seen = {}, [(self.root, "")], {self.root}
        while todo:
            folder, prefix = todo.pop()
            used: set[str] = set()
            for f in sorted(self._children(folder), key=lambda f: (f.get("createdTime", ""), f["id"])):
                name = segment(f["name"])
                if f["mimeType"] == FOLDER:
                    if f["id"] not in seen:  # a folder can sit in two parents; walk it once
                        seen.add(f["id"])
                        todo.append((f["id"], posixpath.join(prefix, name)))
                    continue
                if not _readable(f):
                    continue
                stem = name[:-3] if name.lower().endswith(".md") else name
                path = posixpath.join(prefix, stem + ".md")
                if path.lower() in used:  # two files of one name in one folder
                    path = posixpath.join(prefix, f"{stem}-{f['id'][:8]}.md")
                used.add(path.lower())
                self.files[f["id"]] = f
                entries[f["id"]] = Entry(str(f.get("version") or f.get("modifiedTime")), path,
                                         f.get("webViewLink") or f"{WEB}/file/d/{f['id']}/view")
        return entries

    def fetch(self, ids: list[str]):
        for fid in ids:
            doc = document(self.api, self.files[fid])
            if doc is not None:
                yield fid, doc


# ------------------------------------------------------------------------------ file -> Markdown


def document(api: Client, f: dict) -> Doc | None:
    """The Markdown a Drive file is kept as, or None when there is nothing to read in it."""
    kind = EXPORT.get(f["mimeType"], (None, DOWNLOAD.get(f["mimeType"])))[1]
    if kind is None:
        return None
    if f["mimeType"] in EXPORT:
        data = api.raw(f"/files/{f['id']}/export", {"mimeType": EXPORT[f["mimeType"]][0]})
    else:
        data = api.raw(f"/files/{f['id']}", {"alt": "media", "supportsAllDrives": "true"})
    title = f["name"]
    if kind in ("markdown", "pdf") and posixpath.splitext(title)[1].lower() in (".md", ".pdf"):
        title = posixpath.splitext(title)[0]
    if kind == "pdf":
        from inventio.pdf import pdf_text

        text = pdf_text(data, title)
        if text is None:
            return None
    else:
        body = data.decode("utf-8", "replace").replace("\r\n", "\n").lstrip("\ufeff")
        if kind == "sheet":
            import csv
            import io

            rows = [r for r in csv.reader(io.StringIO(body)) if any(c.strip() for c in r)]
            lines = table(rows[0], rows[1:]) if rows else []
        else:
            if kind == "doc":  # images come as inline base64: a page of noise for the chunker
                body = DATA_URI.sub("(image)", body)
            lines = body.split("\n")
        text = tidy([f"# {title}", "", *lines])
    meta = {"format": kind, "owner": ", ".join(o.get("displayName", "") for o in f.get("owners") or []),
            "editor": (f.get("lastModifyingUser") or {}).get("displayName", ""),
            "created": (f.get("createdTime") or "")[:10], "updated": (f.get("modifiedTime") or "")[:10]}
    return Doc(text, {}, {k: v for k, v in meta.items() if v})
