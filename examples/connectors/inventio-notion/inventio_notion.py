"""Notion for inventio: every page an integration can see, one Markdown file per page, laid out
as the page tree.

    inventio login https://www.notion.so          # the integration's token, kept in the keychain
    inventio init https://www.notion.so           # every page shared with the integration
    inventio init https://www.notion.so/Runbook-0123456789abcdef0123456789abcdef   # one page and its subpages

A page is read block by block (the public API, Notion-Version below) and written the way a reader
would keep it: headings stay headings and each one links back to its block on the web; a subpage,
a link to a page or an @-mention of a mirrored page becomes a relative link, so it is a
`citation` in the map; a database row starts with its properties as `- **Name:** value` lines,
and its status, select, multi-select, people, checkbox and date properties become fields
`-w` filters on (`-w status:Done`, `-w priority:High`). Images and files keep their place as
`![caption](link)`; Notion's own file links expire within the hour, so those point at the block.

Written against the API reference and the official SDK's types (notion-sdk-js, API version
2025-09-03), and tested on responses built from them; not yet run against a real workspace.
"""

import os
import posixpath
import re
import urllib.parse

from inventio import credentials
from inventio.connectors.http import Client, RemoteError
from inventio.connectors.markdown import fence, safe, segment, table, tidy
from inventio.connectors.mirror import Doc, Entry
from inventio.ingest import slug

KIND = "notion"
API = "https://api.notion.com/v1"
VERSION = "2025-09-03"
WEB = "https://www.notion.so"
HOSTS = re.compile(r"^(www\.)?notion\.so$|\.notion\.site$")
ID = re.compile(r"([0-9a-f]{32})$|([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$")
KEY = "notion"  # keychain key: an integration token belongs to one workspace, one login per machine
SKIP = {"table_of_contents", "breadcrumb", "unsupported", "template"}
CONTAINERS = {"column_list", "column", "synced_block", "tab"}  # only their children are content


def _id(text: str) -> str | None:
    """A Notion id, dashed, from the end of a URL path segment or a bare id."""
    m = ID.search(text.lower())
    if not m:
        return None
    h = (m.group(1) or m.group(2)).replace("-", "")
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def _parse(url: str) -> tuple[str | None, str] | None:
    """(page id or None for the whole workspace, fragment) of a Notion URL, or None."""
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not HOSTS.search((u.hostname or "").lower()):
        return None
    q = urllib.parse.parse_qs(u.query)
    pid = _id(q["p"][0]) if q.get("p") else None  # a page opened as a peek: ?p=<id>
    parts = [p for p in u.path.split("/") if p]
    return pid or (_id(parts[-1]) if parts else None), u.fragment


def origin(url: str, query: str | None = None) -> str | None:
    hit = _parse(url)
    if hit is None:
        return None
    if query:
        raise ValueError("--jql applies to a Jira URL")
    return f"{WEB}/{hit[0].replace('-', '')}" if hit[0] else WEB


def locate(url: str) -> tuple[str, str] | None:
    """(page id, block fragment) of a page URL."""
    hit = _parse(url)
    return (hit[0], hit[1]) if hit and hit[0] else None


def heading_url(item: dict, heads: list[str]) -> str:
    return item.get("anchors", {}).get(slug(heads[-1]), item["url"]) if heads else item["url"]


def fetch_url(url: str) -> tuple[str, Doc] | None:
    """A page that is not in any mirror, read live; its links point to the web."""
    hit = locate(url)
    if not hit:
        return None
    api = client()
    page = api.get(f"/pages/{hit[0]}")
    return page["url"], _Page(api, page, "", {}, {}, {}).doc()


# ------------------------------------------------------------------------------ signing in


def _token() -> tuple[str, str] | None:
    """(token, where it came from)."""
    env = os.environ.get("NOTION_TOKEN")
    if env:
        return env, "env"
    kept = credentials.get(KEY)
    return (kept["token"], "keyring") if kept else None


def client(token: str | None = None) -> Client:
    if token is None:
        found = _token()
        if found is None:
            raise credentials.Missing(WEB, "Notion integration token")
        token = found[0]
    return Client(API, {"Authorization": f"Bearer {token}", "Notion-Version": VERSION})


def credential(origin_url: str) -> str:
    found = _token()
    return found[1] if found else "missing"


def login(url: str, ask, ask_secret) -> str:
    token = ask_secret("integration token (https://www.notion.so/profile/integrations, "
                       "then share pages with the integration): ").strip()
    try:
        me = client(token).get("/users/me")
    except RemoteError as e:
        if str(e).startswith(("401", "403")):
            raise RemoteError(f"Notion refused that token ({str(e).split(' from ')[0]}); nothing was kept") from None
        raise
    credentials.put(KEY, {"token": token})
    return f"signed in to Notion as {me.get('name') or 'the integration'}; kept in the keychain"


def logout(url: str) -> str:
    return "removed the Notion login" if credentials.delete(KEY) else "no Notion login kept"


# ------------------------------------------------------------------------------ the mirror


def _plain(rich: list[dict]) -> str:
    return "".join(r.get("plain_text", "") for r in rich or [])


def title_of(page: dict) -> str:
    for p in (page.get("properties") or {}).values():
        if p.get("type") == "title":
            return _plain(p["title"]).strip() or "Untitled"
    return "Untitled"


def _gone(o: dict) -> bool:
    return bool(o.get("in_trash") or o.get("archived") or o.get("is_archived"))


class Remote:
    FORMAT = 1

    def __init__(self, origin_url: str):
        self.origin = origin_url
        tail = origin_url[len(WEB):].strip("/")
        self.root = _id(tail) if tail else None
        self.name = f"notion-{tail[:8]}" if tail else "notion"
        self.api = client()
        self.pages: dict[str, dict] = {}  # page id -> page object from the listing
        self.paths: dict[str, str] = {}   # page id -> mirror path
        self.urls: dict[str, str] = {}    # page id -> web URL
        self.titles: dict[str, str] = {}  # page id -> title, for link text
        self._blocks: dict[str, dict] = {}

    def _search(self, kind: str) -> list[dict]:
        body, out = {"filter": {"property": "object", "value": kind}, "page_size": 100}, []
        while True:
            r = self.api.post("/search", body)
            out += [o for o in r["results"] if o.get("object") in ("page", "data_source") and not _gone(o)]
            if (r.get("request_status") or {}).get("type") == "incomplete":
                raise RemoteError(f"Notion's search stopped before listing every {kind} "
                                  f"({r['request_status'].get('incomplete_reason')}); nothing was synced")
            if not r.get("has_more"):
                return out
            body["start_cursor"] = r["next_cursor"]

    def _up(self, parent: dict, sources: dict) -> tuple[str, str] | None:
        """(kind, id) of the page, data source or nothing above something with this parent."""
        t = parent.get("type")
        if t == "page_id":
            return "page", parent["page_id"]
        if t == "data_source_id":
            return "source", parent["data_source_id"]
        if t == "database_id":  # an older parent form: the database's first data source, if known
            for sid, s in sources.items():
                if (s.get("parent") or {}).get("database_id") == parent["database_id"]:
                    return "source", sid
            return None
        if t == "block_id":  # a page inside a toggle or a column: climb to the page holding the block
            b = self._blocks.get(parent["block_id"])
            if b is None:
                try:
                    b = self._blocks[parent["block_id"]] = self.api.get(f"/blocks/{parent['block_id']}")
                except RemoteError:
                    return None
            return self._up(b.get("parent") or {}, sources)
        return None

    def listing(self) -> dict[str, Entry]:
        pages = {p["id"]: p for p in self._search("page") if "properties" in p}
        self.titles = {pid: title_of(p) for pid, p in pages.items()}
        sources = {s["id"]: s for s in self._search("data_source")}
        chain: dict[str, list[tuple[str, str]]] = {}

        def ancestors(kind: str, oid: str, seen: frozenset = frozenset()) -> list[tuple[str, str]]:
            """(kind, id) from the top down to this object, as far as the integration can see."""
            if (kind, oid) in seen:
                return []
            obj = pages.get(oid) if kind == "page" else sources.get(oid)
            if obj is None:
                return []
            parent = obj.get("database_parent") if kind == "source" else obj.get("parent")
            if kind == "source" and not parent:
                parent = {}
            up = self._up(parent or {}, sources)
            return (ancestors(*up, seen | {(kind, oid)}) if up else []) + [(kind, oid)]

        for pid, p in pages.items():
            chain[pid] = ancestors("page", pid)
        if self.root:
            pages = {pid: p for pid, p in pages.items() if ("page", self.root) in chain[pid]}
            if not pages:
                raise RemoteError(f"Notion page {self.root} is not shared with this integration "
                                  "(Share > Connections in Notion), or does not exist")

        def name(kind: str, oid: str) -> str:
            return segment(title_of(pages[oid]) if kind == "page" and oid in pages
                           else _plain((sources.get(oid) or {}).get("title")) or "Untitled")

        used: set[str] = set()
        for pid in sorted(pages, key=lambda i: (len(chain[i]), pages[i].get("created_time", ""), i)):
            top = chain[pid][chain[pid].index(("page", self.root)):] if self.root else chain[pid]
            folder = "/".join(name(*a) for a in top[:-1])
            path = posixpath.join(folder, name("page", pid) + ".md")
            if path.lower() in used:  # two pages with one title under one parent
                path = path[:-3] + "-" + pid.replace("-", "")[:8] + ".md"
            used.add(path.lower())
            self.paths[pid] = path
            self.urls[pid] = pages[pid].get("url") or f"{WEB}/{pid.replace('-', '')}"
        self.pages = pages
        return {pid: Entry(pages[pid]["last_edited_time"], self.paths[pid], self.urls[pid]) for pid in pages}

    def fetch(self, ids: list[str]):
        for pid in ids:
            page = self.pages[pid]
            yield pid, _Page(self.api, page, self.paths[pid], self.paths, self.urls, self.titles).doc()


# ------------------------------------------------------------------------------ page -> Markdown


def _key(name: str) -> str | None:
    k = re.sub(r"[^\w.\-]+", "-", name.strip().lower()).strip("-")
    return k if re.match(r"^[a-z_][\w.\-]*$", k) else None


def _person(u: dict) -> str:
    return u.get("name") or ""


class _Page:
    """Renders one page. `paths` and `urls` describe the mirror, so links to its pages are relative."""

    def __init__(self, api: Client, page: dict, path: str, paths: dict, urls: dict, titles: dict):
        self.api, self.page, self.path, self.paths, self.urls, self.titles = api, page, path, paths, urls, titles
        self.url = page.get("url") or f"{WEB}/{page['id'].replace('-', '')}"
        self.anchors: dict[str, str] = {}

    def link(self, pid: str, text: str) -> str:
        pid = _id(pid) or pid
        if pid in self.paths and self.path:
            rel = posixpath.relpath(self.paths[pid], posixpath.dirname(self.path) or ".")
            return f"[{text}]({urllib.parse.quote(rel)})"
        return f"[{text}]({self.urls.get(pid) or WEB + '/' + pid.replace('-', '')})"

    def text(self, rich: list[dict]) -> str:
        out = []
        for r in rich or []:
            t, a = r.get("plain_text", ""), r.get("annotations") or {}
            if r.get("type") == "mention":
                m = r["mention"]
                if m.get("type") == "page":
                    out.append(self.link(m["page"]["id"], t))
                    continue
                if m.get("type") == "user":
                    out.append("@" + (_person(m["user"]) or t.lstrip("@")))
                    continue
            if r.get("type") == "equation":
                out.append(f"${r['equation']['expression']}$")
                continue
            if not t.strip():
                out.append(t)
                continue
            lead, core, trail = re.match(r"^(\s*)(.*?)(\s*)$", t, re.S).groups()
            if a.get("code"):
                core = f"`{core}`"
            if a.get("bold"):
                core = f"**{core}**"
            if a.get("italic"):
                core = f"_{core}_"
            if a.get("strikethrough"):
                core = f"~~{core}~~"
            if r.get("href"):
                core = f"[{core}]({r['href']})"
            out.append(lead + core + trail)
        return "".join(out)

    def children(self, block_id: str) -> list[dict]:
        out, params = [], {"page_size": 100}
        while True:
            r = self.api.get(f"/blocks/{block_id}/children", params)
            out += [b for b in r["results"] if not _gone(b)]
            if not r.get("has_more"):
                return out
            params["start_cursor"] = r["next_cursor"]

    def properties(self) -> tuple[list[str], dict]:
        lines, meta = [], {}
        for name, p in (self.page.get("properties") or {}).items():
            t = p.get("type")
            v, field = p.get(t), None
            if t == "title" or v is None:
                continue
            if t in ("select", "status"):
                shown = field = v.get("name") if v else ""
            elif t == "multi_select":
                field = [o["name"] for o in v]
                shown = ", ".join(field)
            elif t == "people":
                field = [n for n in (_person(u) for u in v) if n]
                shown = ", ".join(field)
            elif t == "checkbox":
                shown = field = "yes" if v else "no"
            elif t == "date":
                shown = (v["start"] + (f" to {v['end']}" if v.get("end") else "")) if v else ""
                field = v["start"][:10] if v else None
            elif t == "rich_text":
                shown = self.text(v)
            elif t == "relation":
                shown = ", ".join(self.link(r["id"], self.titles.get(r["id"], "linked page")) for r in v)
            elif t in ("number", "url", "email", "phone_number"):
                shown = "" if v is None else str(v)
            elif t == "unique_id":
                shown = f"{v.get('prefix') or ''}{'-' if v.get('prefix') else ''}{v.get('number')}"
            elif t == "formula":
                shown = str(v.get(v.get("type")) or "") if v.get("type") != "date" else (v.get("date") or {}).get("start", "")
            elif t in ("created_by", "last_edited_by"):
                shown = _person(v)
            else:  # files, rollup, button, verification, place, created/edited time (the page's own)
                continue
            if shown:
                lines.append(f"- **{name}:** {shown}")
            k = _key(name)
            if field and k:
                meta[k] = field
        if "status" not in meta:
            for k in ("state", "stage"):
                if k in meta:
                    meta["status"] = meta[k]
        tags = [x for k in ("tags", "labels", "tag") for x in (meta.get(k) or []) if isinstance(meta.get(k), list)]
        if tags:
            meta["labels"] = tags
        meta["created"] = (self.page.get("created_time") or "")[:10]
        meta["updated"] = (self.page.get("last_edited_time") or "")[:10]
        return lines, meta

    def doc(self) -> Doc:
        title = title_of(self.page)
        props, meta = self.properties()
        lines = [f"# {title}", ""] + (props + [""] if props else [])
        self.blocks(self.page["id"], lines, "")
        return Doc(tidy(lines), self.anchors, meta)

    def blocks(self, parent_id: str, lines: list[str], indent: str) -> None:
        numbered = 0
        for b in self.children(parent_id):
            t = b.get("type")
            numbered = numbered + 1 if t == "numbered_list_item" else 0
            if t in SKIP:
                continue
            if t in CONTAINERS:
                if b.get("has_children"):
                    self.blocks(b["id"], lines, indent)
                continue
            self.block(b, t, numbered, lines, indent)

    def block(self, b: dict, t: str, numbered: int, lines: list[str], indent: str) -> None:
        v = b.get(t) or {}
        anchor = f"{self.url}#{b['id'].replace('-', '')}"
        nest = indent
        if t.startswith("heading_"):
            head = self.text(v.get("rich_text")).strip()
            if head:
                lines += ["", "#" * min(6, int(t[-1]) + 1) + " " + head, ""]
                self.anchors.setdefault(slug(head), anchor)
        elif t == "paragraph":
            lines += [indent + safe(self.text(v.get("rich_text"))), ""]
        elif t in ("bulleted_list_item", "toggle"):
            lines.append(f"{indent}- {self.text(v.get('rich_text'))}")
            nest = indent + "  "
        elif t == "numbered_list_item":
            lines.append(f"{indent}{numbered}. {self.text(v.get('rich_text'))}")
            nest = indent + "   "
        elif t == "to_do":
            lines.append(f"{indent}- [{'x' if v.get('checked') else ' '}] {self.text(v.get('rich_text'))}")
            nest = indent + "  "
        elif t in ("quote", "callout"):
            icon = (v.get("icon") or {}).get("emoji", "") if t == "callout" else ""
            body = self.text(v.get("rich_text"))
            lines += [f"{indent}> {(icon + ' ') if icon else ''}{line}" for line in body.split("\n")] + [""]
        elif t == "code":
            lang = (v.get("language") or "").replace("plain text", "")
            lines += [indent + x for x in fence(_plain(v.get("rich_text")), lang.split(" ")[0])]
        elif t == "equation":
            lines += [f"{indent}$$ {v.get('expression', '')} $$", ""]
        elif t == "divider":
            lines += ["", "***", ""]
        elif t == "child_page":
            lines += [indent + self.link(b["id"], v.get("title") or "Untitled"), ""]
            return  # a subpage is a page of its own
        elif t == "child_database":
            lines += [f"{indent}**Database:** [{v.get('title') or 'Untitled'}]({anchor})", ""]
            return  # its rows are pages of their own
        elif t == "link_to_page":
            target = v.get("page_id") or v.get("database_id")
            if target:
                lines += [indent + self.link(target, self.titles.get(target, "linked page")), ""]
        elif t == "table":
            rows = [[self.text(c) for c in r["table_row"]["cells"]]
                    for r in self.children(b["id"]) if r.get("type") == "table_row"]
            if rows:
                head = rows.pop(0) if v.get("has_column_header") else [""] * len(rows[0])
                lines += [""] + [indent + x for x in table(head, rows)]
            return
        elif t in ("image", "video", "file", "pdf", "audio"):
            caption = self.text(v.get("caption")) or v.get("name") or t
            url = (v.get("external") or {}).get("url") if v.get("type") == "external" else anchor
            lines += [f"{indent}![{caption}]({url})", ""]
        elif t in ("bookmark", "embed", "link_preview"):
            url = v.get("url", "")
            lines += [f"{indent}[{self.text(v.get('caption')) or url}]({url})", ""]
        else:  # a block kind this connector does not know yet: keep its text, if any
            text = self.text(v.get("rich_text")) if isinstance(v, dict) else ""
            if text:
                lines += [indent + safe(text), ""]
        if b.get("has_children"):
            self.blocks(b["id"], lines, nest)
            if nest != indent:
                lines.append("")
