"""Confluence Cloud: a space, one Markdown file per page, laid out as the page tree.

A page is fetched in its storage format (XHTML with `ac:` macros and `ri:` resources) and
written as Markdown the way a reader would want it kept:
- links to pages of the same space become relative links, so they are `citation` links in the map;
- `include` and `excerpt-include` become a link to the page they pull from, so a rule lives once;
- a code macro stays code; a `jira` macro keeps the ticket key, which links to the ticket;
- images and embedded diagrams (Lucidchart, Figma, draw.io) are not read, but keep their place as
  `![label](url)`: the chunk says a diagram is there and where to open it;
- tables stay tables, long ones cut with the header repeated; two-column key/value tables become
  `- **key:** value` lines;
- navigation macros (table of contents, child lists, recently updated) are dropped.
"""

import posixpath
import re
import urllib.parse
from html.parser import HTMLParser

from ..ingest import FENCE, HEADING, slug
from . import atlassian
from .markdown import fence, safe, segment, table, tidy
from .mirror import Doc, Entry

KIND = "confluence"
SPACE_URL = re.compile(r"/wiki/spaces/([^/?#]+)")
PAGE_URL = re.compile(r"/wiki/spaces/[^/?#]+/pages/(\d+)")
MAX_PATH = 180  # characters of a mirror path; deeper trees lose their top folders first
BATCH = 100     # pages fetched per request (the API takes up to 250 ids)


def origin(url: str, query: str | None = None) -> str | None:
    s, m = atlassian.site(url), SPACE_URL.search(url)
    if not s or not m or "/browse/" in url:
        return None
    if query:
        raise ValueError("--jql applies to Jira sources, not a Confluence space")
    return f"{s}/wiki/spaces/{m.group(1)}"


def locate(url: str) -> tuple[str, str] | None:
    """(page id, heading fragment) of a page URL."""
    if not atlassian.site(url):
        return None
    u = urllib.parse.urlparse(url)
    m = PAGE_URL.search(u.path)
    pid = m.group(1) if m else urllib.parse.parse_qs(u.query).get("pageId", [None])[0]
    return (pid, urllib.parse.unquote(u.fragment)) if pid else None


def heading_url(item: dict, heads: list[str]) -> str:
    """Confluence anchors a heading by its text with spaces as dashes; the first heading is the
    page title, which has none."""
    return item["url"] + ("#" + heads[-1].replace(" ", "-") if len(heads) > 1 else "")


def fetch_url(url: str) -> tuple[str, Doc] | None:
    """A page that is not in any mirror, read live; its links point to the web."""
    hit = locate(url)
    if not hit:
        return None
    site = atlassian.site(url)
    api = atlassian.client(site)
    p = api.get(f"/wiki/api/v2/pages/{hit[0]}", {"body-format": "storage"})
    key = api.get(f"/wiki/api/v2/spaces/{p['spaceId']}")["key"]
    users: dict[str, str] = {}
    people(api, [p["body"]["storage"]["value"]], users)
    page = _Page(site, key, p["id"], "", {}, {}, users)
    return site + "/wiki" + p["_links"]["webui"], Doc(page.markdown(p["title"], p["body"]["storage"]["value"]))


class Remote:
    FORMAT = 1  # raise when the Markdown written changes, so every mirror is written again

    def __init__(self, origin_url: str):
        self.origin = origin_url
        self.site, self.key = origin_url.split("/wiki/spaces/")
        self.name = f"wiki-{self.key}"
        self.api = atlassian.client(self.site)
        self.paths: dict[str, str] = {}     # page id -> mirror path
        self.by_title: dict[str, str] = {}  # title -> page id (titles are unique in a space)
        self.users: dict[str, str] = {}     # account id -> display name

    def _paged(self, path: str, params: dict):
        page = self.api.get(path, params)
        while True:
            yield from page["results"]
            nxt = page.get("_links", {}).get("next")
            if not nxt:
                return
            page = self.api.get(nxt)

    def listing(self) -> dict[str, Entry]:
        spaces = self.api.get("/wiki/api/v2/spaces", {"keys": self.key})["results"]
        if not spaces:
            raise atlassian.RemoteError(f"no Confluence space {self.key!r} on {self.site} (or no access)")
        pages = list(self._paged(f"/wiki/api/v2/spaces/{spaces[0]['id']}/pages", {"limit": 250, "status": "current"}))
        self.paths = tree_paths(pages)
        self.by_title = {p["title"]: p["id"] for p in pages}
        return {p["id"]: Entry(str(p["version"]["number"]), self.paths[p["id"]], self.site + "/wiki" + p["_links"]["webui"])
                for p in pages}

    def fetch(self, ids: list[str]):
        for i in range(0, len(ids), BATCH):
            params = {"id": ",".join(ids[i:i + BATCH]), "body-format": "storage", "limit": 250}
            batch = list(self._paged("/wiki/api/v2/pages", params))
            people(self.api, [p["body"]["storage"]["value"] for p in batch], self.users)
            for p in batch:
                page = _Page(self.site, self.key, p["id"], self.paths[p["id"]], self.paths, self.by_title, self.users)
                yield p["id"], Doc(page.markdown(p["title"], p["body"]["storage"]["value"]))


ACCOUNT = re.compile(r'ri:account-id="([^"]+)"')


def people(api, bodies: list[str], known: dict[str, str]) -> None:
    """Display names for the people pages mention (`@Name`, an owner, a reviewer), added to `known`;
    one request per hundred people not seen before."""
    ids = sorted({a for b in bodies for a in ACCOUNT.findall(b)} - set(known))
    for i in range(0, len(ids), 100):
        part = ids[i:i + 100]
        try:
            found = api.get("/wiki/rest/api/user/bulk", {"accountId": part, "limit": 100}).get("results", [])
        except atlassian.RemoteError:
            found = []
        known.update(dict.fromkeys(part, ""))  # asked once, found or not
        known.update({u["accountId"]: u.get("displayName") or u.get("publicName") or "" for u in found})


def tree_paths(pages: list[dict]) -> dict[str, str]:
    """Page id -> `Parent/Child/Page.md`, from the titles up the tree. A parent outside the
    listing (a folder, a page of another space) ends the chain."""
    by_id = {p["id"]: p for p in pages}
    out, taken = {}, set()
    for p in pages:
        segs, cur, seen = [], p, set()
        while cur and cur["id"] not in seen:
            seen.add(cur["id"])
            segs.append(segment(cur["title"]))
            cur = by_id.get(cur.get("parentId"))
        segs.reverse()
        while len(segs) > 1 and len("/".join(segs)) > MAX_PATH:
            segs.pop(0)
        path = "/".join(segs) + ".md"
        if path.lower() in taken:  # two titles that differ only in characters a path cannot hold
            path = path[:-3] + f"-{p['id']}.md"
        taken.add(path.lower())
        out[p["id"]] = path
    return out


# ---------------------------------------------------------------- storage format -> Markdown


class _Node:
    __slots__ = ("tag", "attrs", "kids")

    def __init__(self, tag: str, attrs: dict):
        self.tag, self.attrs, self.kids = tag, attrs, []

    def nodes(self):
        return (k for k in self.kids if isinstance(k, _Node))

    def child(self, tag: str) -> "_Node | None":
        return next((k for k in self.nodes() if k.tag == tag), None)

    def find(self, tag: str) -> "_Node | None":
        for k in self.nodes():
            if k.tag == tag:
                return k
            hit = k.find(tag)
            if hit:
                return hit
        return None

    def param(self, name: str) -> "_Node | None":
        return next((k for k in self.nodes() if k.tag == "ac:parameter" and k.attrs.get("ac:name") == name), None)

    def text(self) -> str:
        return "".join(k if isinstance(k, str) else k.text() for k in self.kids)


class _Tree(HTMLParser):
    VOID = {"br", "hr", "img", "col", "input", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Node("root", {})
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        n = _Node(tag, dict(attrs))
        self.stack[-1].kids.append(n)
        if tag not in self.VOID:
            self.stack.append(n)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].kids.append(_Node(tag, dict(attrs)))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        self.stack[-1].kids.append(data)

    def unknown_decl(self, data):  # <![CDATA[...]]>, the body of code macros
        if data.startswith("CDATA["):
            self.stack[-1].kids.append(data[6:])


def parse(xhtml: str) -> _Node:
    t = _Tree()
    t.feed(xhtml)
    t.close()
    return t.root


INLINE = {"a", "strong", "b", "em", "i", "u", "s", "del", "code", "span", "br", "sub", "sup", "font", "small",
          "mark", "time", "ac:link", "ac:image", "ac:emoticon", "ac:inline-comment-marker", "ac:placeholder"}
INLINE_MACROS = {"status", "jira", "anchor"}
DROP_MACROS = {"toc", "toc-zone", "children", "pagetree", "contributors", "contributors-summary", "profile",
               "livesearch", "blog-posts", "recently-updated", "jirachart", "roadmap", "create-from-template",
               "page-index", "listlabels", "popular-labels", "change-history"}
EMBEDS = {"lucidchart": "Lucidchart diagram", "figma-for-confluence-lite": "Figma design", "figma": "Figma design",
          "drawio": "draw.io diagram", "inc-drawio": "draw.io diagram", "gliffy": "Gliffy diagram",
          "iframe": "embedded page", "widget": "embedded page", "view-file": "attached file",
          "viewpdf": "attached file", "multimedia": "attached media"}
NOTES = {"info": "Info", "note": "Note", "tip": "Tip", "warning": "Warning"}
HEADINGS = {f"h{i}" for i in range(1, 7)}
WS = re.compile(r"\s+")


class _Page:
    """Renders one page. `paths` and `by_title` describe the mirrored space, so links to its pages
    can be written as relative links; both empty for a page read live."""

    def __init__(self, site: str, space: str, page_id: str, path: str, paths: dict, by_title: dict,
                 users: dict | None = None):
        self.site, self.space, self.page_id, self.path = site, space, page_id, path
        self.paths, self.by_title, self.users = paths, by_title, users or {}

    def markdown(self, title: str, xhtml: str) -> str:
        return tidy([f"# {title}", "", *self.blocks(parse(xhtml).kids)])

    # ---- targets

    def _relative(self, page_id: str) -> str | None:
        if page_id not in self.paths or not self.path:
            return None
        return posixpath.relpath(self.paths[page_id], posixpath.dirname(self.path) or ".")

    def page_target(self, title: str, space: str | None) -> str:
        space = space or self.space
        if space == self.space and title in self.by_title:
            rel = self._relative(self.by_title[title])
            if rel:
                return rel
        return f"{self.site}/wiki/display/{space}/{urllib.parse.quote_plus(title)}"

    def attachment(self, filename: str) -> str:
        return f"{self.site}/wiki/download/attachments/{self.page_id}/{urllib.parse.quote(filename)}"

    def href(self, url: str) -> str:
        m = PAGE_URL.search(url)
        if m and url.startswith(self.site):
            rel = self._relative(m.group(1))
            if rel:
                return rel
        return url.replace(" ", "%20")

    def resource(self, node: _Node | None) -> str:
        """The URL an `ri:` resource inside `node` points at."""
        if node is None:
            return ""
        if att := node.find("ri:attachment") if node.tag != "ri:attachment" else node:
            return self.attachment(att.attrs.get("ri:filename", ""))
        if url := node.find("ri:url") if node.tag != "ri:url" else node:
            return url.attrs.get("ri:value", "")
        return node.text().strip()

    # ---- blocks

    def blocks(self, kids) -> list[str]:
        out, run = [], []

        def flush():
            text = "".join(run)
            run.clear()
            lines = [l.strip() for l in text.split("\n")]
            if any(lines):
                out.extend(safe(l) for l in lines if l)
                out.append("")

        for k in kids:
            if isinstance(k, str) or k.tag in INLINE or (
                    k.tag == "ac:structured-macro" and k.attrs.get("ac:name") in INLINE_MACROS):
                run.append(self.inline(k))
            else:
                flush()
                out.extend(self.block(k))
        flush()
        return out

    def block(self, n: _Node) -> list[str]:
        t = n.tag
        if t in HEADINGS:
            text = " ".join(self.inlines(n.kids).split())
            return [f"{'#' * min(int(t[1]) + 1, 6)} {text}", ""] if text else []  # h1 sits under the page title
        if t in ("ul", "ol"):
            return self.list(n, t == "ol")
        if t == "table":
            return self.table(n)
        if t == "pre":
            return fence(n.text())
        if t == "blockquote":
            return ["> " + l if l else "" for l in self.blocks(n.kids)]
        if t == "ac:structured-macro":
            return self.macro(n)
        if t == "ac:task-list":
            return self.tasks(n)
        if t == "ac:adf-extension":  # newer editor nodes; the fallback beside it is a rendered copy
            content = n.find("ac:adf-content")
            return self.blocks(content.kids) if content else []
        if t in ("hr", "colgroup", "ac:parameter", "ac:placeholder", "style", "script"):
            return []
        return self.blocks(n.kids)  # p, div, layouts, sections

    def list(self, n: _Node, ordered: bool) -> list[str]:
        out, i = [], 0
        for li in (k for k in n.nodes() if k.tag == "li"):
            lines = [l for l in self.blocks(li.kids) if l]
            if not lines:
                continue
            i += 1
            mark = f"{i}. " if ordered else "- "
            out += [mark + lines[0], *(" " * len(mark) + l for l in lines[1:])]
        return out + [""]

    def table(self, n: _Node) -> list[str]:
        rows = []
        for part in [n, *(k for k in n.nodes() if k.tag in ("thead", "tbody", "tfoot"))]:
            for tr in (k for k in part.nodes() if k.tag == "tr"):
                cells = [(c.tag == "th", [l for l in self.blocks(c.kids) if l])
                         for c in tr.nodes() if c.tag in ("th", "td")]
                if cells:
                    rows.append(cells)
        if not rows:
            return []
        label = lambda lines: " ".join(lines).replace("*", "").strip()  # noqa: E731  a header cell's text
        if any(len(ls) > 3 or any(FENCE.match(l) for l in ls) for r in rows for _, ls in r):
            return self.records(rows, label)
        if all(len(r) == 2 and r[0][0] for r in rows):  # a details table: key in a header cell, then its value
            return [f"- **{label(k)}:** {' '.join(v)}".rstrip() for (_, k), (_, v) in rows if label(k)] + [""]
        return table([" ".join(ls) for _, ls in rows[0]], [[" ".join(ls) for _, ls in r] for r in rows[1:]])

    @staticmethod
    def records(rows: list, label) -> list[str]:
        """A table used for layout, with code, lists or paragraphs in its cells: each row becomes a
        record, each cell its blocks under the column's name, so code stays code."""
        names = [label(ls) for _, ls in rows[0]] if len(rows) > 1 and all(th for th, _ in rows[0]) else []
        out = []
        for r in rows[1:] if names else rows:
            for j, (th, ls) in enumerate(r):
                if not ls:
                    continue
                if j < len(names) and names[j]:
                    out.append(f"**{names[j]}:**")
                out += [*ls, ""]
        return out

    def tasks(self, n: _Node) -> list[str]:
        out = []
        for task in (k for k in n.nodes() if k.tag == "ac:task"):
            status, body = task.child("ac:task-status"), task.child("ac:task-body")
            done = "x" if status is not None and status.text().strip() == "complete" else " "
            text = " ".join(self.inlines(body.kids).split()) if body else ""
            if text:
                out.append(f"- [{done}] {text}")
        return out + [""]

    def macro(self, n: _Node) -> list[str]:
        name = n.attrs.get("ac:name", "")
        body, plain = n.child("ac:rich-text-body"), n.child("ac:plain-text-body")
        if name in DROP_MACROS:
            return []
        if name in ("code", "noformat"):
            lang = n.param("language")
            return fence(plain.text() if plain else "", lang.text().strip() if lang else "")
        if name in ("include", "excerpt-include"):
            link = (n.param("") or n).find("ri:page")
            excerpt = n.param("name")
            what = f'Excerpt "{excerpt.text().strip()}"' if excerpt else "Included"
            if link is None:
                return [f"{what} of this page", ""] if excerpt else []
            title = link.attrs.get("ri:content-title", "")
            return [f"{what} from [{title}]({self.page_target(title, link.attrs.get('ri:space-key'))})", ""]
        if name in EMBEDS:
            p = n.param("url") or n.param("nodeUrl") or n.param("src") or n.param("name") or n.param("")
            title = n.param("title") or n.param("diagramDisplayName") or n.param("diagramName")
            label = EMBEDS[name] + (f": {title.text().strip()}" if title and title.text().strip() else "")
            if p is not None and p.find("ri:attachment") is not None and not title:
                label += f": {p.find('ri:attachment').attrs.get('ri:filename', '')}"
            url = self.resource(p)
            return [f"![{label}]({url.replace(' ', '%20')})" if url else f"[{label}]", ""]
        if name in NOTES:  # the label only where the panel does not open with its own heading
            lines = self.blocks(body.kids) if body else []
            return lines if not lines or HEADING.match(lines[0]) else [f"**{NOTES[name]}:**", *lines]
        if name == "expand":
            title = n.param("title")
            head = [f"**{title.text().strip()}**"] if title and title.text().strip() else []
            return head + self.blocks(body.kids) if body else []
        if body:
            return self.blocks(body.kids)
        if plain:
            return fence(plain.text())
        return []

    # ---- inline

    def inlines(self, kids) -> str:
        return "".join(self.inline(k) for k in kids)

    def inline(self, k) -> str:
        if isinstance(k, str):
            return WS.sub(" ", k.replace("\xa0", " "))
        t = k.tag
        if t == "br":
            return "\n"
        if t in ("strong", "b", "em", "i"):
            s = self.inlines(k.kids)
            mark = "**" if t in ("strong", "b") else "*"
            return f"{mark}{s.strip()}{mark}" if s.strip() else s
        if t == "code":
            return f"`{k.text()}`"
        if t == "a":
            href, text = k.attrs.get("href", ""), self.inlines(k.kids).strip()
            if not href or text == href:
                return text or href
            return f"[{text.replace(']', ')')}]({self.href(href)})"
        if t == "ac:link":
            return self.link(k)
        if t == "ac:image":
            return self.image(k)
        if t == "time":
            return k.attrs.get("datetime", "")
        if t == "ac:structured-macro":
            name = k.attrs.get("ac:name")
            if name == "status":
                title = k.param("title")
                return f"[{title.text().strip()}]" if title else ""
            if name == "jira":
                key, jql = k.param("key"), k.param("jqlQuery")
                return key.text().strip() if key else (f"(Jira: {jql.text().strip()})" if jql else "")
            return ""
        if t in ("ac:emoticon", "ac:placeholder"):
            return ""
        return self.inlines(k.kids)

    def link(self, k: _Node) -> str:
        body = k.child("ac:link-body") or k.child("ac:plain-text-link-body")
        text = " ".join(self.inlines(body.kids).split()) if body else ""
        anchor = k.attrs.get("ac:anchor", "")
        if page := k.child("ri:page"):
            title = page.attrs.get("ri:content-title", "")
            target, text = self.page_target(title, page.attrs.get("ri:space-key")), text or title
        elif att := k.child("ri:attachment"):
            name = att.attrs.get("ri:filename", "")
            target, text = self.attachment(name), text or name
        elif url := k.child("ri:url"):
            target = url.attrs.get("ri:value", "")
        elif user := k.child("ri:user"):
            return text or f"@{self.users.get(user.attrs.get('ri:account-id', '')) or 'user'}"
        elif anchor:
            target = ""
        else:
            return text
        if anchor:  # a relative link names the heading the way the map anchors it (ingest.slug)
            target += "#" + (anchor if target.startswith("http") else slug(anchor.replace("-", " ")))
        return f"[{(text or target).replace(']', ')')}]({target})" if target else text

    def image(self, k: _Node) -> str:
        src = self.resource(k.child("ri:attachment") or k.child("ri:url"))
        att = k.child("ri:attachment")
        alt = k.attrs.get("ac:alt") or k.attrs.get("ac:title") or (att.attrs.get("ri:filename", "") if att else "image")
        cap = k.child("ac:caption")
        if cap is not None and cap.text().strip():
            alt = f"{alt}: {' '.join(cap.text().split())}"
        return f"![{alt.replace(']', ')')}]({src.replace(' ', '%20')})" if src else f"[{alt}]"
