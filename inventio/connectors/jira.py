"""Jira Cloud: the tickets of a JQL query, one Markdown file per ticket.

A ticket is a record: its title line, a line of facts (type, status, people, dates), its links
to other tickets, the description, then every comment under its own heading, because the answer
to a question is more often in a comment ("reran it from the replica") than in the description.
Descriptions and comments are Atlassian Document Format (JSON), written here as Markdown.

The file is `<PROJECT>/<KEY>.md`; a Markdown file named after a ticket key defines that key
(ingest.chunk_file), so a page, commit message or code comment naming `SHOP-812` links to it."""

import re
import urllib.parse
from datetime import datetime, timezone

from ..ingest import slug
from . import atlassian
from .markdown import fence, safe, table, tidy
from .mirror import Doc, Entry

KIND = "jira"
PROJECT_URL = re.compile(r"/(?:browse|projects)/([A-Z][A-Z0-9_]+)(?:[/?#]|$)")
ISSUE_URL = re.compile(r"/browse/([A-Z][A-Z0-9_]+-\d+)")
FIELDS = "summary,description,comment,issuetype,status,resolution,priority,assignee,reporter,created,updated," \
         "labels,issuelinks,parent"
BATCH = 50  # tickets per search request when fetching bodies


def origin(url: str, query: str | None = None) -> str | None:
    """`<site>/issues/?jql=...`, a URL Jira itself opens, from a project URL or a search URL."""
    s = atlassian.site(url)
    if not s:
        return None
    u = urllib.parse.urlparse(url)
    jql = urllib.parse.parse_qs(u.query).get("jql", [None])[0]
    if jql is None:
        m = PROJECT_URL.search(u.path)
        if not m:
            return None
        jql = f"project = {m.group(1)}"
    jql = re.split(r"\border\s+by\b", jql, flags=re.I)[0].strip()  # order is meaningless to a mirror
    if query:
        jql = f"({jql}) AND ({query})"
    return f"{s}/issues/?jql={urllib.parse.quote(jql)}"


def locate(url: str) -> tuple[str, str] | None:
    """(ticket key, `focusedCommentId=<id>` or "") of a ticket URL."""
    if not atlassian.site(url):
        return None
    u = urllib.parse.urlparse(url)
    q = urllib.parse.parse_qs(u.query)
    m = ISSUE_URL.search(u.path)
    key = m.group(1) if m else q.get("selectedIssue", [None])[0]
    if not key:
        return None
    comment = q.get("focusedCommentId", [None])[0]
    return key, f"focusedCommentId={comment}" if comment else ""


def heading_url(item: dict, heads: list[str]) -> str:
    return item.get("anchors", {}).get(slug(heads[-1]), item["url"]) if heads else item["url"]


def fetch_url(url: str) -> tuple[str, Doc] | None:
    hit = locate(url)
    if not hit:
        return None
    site = atlassian.site(url)
    api = atlassian.client(site)
    issue = api.get(f"/rest/api/3/issue/{hit[0]}", {"fields": FIELDS})
    return f"{site}/browse/{issue['key']}", issue_doc(site, issue, _comments(api, issue))


def _comments(api, issue: dict) -> list[dict]:
    """Every comment; a search returns only the first ones of a long thread."""
    got = issue["fields"].get("comment") or {"comments": [], "total": 0}
    comments = list(got.get("comments", []))
    while len(comments) < got.get("total", 0):
        page = api.get(f"/rest/api/3/issue/{issue['key']}/comment", {"startAt": len(comments), "maxResults": 100})
        if not page.get("comments"):
            break
        comments += page["comments"]
    return comments


class Remote:
    FORMAT = 3  # raise when the Markdown or the fields written change, so every mirror is written again

    def __init__(self, origin_url: str):
        self.origin = origin_url
        self.site = origin_url.split("/issues/")[0]
        self.jql = urllib.parse.parse_qs(urllib.parse.urlparse(origin_url).query)["jql"][0]
        projects = set(re.findall(r"\bproject\s*=\s*\"?([A-Z][A-Z0-9_]+)\"?", self.jql, re.I))
        self.name = f"jira-{projects.pop()}" if len(projects) == 1 else "jira"
        self.api = atlassian.client(self.site)

    def _search(self, jql: str, fields: str, size: int):
        params = {"jql": jql, "fields": fields, "maxResults": size}
        while True:
            page = self.api.get("/rest/api/3/search/jql", params)
            yield from page.get("issues", [])
            if page.get("isLast", True) or not page.get("nextPageToken"):
                return
            params["nextPageToken"] = page["nextPageToken"]

    def listing(self) -> dict[str, Entry]:
        return {i["key"]: Entry(i["fields"]["updated"], f"{i['key'].split('-')[0]}/{i['key']}.md",
                                f"{self.site}/browse/{i['key']}")
                for i in self._search(self.jql, "updated", 1000)}

    def fetch(self, keys: list[str]):
        for n in range(0, len(keys), BATCH):
            for issue in self._search(f"key in ({','.join(keys[n:n + BATCH])})", FIELDS, BATCH):
                yield issue["key"], issue_doc(self.site, issue, _comments(self.api, issue))


# ---------------------------------------------------------------- ticket -> Markdown


def _name(user: dict | None) -> str:
    return (user or {}).get("displayName", "")


def _day(stamp: str | None) -> str:
    return (stamp or "")[:10]


def issue_doc(site: str, issue: dict, comments: list[dict]) -> Doc:
    f, key = issue["fields"], issue["key"]
    url = f"{site}/browse/{key}"
    status = (f.get("status") or {}).get("name", "")
    if f.get("resolution"):
        status += f" ({f['resolution']['name']})"
    facts = [(f.get("issuetype") or {}).get("name", ""), status, (f.get("priority") or {}).get("name", ""),
             f"reporter {_name(f.get('reporter'))}" if f.get("reporter") else "",
             f"assignee {_name(f.get('assignee'))}" if f.get("assignee") else "",
             f"created {_day(f.get('created'))}", f"updated {_day(f.get('updated'))}"]
    lines = [f"# {key} {f.get('summary', '')}", f"<!-- defines: {key} -->", "",
             " · ".join(x for x in facts if x), ""]
    if f.get("labels"):
        lines += [f"Labels: {', '.join(f['labels'])}", ""]
    if f.get("parent"):
        p = f["parent"]
        lines += [f"Parent: {p['key']} {p.get('fields', {}).get('summary', '')}".rstrip(), ""]
    links = []
    for l in f.get("issuelinks") or []:
        other, verb = (l["outwardIssue"], l["type"]["outward"]) if "outwardIssue" in l else (l["inwardIssue"], l["type"]["inward"])
        links.append(f"- {verb} {other['key']} {other.get('fields', {}).get('summary', '')}".rstrip())
    if links:
        lines += ["Links:", *links, ""]
    if f.get("description"):
        lines += ["## Description", "", *adf_blocks(f["description"])]
    anchors = {}
    for c in comments:
        stamp = (c.get("created") or "")[:16].replace("T", " ")
        head = f"Comment {stamp} {_name(c.get('author'))}".rstrip()
        anchors[slug(head)] = f"{url}?focusedCommentId={c['id']}"
        lines += [f"## {head}", "", *adf_blocks(c.get("body") or {})]
    meta = {"project": key.split("-")[0], "status": (f.get("status") or {}).get("name", ""),
            "resolution": (f.get("resolution") or {}).get("name", ""),
            "issuetype": (f.get("issuetype") or {}).get("name", ""),
            "priority": (f.get("priority") or {}).get("name", ""),
            "assignee": _name(f.get("assignee")), "reporter": _name(f.get("reporter")),
            "labels": list(f.get("labels") or []), "parent": (f.get("parent") or {}).get("key", ""),
            "created": _day(f.get("created")), "updated": _day(f.get("updated"))}
    return Doc(tidy(lines), anchors, meta)


def adf_blocks(node: dict, shift: int = 2) -> list[str]:
    """Atlassian Document Format -> Markdown lines. Headings move down `shift` levels, under the
    ticket title and its section."""
    t, kids = node.get("type"), node.get("content", [])
    attrs = node.get("attrs", {})
    if t == "paragraph":
        text = adf_inline(kids)
        return [safe(l.strip()) for l in text.split("\n") if l.strip()] + [""]
    if t == "heading":
        text = " ".join(adf_inline(kids).split())
        return [f"{'#' * min(attrs.get('level', 1) + shift, 6)} {text}", ""] if text else []
    if t in ("bulletList", "orderedList", "taskList", "decisionList"):
        out = []
        for i, item in enumerate(kids, attrs.get("order", 1)):
            mark = {"orderedList": f"{i}. ", "taskList": "- [x] " if item.get("attrs", {}).get("state") == "DONE"
                    else "- [ ] "}.get(t, "- ")
            inner = item.get("content", [])
            body = adf_blocks({"type": "doc", "content": inner}, shift) if inner and inner[0].get("type") in BLOCKS \
                else [adf_inline(inner)]
            body = [l for l in body if l.strip()]
            if body:
                out += [mark + body[0], *(" " * len(mark) + l for l in body[1:])]
        return out + [""]
    if t == "codeBlock":
        return fence("".join(k.get("text", "") for k in kids), attrs.get("language", "") or "")
    if t == "blockquote":
        return ["> " + l if l else "" for l in _concat(kids, shift)]
    if t == "panel":
        return [f"**{attrs.get('panelType', 'note').capitalize()}:**", *_concat(kids, shift)]
    if t in ("expand", "nestedExpand"):
        title = attrs.get("title", "")
        return ([f"**{title}**"] if title else []) + _concat(kids, shift)
    if t == "table":
        rows = [[" ".join(l for l in _concat(c.get("content", []), shift) if l) for c in r.get("content", [])]
                for r in kids]
        return table(rows[0], rows[1:]) if rows else []
    if t in ("mediaSingle", "mediaGroup", "media"):
        media = [m for m in ([node] if t == "media" else kids) if m.get("type") == "media"]
        return [f"[image: {m.get('attrs', {}).get('alt', '')}]".replace(": ]", "]") for m in media] + [""]
    if t in ("blockCard", "embedCard"):
        return [attrs.get("url", ""), ""]
    if t == "rule":
        return []
    return _concat(kids, shift)  # doc, layouts, extensions


BLOCKS = {"paragraph", "heading", "bulletList", "orderedList", "taskList", "decisionList", "codeBlock",
          "blockquote", "panel", "expand", "nestedExpand", "table", "mediaSingle", "mediaGroup", "rule"}


def _concat(kids: list[dict], shift: int) -> list[str]:
    return [l for k in kids for l in adf_blocks(k, shift)]


def adf_inline(nodes: list[dict]) -> str:
    out = []
    for n in nodes:
        t, attrs = n.get("type"), n.get("attrs", {})
        if t == "text":
            s = n.get("text", "")
            for m in n.get("marks", []):
                kind = m.get("type")
                if kind == "code":
                    s = f"`{s}`"
                elif kind == "strong" and s.strip():
                    s = f"**{s}**"
                elif kind == "em" and s.strip():
                    s = f"*{s}*"
                elif kind == "link":
                    s = f"[{s.replace(']', ')')}]({m.get('attrs', {}).get('href', '').replace(' ', '%20')})"
            out.append(s)
        elif t == "hardBreak":
            out.append("\n")
        elif t == "mention":
            out.append(attrs.get("text") or "@user")
        elif t == "emoji":
            out.append(attrs.get("text") or attrs.get("shortName", ""))
        elif t == "inlineCard":
            out.append(attrs.get("url", ""))
        elif t == "status":
            out.append(f"[{attrs.get('text', '')}]")
        elif t == "date":
            ts = int(attrs.get("timestamp", 0)) / 1000
            out.append(datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d"))
        elif n.get("content"):
            out.append(adf_inline(n["content"]))
    return "".join(out)
