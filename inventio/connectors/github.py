"""GitHub: a repository's issues and pull requests, one Markdown file per item.

    inventio init https://github.com/acme/shop       -> source gh-shop

Inventio keeps no GitHub token: every run asks the GitHub CLI for the one it already holds
(`gh auth token --hostname <host>`), so `gh auth login` once is all it takes, for github.com and
for GitHub Enterprise hosts alike. Without the CLI, GH_TOKEN or GITHUB_TOKEN is used.

A file holds the item's description, its comments, and for a pull request its reviews and the
review comments with the file and line they are on. `#123` and links to this repository's
issues become relative links, so they are `citation` links in the map; the first line defines
`shop#123` and `acme/shop#123`, so a page or ticket naming `acme/shop#123` links to it. A bare
`#123` outside the repository's own items names no repository and links nowhere.

The code itself is not fetched: index a clone with `inventio init <dir>`.
"""

import os
import re
import shutil
import subprocess
import urllib.parse

from ..ingest import FENCE, HEADING, slug
from .http import Client, RemoteError
from .markdown import fence, tidy
from .mirror import Doc, Entry

KIND = "github"
PAGE = 100
REPO_PATH = re.compile(r"^/([\w.\-]+)/([\w.\-]+?)(?:\.git)?(?:/|$)")
ITEM_PATH = re.compile(r"^/[\w.\-]+/[\w.\-]+/(?:issues|pull)/(\d+)")
_tokens: dict[str, str | None] = {}  # host -> the GitHub CLI's token, asked once per process


def _gh(host: str) -> str | None:
    """The GitHub CLI's token for this host, or None (not installed, not signed in there)."""
    if host not in _tokens:
        found = None
        if shutil.which("gh"):
            r = subprocess.run(["gh", "auth", "token", "--hostname", host], capture_output=True, text=True)
            found = r.stdout.strip() if r.returncode == 0 else None
        _tokens[host] = found or None
    return _tokens[host]


def token(host: str, quiet: bool = False) -> str | None:
    """The token requests to this host carry: the GitHub CLI's, else GH_TOKEN or GITHUB_TOKEN
    for github.com and GH_ENTERPRISE_TOKEN for another host."""
    env = ("GH_TOKEN", "GITHUB_TOKEN") if host == "github.com" else ("GH_ENTERPRISE_TOKEN",)
    found = _gh(host) or next((os.environ[v] for v in env if os.environ.get(v)), None)
    if found is None and not quiet:
        raise RemoteError(f"GitHub on {host} needs the GitHub CLI signed in: gh auth login --hostname {host}")
    return found


def host_of(url: str) -> str:
    return (urllib.parse.urlparse(url).hostname or "").lower()


def _is_github(url: str) -> bool:
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        return False
    host = u.hostname.lower()
    # another host is GitHub Enterprise only if the CLI holds a login for it
    return host == "github.com" or _gh(host) is not None


def origin(url: str, query: str | None = None) -> str | None:
    """`https://<host>/<owner>/<repo>` from any URL inside the repository."""
    if "://" not in url or not REPO_PATH.match(urllib.parse.urlparse(url).path) or not _is_github(url):
        return None
    if query:
        raise ValueError("--jql applies to a Jira URL")
    owner, repo = REPO_PATH.match(urllib.parse.urlparse(url).path).groups()
    return f"https://{host_of(url)}/{owner}/{repo}"


def locate(url: str) -> tuple[str, str] | None:
    """(item number, `issuecomment-<id>` or "") of an issue or pull request URL."""
    u = urllib.parse.urlparse(url)
    m = ITEM_PATH.match(u.path)
    if not m or not _is_github(url):
        return None
    return m.group(1), u.fragment


def heading_url(item: dict, heads: list[str]) -> str:
    return item.get("anchors", {}).get(slug(heads[-1]), item["url"]) if heads else item["url"]


def _api(origin_url: str) -> tuple[Client, str]:
    host = host_of(origin_url)
    base = "https://api.github.com" if host == "github.com" else f"https://{host}/api/v3"
    owner_repo = urllib.parse.urlparse(origin_url).path.strip("/")
    client = Client(base, {"Authorization": f"Bearer {token(host)}", "Accept": "application/vnd.github+json",
                           "X-GitHub-Api-Version": "2022-11-28"})
    return client, f"/repos/{owner_repo}"


def _pages(api: Client, path: str, params: dict | None = None):
    n = 1
    while True:
        page = api.get(path, {**(params or {}), "per_page": PAGE, "page": n})
        yield from page
        if len(page) < PAGE:
            return
        n += 1


def fetch_url(url: str) -> tuple[str, Doc] | None:
    hit = locate(url)
    if not hit:
        return None
    o = origin(url)
    api, repo = _api(o)
    issue = api.get(f"{repo}/issues/{hit[0]}")
    return issue["html_url"], item_doc(o, issue, *_threads(api, repo, issue), known={})


def _threads(api: Client, repo: str, issue: dict) -> tuple[list, list, list]:
    """(comments, reviews, review comments) of an issue or pull request."""
    comments = list(_pages(api, f"{repo}/issues/{issue['number']}/comments")) if issue.get("comments") else []
    if "pull_request" not in issue:
        return comments, [], []
    n = issue["number"]
    return comments, list(_pages(api, f"{repo}/pulls/{n}/reviews")), list(_pages(api, f"{repo}/pulls/{n}/comments"))


class Remote:
    FORMAT = 1  # raise when the Markdown or the fields written change, so every mirror is written again

    def __init__(self, origin_url: str):
        self.origin = origin_url
        self.api, self.repo = _api(origin_url)
        self.name = "gh-" + origin_url.rstrip("/").rsplit("/", 1)[1]
        self.items: dict[str, dict] = {}

    def listing(self) -> dict[str, Entry]:
        # the issues listing holds pull requests too, with their bodies: fetch then only adds the threads
        self.items = {str(i["number"]): i for i in _pages(self.api, f"{self.repo}/issues",
                                                          {"state": "all", "sort": "updated", "direction": "desc"})}
        return {n: Entry(i["updated_at"], _path(i), i["html_url"]) for n, i in self.items.items()}

    def fetch(self, ids: list[str]):
        known = {n: _path(i) for n, i in self.items.items()}
        for n in ids:
            issue = self.items[n]
            yield n, item_doc(self.origin, issue, *_threads(self.api, self.repo, issue), known=known)


def _path(issue: dict) -> str:
    return f"{'pulls' if 'pull_request' in issue else 'issues'}/{issue['number']}.md"


# ---------------------------------------------------------------- item -> Markdown


def _login(user: dict | None) -> str:
    return (user or {}).get("login", "")


def _stamp(s: str | None) -> str:
    return (s or "")[:16].replace("T", " ")


def _state(issue: dict) -> str:
    pr = issue.get("pull_request")
    if pr and pr.get("merged_at"):
        return "merged"
    if issue.get("draft"):
        return "draft"
    return issue.get("state", "")


REF = re.compile(r"(?<![\w/&#\[])#(\d+)\b")  # not `[#7]`: a link already, maybe one just written
INLINE_CODE = re.compile(r"`[^`\n]*`")


def body_lines(text: str | None, origin_url: str, here: str, known: dict[str, str]) -> list[str]:
    """GitHub Markdown under a section of the item: headings two levels down, `#12` and URLs of
    this repository's items as relative links to their mirrored files."""
    items = re.compile(re.escape(origin_url) + r"/(?:issues|pull)/(\d+)(?:#[\w\-]+)?", re.I)
    depth = here.count("/")

    def rel(n: str) -> str:
        return "../" * depth + known[n]

    def link(line: str) -> str:
        parts, last = [], 0
        for m in INLINE_CODE.finditer(line):  # code spans stay as written
            parts += [_links(line[last:m.start()]), m.group(0)]
            last = m.end()
        return "".join(parts) + _links(line[last:])

    def _links(s: str) -> str:
        s = items.sub(lambda m: f"[#{m.group(1)}]({rel(m.group(1))})" if m.group(1) in known else m.group(0), s)
        return REF.sub(lambda m: f"[#{m.group(1)}]({rel(m.group(1))})" if m.group(1) in known else m.group(0), s)

    out, in_fence = [], False
    for line in (text or "").replace("\r\n", "\n").split("\n"):
        if FENCE.match(line):
            in_fence = not in_fence
            out.append(line)
        elif in_fence:
            out.append(line)
        elif m := HEADING.match(line):
            out.append("#" * min(len(m.group(1)) + 2, 6) + " " + m.group(2))
        else:
            out.append(link(line))
    return out + [""]


def item_doc(origin_url: str, issue: dict, comments: list[dict], reviews: list[dict],
             review_comments: list[dict], known: dict[str, str]) -> Doc:
    owner_repo = urllib.parse.urlparse(origin_url).path.strip("/")
    repo, n = owner_repo.split("/")[1], issue["number"]
    here = _path(issue)
    is_pr = "pull_request" in issue
    state = _state(issue)
    facts = ["Pull request" if is_pr else "Issue", state, f"by {_login(issue.get('user'))}",
             f"created {issue.get('created_at', '')[:10]}", f"updated {issue.get('updated_at', '')[:10]}"]
    if issue.get("closed_at") and state != "open":
        facts.append(f"closed {issue['closed_at'][:10]}")
    labels = [l["name"] for l in issue.get("labels") or []]
    assignees = [_login(a) for a in issue.get("assignees") or []]
    lines = [f"# {repo}#{n} {issue.get('title', '')}", f"<!-- defines: {repo}#{n} {owner_repo}#{n} -->", "",
             " · ".join(x for x in facts if x), ""]
    if labels:
        lines += [f"Labels: {', '.join(labels)}", ""]
    if assignees:
        lines += [f"Assignees: {', '.join(assignees)}", ""]
    if (issue.get("milestone") or {}).get("title"):
        lines += [f"Milestone: {issue['milestone']['title']}", ""]
    if issue.get("body"):
        lines += ["## Description", "", *body_lines(issue["body"], origin_url, here, known)]
    anchors: dict[str, str] = {}
    thread = [(c["created_at"], "comment", c) for c in comments]
    thread += [(r.get("submitted_at") or "", "review", r) for r in reviews if r.get("body")]
    thread += [(c["created_at"], "review comment", c) for c in review_comments]
    for _, what, c in sorted(thread, key=lambda t: t[0]):
        who, when = _login(c.get("user")), _stamp(c.get("created_at") or c.get("submitted_at"))
        if what == "review":
            head = f"Review {when} {who} ({c.get('state', '').lower().replace('_', ' ')})"
        elif what == "review comment":
            line = c.get("line") or c.get("original_line")
            head = f"Review comment {when} {who} on {c.get('path', '')}" + (f":{line}" if line else "")
        else:
            head = f"Comment {when} {who}"
        anchors[slug(head)] = c.get("html_url", issue["html_url"])
        lines += [f"## {head}", ""]
        if what == "review comment" and c.get("diff_hunk"):
            lines += fence("\n".join(c["diff_hunk"].split("\n")[-6:]), "diff")
        lines += body_lines(c.get("body"), origin_url, here, known)
    meta = {"repo": owner_repo, "item": "pull" if is_pr else "issue", "state": state,
            "author": _login(issue.get("user")), "assignee": assignees, "labels": labels,
            "milestone": (issue.get("milestone") or {}).get("title", ""),
            "created": issue.get("created_at", "")[:10], "updated": issue.get("updated_at", "")[:10],
            "closed": (issue.get("closed_at") or "")[:10]}
    return Doc(tidy(lines), anchors, meta)
