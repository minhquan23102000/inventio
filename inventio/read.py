"""`inventio read`: the lines behind a coordinate or a URL.

A person or an agent follows a result (`wiki-GD:Ops/Review-policy.md:40-58`), a link it printed,
or a URL met in a page or ticket. A mirrored page or ticket is read from the copy on this machine;
an Atlassian URL outside every mirror is fetched live with the reader's own credentials and
printed, without being added to the map."""

import re
from dataclasses import dataclass
from pathlib import Path

from . import connectors
from .connectors import RemoteError
from .ingest import FENCE, HEADING, file_text, slug

COORD = re.compile(r"^(?P<source>[^:/\\]+):(?P<path>.+?)(?::(?P<start>\d+)(?:-(?P<end>\d+))?)?$")


@dataclass
class Passage:
    source: str | None  # None when read live
    path: str
    start: int
    end: int
    lines: list[str]
    url: str | None = None
    version: str | None = None

    def as_dict(self) -> dict:
        return {"source": self.source, "path": self.path, "start_line": self.start, "end_line": self.end,
                "url": self.url, "version": self.version, "text": "\n".join(self.lines)}


def resolve(con, target: str) -> Passage:
    return _from_url(con, target) if target.startswith(("http://", "https://")) else _from_coord(con, target)


def _from_coord(con, target: str) -> Passage:
    m = COORD.match(target)
    if not m:
        raise RemoteError(f"not a coordinate (source:path:start-end) or a URL: {target}")
    row = con.execute("SELECT root FROM sources WHERE name = ?", (m["source"],)).fetchone()
    if not row:
        raise RemoteError(f"no source named {m['source']!r}; see `inventio sources`")
    file = Path(row["root"]) / m["path"]
    if not file.is_file():
        raise RemoteError(f"{m['source']} has no file {m['path']}")
    text = file_text(file).split("\n")
    start = int(m["start"] or 1)
    end = int(m["end"] or (m["start"] and start) or len(text))
    start, end = max(1, start), min(len(text), max(start, end))
    head = con.execute(
        "SELECT c.heading_path FROM chunks c JOIN files f ON f.id = c.file_id JOIN sources s ON s.id = f.source_id "
        "WHERE s.name = ? AND f.path = ? AND c.start_line <= ? ORDER BY c.start_line DESC LIMIT 1",
        (m["source"], m["path"], start)).fetchone()
    url = connectors.web_url(row["root"], m["path"], head["heading_path"] if head else "")
    return Passage(m["source"], m["path"], start, end, text[start - 1:end], url)


def _from_url(con, url: str) -> Passage:
    hit = connectors.locate(url)
    if not hit:
        raise RemoteError(f"not a Confluence page or Jira ticket URL: {url}")
    kind, item_id, fragment = hit
    found = connectors.mirrored(con, kind, item_id)
    if found:
        name, file, item = found
        text = file.read_text(encoding="utf-8").split("\n")
        start, end = section(text, fragment, item.get("anchors", {}))
        rel = Path(item["path"]).as_posix()
        return Passage(name, rel, start, end, text[start - 1:end], url, item["version"])
    web, doc = connectors.KINDS[kind].fetch_url(url)
    text = doc.text.split("\n")
    start, end = section(text, fragment, doc.anchors)
    return Passage(None, item_id, start, end, text[start - 1:end], web)


def section(lines: list[str], fragment: str, anchors: dict[str, str]) -> tuple[int, int]:
    """1-based lines of the section a URL fragment names: a heading anchor (`#Review-deadline`), or
    a place recorded in `anchors` (`focusedCommentId=...`). The whole text when nothing matches."""
    if not fragment:
        return 1, len(lines)
    want = next((s for s, u in anchors.items() if fragment in u), None)
    norm = lambda s: re.sub(r"[\W_]+", "", s.lower())  # noqa: E731
    in_fence, level, start = False, 0, 0
    for i, line in enumerate(lines, 1):
        if FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else HEADING.match(line)
        if not m:
            continue
        if start:
            if len(m.group(1)) <= level:
                return start, i - 1
        elif (slug(m.group(2)) == want) if want else (norm(m.group(2)) == norm(fragment)):
            level, start = len(m.group(1)), i
    return (start, len(lines)) if start else (1, len(lines))
