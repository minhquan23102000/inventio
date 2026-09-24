"""A remote source's copy on this machine: one Markdown file per item, and a manifest.

The manifest (`.inventio/manifest.json` inside the mirror, a directory ingest never reads) holds
per item the version it was fetched at, its file, and where it lives on the web. A sync lists
what exists now, fetches only items whose version changed, moves files whose place changed (a
page renamed, or its parent), and deletes files of items that are gone. Ingest then sees only
the files that changed, exactly as with a directory someone edited."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..store import data_home

MANIFEST = Path(".inventio") / "manifest.json"
SAVE_EVERY = 100  # fetched items between manifest writes, so a crash does not refetch everything


@dataclass
class Entry:
    """An item as a listing sees it, without its body."""
    version: str  # anything that changes when the item does (a page version, an update time)
    path: str     # POSIX, relative to the mirror
    url: str


@dataclass
class Doc:
    text: str  # Markdown, starting with a `# title` line
    # heading slug (ingest.slug) -> URL, for headings that are a place of their own (a Jira comment)
    anchors: dict[str, str] = field(default_factory=dict)


def mirror_dir(name: str) -> Path:
    return data_home() / "mirrors" / name


def load(root: Path) -> dict:
    p = root / MANIFEST
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"kind": "", "origin": "", "items": {}}


def save(root: Path, manifest: dict) -> None:
    p = root / MANIFEST
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")


def _prune(root: Path, rel: str) -> None:
    """Remove a file and the directories it leaves empty."""
    p = root / rel
    p.unlink(missing_ok=True)
    for d in p.parents:
        if d == root or not d.is_relative_to(root) or any(d.iterdir()):
            break
        d.rmdir()


def sync(kind: str, remote, root: Path, log=print) -> dict:
    """Bring the mirror in line with the remote; returns what it did."""
    m = load(root)
    if m.get("format") != remote.FORMAT:  # the Markdown this connector writes has changed: write every item again
        for it in m["items"].values():
            it["version"] = None
    m["kind"], m["origin"], m["format"] = kind, remote.origin, remote.FORMAT
    items: dict[str, dict] = m["items"]
    now: dict[str, Entry] = remote.listing()
    counts = {"fetched": 0, "moved": 0, "deleted": 0, "same": 0}
    for gone in set(items) - set(now):
        _prune(root, items.pop(gone)["path"])
        counts["deleted"] += 1
    stale = []
    for iid, e in now.items():
        old = items.get(iid)
        if old is None or old["version"] != e.version:
            stale.append(iid)
            continue
        if old["path"] != e.path:
            (root / e.path).parent.mkdir(parents=True, exist_ok=True)
            os.replace(root / old["path"], root / e.path)
            _prune(root, old["path"])
            counts["moved"] += 1
        else:
            counts["same"] += 1
        old.update(path=e.path, url=e.url)
    if stale:
        log(f"{remote.origin}: fetching {len(stale)} of {len(now)} items")
    for iid, doc in remote.fetch(stale):
        e, old = now[iid], items.get(iid)
        if old and old["path"] != e.path:
            _prune(root, old["path"])
        _write(root, e.path, doc.text)
        items[iid] = {"version": e.version, "path": e.path, "url": e.url, "anchors": doc.anchors}
        counts["fetched"] += 1
        if counts["fetched"] % SAVE_EVERY == 0:
            save(root, m)
            log(f"  {counts['fetched']}/{len(stale)}")
    save(root, m)
    return counts


_by_path: dict[str, tuple[str, dict]] = {}


def item_at(root: str, path: str) -> tuple[str, dict] | None:
    """(kind, manifest item) of the file at `path` in a mirror; None for a plain directory source."""
    if root not in _by_path:
        m = load(Path(root))
        _by_path[root] = (m["kind"], {it["path"]: it for it in m["items"].values()})
    kind, index = _by_path[root]
    return (kind, index[path]) if path in index else None
