"""Turn a directory into chunks with exact line coordinates, plus the identifiers and references
each chunk carries. Everything here is code reading structure the source already has: headings,
definitions, links. No model is asked what code can know.
"""

import ast
import fnmatch
import hashlib
import keyword
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

MAX_CHARS = 1500

LANGS = {
    ".md": "markdown", ".markdown": "markdown", ".mdx": "markdown",
    ".py": "python",
    ".txt": "text", ".rst": "text",
    ".sql": "code", ".yml": "code", ".yaml": "code", ".toml": "code", ".ini": "code", ".cfg": "code",
    ".js": "code", ".mjs": "code", ".ts": "code", ".tsx": "code", ".jsx": "code",
    ".java": "code", ".scala": "code", ".kt": "code", ".go": "code", ".rs": "code",
    ".sh": "code", ".ps1": "code", ".json": "code",
}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".mypy_cache", ".pytest_cache"}
MAX_FILE_BYTES = 1_000_000


@dataclass
class Chunk:
    kind: str
    heading_path: str
    start_line: int
    end_line: int
    text: str
    anchor: str = ""
    parent: int | None = None  # index into the file's chunk list
    defines: set[str] = field(default_factory=set)
    mentions: set[str] = field(default_factory=set)
    refs: list[tuple[str, str]] = field(default_factory=list)  # (target path relative to source root, anchor)


def list_files(root: Path, excludes: list[str]) -> list[str]:
    """Relative POSIX paths. Inside a git work tree, git decides what is ignored."""
    rels: list[str] | None = None
    try:
        out = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard", "-z"],
            cwd=root, capture_output=True, check=True,
        ).stdout.decode("utf-8", "replace")
        rels = [p for p in out.split("\0") if p]
    except (OSError, subprocess.CalledProcessError):
        rels = None
    if rels is None:
        rels = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for name in filenames:
                rels.append(Path(dirpath, name).relative_to(root).as_posix())
    keep = []
    for rel in rels:
        if Path(rel).suffix.lower() not in LANGS:
            continue
        if any(fnmatch.fnmatch(rel, pat) for pat in excludes):
            continue
        if any(part in SKIP_DIRS for part in Path(rel).parts):
            continue
        p = root / rel
        try:
            if not p.is_file() or p.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        keep.append(rel)
    return sorted(set(keep))


def decode_text(data: bytes) -> str | None:
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", "replace").replace("\r\n", "\n")


# ---------------------------------------------------------------- identifiers and references

BACKTICK = re.compile(r"`([^`\n]{2,80})`")
IDENT_SHAPES = [
    re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"),            # snake_case
    re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]+)+\b"),                 # camelCase
    re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b"),         # PascalCase
    re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b"),                         # ticket ids: FRAML-123
    re.compile(r"\b[a-z0-9_]+\.[a-z0-9_]+\.[a-z0-9_]+\b"),        # schema.table.column
]
TOKEN_IN_TICKS = re.compile(r"^[A-Za-z_@][\w.\-/:@]*$")
MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def norm_ident(s: str) -> str:
    return s.strip().strip("()").lower()


def mentions_in(text: str) -> set[str]:
    found = set()
    for m in BACKTICK.finditer(text):
        inner = m.group(1).strip()
        if TOKEN_IN_TICKS.match(inner) and len(inner) >= 3:
            found.add(norm_ident(inner))
    for rx in IDENT_SHAPES:
        found.update(norm_ident(x) for x in rx.findall(text))
    return found


def slug(heading: str) -> str:
    s = heading.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"\s+", "-", s)


def md_refs(text: str, file_rel: str) -> list[tuple[str, str]]:
    out = []
    base = Path(file_rel).parent
    for m in MD_LINK.finditer(text):
        target = m.group(1)
        if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I):  # http:, skill:, mailto: ...
            continue
        path, _, anchor = target.partition("#")
        if not path:
            out.append((file_rel, anchor))
            continue
        resolved = os.path.normpath((base / path).as_posix()).replace("\\", "/")
        out.append((resolved, anchor))
    return out


# ---------------------------------------------------------------- chunkers


def _trimmed(start: int, ls: list[str]) -> tuple[int, list[str]]:
    """Drop blank lines at both ends so coordinates cover only lines that carry text."""
    lo, hi = 0, len(ls)
    while lo < hi and not ls[lo].strip():
        lo += 1
    while hi > lo and not ls[hi - 1].strip():
        hi -= 1
    return start + lo, ls[lo:hi]


def split_long(lines: list[str], first: int, kind: str, heading_path: str, anchor: str) -> list[Chunk]:
    """Split a span into pieces of at most MAX_CHARS, cutting at blank lines where possible."""
    pieces, cur, cur_start, size = [], [], 0, 0
    for i, line in enumerate(lines):
        if not cur:
            cur_start = i
        cur.append(line)
        size += len(line) + 1
        at_break = line.strip() == ""
        if size >= MAX_CHARS and (at_break or size >= 2 * MAX_CHARS):
            pieces.append((cur_start, cur))
            cur, size = [], 0
    if cur:
        pieces.append((cur_start, cur))
    out = []
    for s, ls in pieces:
        s, ls = _trimmed(s, ls)
        if ls:
            out.append(Chunk(kind, heading_path, first + s, first + s + len(ls) - 1, "\n".join(ls), anchor))
    return out


HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")


def chunk_markdown(text: str) -> list[Chunk]:
    lines = text.split("\n")
    stack: list[tuple[int, str]] = []
    in_fence = False
    starts = [(0, 0, [])]
    for i, line in enumerate(lines):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEADING.match(line)
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            starts.append((i, level, [t for _, t in stack]))
    chunks: list[Chunk] = []
    section_index: dict[tuple[str, ...], int] = {}
    for n, (start, level, heads) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        body = lines[start:end]
        if not "\n".join(body).strip():
            continue
        # a heading directly followed by a sub-heading carries no body of its own
        if heads and len([l for l in body[1:] if l.strip()]) == 0:
            continue
        hp = " > ".join(heads)
        anchor = slug(heads[-1]) if heads else ""
        parent = None
        for k in range(len(heads) - 1, 0, -1):
            parent = section_index.get(tuple(heads[:k]))
            if parent is not None:
                break
        pieces = split_long(body, start + 1, "section", hp, anchor)
        for p in pieces:
            p.parent = parent
        if pieces:
            section_index[tuple(heads)] = len(chunks)
        chunks.extend(pieces)
    return chunks


PY_NAME = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
PY_KEYWORDS = set(keyword.kwlist) | {"self", "cls"}


def chunk_python(text: str) -> list[Chunk]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return chunk_generic(text)
    lines = text.split("\n")
    chunks: list[Chunk] = []
    covered = [False] * (len(lines) + 2)

    def span(node) -> tuple[int, int]:
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        return start, node.end_lineno

    def add(node, qual: str, parent: int | None) -> None:
        s, e = span(node)
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        body_nodes = [b for b in node.body if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        if kind == "class" and body_nodes and len("\n".join(lines[s - 1:e])) > MAX_CHARS:
            head_end = span(body_nodes[0])[0] - 1
            c = Chunk(kind, qual, s, head_end, "\n".join(lines[s - 1:head_end]), qual.split(".")[-1], parent)
            c.defines.add(norm_ident(node.name))
            me = len(chunks)
            chunks.append(c)
            for i in range(s, head_end + 1):
                covered[i] = True
            for b in body_nodes:
                add(b, f"{qual}.{b.name}", me)
            return
        for p in split_long(lines[s - 1:e], s, kind, qual, qual.split(".")[-1]):
            p.parent = parent
            p.defines.add(norm_ident(node.name))
            chunks.append(p)
        for i in range(s, e + 1):
            covered[i] = True

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add(node, node.name, None)
    # module-level code between definitions
    run: list[int] = []
    for i in range(1, len(lines) + 1):
        if not covered[i]:
            run.append(i)
        if (covered[i] or i == len(lines)) and run:
            seg = lines[run[0] - 1:run[-1]]
            if "\n".join(seg).strip():
                chunks.extend(split_long(seg, run[0], "module", "(module)", ""))
            run = []
    for c in chunks:
        c.mentions.update(norm_ident(x) for x in PY_NAME.findall(c.text) if x not in PY_KEYWORDS)
    return chunks


def chunk_generic(text: str) -> list[Chunk]:
    lines = text.split("\n")
    return split_long(lines, 1, "block", "", "")


def chunk_file(text: str, lang: str, rel: str) -> list[Chunk]:
    if lang == "markdown":
        chunks = chunk_markdown(text)
    elif lang == "python":
        chunks = chunk_python(text)
    else:
        chunks = chunk_generic(text)
    for c in chunks:
        c.mentions |= mentions_in(c.text)
        if lang == "markdown":
            c.refs = md_refs(c.text, rel)
    return chunks


def ingest_source(con, name: str, root: Path, public: bool, excludes: list[str], full: bool = False) -> dict:
    """Bring one source's part of the map in line with the files on disk.

    A file whose size and modification time match the map is skipped without being read; one
    whose bytes hash the same is only re-stamped. Only new and edited files are chunked again,
    so unchanged chunks keep their ids. `full` drops the source first and rebuilds everything.
    """
    from .store import drop_file, drop_source

    root = root.resolve()
    row = con.execute("SELECT id FROM sources WHERE name = ?", (name,)).fetchone()
    if row:
        sid = row["id"]
        if full:
            drop_source(con, sid)
        con.execute(
            "UPDATE sources SET root = ?, public = ?, excludes = ?, indexed_at = datetime('now') WHERE id = ?",
            (str(root), int(public), "\n".join(excludes), sid),
        )
    else:
        sid = con.execute(
            "INSERT INTO sources (name, root, public, excludes, indexed_at) VALUES (?, ?, ?, ?, datetime('now'))",
            (name, str(root), int(public), "\n".join(excludes)),
        ).lastrowid
    known = {
        r["path"]: r
        for r in con.execute("SELECT id, path, size, mtime_ns, sha1 FROM files WHERE source_id = ?", (sid,))
    }
    counts = {"added": 0, "changed": 0, "removed": 0, "unchanged": 0}
    for rel in list_files(root, excludes):
        path = root / rel
        st = path.stat()
        old = known.pop(rel, None)
        if old is not None and old["size"] == st.st_size and old["mtime_ns"] == st.st_mtime_ns:
            counts["unchanged"] += 1
            continue
        data = path.read_bytes()
        digest = hashlib.sha1(data).hexdigest()
        if old is not None and old["sha1"] == digest:
            con.execute("UPDATE files SET size = ?, mtime_ns = ? WHERE id = ?", (st.st_size, st.st_mtime_ns, old["id"]))
            counts["unchanged"] += 1
            continue
        if old is not None:
            drop_file(con, old["id"])
        text = decode_text(data)
        if text is None or not text.strip():  # binary or empty: not in the map
            counts["removed"] += old is not None
            continue
        counts["changed" if old is not None else "added"] += 1
        _insert_file(con, sid, rel, text, st.st_size, st.st_mtime_ns, digest)
    for old in known.values():  # indexed before, gone from disk or now excluded
        drop_file(con, old["id"])
        counts["removed"] += 1
    n_files, n_chunks = con.execute(
        "SELECT count(DISTINCT f.id), count(c.id) FROM files f LEFT JOIN chunks c ON c.file_id = f.id "
        "WHERE f.source_id = ?",
        (sid,),
    ).fetchone()
    return {"source": name, "files": n_files, "chunks": n_chunks, **counts}


def _insert_file(con, sid: int, rel: str, text: str, size: int, mtime_ns: int, digest: str) -> None:
    lang = LANGS[Path(rel).suffix.lower()]
    fid = con.execute(
        "INSERT INTO files (source_id, path, lang, size, mtime_ns, sha1) VALUES (?, ?, ?, ?, ?, ?)",
        (sid, rel, lang, size, mtime_ns, digest),
    ).lastrowid
    ids: list[int] = []
    for c in chunk_file(text, lang, rel):
        parent_id = ids[c.parent] if c.parent is not None and c.parent < len(ids) else None
        cid = con.execute(
            "INSERT INTO chunks (file_id, parent_id, kind, heading_path, anchor, start_line, end_line, text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (fid, parent_id, c.kind, c.heading_path, c.anchor, c.start_line, c.end_line, c.text),
        ).lastrowid
        ids.append(cid)
        head = f"{rel} {c.heading_path}".replace("/", " ").replace("_", " ").replace("-", " ")
        con.execute("INSERT INTO chunks_fts (rowid, head, body) VALUES (?, ?, ?)", (cid, head, c.text))
        con.executemany(
            "INSERT INTO idents (chunk_id, ident, role) VALUES (?, ?, ?)",
            [(cid, x, "defines") for x in c.defines] + [(cid, x, "mentions") for x in c.mentions - c.defines],
        )
        con.executemany(
            "INSERT INTO refs (chunk_id, target_path, target_anchor) VALUES (?, ?, ?)",
            [(cid, p, a) for p, a in c.refs],
        )
