"""One SQLite file holds the whole map: sources, files, chunks, the BM25 index, links, labels,
content categories and every model judgment behind them.

The file lives in the user's cache directory, never inside an indexed repository: every derived
table carries source text, and a map built over private sources must not be committed anywhere.
"""

import os
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    root TEXT NOT NULL,
    public INTEGER NOT NULL DEFAULT 0,
    excludes TEXT NOT NULL DEFAULT '',
    indexed_at TEXT
);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    lang TEXT NOT NULL,
    type TEXT,
    size INTEGER,
    mtime_ns INTEGER,
    sha1 TEXT,
    UNIQUE (source_id, path)
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    parent_id INTEGER,
    kind TEXT NOT NULL,
    heading_path TEXT NOT NULL,
    anchor TEXT NOT NULL DEFAULT '',
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_file ON chunks(file_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    head, body, tokenize = 'unicode61 remove_diacritics 2'
);
CREATE TABLE IF NOT EXISTS idents (
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    ident TEXT NOT NULL,
    role TEXT NOT NULL
);
-- (ident, role): the mentions join looks up the few definers of an identifier; on ident alone
-- it scanned every mention of it too, which is quadratic on common names (26 s -> 0.1 s on astropy).
DROP INDEX IF EXISTS idents_ident;
CREATE INDEX IF NOT EXISTS idents_ident_role ON idents(ident, role);
-- chunk_id: re-indexing one edited file cascades its chunk deletes here; without it every
-- deleted chunk scanned the whole table (6.9 s for one file on astropy).
CREATE INDEX IF NOT EXISTS idents_chunk ON idents(chunk_id);
CREATE TABLE IF NOT EXISTS refs (
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    target_path TEXT NOT NULL,
    target_anchor TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS refs_chunk ON refs(chunk_id);
CREATE TABLE IF NOT EXISTS links (
    src INTEGER NOT NULL,
    dst INTEGER NOT NULL,
    rel TEXT NOT NULL,
    via TEXT NOT NULL,
    PRIMARY KEY (src, dst, rel)
);
CREATE INDEX IF NOT EXISTS links_dst ON links(dst);
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY,
    query TEXT NOT NULL,
    passage TEXT NOT NULL,
    noul REAL NOT NULL,
    model TEXT NOT NULL,
    source TEXT NOT NULL,
    at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (query, passage, model)
);
-- Every judgment a decision model made at index or query time, with the text it read: the
-- teacher data for fine-tuning Laya. `key` is the sha1 of (kind, question, passage, other), so a
-- rebuilt map or a rerun reuses a judgment instead of paying for it again.
--   kind 'category'        passage = a chunk, question = a schema.org type
--   kind 'query_category'  passage = a query, question = a schema.org type
--   kind 'same_thing'      passage = a chunk, other = a neighbour chunk
CREATE TABLE IF NOT EXISTS judgments (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    question TEXT NOT NULL,
    passage TEXT NOT NULL,
    other TEXT NOT NULL DEFAULT '',
    key TEXT NOT NULL,
    p REAL NOT NULL,
    model TEXT NOT NULL,
    source TEXT NOT NULL,
    at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (key, model)
);
-- one row per (chunk, type); `kept` marks the types the chunk keeps (p >= threshold, at most 3)
CREATE TABLE IF NOT EXISTS chunk_categories (
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    category TEXT NOT NULL,
    p REAL NOT NULL,
    kept INTEGER NOT NULL,
    model TEXT NOT NULL,
    PRIMARY KEY (chunk_id, category)
);
CREATE INDEX IF NOT EXISTS chunk_categories_kept ON chunk_categories(category, chunk_id) WHERE kept = 1;
"""


def cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "inventio"


def default_db() -> Path:
    env = os.environ.get("INVENTIO_DB")
    if env:
        return Path(env)
    return cache_dir() / "map.db"


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = Path(path) if path else default_db()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    con.executescript(SCHEMA)
    # maps built by earlier versions lack these columns; NULL reads as "changed" or "retype"
    have = {r["name"] for r in con.execute("PRAGMA table_info(files)")}
    for col, kind in (("size", "INTEGER"), ("mtime_ns", "INTEGER"), ("sha1", "TEXT"), ("type", "TEXT")):
        if col not in have:
            con.execute(f"ALTER TABLE files ADD COLUMN {col} {kind}")
    return con


# fact links (`about`) are judged, not rebuilt from the sources, so they are removed with their chunks
_ABOUT = "DELETE FROM links WHERE rel = 'about' AND (src IN ({ids}) OR dst IN ({ids}))"


def drop_source(con: sqlite3.Connection, source_id: int) -> None:
    """Remove every derived row of one source; FTS rows are keyed by chunk id, so clear them first."""
    ids = "SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.source_id = ?"
    con.execute(_ABOUT.format(ids=ids), (source_id, source_id))
    con.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({ids})", (source_id,))
    con.execute("DELETE FROM files WHERE source_id = ?", (source_id,))


def drop_file(con: sqlite3.Connection, file_id: int) -> None:
    """Remove one file and everything derived from it (chunks, BM25 rows, identifiers, references)."""
    ids = "SELECT id FROM chunks WHERE file_id = ?"
    con.execute(_ABOUT.format(ids=ids), (file_id, file_id))
    con.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({ids})", (file_id,))
    con.execute("DELETE FROM files WHERE id = ?", (file_id,))
