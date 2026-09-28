"""Links between chunks, all drawn by code from what the sources already say.

Relation names come from schema.org's CreativeWork vocabulary:
- `citation`: a Markdown link points at a file or heading; an absolute URL points at the item
  another source mirrors (a ticket linking a page, a page linking a ticket or a GitHub issue).
- `mentions`: a chunk names an identifier another chunk defines (a function, a class), or two
  chunks in different files share a rare identifier-shaped token (`nightly_backup`, `SHOP-123`), or
  two prose chunks in different files both name a defined identifier used in few places (the runbook
  and the incident report that both name the job). This is the bridge between a repository and the
  prose written about it, and between the pages written about the same code.
Parent sections (`isPartOf`) are kept on the chunk row itself. Links judged by a model (`about`,
see facts.py) are not drawn from the sources, so a rebuild leaves them alone.
"""

from pathlib import Path

MAX_DEFINERS = 3     # an identifier defined in more places than this is too ambiguous to link
MAX_SHARED = 5       # a shared token found in more chunks than this is vocabulary, not a link


def rebuild_links(con) -> dict:
    con.execute("DELETE FROM links WHERE rel IN ('citation', 'mentions')")
    # citation: resolve against files of the same source, then the heading anchor inside the file
    con.execute(
        """
        INSERT OR IGNORE INTO links (src, dst, rel, via)
        SELECT r.chunk_id,
               COALESCE(
                   (SELECT c2.id FROM chunks c2 WHERE c2.file_id = f2.id AND r.target_anchor != ''
                      AND c2.anchor = lower(r.target_anchor) ORDER BY c2.start_line LIMIT 1),
                   (SELECT c3.id FROM chunks c3 WHERE c3.file_id = f2.id ORDER BY c3.start_line LIMIT 1)
               ),
               'citation',
               r.target_path || CASE WHEN r.target_anchor != '' THEN '#' || r.target_anchor ELSE '' END
        FROM refs r
        JOIN chunks c ON c.id = r.chunk_id
        JOIN files f ON f.id = c.file_id
        JOIN files f2 ON f2.source_id = f.source_id AND f2.path = r.target_path
        """
    )
    con.execute("DELETE FROM links WHERE dst IS NULL OR dst = src")
    # a source that writes its links as absolute URLs (a Jira ticket linking a Confluence page or
    # another ticket, a page linking a ticket, a GitHub body linking either): resolved against the
    # URL the mirror recorded for the item it names
    _url_citations(con)
    # mentions of a defined identifier
    con.execute(
        f"""
        WITH defs AS (
            SELECT ident FROM idents WHERE role = 'defines' GROUP BY ident HAVING count(*) <= {MAX_DEFINERS}
        )
        INSERT OR IGNORE INTO links (src, dst, rel, via)
        SELECT m.chunk_id, d.chunk_id, 'mentions', m.ident
        FROM idents m
        JOIN defs USING (ident)
        JOIN idents d ON d.ident = m.ident AND d.role = 'defines'
        WHERE m.role = 'mentions' AND m.chunk_id != d.chunk_id
        """
    )
    # shared rare identifier-shaped tokens across files, both directions
    con.execute(
        f"""
        WITH shared AS (
            SELECT i.ident FROM idents i JOIN chunks c ON c.id = i.chunk_id
            WHERE i.role = 'mentions'
              AND (instr(i.ident, '_') OR instr(i.ident, '.') OR instr(i.ident, '-') OR instr(i.ident, '/'))
              AND i.ident NOT IN (SELECT ident FROM idents WHERE role = 'defines')
            GROUP BY i.ident
            HAVING count(DISTINCT i.chunk_id) BETWEEN 2 AND {MAX_SHARED} AND count(DISTINCT c.file_id) >= 2
        ),
        hits AS (
            SELECT i.chunk_id, i.ident, c.file_id FROM idents i JOIN shared USING (ident)
            JOIN chunks c ON c.id = i.chunk_id WHERE i.role = 'mentions'
        )
        INSERT OR IGNORE INTO links (src, dst, rel, via)
        SELECT a.chunk_id, b.chunk_id, 'mentions', a.ident
        FROM hits a JOIN hits b ON a.ident = b.ident AND a.file_id != b.file_id
        """
    )
    # prose that names the same defined identifier: the definer is linked from each already, and the pages
    # themselves (a runbook step, the incident it answers) are linked to each other only here
    con.execute(
        f"""
        WITH defs AS (
            SELECT ident FROM idents WHERE role = 'defines' GROUP BY ident HAVING count(*) <= {MAX_DEFINERS}
        ),
        prose AS (
            SELECT i.chunk_id, i.ident, c.file_id FROM idents i JOIN defs USING (ident)
            JOIN chunks c ON c.id = i.chunk_id JOIN files f ON f.id = c.file_id
            WHERE i.role = 'mentions' AND f.lang IN ('markdown', 'text')
        ),
        rare AS (
            SELECT ident FROM prose GROUP BY ident
            HAVING count(DISTINCT chunk_id) BETWEEN 2 AND {MAX_SHARED} AND count(DISTINCT file_id) >= 2
        )
        INSERT OR IGNORE INTO links (src, dst, rel, via)
        SELECT a.chunk_id, b.chunk_id, 'mentions', a.ident
        FROM prose a JOIN rare USING (ident) JOIN prose b ON b.ident = a.ident AND a.file_id != b.file_id
        """
    )
    counts = {r["rel"]: r["n"] for r in con.execute("SELECT rel, count(*) n FROM links GROUP BY rel")}
    return counts


def _url_citations(con) -> None:
    """A citation for every ref written as an absolute URL that names an item some source mirrors.

    The same item is reached however the URL is spelled: the connector's `locate` reads the ticket
    key, the page id or the item number out of `/browse/KEY`, `/pages/<id>/...`,
    `?pageId=<id>` and `?focusedCommentId=<id>` alike, so the query string and the host's case do
    not have to match the mirror's recorded URL. The target is the item's first chunk, or the
    heading a fragment or comment id names, when the mirror recorded that heading. `via` is the URL
    as the source wrote it. A URL whose host no source mirrors, or that no connector recognises,
    stays dropped.
    """
    import urllib.parse

    from .connectors import KINDS
    from .connectors.mirror import load

    hosts, items, anchors = {}, {}, {}   # host -> {kind: connector}, so only mirrored kinds are asked
    for r in con.execute("SELECT name, kind, origin, root FROM sources WHERE kind <> 'dir'"):
        m = KINDS.get(r["kind"])
        host = (urllib.parse.urlparse(r["origin"]).hostname or "").lower()
        if m is None or not host:
            continue
        hosts.setdefault(host, {})[r["kind"]] = m
        files = {f["path"]: f["id"] for f in con.execute(
            "SELECT id, path FROM files WHERE source_id = (SELECT id FROM sources WHERE name = ?)", (r["name"],))}
        for item in load(Path(r["root"]))["items"].values():
            fid = files.get(item["path"])
            hit = m.locate(item["url"]) if fid is not None and item.get("url") else None
            if not hit:
                continue
            items[(host, r["kind"], hit[0])] = fid
            for anchor, url in (item.get("anchors") or {}).items():
                a = m.locate(url)
                if a:
                    anchors[(host, r["kind"], a[0], a[1])] = anchor
    rows = con.execute("SELECT chunk_id, target_path FROM refs WHERE target_path LIKE 'http://%' "
                       "OR target_path LIKE 'https://%'").fetchall()
    seen, add = {}, []
    for r in rows:
        url = r["target_path"]
        if url not in seen:
            seen[url] = _url_target(con, url, hosts, items, anchors)
        dst = seen[url]
        if dst is not None and dst != r["chunk_id"]:
            add.append((r["chunk_id"], dst, url))
    con.executemany("INSERT OR IGNORE INTO links (src, dst, rel, via) VALUES (?, ?, 'citation', ?)", add)


def _url_target(con, url: str, hosts: dict, items: dict, anchors: dict) -> int | None:
    """The chunk an absolute URL names: the heading its fragment or comment id records, else the
    first chunk of the mirrored item; None when no mirror holds it."""
    import urllib.parse

    host = (urllib.parse.urlparse(url).hostname or "").lower()
    kinds = hosts.get(host)   # nothing this map mirrors lives on that host
    if not kinds:
        return None
    for kind, m in kinds.items():
        hit = m.locate(url)
        if hit:
            break
    else:
        return None
    item_id, fragment = hit
    fid = items.get((host, kind, item_id))
    if fid is None:
        return None
    anchor = anchors.get((host, kind, item_id, fragment),
                         fragment.replace("+", "-").replace(" ", "-").lower())
    if anchor:
        row = con.execute("SELECT id FROM chunks WHERE file_id = ? AND anchor = ? ORDER BY start_line LIMIT 1",
                          (fid, anchor)).fetchone()
        if row:
            return row["id"]
    row = con.execute("SELECT id FROM chunks WHERE file_id = ? ORDER BY start_line LIMIT 1", (fid,)).fetchone()
    return row["id"] if row else None
