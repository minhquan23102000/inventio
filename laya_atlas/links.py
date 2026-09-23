"""Links between chunks, all drawn by code from what the sources already say.

Relation names come from schema.org's CreativeWork vocabulary:
- `citation`: a Markdown link points at a file or heading.
- `mentions`: a chunk names an identifier another chunk defines (a function, a class), or two
  chunks in different files share a rare identifier-shaped token (`fraud_score_daily`, `FRAML-123`).
  This is the bridge between a repository and the prose written about it.
Parent sections (`isPartOf`) are kept on the chunk row itself.
"""

MAX_DEFINERS = 3     # an identifier defined in more places than this is too ambiguous to link
MAX_SHARED = 5       # a shared token found in more chunks than this is vocabulary, not a link


def rebuild_links(con) -> dict:
    con.execute("DELETE FROM links")
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
    counts = {r["rel"]: r["n"] for r in con.execute("SELECT rel, count(*) n FROM links GROUP BY rel")}
    return counts
