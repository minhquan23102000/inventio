"""How well the ranker names what one passage is to another (systemone.RELATIONS), on the webshop example.

    python benchmarks/relations_probe.py

Ten pairs of the webshop map, each with the relation a reader gives it (the reader's labels are written
below; two answers count when a reader would accept either), asked as the zero-shot baseline for any model
that names links (`SystemOneRanker.relations`): every pair one choice question over one state holding all
nine chunks, under two questions. `inventio query` does not ask it; links print with code-known names.
Prints each pair's answer and how many agree with the reader. v4 (2026-09-28): 2/10 and 3/10 under the two
questions, mostly `nothing to follow`, and 2/10 with the user's question left out of the wording.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inventio.ingest import ingest_source  # noqa: E402
from inventio.links import rebuild_links  # noqa: E402
from inventio.rankers import SystemOneRanker  # noqa: E402
from inventio.search import HIT_SQL, Hit  # noqa: E402
from inventio.store import connect  # noqa: E402

WEBSHOP = Path(__file__).resolve().parent.parent / "examples" / "webshop"
READER = [   # (result, what it leads to, the relations a reader would accept)
    ("wiki/policy.md:3", "app/jobs/backup.py:1", {"carries it out", "defines a name in it"}),
    ("app/jobs/backup.py:1", "wiki/policy.md:3", {"rule it follows"}),
    ("wiki/runbook.md:3", "app/jobs/backup.py:4", {"carries it out"}),
    ("app/jobs/backup.py:4", "wiki/runbook.md:3", {"nothing to follow", "why it is so"}),
    ("wiki/runbook.md:8", "wiki/incidents/2026-03-14.md:10", {"why it is so"}),
    ("wiki/incidents/2026-03-14.md:10", "wiki/runbook.md:8", {"carries it out"}),
    ("app/jobs/backup.py:4", "wiki/incidents/2026-03-14.md:3", {"where it failed"}),
    ("wiki/runbook.md:3", "wiki/incidents/2026-03-14.md:3", {"where it failed"}),
    ("wiki/policy.md:10", "app/jobs/backup.py:12", {"nothing to follow"}),
    ("wiki/policy.md:3", "app/jobs/backup.py:12", {"carries it out"}),
]
QUESTIONS = ("why could we not restore the orders on March 14?", "how many days are backups kept?")


def main() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        con = connect(Path(tmp) / "map.db")
        with con:
            ingest_source(con, "webshop", WEBSHOP, True, [])
            rebuild_links(con)
        hits = [Hit(**{**dict(r), "public": bool(r["public"])})
                for r in con.execute(HIT_SQL + " ORDER BY f.path, c.start_line")]
        name = [f"{h.path}:{h.start_line}" for h in hits]
        pairs = [(name.index(a), name.index(b)) for a, b, _ in READER]
        ranker = SystemOneRanker()
        for q in QUESTIONS:
            print(f"== {q}")
            ok = 0
            for (a, b, want), p in zip(READER, ranker.relations(q, hits, pairs)):
                best = max(p, key=p.get)
                ok += best in want
                print(f"  {a} -> {b}: {best} {p[best]:.2f}   (reader: {' / '.join(sorted(want))})")
            print(f"  agrees with the reader: {ok}/{len(READER)}")
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
