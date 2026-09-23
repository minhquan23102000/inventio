import json
import textwrap

import pytest

from inventio import cli
from inventio.ingest import chunk_file


def write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return p


def run(capsys, *argv):
    code = cli.main(list(argv))
    return code, capsys.readouterr()


def test_coordinates_point_at_the_exact_lines(tmp_path):
    text = textwrap.dedent("""
        # Title
        intro line

        ## Setup
        ```sh
        # not a heading, a shell comment
        pip install x
        ```
        after the fence

        ## Usage
        run it
    """).lstrip("\n")
    lines = text.split("\n")
    chunks = chunk_file(text, "markdown", "a.md")
    assert [c.heading_path for c in chunks] == ["Title", "Title > Setup", "Title > Usage"]
    for c in chunks:
        assert c.text == "\n".join(lines[c.start_line - 1:c.end_line])


def test_python_method_coordinates(tmp_path):
    body = "\n".join(f"        x{i} = {i}" for i in range(120))
    text = f"import os\n\nclass Big:\n    def a(self):\n{body}\n\n    def b(self):\n        return 1\n"
    lines = text.split("\n")
    chunks = {c.heading_path: c for c in chunk_file(text, "python", "m.py")}
    assert "Big.b" in chunks
    b = chunks["Big.b"]
    assert lines[b.start_line - 1].strip() == "def b(self):"
    assert b.text.splitlines()[-1].strip() == "return 1"


def test_prose_links_to_the_code_that_defines_what_it_names(tmp_path, capsys):
    db = str(tmp_path / "map.db")
    repo, wiki = tmp_path / "repo", tmp_path / "wiki"
    write(repo, "jobs/score.py", """
        def fraud_score_daily(txns):
            return sum(t.amount for t in txns)
    """)
    write(wiki, "runbook.md", """
        # Daily scoring
        The nightly job `fraud_score_daily` recomputes customer risk before the morning review.
    """)
    assert run(capsys, "--db", db, "init", str(repo), "--name", "repo")[0] == 0
    assert run(capsys, "--db", db, "init", str(wiki), "--name", "wiki")[0] == 0
    code, out = run(capsys, "--db", db, "query", "nightly job customer risk", "--json", "--source", "wiki")
    assert code == 0
    top = json.loads(out.out)[0]
    assert {"rel": "mentions", "dir": "out", "via": "fraud_score_daily", "source": "repo", "coord": "jobs/score.py:1-2"} in top["links"]


def test_markdown_link_resolves_to_the_heading(tmp_path, capsys):
    db = str(tmp_path / "map.db")
    write(tmp_path / "d", "a.md", """
        # A
        See [the limits](sub/b.md#known-limits) before tuning thresholds.
    """)
    write(tmp_path / "d", "sub/b.md", """
        # B
        overview

        ## Known limits
        recall drops below 0.4 on new merchants
    """)
    run(capsys, "--db", db, "init", str(tmp_path / "d"), "--name", "d")
    code, out = run(capsys, "--db", db, "query", "tuning thresholds", "--json", "-k", "1")
    links = json.loads(out.out)[0]["links"]
    assert {"rel": "citation", "dir": "out", "via": "sub/b.md#known-limits", "source": "d", "coord": "sub/b.md:4-5"} in links


def test_reindex_touches_only_what_changed(tmp_path, capsys):
    import sqlite3

    db = str(tmp_path / "map.db")
    d = tmp_path / "d"
    keep = write(d, "keep.md", "# Keep\nThe nightly job `fraud_score_daily` recomputes customer risk.\n")
    write(d, "edit.md", "# Edit\nthresholds are tuned per wombat segment\n")
    gone = write(d, "jobs/score.py", "def fraud_score_daily(txns):\n    return 0\n")
    run(capsys, "--db", db, "init", str(d), "--name", "d")

    def keep_ids():
        con = sqlite3.connect(db)
        return con.execute(
            "SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.path = 'keep.md'"
        ).fetchall()

    before = keep_ids()
    keep.write_bytes(keep.read_bytes())  # same bytes, new mtime: must not be re-chunked
    write(d, "edit.md", "# Edit\nthresholds are now tuned per quokka merchant segment\n")
    gone.unlink()
    code, out = run(capsys, "--db", db, "init", str(d), "--name", "d")
    assert code == 0
    assert "+0 new, ~1 edited, -1 removed, 1 unchanged" in out.out
    assert keep_ids() == before

    hits = json.loads(run(capsys, "--db", db, "query", "quokka", "--json")[1].out)
    assert [h["path"] for h in hits] == ["edit.md"]
    assert run(capsys, "--db", db, "query", "wombat")[0] == 1
    hits = json.loads(run(capsys, "--db", db, "query", "nightly customer risk", "--json")[1].out)
    assert hits[0]["links"] == []  # the definer was deleted, so its link went with it


def test_cloud_ranker_refuses_private_sources(tmp_path, capsys, monkeypatch):
    pytest.importorskip("typesafe_sdk")
    monkeypatch.setenv("TYPESAFE_API_KEY", "not-a-real-key")
    db = str(tmp_path / "map.db")
    write(tmp_path / "private", "notes.md", "# Notes\ncustomer 4411 flagged for mule activity\n")
    run(capsys, "--db", db, "init", str(tmp_path / "private"), "--name", "bank")
    code, out = run(capsys, "--db", db, "query", "mule activity", "--ranker", "typesafe")
    assert code == 3
    assert "bank" in out.err


def test_tree_sitter_definitions_carry_coordinates_and_links(tmp_path, capsys):
    pytest.importorskip("tree_sitter_language_pack")
    db = str(tmp_path / "map.db")
    repo, wiki = tmp_path / "repo", tmp_path / "wiki"
    write(repo, "models/score_daily.sql", """
        -- one row per customer per day
        create table risk.daily_score as
        select customer_id, sum(amount) amt from txn group by 1;

        create view risk.v_alerts as select * from risk.daily_score where amt > 1000;
    """)
    ts = write(repo, "src/rules.ts", """
        import { Txn } from "./txn";

        export class Velocity {
          limit = 5;

          /** txns in the last hour above the limit */
          breached(txns: Txn[]): boolean {
            return txns.length > this.limit;
          }
        }

        export const muleScore = (t: Txn) => t.amount * 0.3;
    """)
    write(wiki, "alerts.md", "# Alerts\nAlerts read `risk.daily_score` every morning.\n")
    run(capsys, "--db", db, "init", str(repo), "--name", "repo")
    run(capsys, "--db", db, "init", str(wiki), "--name", "wiki")

    lines = ts.read_text(encoding="utf-8").split("\n")
    by_name = {c.heading_path: c for c in chunk_file("\n".join(lines), "typescript", "src/rules.ts")}
    assert (by_name["Velocity"].start_line, by_name["Velocity"].end_line) == (3, 10)
    assert (by_name["muleScore"].kind, by_name["muleScore"].start_line) == ("function", 12)

    top = json.loads(run(capsys, "--db", db, "query", "alerts every morning", "--json", "--source", "wiki")[1].out)[0]
    assert ("mentions", "repo", "models/score_daily.sql:1-3") in {(l["rel"], l["source"], l["coord"]) for l in top["links"]}


class FakeRanker:
    """Says the answer is in a test; scores nothing, so the order stays BM25 then widened."""

    def __init__(self, cloud=False):
        self.cloud = cloud

    def types(self, query, types):
        return {t: (0.9 if t == "Test" else 0.1 / len(types)) for t in types}

    def score(self, query, hits):
        return [None] * len(hits)


def test_type_widening_only_adds_and_respects_privacy(tmp_path, capsys):
    from inventio.search import search
    from inventio.rankers import CloudRefused
    from inventio.store import connect

    db = tmp_path / "map.db"
    repo = tmp_path / "repo"
    for i in range(3):
        write(repo, f"docs/limits{i}.md", f"# Limits {i}\nthe velocity limit is five per hour, note {i}\n")
    write(repo, "tests/test_cap.py", """
        def test_velocity_cap():
            txns = make_txns(count=6, window="1h")
            assert flagged(txns)
            assert not flagged(txns[:5])
    """)
    run(capsys, "--db", str(db), "init", str(repo), "--name", "repo", "--public")
    con = connect(db)

    plain = search(con, "velocity limit", k=100, pool=2)
    wide = search(con, "velocity limit", k=100, pool=2, ranker=FakeRanker(), by_type=True)
    assert [h.id for h in wide[:len(plain)]] == [h.id for h in plain]
    added = wide[len(plain):]
    assert [(h.path, h.type, h.via) for h in added] == [("tests/test_cap.py", "Test", "type:Test")]

    run(capsys, "--db", str(db), "init", str(tmp_path / "repo" / "docs"), "--name", "notes")  # private
    with pytest.raises(CloudRefused, match="notes"):
        search(con, "velocity limit", pool=2, ranker=FakeRanker(cloud=True), by_type=True)


class FakeJudge:
    """Counts calls. Categories: a chunk naming a card is about a Product, one naming a merchant
    about an Organization; every query is about a Product. Two passages are about the same thing
    when both name the same card."""

    packs = True

    def __init__(self, cloud=False):
        self.cloud, self.name, self.calls, self.refused, self.failed = cloud, "fake", 0, 0, 0

    def batch(self, jobs):
        for i, (state, qs) in enumerate(jobs):
            self.calls += 1
            if "query" in state:
                yield i, {c: (0.9 if c == "Product" else 0.1) for c in qs}
            elif "n0" not in state:
                t = state["passage"]
                yield i, {c: (0.9 if (c == "Product" and "card" in t) or (c == "Organization" and "merchant" in t)
                              else 0.1) for c in qs}
            else:
                card = lambda t: next((w for w in ("gold card", "blue card") if w in t), None)
                yield i, {s: (0.9 if card(state[s]) and card(state[s]) == card(state["passage"]) else 0.1) for s in qs}


def test_facts_categories_links_and_query_widening(tmp_path, capsys):
    from inventio.facts import build
    from inventio.links import rebuild_links
    from inventio.search import search
    from inventio.store import connect

    db = tmp_path / "map.db"
    repo = tmp_path / "repo"
    write(repo, "a.md", "# Gold card\nthe gold card has a monthly spend cap of 5000\n")
    write(repo, "b.md", "# Disputes\nchargebacks on the gold card go to the disputes desk within 30 days\n")
    write(repo, "c.md", "# Blue card\nthe blue card is issued to students\n")
    write(repo, "d.md", "# Merchants\na merchant is onboarded after a site visit\n")
    write(repo, "e.md", "# Cap\nspend cap limits reset at month end for every product\n")
    write(repo, "code.py", "def gold_card_cap():\n    return 5000\n")
    run(capsys, "--db", str(db), "init", str(repo), "--name", "repo", "--public")
    con = connect(db)
    judge = FakeJudge()
    res = build(con, judge)

    # every prose chunk has all eight p; code is not categorized; kept = p >= 0.5
    per_chunk = con.execute(
        "SELECT f.path, count(*) n, group_concat(CASE WHEN kept THEN category END) kept FROM chunk_categories cc "
        "JOIN chunks c ON c.id = cc.chunk_id JOIN files f ON f.id = c.file_id GROUP BY c.id").fetchall()
    assert {r["path"]: (r["n"], r["kept"]) for r in per_chunk} == {
        "a.md": (8, "Product"), "b.md": (8, "Product"), "c.md": (8, "Product"), "d.md": (8, "Organization"), "e.md": (8, None)}
    # the gold card chunks are linked; the blue card shares their type but not their card
    about = con.execute("SELECT fa.path a, fb.path b, l.via FROM links l JOIN chunks ca ON ca.id = l.src "
                        "JOIN files fa ON fa.id = ca.file_id JOIN chunks cb ON cb.id = l.dst "
                        "JOIN files fb ON fb.id = cb.file_id WHERE l.rel = 'about'").fetchall()
    assert [tuple(sorted((r["a"], r["b"]))) + (r["via"],) for r in about] == [("a.md", "b.md", "Product")]
    # every judgment is kept with the text it read
    j = con.execute("SELECT kind, passage, other, model FROM judgments WHERE kind = 'same_thing' AND p >= 0.5").fetchone()
    assert j["model"] == "fake" and "gold card" in j["passage"] and "gold card" in j["other"]

    # a rerun pays nothing: the judgments answer again, and code-drawn links do not remove judged ones
    calls = judge.calls
    con.execute("DELETE FROM chunk_categories")
    build(con, judge, relink=True)
    rebuild_links(con)
    assert judge.calls == calls
    assert con.execute("SELECT count(*) FROM links WHERE rel = 'about'").fetchone()[0] == 1

    # query time only adds: the plain pool is a prefix, the rest came through a category or a link
    plain = search(con, "spend cap", k=100, pool=1)
    wide = search(con, "spend cap", k=100, pool=1, facts=judge, ranker=None)
    assert [h.id for h in wide[:len(plain)]] == [h.id for h in plain]
    assert plain[0].path == "a.md" or plain[0].path == "e.md"
    added = {h.path: h.via for h in wide[len(plain):]}
    assert set(added) <= {"a.md", "b.md", "c.md"} and "b.md" in added
    assert all(v.startswith(("category:Product", "about:Product")) for v in added.values())

    # editing a file drops its judged links with its chunks
    write(repo, "b.md", "# Disputes\nchargebacks go to the disputes desk\n")
    run(capsys, "--db", str(db), "init", str(repo), "--name", "repo", "--public")
    assert con.execute("SELECT count(*) FROM links WHERE rel = 'about'").fetchone()[0] == 0


def test_cloud_judge_refuses_private_sources(tmp_path, capsys):
    from inventio.facts import build
    from inventio.rankers import CloudRefused
    from inventio.store import connect

    db = tmp_path / "map.db"
    write(tmp_path / "private", "notes.md", "# Notes\ncustomer 4411 flagged for mule activity\n")
    run(capsys, "--db", str(db), "init", str(tmp_path / "private"), "--name", "bank")
    judge = FakeJudge(cloud=True)
    with pytest.raises(CloudRefused, match="bank"):
        build(connect(db), judge)
    assert judge.calls == 0
