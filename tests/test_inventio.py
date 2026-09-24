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


@pytest.fixture(autouse=True)
def bm25_order(monkeypatch):
    """The CLI ranks with dispositio when the laya extra is installed; these tests pin BM25 order
    so they run offline and do not depend on a model."""
    monkeypatch.setenv("INVENTIO_RANKER", "none")


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
    write(repo, "jobs/backup.py", """
        def nightly_backup(db):
            return db.snapshot()
    """)
    write(wiki, "runbook.md", """
        # Nightly backup
        The nightly job `nightly_backup` copies the orders database before the morning peak.
    """)
    assert run(capsys, "--db", db, "init", str(repo), "--name", "repo")[0] == 0
    assert run(capsys, "--db", db, "init", str(wiki), "--name", "wiki")[0] == 0
    code, out = run(capsys, "--db", db, "query", "nightly job orders database", "--json", "--source", "wiki")
    assert code == 0
    top = json.loads(out.out)[0]
    assert {"rel": "mentions", "dir": "out", "via": "nightly_backup", "source": "repo", "coord": "jobs/backup.py:1-2"} in top["links"]


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
    keep = write(d, "keep.md", "# Keep\nThe nightly job `nightly_backup` copies the orders database.\n")
    write(d, "edit.md", "# Edit\nthresholds are tuned per wombat segment\n")
    gone = write(d, "jobs/backup.py", "def nightly_backup(db):\n    return 0\n")
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
    hits = json.loads(run(capsys, "--db", db, "query", "nightly orders database", "--json")[1].out)
    assert hits[0]["links"] == []  # the definer was deleted, so its link went with it


def test_cloud_ranker_refuses_private_sources(tmp_path, capsys, monkeypatch):
    pytest.importorskip("typesafe_sdk")
    monkeypatch.setenv("TYPESAFE_API_KEY", "not-a-real-key")
    db = str(tmp_path / "map.db")
    write(tmp_path / "private", "notes.md", "# Notes\ncustomer 4411 emailed their card number\n")
    run(capsys, "--db", db, "init", str(tmp_path / "private"), "--name", "crm")
    code, out = run(capsys, "--db", db, "query", "card number", "--ranker", "typesafe")
    assert code == 3
    assert "crm" in out.err


def test_tree_sitter_definitions_carry_coordinates_and_links(tmp_path, capsys):
    pytest.importorskip("tree_sitter_language_pack")
    db = str(tmp_path / "map.db")
    repo, wiki = tmp_path / "repo", tmp_path / "wiki"
    write(repo, "models/daily_orders.sql", """
        -- one row per customer per day
        create table shop.daily_orders as
        select customer_id, sum(total) amt from orders group by 1;

        create view shop.v_big_orders as select * from shop.daily_orders where amt > 1000;
    """)
    ts = write(repo, "src/limits.ts", """
        import { Req } from "./req";

        export class RateLimit {
          limit = 5;

          /** requests in the last second above the limit */
          breached(reqs: Req[]): boolean {
            return reqs.length > this.limit;
          }
        }

        export const shippingCost = (r: Req) => r.weight * 0.3;
    """)
    write(wiki, "reports.md", "# Reports\nReports read `shop.daily_orders` every morning.\n")
    run(capsys, "--db", db, "init", str(repo), "--name", "repo")
    run(capsys, "--db", db, "init", str(wiki), "--name", "wiki")

    lines = ts.read_text(encoding="utf-8").split("\n")
    by_name = {c.heading_path: c for c in chunk_file("\n".join(lines), "typescript", "src/limits.ts")}
    assert (by_name["RateLimit"].start_line, by_name["RateLimit"].end_line) == (3, 10)
    assert (by_name["shippingCost"].kind, by_name["shippingCost"].start_line) == ("function", 12)

    top = json.loads(run(capsys, "--db", db, "query", "reports every morning", "--json", "--source", "wiki")[1].out)[0]
    assert ("mentions", "repo", "models/daily_orders.sql:1-3") in {(l["rel"], l["source"], l["coord"]) for l in top["links"]}


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
        write(repo, f"docs/limits{i}.md", f"# Limits {i}\nthe rate limit is five per second, note {i}\n")
    write(repo, "tests/test_cap.py", """
        def test_rate_cap():
            reqs = make_requests(count=6, window="1s")
            assert throttled(reqs)
            assert not throttled(reqs[:5])
    """)
    run(capsys, "--db", str(db), "init", str(repo), "--name", "repo", "--public")
    con = connect(db)

    plain = search(con, "rate limit", k=100, pool=2)
    wide = search(con, "rate limit", k=100, pool=2, ranker=FakeRanker(), by_type=True)
    assert [h.id for h in wide[:len(plain)]] == [h.id for h in plain]
    added = wide[len(plain):]
    assert [(h.path, h.type, h.via) for h in added] == [("tests/test_cap.py", "Test", "type:Test")]

    run(capsys, "--db", str(db), "init", str(tmp_path / "repo" / "docs"), "--name", "notes")  # private
    with pytest.raises(CloudRefused, match="notes"):
        search(con, "rate limit", pool=2, ranker=FakeRanker(cloud=True), by_type=True)


def test_query_names_bring_their_file_and_definition(tmp_path, capsys):
    from inventio.search import search
    from inventio.store import connect

    db = tmp_path / "map.db"
    repo = tmp_path / "repo"
    for i in range(6):  # prose that outranks the code on every word of the question
        write(repo, f"docs/jobs-report-nightly-{i}.md",
              f"# Nightly jobs report {i}\nnightly job crash: rebuild index fails; see the jobs report, then run it again\n")
    write(repo, "jobs/nightly.py", "def rebuild_index(rows):\n    return sum(r.size for r in rows)\n")
    write(repo, "jobs/report.py", "def render(rows):\n    return str(rows)\n")
    for i in range(4):  # a name defined in many places names nothing in particular
        write(repo, f"lib/m{i}.py", "def run():\n    return 1\n")
    run(capsys, "--db", str(db), "init", str(repo), "--name", "repo")
    con = connect(db)

    q = "the nightly job crash: rebuild_index fails, see jobs/report.py, then run"
    plain = search(con, q, k=100, pool=2, symbols=False)
    assert {h.path for h in plain} <= {f"docs/jobs-report-nightly-{i}.md" for i in range(6)}
    named = [("jobs/report.py", "path"), ("jobs/nightly.py", "defines:rebuild_index")]
    wide = search(con, q, k=100, pool=2)  # no ranker: the named ones go first, BM25's order after
    assert [(h.path, h.via) for h in wide] == named + [(h.path, h.via) for h in plain]
    ranked = search(con, q, k=100, pool=2, ranker=FakeRanker())  # a ranker sees all of them
    assert {(h.path, h.via) for h in ranked} == set(named) | {(h.path, h.via) for h in plain}


def test_vietnamese_query_matches_words_not_scattered_syllables(tmp_path, capsys):
    from inventio.search import bm25
    from inventio.store import connect

    db = tmp_path / "map.db"
    write(tmp_path / "d", "a.md", "# A\nHợp đồng lao động phải lập thành văn bản.\n")
    # the same four syllables, never adjacent (FTS5 phrases ignore punctuation, so a comma is not a gap)
    write(tmp_path / "d", "b.md", "# B\nHợp lý, đồng ý, lao xao, động viên; hợp tác, đồng hồ, lao công, động cơ.\n")
    for i in range(8):  # BM25 needs a corpus in which the words are rare
        write(tmp_path / "d", f"f{i}.md", f"# F{i}\nNgân hàng mở cửa lúc {i} giờ sáng.\n")
    run(capsys, "--db", str(db), "init", str(tmp_path / "d"), "--name", "d")
    con = connect(db)

    q = "hợp đồng lao động là gì"
    assert bm25(con, q, 2, phrases=False)[0].path == "b.md"  # syllables alone favour the noisy text
    assert bm25(con, q, 2)[0].path == "a.md"


class FakeJudge:
    """Counts calls. A chunk naming a cap or a limit states a Rule, one saying where things go a
    Procedure, one saying who a card is for is Reference, anything else Other; every query asks
    for a Rule. Two passages are about the same thing when both name the same card."""

    packs = True

    def __init__(self, cloud=False):
        self.cloud, self.name, self.calls, self.refused, self.failed = cloud, "fake", 0, 0, 0

    @staticmethod
    def category(t: str) -> dict:
        from inventio.facts import CATEGORIES

        c = ("Rule" if "cap" in t or "limit" in t else "Procedure" if " go to " in t
             else "Reference" if "issued to" in t else "Other")
        return {x: (0.7 if x == c else 0.05) for x in CATEGORIES}

    def batch(self, jobs):
        for i, (state, qs) in enumerate(jobs):
            self.calls += 1
            if "n0" not in state:
                yield i, {"category": self.category("cap" if "query" in state else state["passage"])}
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
    write(repo, "f.md", "# Holders\nevery gold card holder has the same spend cap\n")
    write(repo, "code.py", "def gold_card_cap():\n    return 5000\n")
    run(capsys, "--db", str(db), "init", str(repo), "--name", "repo", "--public")
    con = connect(db)
    judge = FakeJudge()
    build(con, judge)

    # every prose chunk has a p per category and keeps the likeliest one, none when that is Other;
    # code is not categorized
    per_chunk = con.execute(
        "SELECT f.path, count(*) n, group_concat(CASE WHEN kept THEN category END) kept FROM chunk_categories cc "
        "JOIN chunks c ON c.id = cc.chunk_id JOIN files f ON f.id = c.file_id GROUP BY c.id").fetchall()
    assert {r["path"]: (r["n"], r["kept"]) for r in per_chunk} == {
        "a.md": (7, "Rule"), "b.md": (7, "Procedure"), "c.md": (7, "Reference"), "d.md": (7, None),
        "e.md": (7, "Rule"), "f.md": (7, "Rule")}
    # the gold card rule is linked to the gold card procedure; the other gold card rule is not,
    # being of the same category, nor is the blue card
    about = con.execute("SELECT fa.path a, fb.path b, l.via FROM links l JOIN chunks ca ON ca.id = l.src "
                        "JOIN files fa ON fa.id = ca.file_id JOIN chunks cb ON cb.id = l.dst "
                        "JOIN files fb ON fb.id = cb.file_id WHERE l.rel = 'about'").fetchall()
    assert sorted(tuple(sorted((r["a"], r["b"]))) + (r["via"],) for r in about) == [
        ("a.md", "b.md", "Procedure~Rule"), ("b.md", "f.md", "Procedure~Rule")]
    # every judgment is kept with the text it read
    j = con.execute("SELECT kind, passage, other, model FROM judgments WHERE kind = 'same_thing' AND p >= 0.5").fetchone()
    assert j["model"] == "fake" and "gold card" in j["passage"] and "gold card" in j["other"]

    # a rerun pays nothing: the judgments answer again, and code-drawn links do not remove judged ones
    calls = judge.calls
    con.execute("DELETE FROM chunk_categories")
    build(con, judge, relink=True)
    rebuild_links(con)
    assert judge.calls == calls
    assert con.execute("SELECT count(*) FROM links WHERE rel = 'about'").fetchone()[0] == 2

    # categories of an earlier scheme are replaced, with the links they gated
    con.execute("UPDATE chunk_categories SET category = 'Product' WHERE category = 'Rule'")
    build(con, judge)
    assert con.execute("SELECT count(*) FROM chunk_categories WHERE category = 'Product'").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM links WHERE rel = 'about'").fetchone()[0] == 2

    # query time only adds: the plain pool is a prefix, the rest came through a category or a link
    plain = search(con, "spend cap", k=100, pool=1)
    wide = search(con, "spend cap", k=100, pool=1, facts=judge, ranker=None)
    assert [h.id for h in wide[:len(plain)]] == [h.id for h in plain]
    added = {h.path: h.via for h in wide[len(plain):]}
    assert "b.md" in added and added["b.md"].startswith("about:")
    assert all(v.startswith("category:Rule") for p, v in added.items() if p != "b.md")

    # editing a file drops its judged links with its chunks
    write(repo, "b.md", "# Disputes\nchargebacks go to the disputes desk\n")
    run(capsys, "--db", str(db), "init", str(repo), "--name", "repo", "--public")
    assert con.execute("SELECT count(*) FROM links WHERE rel = 'about'").fetchone()[0] == 0


def test_cloud_judge_refuses_private_sources(tmp_path, capsys):
    from inventio.facts import build
    from inventio.rankers import CloudRefused
    from inventio.store import connect

    db = tmp_path / "map.db"
    write(tmp_path / "private", "notes.md", "# Notes\ncustomer 4411 emailed their card number\n")
    run(capsys, "--db", str(db), "init", str(tmp_path / "private"), "--name", "crm")
    judge = FakeJudge(cloud=True)
    with pytest.raises(CloudRefused, match="crm"):
        build(connect(db), judge)
    assert judge.calls == 0


def test_grep_and_ls_print_coordinates_that_read_opens(tmp_path, capsys):
    db = str(tmp_path / "map.db")
    write(tmp_path / "d", "ops/runbook.md", """
        # Runbook

        ## Late backup
        Rerun nightly_backup from the replica.
    """)
    write(tmp_path / "d", "jobs/backup.py", """
        import os


        def nightly_backup(db):
            return db
    """)
    assert run(capsys, "--db", db, "init", str(tmp_path / "d"), "--name", "d")[0] == 0

    code, out = run(capsys, "--db", db, "grep", "NIGHTLY_BACKUP", "-i", "--json")
    assert code == 0
    got = json.loads(out.out)
    assert [(m["path"], m["line"]) for m in got["matches"]] == [("jobs/backup.py", 4), ("ops/runbook.md", 4)]
    code, out = run(capsys, "--db", db, "read", "d:jobs/backup.py:4")
    assert out.out.splitlines()[1].endswith("def nightly_backup(db):")
    assert run(capsys, "--db", db, "grep", "absent_name")[0] == 1

    code, out = run(capsys, "--db", db, "ls", "d")
    assert out.out.splitlines() == ["d:jobs/  1 file, 2 chunks", "d:ops/  1 file, 1 chunk"]
    code, out = run(capsys, "--db", db, "ls", "d:ops/runbook.md")
    assert out.out.splitlines() == ["d:ops/runbook.md:3-4  section  Runbook > Late backup"]


def test_show_names_who_decided_each_fact_and_every_coordinate_opens(tmp_path, capsys):
    db = str(tmp_path / "map.db")
    write(tmp_path / "d", "ops/runbook.md", """
        # Runbook

        ## Late backup
        Rerun nightly_backup from the replica.

        ## Backup retention
        Every nightly backup is kept for 35 days.
    """)
    write(tmp_path / "d", "jobs/backup.py", """
        def nightly_backup(db):
            return db
    """)
    assert run(capsys, "--db", db, "init", str(tmp_path / "d"), "--name", "d")[0] == 0
    from inventio.facts import SAME, key, passage
    from inventio.store import connect

    con = connect(db)
    ids = {r["start_line"]: r["id"] for r in con.execute(
        "SELECT c.id, c.start_line FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.path = 'ops/runbook.md'")}
    late, retention = ids[3], ids[6]
    con.execute("INSERT INTO chunk_categories VALUES (?, 'Procedure', 0.91, 1, 'dispositio')", (late,))
    con.execute("INSERT INTO chunk_categories VALUES (?, 'Rule', 0.88, 1, 'dispositio')", (retention,))
    con.execute("INSERT INTO links VALUES (?, ?, 'about', 'Procedure~Rule')", (late, retention))
    text = {i: passage(*con.execute("SELECT f.path, c.heading_path, c.text FROM chunks c JOIN files f "
                                    "ON f.id = c.file_id WHERE c.id = ?", (i,)).fetchone()) for i in (late, retention)}
    con.execute("INSERT INTO judgments (kind, question, passage, other, key, p, model, source) "
                "VALUES (?, ?, ?, ?, ?, 0.84, 'dispositio', 'd')",
                (SAME, SAME, text[late], text[retention], key(SAME, SAME, text[late], text[retention])))
    con.commit()

    code, out = run(capsys, "--db", db, "show", "d:ops/runbook.md:3-4")
    lines = out.out.splitlines()
    assert code == 0 and lines[0] == "d:ops/runbook.md:3-4  Runbook > Late backup"
    assert lines[1] == "  Article (by path) · markdown · private"
    assert lines[2] == "  section 3-4 · Procedure p=0.91 (dispositio) · mentions 1 name"
    assert "  -> mentions d:jobs/backup.py:1-2  nightly_backup  · SoftwareSourceCode  (nightly_backup)" in lines
    assert "  ~  about    d:ops/runbook.md:6-7  Runbook > Backup retention  · Article · Rule p=0.88 (dispositio)  p=0.84 (dispositio)" in lines
    assert "  > d:ops/runbook.md:6-7  Runbook > Backup retention  · Article · Rule p=0.88 (dispositio)" in lines

    node = json.loads(run(capsys, "--db", db, "show", "d:ops/runbook.md:3", "--json")[1].out)
    for other in [*node["links"], *node["structure"], *node["similar"]]:
        assert run(capsys, "--db", db, "show", other["coord"])[0] == 0
    assert run(capsys, "--db", db, "show", "d:ops/absent.md")[0] == 2
