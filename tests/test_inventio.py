import json
import textwrap
from pathlib import Path

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
    """The CLI ranks with dispositio when the dispositio extra is installed; these tests pin BM25 order
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
    assert {"rel": "mentions", "dir": "out", "via": "nightly_backup", "source": "repo", "coord": "jobs/backup.py:1-2",
            "kind": "code", "fact": "defines nightly_backup", "title": "nightly_backup"} in top["links"]
    code, out = run(capsys, "--db", db, "query", "nightly_backup snapshot", "--json", "--source", "repo")
    back = json.loads(out.out)[0]["links"]   # the other way round: the page is what uses the name
    assert [(l["kind"], l["fact"], l["coord"]) for l in back] == [("page", "uses nightly_backup", "runbook.md:1-2")]


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
    assert {"rel": "citation", "dir": "out", "via": "sub/b.md#known-limits", "source": "d", "coord": "sub/b.md:4-5",
            "kind": "page", "fact": "links to", "title": "B > Known limits"} in links


def test_links_are_named_by_what_the_map_knows(tmp_path, capsys):
    from inventio.search import search
    from inventio.store import connect

    db = tmp_path / "map.db"
    write(tmp_path / "jira", "SHOP-1.md", """
        # SHOP-1 Orders lost after the migration
        <!-- defines: SHOP-1 -->

        Incident · Done

        Links:
        - is caused by SHOP-2 Bucket full
    """)
    write(tmp_path / "jira", "SHOP-2.md", """
        # SHOP-2 Bucket full
        <!-- defines: SHOP-2 -->

        Bug · Done
    """)
    write(tmp_path / "wiki", "restore.md", """
        # Restore
        - check SHOP-2 is closed before the restore
    """)
    write(tmp_path / "wiki", "notes.md", """
        # Notes
        - look at SHOP-2 when the bucket fills
    """)
    for d in ("jira", "wiki"):
        run(capsys, "--db", str(db), "init", str(tmp_path / d), "--name", d)
    con = connect(db)
    ids = {r["path"]: r["id"] for r in con.execute("SELECT f.path, c.id FROM chunks c JOIN files f ON f.id = c.file_id")}
    tickets = {r["path"]: r["id"] for r in con.execute("SELECT path, id FROM files WHERE path LIKE 'SHOP-%'")}
    con.executemany("INSERT INTO file_meta (file_id, key, value) VALUES (?, 'issuetype', ?)",
                    [(tickets["SHOP-1.md"], "Incident"), (tickets["SHOP-2.md"], "Bug")])
    # a confident Procedure prints as steps; the same category below its threshold prints as page
    con.executemany("INSERT INTO chunk_categories (chunk_id, category, p, kept, model) VALUES (?, 'Procedure', ?, 1, 'x')",
                    [(ids["restore.md"], 0.9), (ids["notes.md"], 0.6)])
    con.commit()

    def named(q):
        return [(l["kind"], l["fact"], l["coord"].split(":")[0]) for l in search(con, q, k=1)[0].links]

    # between two tickets the link verb is the ticket's own line, both ways round
    assert ("bug", "is caused by", "SHOP-2.md") in named("orders lost after the migration")
    assert ("incident", "is caused by this", "SHOP-1.md") in named("bucket full bug")
    # a runbook bullet naming the key is a step, not a link verb
    got = named("bucket full bug")
    assert ("steps", "uses shop-2", "restore.md") in got and ("page", "uses shop-2", "notes.md") in got
    # what the step names, stated by one side, before pages that only name the same ticket too
    assert named("check closed before the restore") == [("bug", "defines shop-2", "SHOP-2.md"),
                                                        ("incident", "names shop-2 too", "SHOP-1.md"),
                                                        ("page", "names shop-2 too", "notes.md")]


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


class PassesRanker:
    """Reads like dispositio: passes of 15, BM25's pool first (`first`), what widening added after. Favours
    any passage that defines a name, so a named definition can only win if it is read."""

    passes = True

    def __init__(self):
        self.read = []

    def score(self, q, hits, first=None):
        from inventio.rankers import SystemOneRanker

        r = SystemOneRanker.__new__(SystemOneRanker)
        r.url, r.host = "", ""

        def one(query, hs):
            self.read.append([h.path for h in hs])
            r.last = {"dropped_passages": 0}
            return [0.9 if "def " in h.text else 0.1 for h in hs]

        r._pass = one
        return SystemOneRanker.score(r, q, hits, first)


def test_a_passes_ranker_reads_what_widening_added(tmp_path, capsys):
    from inventio.search import search
    from inventio.store import connect

    db = tmp_path / "map.db"
    repo = tmp_path / "repo"
    for i in range(20):  # prose that fills BM25's whole pool of 15 and more
        write(repo, f"docs/crash-{i}.md", f"# Crash {i}\nthe nightly job crash: rebuild index fails, run it again {i}\n")
    write(repo, "jobs/nightly.py", "def rebuild_index(rows):\n    return sum(r.size for r in rows)\n")
    run(capsys, "--db", str(db), "init", str(repo), "--name", "repo")
    con = connect(db)

    q = "the nightly job crash: rebuild_index fails"
    assert "jobs/nightly.py" not in {h.path for h in search(con, q, k=100, pool=15, symbols=False)}
    ranker = PassesRanker()
    top = search(con, q, k=3, pool=15, ranker=ranker)
    assert (top[0].path, top[0].via) == ("jobs/nightly.py", "defines:rebuild_index")
    assert len(ranker.read) == 2 and len(ranker.read[1]) <= 15   # the pool, then one final with the named file


def test_a_law_section_without_blank_lines_is_cut_at_its_clauses():
    from inventio.ingest import MAX_CHARS, chunk_markdown

    items = [f"{c}) Người sử dụng lao động phải thông báo bằng văn bản cho người lao động trước {n} ngày làm việc;"
             for n in range(3) for c in "abcdđeghik"]
    text = "# Điều 36. Quyền đơn phương chấm dứt hợp đồng\n\n1. Trong các trường hợp sau:\n" + "\n".join(items) + "\n"
    chunks = chunk_markdown(text)
    assert len(chunks) > 1
    assert all(len(c.text) < MAX_CHARS + 200 for c in chunks)   # a clause past the size, not 2 x MAX_CHARS
    assert all(c.text.split("\n")[0][1] == ")" for c in chunks[1:])   # each piece starts at a clause
    assert [c.start_line for c in chunks[1:]] == [c.end_line + 1 for c in chunks[:-1]]   # no line lost


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


def test_grep_searches_the_kept_text_and_opens_only_files_changed_since_sync(tmp_path, capsys, monkeypatch):
    """grep reads the text the map keeps (opening thousands of files took 24 s on a Mac where their stat took
    0.05 s), so a file edited after the last sync must still be read from disk, or grep shows lines that
    `read` no longer has. A map built before the text was kept gets it on its next sync."""
    import sqlite3

    import inventio.ingest as ingest

    db = str(tmp_path / "map.db")
    write(tmp_path / "d", "a.md", "# A\n\nnightly backup runs at 07:00\n")
    edited = write(tmp_path / "d", "b.md", "# B\n\nnothing here\n")
    assert run(capsys, "--db", db, "init", str(tmp_path / "d"), "--name", "d")[0] == 0
    edited.write_text("# B\n\nthe nightly backup moved to 09:00\n", encoding="utf-8")

    opened = []
    real = ingest.file_text
    monkeypatch.setattr(ingest, "file_text", lambda p: opened.append(p.name) or real(p))
    code, out = run(capsys, "--db", db, "grep", "nightly backup", "--json")
    got = json.loads(out.out)["matches"]
    assert [(m["path"], m["line"], m["text"]) for m in got] == [
        ("a.md", 3, "nightly backup runs at 07:00"), ("b.md", 3, "the nightly backup moved to 09:00")]
    assert opened == ["b.md"]

    con = sqlite3.connect(db)
    con.execute("DELETE FROM file_texts")
    con.commit()
    assert run(capsys, "--db", db, "sync")[0] == 0
    assert con.execute("SELECT count(*) FROM file_texts").fetchone()[0] == 2
    opened.clear()
    code, out = run(capsys, "--db", db, "grep", "09:00")
    assert code == 0 and opened == []


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


def test_systemone_state_and_questions_are_the_trained_shape():
    """The words a served model reads are the words it was trained on: one question per passage, the
    passage and line choices over the ids in the state, and the question capped where the records cap it."""
    from types import SimpleNamespace

    from inventio.systemone import MAX_OPTIONS, MAX_QUERY_CHARS, PASSAGE_ASKS, questions, render

    hits = [SimpleNamespace(path=f"src/f{i}.py", heading_path="", text=f"one {i}\n\ntwo {i}") for i in range(3)]
    state, pids, lids, owner = render(hits)
    assert pids == ["P01", "P02", "P03"]
    assert lids == ["L000", "L001", "L002", "L003", "L004", "L005"] and owner == [0, 0, 1, 1, 2, 2]
    assert "P02 [src/f1.py]" in state and "L005| two 2" in state

    served = questions("q", pids, lids)          # the served shape: three branches, no per-passage asks
    assert sorted(served) == ["exists", "where_line", "where_passage"]
    qs = questions("q" * (MAX_QUERY_CHARS + 40), pids, lids, with_passage_asks=True)
    assert sorted(k for k in qs if k.startswith("P")) == pids
    assert qs["P01"]["instructions"] == PASSAGE_ASKS[0].format(p="P01", q="q" * MAX_QUERY_CHARS)
    assert list(qs["where_passage"]["criteria"]) == pids and list(qs["where_line"]["criteria"]) == lids
    assert "exists" in qs
    assert "where_line" not in questions("q", pids, [f"L{i:03d}" for i in range(MAX_OPTIONS + 1)])


def test_honesty_speaks_only_when_asked_and_only_below():
    """The caveat is off unless the operator sets a threshold, and then it says what it knows: the
    passages read (not the map) do not seem to answer."""
    from inventio.systemone import EXISTS_READING, honesty

    key = "exists_max" if EXISTS_READING == "max" else "exists_head"
    assert honesty(None, 0.3) is None and honesty({}, 0.3) is None
    assert honesty({key: 0.5}, None) is None                     # no threshold: no caveat
    assert honesty({key: 0.35}, 0.30) is None                    # above it
    note = honesty({key: 0.20, "line": {"text": "the answer is here"}}, 0.30)
    assert "do not seem to answer" in note and "the answer is here" in note and "p=0.20" in note


def test_bench_rows_say_how_each_answer_entered(tmp_path, capsys):
    """The webshop example, one row per question: the gold's rank, the `via` that put it in the pool,
    whether BM25's own pool held it, and the top-10 coordinates; the summary is followed by the count
    of how the top-10 gold answers arrived."""
    example = Path(__file__).resolve().parent.parent / "examples" / "webshop"
    db = str(tmp_path / "map.db")
    for sub in ("app", "wiki"):
        assert run(capsys, "--db", db, "init", str(example / sub), "--name", sub)[0] == 0
    rows_path = tmp_path / "rows.jsonl"
    code, out = run(capsys, "--db", db, "bench", str(example / "questions.jsonl"),
                    "--ranker", "none", "--rows", str(rows_path))
    assert code == 0
    rows = [json.loads(l) for l in rows_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 13
    assert set(rows[0]) == {"question", "gold", "rank", "via", "in_bm25", "seconds", "top10"}
    assert rows[0]["gold"] == {"source": "wiki", "path": "runbook.md", "start_line": 8, "end_line": 8}
    assert isinstance(rows[0]["in_bm25"], bool) and len(rows[0]["top10"]) <= 10
    assert "top-10 gold came by:" in out.out
