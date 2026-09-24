"""Remote sources, offline: what a page or ticket becomes as Markdown, and a mirror that follows
the remote through edits, renames and deletions."""

import sqlite3
from types import SimpleNamespace

import pytest

from inventio import cli, connectors
from inventio.connectors.confluence import _Page
from inventio.connectors.jira import issue_doc
from inventio.connectors.mirror import Doc, Entry
from inventio.read import section


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("INVENTIO_RANKER", "none")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "data"))  # mirrors go under the data directory
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))


def test_confluence_page_keeps_code_links_and_diagrams():
    xhtml = (
        '<h1>Backup retention</h1><p># of orders rebuilt by hand: 1870.</p>'
        '<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">python</ac:parameter>'
        '<ac:plain-text-body><![CDATA[if age_days > 35:\n    storage.delete(backup)]]></ac:plain-text-body>'
        '</ac:structured-macro>'
        '<ac:structured-macro ac:name="include"><ac:parameter ac:name=""><ac:link>'
        '<ri:page ri:content-title="Escalation" /></ac:link></ac:parameter></ac:structured-macro>'
        '<p>Tracked in <ac:structured-macro ac:name="jira"><ac:parameter ac:name="key">SHOP-812</ac:parameter>'
        '</ac:structured-macro> <ac:structured-macro ac:name="status"><ac:parameter ac:name="title">Done'
        '</ac:parameter></ac:structured-macro></p>'
        '<ac:structured-macro ac:name="lucidchart"><ac:parameter ac:name="url">https://lucid.app/d/1'
        '</ac:parameter></ac:structured-macro>'
        '<table><tbody><tr><th><p>Job</p></th><th><p>Code</p></th></tr>'
        '<tr><td><p>daily</p></td><td><ac:structured-macro ac:name="code"><ac:plain-text-body>'
        '<![CDATA[run()\nnotify()]]></ac:plain-text-body></ac:structured-macro></td></tr></tbody></table>'
    )
    paths = {"1": "Ops/Policy.md", "2": "Ops/Escalation/Escalation.md"}
    md = _Page("https://x.atlassian.net", "GD", "1", paths["1"], paths, {"Policy": "1", "Escalation": "2"}) \
        .markdown("Policy", xhtml).split("\n")
    assert md[0] == "# Policy" and "## Backup retention" in md  # page h1 sits under the title
    assert "\\# of orders rebuilt by hand: 1870." in md  # body text never becomes a heading
    i = md.index("```python")
    assert md[i + 1:i + 4] == ["if age_days > 35:", "    storage.delete(backup)", "```"]
    assert "Included from [Escalation](Escalation/Escalation.md)" in md  # a relative link: a citation in the map
    assert "Tracked in SHOP-812 [Done]" in md
    assert "![Lucidchart diagram](https://lucid.app/d/1)" in md
    j = md.index("**Code:**")  # code in a table cell stays code, under its column name
    assert md[j + 1:j + 5] == ["```", "run()", "notify()", "```"]


def test_jira_comment_url_reads_that_comment():
    adf = lambda *content: {"type": "doc", "content": list(content)}  # noqa: E731
    para = lambda text: {"type": "paragraph", "content": [{"type": "text", "text": text}]}  # noqa: E731
    issue = {"key": "SHOP-812", "fields": {
        "summary": "No backup to restore", "status": {"name": "Done"}, "issuetype": {"name": "Incident"},
        "created": "2026-03-14T07:10:00.000+0700", "updated": "2026-03-15T09:00:00.000+0700",
        "description": adf({"type": "heading", "attrs": {"level": 1}, "content": [{"type": "text", "text": "Impact"}]},
                           para("1870 orders rebuilt by hand.")),
    }}
    comments = [{"id": "10", "created": "2026-03-14T08:00:00.000+0700", "author": {"displayName": "An"},
                 "body": adf(para("Looking."))},
                {"id": "11", "created": "2026-03-14T13:30:00.000+0700", "author": {"displayName": "Binh"},
                 "body": adf(para("Reran from the replica."))}]
    doc = issue_doc("https://x.atlassian.net", issue, comments)
    lines = doc.text.split("\n")
    assert lines[0] == "# SHOP-812 No backup to restore" and "### Impact" in lines
    start, end = section(lines, "focusedCommentId=11", doc.anchors)
    assert lines[start - 1] == "## Comment 2026-03-14 13:30 Binh"
    assert "Reran from the replica." in lines[start - 1:end] and "Looking." not in lines[start - 1:end]


class FakeRemote:
    """A remote whose items the test edits between syncs."""
    FORMAT = 1
    items: dict = {}

    def __init__(self, origin):
        self.origin, self.name = origin, "notes"

    def listing(self):
        return {i: Entry(v, p, f"https://wiki/{i}") for i, (v, p, _) in self.items.items()}

    def fetch(self, ids):
        for i in ids:
            yield i, Doc(self.items[i][2])


def test_mirror_follows_edits_renames_and_deletions(monkeypatch, tmp_path, capsys):
    site = "https://wiki.test/space"
    fake = SimpleNamespace(KIND="fake", Remote=FakeRemote, heading_url=lambda item, heads: item["url"],
                           origin=lambda url, q=None: url if url == site else None, locate=lambda url: None)
    monkeypatch.setitem(connectors.KINDS, "fake", fake)
    db = tmp_path / "map.db"
    FakeRemote.items = {
        "1": ("v1", "Ops/Policy.md", "# Policy\n\nKeep backups for 35 days; see [runbook](Runbook.md).\n"),
        "2": ("v1", "Ops/Runbook.md", "# Runbook\n\nRerun nightly_backup from the replica.\n"),
        "3": ("v1", "Old.md", "# Old\n\nThe retired quarterly sweep.\n"),
    }
    assert cli.main(["--db", str(db), "init", site]) == 0
    assert "citation 1" in capsys.readouterr().out  # a relative link between mirrored pages resolves

    FakeRemote.items["1"] = ("v2", "Ops/Policy.md", "# Policy\n\nKeep backups for 14 days; see [runbook](Runbook.md).\n")
    FakeRemote.items["2"] = ("v1", "Ops/Runbook-2026.md", FakeRemote.items["2"][2])  # renamed, same version
    del FakeRemote.items["3"]
    assert cli.main(["--db", str(db), "sync"]) == 0
    out = capsys.readouterr().out
    assert "1 fetched, 1 moved, 1 deleted" in out

    paths = {r[0] for r in sqlite3.connect(db).execute("SELECT path FROM files")}
    assert paths == {"Ops/Policy.md", "Ops/Runbook-2026.md"}
    assert cli.main(["--db", str(db), "query", "keep backups for days", "-k", "1"]) == 0
    out = capsys.readouterr().out
    assert "14 days" in out and "https://wiki/1" in out
    assert cli.main(["--db", str(db), "read", "notes:Ops/Runbook-2026.md:3"]) == 0
    assert capsys.readouterr().out.split("\n")[1] == "3  Rerun nightly_backup from the replica."


def test_database_schema_becomes_linked_cards_without_rows_or_password(tmp_path, capsys):
    pytest.importorskip("sqlalchemy")
    from inventio.connectors import sql

    assert sql.origin("postgresql://reader:s3cret@db:5432/core") == "postgresql://reader@db:5432/core"
    con = sqlite3.connect(tmp_path / "core.sqlite")
    con.executescript("""
        CREATE TABLE customer (id INTEGER PRIMARY KEY, country TEXT NOT NULL);
        CREATE TABLE orders (customer_id INTEGER NOT NULL REFERENCES customer(id), total REAL);
        INSERT INTO customer VALUES (991, 'VN');
    """)
    con.close()
    db = str(tmp_path / "map.db")
    assert cli.main(["--db", db, "init", f"sqlite:///{(tmp_path / 'core.sqlite').as_posix()}"]) == 0
    capsys.readouterr()
    assert cli.main(["--db", db, "read", "sqlite-core:main/orders.md"]) == 0
    card = capsys.readouterr().out
    assert "<!-- defines: main.orders orders -->" in card
    assert "- `customer_id` → [customer](customer.md) (`id`)" in card
    assert "991" not in card  # the catalog only, never a row
    assert cli.main(["--db", db, "show", "sqlite-core:main/orders.md"]) == 0
    out = capsys.readouterr().out
    assert "Dataset (by connector)" in out
    assert "<> citation sqlite-core:main/customer.md:" in out


def test_kafka_card_keeps_field_names_never_values():
    from inventio.connectors import kafka

    class Message:
        def value(self):
            return b'{"account_id": 991, "note": "customer phone 0901234567"}'

    remote = kafka.Remote("kafka://broker:9092")
    card = remote._card("alerts.raw", 3, {"retention.ms": "86400000"}, set(), Message())
    assert "| account_id | integer |  |" in card and "| note | string |  |" in card
    assert "991" not in card and "0901234567" not in card
