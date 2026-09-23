import json
import textwrap

import pytest

from laya_atlas import cli
from laya_atlas.ingest import chunk_file


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
    db = str(tmp_path / "atlas.db")
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
    db = str(tmp_path / "atlas.db")
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


def test_cloud_ranker_refuses_private_sources(tmp_path, capsys, monkeypatch):
    pytest.importorskip("typesafe_sdk")
    monkeypatch.setenv("TYPESAFE_API_KEY", "not-a-real-key")
    db = str(tmp_path / "atlas.db")
    write(tmp_path / "private", "notes.md", "# Notes\ncustomer 4411 flagged for mule activity\n")
    run(capsys, "--db", db, "init", str(tmp_path / "private"), "--name", "bank")
    code, out = run(capsys, "--db", db, "query", "mule activity", "--ranker", "typesafe")
    assert code == 3
    assert "bank" in out.err
