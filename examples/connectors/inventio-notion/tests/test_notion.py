"""The connector against a local server that answers as the Notion API documents (request and
response shapes from notion-sdk-js, API version 2025-09-03). Not a real workspace."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import inventio_notion as notion
from inventio import cli, connectors

ROOT = "11111111-1111-1111-1111-111111111111"
CHILD = "22222222-2222-2222-2222-222222222222"
ROW = "33333333-3333-3333-3333-333333333333"
OTHER = "44444444-4444-4444-4444-444444444444"
SOURCE = "55555555-5555-5555-5555-555555555555"
TOGGLE = "66666666-6666-6666-6666-666666666666"


def rich(text, **ann):
    return {"type": "text", "text": {"content": text, "link": None}, "plain_text": text, "href": None,
            "annotations": {"bold": False, "italic": False, "strikethrough": False, "underline": False,
                            "code": False, "color": "default", **ann}}


def mention(pid, text):
    return {"type": "mention", "mention": {"type": "page", "page": {"id": pid}}, "plain_text": text,
            "href": f"https://www.notion.so/{pid.replace('-', '')}", "annotations": rich("")["annotations"]}


def page(pid, title, parent, edited="2026-09-01T10:00:00.000Z", props=None):
    return {"object": "page", "id": pid, "created_time": "2026-08-01T10:00:00.000Z", "last_edited_time": edited,
            "in_trash": False, "archived": False, "is_archived": False,
            "url": f"https://www.notion.so/{title.replace(' ', '-')}-{pid.replace('-', '')}", "parent": parent,
            "properties": {"Name": {"id": "title", "type": "title", "title": [rich(title)]}, **(props or {})}}


def block(bid, kind, has_children=False, **body):
    return {"object": "block", "id": bid, "type": kind, kind: body, "has_children": has_children,
            "in_trash": False, "archived": False}


class Notion:
    """The pages, blocks and data sources the fake workspace holds; `calls` records requests."""

    def __init__(self):
        self.pages = {
            ROOT: page(ROOT, "Runbook", {"type": "workspace", "workspace": True}),
            CHILD: page(CHILD, "Backup", {"type": "block_id", "block_id": TOGGLE}),
            ROW: page(ROW, "Fix replica lag", {"type": "data_source_id", "data_source_id": SOURCE, "database_id": "d"},
                      props={"Status": {"id": "s", "type": "status", "status": {"id": "1", "name": "In progress", "color": "blue"}},
                             "Tags": {"id": "t", "type": "multi_select", "multi_select": [{"id": "a", "name": "db", "color": "red"}]},
                             "Owner": {"id": "o", "type": "people", "people": [{"object": "user", "id": "u", "name": "An"}]}}),
            OTHER: page(OTHER, "Elsewhere", {"type": "workspace", "workspace": True}),
        }
        self.sources = [{"object": "data_source", "id": SOURCE, "title": [rich("Tasks")], "in_trash": False,
                         "parent": {"type": "database_id", "database_id": "d"},
                         "database_parent": {"type": "page_id", "page_id": ROOT}}]
        self.blocks = {TOGGLE: {"object": "block", "id": TOGGLE, "type": "toggle", "parent": {"type": "page_id", "page_id": ROOT}}}
        self.children = {
            ROOT: [block("h1", "heading_1", rich_text=[rich("Nightly backup")], is_toggleable=False),
                   block("p1", "paragraph", rich_text=[rich("Runs at "), rich("02:00", code=True), rich(" from the "),
                                                       rich("replica", bold=True), rich(".")]),
                   block(TOGGLE, "toggle", True, rich_text=[rich("Details")]),
                   block("c1", "code", rich_text=[rich("pg_dump -h replica")], caption=[], language="shell"),
                   block("tb", "table", True, table_width=2, has_column_header=True, has_row_header=False),
                   block(CHILD, "child_page", title="Backup"),
                   block("db", "child_database", title="Tasks"),
                   block("im", "image", type="file", file={"url": "https://s3/secret", "expiry_time": "x"}, caption=[])],
            TOGGLE: [block("b1", "bulleted_list_item", rich_text=[rich("see "), mention(ROW, "Fix replica lag")])],
            "tb": [block("r1", "table_row", cells=[[rich("Step")], [rich("Owner")]]),
                   block("r2", "table_row", cells=[[rich("dump")], [rich("An")]])],
            CHILD: [block("n1", "numbered_list_item", rich_text=[rich("stop writes")]),
                    block("n2", "numbered_list_item", rich_text=[rich("dump")]),
                    block("t1", "to_do", rich_text=[rich("verify")], checked=True)],
            ROW: [block("p2", "paragraph", rich_text=[rich("Lag above 30 s since Monday.")])],
            OTHER: [],
        }
        self.calls: list[tuple[str, str]] = []

    def answer(self, method, path, body):
        self.calls.append((method, path))
        if path == "/v1/users/me":
            return {"object": "user", "id": "bot", "type": "bot", "name": "inventio"}
        if path == "/v1/search":
            items = list(self.pages.values()) if body["filter"]["value"] == "page" else self.sources
            start = int(body.get("start_cursor") or 0)
            part = items[start:start + 2]  # two per page, so paging is exercised
            more = start + 2 < len(items)
            return {"object": "list", "results": part, "has_more": more, "next_cursor": str(start + 2) if more else None,
                    "type": "page_or_data_source", "page_or_data_source": {}}
        if path.startswith("/v1/blocks/") and path.endswith("/children"):
            return {"object": "list", "results": self.children[path.split("/")[3]], "has_more": False, "next_cursor": None}
        if path.startswith("/v1/blocks/"):
            return self.blocks[path.split("/")[3]]
        if path.startswith("/v1/pages/"):
            return self.pages[path.split("/")[3]]
        raise KeyError(path)


@pytest.fixture
def workspace(monkeypatch):
    ws = Notion()

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, body=None):
            assert self.headers["Notion-Version"] == notion.VERSION
            if self.headers["Authorization"] != "Bearer secret":
                self.send_response(401)
                self.end_headers()
                return
            out = json.dumps(ws.answer(self.command, self.path.split("?")[0], body)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def do_GET(self):
            self._reply()

        def do_POST(self):
            self._reply(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))

        def log_message(self, format, *args):  # noqa: A002 - the base's name
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(notion, "API", f"http://127.0.0.1:{srv.server_address[1]}/v1")
    monkeypatch.setenv("NOTION_TOKEN", "secret")
    monkeypatch.setenv("INVENTIO_RANKER", "none")
    monkeypatch.setitem(connectors.KINDS, "notion", notion)
    yield ws
    srv.shutdown()


def test_urls_name_the_workspace_or_one_page():
    assert notion.origin("https://www.notion.so/acme") == "https://www.notion.so"
    assert notion.origin(f"https://www.notion.so/acme/Runbook-{ROOT.replace('-', '')}") == \
        f"https://www.notion.so/{ROOT.replace('-', '')}"
    assert notion.locate(f"https://acme.notion.site/Runbook-{ROOT.replace('-', '')}#abc") == (ROOT, "abc")
    assert notion.locate(f"https://www.notion.so/acme/x?p={CHILD.replace('-', '')}") == (CHILD, "")
    assert notion.origin("https://github.com/acme/shop") is None


def test_a_page_tree_is_mirrored_with_relative_links_fields_and_headings(workspace, tmp_path, capsys):
    db = str(tmp_path / "map.db")
    root_url = f"https://www.notion.so/Runbook-{ROOT.replace('-', '')}"
    assert cli.main(["--db", db, "init", root_url]) == 0
    capsys.readouterr()
    mirror = connectors.mirror.mirror_dir(f"notion-{ROOT.replace('-', '')[:8]}")
    files = sorted(p.relative_to(mirror).as_posix() for p in mirror.rglob("*.md"))
    # the page outside the one asked for is left out; the page inside a toggle sits under its page,
    # a database row under the database's title
    assert files == ["Runbook.md", "Runbook/Backup.md", "Runbook/Tasks/Fix-replica-lag.md"]
    text = (mirror / "Runbook.md").read_text()
    assert "Runs at `02:00` from the **replica**." in text
    assert "## Nightly backup" in text
    assert "- see [Fix replica lag](Runbook/Tasks/Fix-replica-lag.md)" in text
    assert "[Backup](Runbook/Backup.md)" in text
    assert "| Step | Owner |" in text and "| dump | An |" in text
    assert "```shell\npg_dump -h replica\n```" in text
    assert "https://s3/secret" not in text  # Notion's file URLs expire; the block's link stays
    assert "1. stop writes\n2. dump\n- [x] verify" in (mirror / "Runbook/Backup.md").read_text()
    row = (mirror / "Runbook/Tasks/Fix-replica-lag.md").read_text()
    assert "- **Status:** In progress" in row and "- **Owner:** An" in row

    assert cli.main(["--db", db, "query", "replica lag", "-w", 'status:"In progress"', "--json"]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert {h["path"] for h in hits} == {"Runbook/Tasks/Fix-replica-lag.md"}
    assert cli.main(["--db", db, "query", "lag", "-w", "labels:db owner:An", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)
    assert cli.main(["--db", db, "query", "nightly backup", "--json"]) == 0
    top = json.loads(capsys.readouterr().out)[0]
    assert top["url"].endswith("#h1")  # the heading's own block on the web


def test_only_edited_pages_are_fetched_again(workspace, tmp_path, capsys):
    db = str(tmp_path / "map.db")
    assert cli.main(["--db", db, "init", "https://www.notion.so"]) == 0
    workspace.calls.clear()
    workspace.pages[CHILD]["last_edited_time"] = "2026-09-20T10:00:00.000Z"
    workspace.children[CHILD].append(block("p9", "paragraph", rich_text=[rich("new step")]))
    assert cli.main(["--db", db, "sync"]) == 0
    read = {p.split("/")[3] for m, p in workspace.calls if p.endswith("/children")}
    assert read == {CHILD}
    assert "1 fetched" in capsys.readouterr().out


def test_login_keeps_a_token_only_after_notion_accepts_it(workspace, monkeypatch):
    kept = {}
    monkeypatch.setattr(notion.credentials, "put", lambda k, v: kept.update({k: v}))
    with pytest.raises(connectors.RemoteError, match="refused"):
        notion.login("https://www.notion.so", input, lambda _: "wrong")
    assert kept == {}
    assert "as inventio" in notion.login("https://www.notion.so", input, lambda _: "secret")
    assert kept == {"notion": {"token": "secret"}}
