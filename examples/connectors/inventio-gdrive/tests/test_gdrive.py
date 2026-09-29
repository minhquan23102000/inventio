"""The connector against a local server that answers as the Drive API v3 and Google's token
endpoint document (files.list, files.get, files.export, about.get, the refresh grant). Not a real Drive."""

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import inventio_gdrive as gdrive
from inventio import cli, connectors

TOP = "folder_top_0000000001"
SUB = "folder_sub_0000000002"
DOC = "doc_runbook_000000003"
SHEET = "sheet_oncall_00000004"
PDF = "pdf_handbook_00000005"
IMG = "img_diagram_000000006"
DOC2 = "doc_elsewhere_0000007"
GDOC, GSHEET = "application/vnd.google-apps.document", "application/vnd.google-apps.spreadsheet"


def _pdf(text: str) -> bytes:
    """A one-page PDF whose text is drawn in Helvetica."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [5 0 R] /Count 1 >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 3 0 R >> >> >>"]
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode() + b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    return out + f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()


def file(fid, name, mime, parent, version=1, size=None):
    f = {"id": fid, "name": name, "mimeType": mime, "parents": [parent], "version": str(version),
         "createdTime": "2026-08-01T10:00:00.000Z", "modifiedTime": "2026-09-01T10:00:00.000Z",
         "webViewLink": f"https://docs.google.com/document/d/{fid}/edit", "owners": [{"displayName": "An"}],
         "lastModifyingUser": {"displayName": "Binh"}}
    if size is not None:
        f["size"] = str(size)
    return f


class Drive:
    """The files the fake Drive holds, their exported bodies, and the requests it was sent."""

    def __init__(self):
        pdf = _pdf("Restore the nightly backup from the replica")
        self.files = [
            file(TOP, "Ops", "application/vnd.google-apps.folder", "root"),
            file(SUB, "On call", "application/vnd.google-apps.folder", TOP),
            file(DOC, "Runbook", GDOC, TOP),
            file(SHEET, "Rota", GSHEET, SUB),
            file(PDF, "handbook.pdf", "application/pdf", SUB, size=len(pdf)),
            file(IMG, "diagram.png", "image/png", SUB, size=10),
            file(DOC2, "Elsewhere", GDOC, "root"),
        ]
        self.bodies = {
            (DOC, "text/markdown"): ("# Nightly backup\n\nRuns at `02:00` from the **replica**.\n\n"
                                     "![][image1]\n\n[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==>\n").encode(),
            (SHEET, "text/csv"): b"Week,Owner\r\n38,An\r\n39,Binh\r\n",
            (PDF, "media"): pdf,
            (DOC2, "text/markdown"): b"Not under Ops.\n",
        }
        self.calls: list[str] = []
        self.refreshed = 0

    def answer(self, path, q):
        self.calls.append(path)
        if path == "/drive/v3/about":
            return {"user": {"displayName": "An", "emailAddress": "an@example.com"}}
        if path == "/drive/v3/files":
            parent = q["q"].split("'")[1]
            kids = [f for f in self.files if f["parents"] == [parent]]
            start = int(q.get("pageToken", 0))
            part = kids[start:start + 2]  # two per page, so paging is exercised
            out = {"kind": "drive#fileList", "incompleteSearch": False, "files": part}
            if start + 2 < len(kids):
                out["nextPageToken"] = str(start + 2)
            return out
        fid = path.split("/")[4]
        if path.endswith("/export"):
            return self.bodies[(fid, q["mimeType"])]
        if q.get("alt") == "media":
            return self.bodies[(fid, "media")]
        return next(f for f in self.files if f["id"] == fid)


@pytest.fixture
def drive(monkeypatch):
    d = Drive()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            if self.headers["Authorization"] != "Bearer secret":
                self.send_response(401)
                self.end_headers()
                return
            out = d.answer(u.path, {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(out if isinstance(out, bytes) else json.dumps(out).encode())

        def do_POST(self):  # the token endpoint's refresh grant
            form = urllib.parse.parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())
            ok = form.get("refresh_token") == ["kept-refresh"] and form.get("grant_type") == ["refresh_token"]
            d.refreshed += ok
            self.send_response(200 if ok else 400)
            self.end_headers()
            self.wfile.write(json.dumps({"access_token": "secret", "expires_in": 3599} if ok else
                                        {"error": "invalid_grant", "error_description": "Token has been expired or revoked."}).encode())

        def log_message(self, format, *args):  # noqa: A002 - the base's name
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    monkeypatch.setattr(gdrive, "API", base + "/drive/v3")
    monkeypatch.setattr(gdrive, "TOKEN_URL", base + "/token")
    monkeypatch.setenv("GOOGLE_DRIVE_TOKEN", "secret")
    monkeypatch.setenv("INVENTIO_RANKER", "none")
    monkeypatch.setitem(connectors.KINDS, "gdrive", gdrive)
    yield d
    srv.shutdown()


def test_urls_name_a_folder_my_drive_or_one_file():
    assert gdrive.origin(f"https://drive.google.com/drive/u/0/folders/{TOP}?usp=sharing") == \
        f"https://drive.google.com/drive/folders/{TOP}"
    assert gdrive.origin("https://drive.google.com/drive/my-drive") == "https://drive.google.com/drive/my-drive"
    assert gdrive.locate(f"https://docs.google.com/document/d/{DOC}/edit#heading=h.x") == (DOC, "")
    assert gdrive.locate(f"https://drive.google.com/open?id={PDF}") == (PDF, "")
    assert gdrive.origin("https://github.com/acme/shop") is None


def test_a_folder_is_mirrored_as_its_tree_and_searched(drive, tmp_path, capsys):
    db = str(tmp_path / "map.db")
    assert cli.main(["--db", db, "init", f"https://drive.google.com/drive/folders/{TOP}"]) == 0
    capsys.readouterr()
    mirror = connectors.mirror.mirror_dir(f"gdrive-{TOP[:8]}")
    files = sorted(p.relative_to(mirror).as_posix() for p in mirror.rglob("*.md"))
    # the image is left out, and so is the document outside the folder
    assert files == ["On-call/Rota.md", "On-call/handbook.pdf.md", "Runbook.md"]
    runbook = (mirror / "Runbook.md").read_text()
    assert runbook.startswith("# Runbook\n") and "Runs at `02:00` from the **replica**." in runbook
    assert "base64" not in runbook
    assert "| Week | Owner |" in (mirror / "On-call/Rota.md").read_text()
    assert cli.main(["--db", db, "query", "restore nightly backup replica", "--json"]) == 0
    hits = json.loads(capsys.readouterr().out)
    pdf = next(h for h in hits if h["path"].endswith("handbook.pdf.md"))
    assert pdf["heading_path"] == "handbook > Page 1"
    assert cli.main(["--db", db, "query", "rota", "-w", "owner:An format:sheet", "--json"]) == 0
    assert {h["path"] for h in json.loads(capsys.readouterr().out)} == {"On-call/Rota.md"}


def test_only_changed_files_are_downloaded_again(drive, tmp_path, capsys):
    db = str(tmp_path / "map.db")
    assert cli.main(["--db", db, "init", f"https://drive.google.com/drive/folders/{TOP}"]) == 0
    drive.calls.clear()
    next(f for f in drive.files if f["id"] == SHEET)["version"] = "2"
    drive.bodies[(SHEET, "text/csv")] += b"40,Chi\r\n"
    assert cli.main(["--db", db, "sync"]) == 0
    assert [c for c in drive.calls if c != "/drive/v3/files"] == [f"/drive/v3/files/{SHEET}/export"]
    assert "1 fetched" in capsys.readouterr().out


def test_a_kept_login_is_refreshed_and_a_revoked_one_says_to_sign_in_again(drive, monkeypatch):
    monkeypatch.delenv("GOOGLE_DRIVE_TOKEN")
    kept = {"client_id": "c", "client_secret": "s", "refresh_token": "kept-refresh"}
    monkeypatch.setattr(gdrive.credentials, "get", lambda k: kept if k == "gdrive" else None)
    gdrive._access.clear()
    assert gdrive.client().get("/about", {"fields": "user"})["user"]["emailAddress"] == "an@example.com"
    gdrive.client()
    assert drive.refreshed == 1  # the access token is reused until it nears expiry
    gdrive._access.clear()
    kept["refresh_token"] = "revoked"
    with pytest.raises(connectors.RemoteError, match="run `inventio login https://drive.google.com` again"):
        gdrive.client()
