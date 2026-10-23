"""A background process that keeps the model ranker (dispositio) loaded, so a query pays only for
reading its candidates: importing PyTorch and loading the checkpoint take seconds, reading 15
candidates about 0.16 s on a laptop GPU. The first `inventio query` with a local ranker starts it, later ones hand
it their command line, and it exits after INVENTIO_SERVE_IDLE seconds (default 900) without a
request. INVENTIO_SERVE=0 runs every query in its own process, as before.

It listens on 127.0.0.1 only and answers a request only when it carries the token written, with
the port, to `serve.json` in the data directory (readable by its owner alone). A server started
by an older copy of the code is stopped and replaced.

`inventio serve --http PORT` is the other shape: a server that runs until stopped and answers
`POST /query` with what `query --json` prints (see serve_http).
"""

import contextlib
import io
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from .store import data_home

IDLE = 900
ENV = ("INVENTIO_", "HF_", "TYPESAFE_")  # what a request carries of the client's environment


def state_file() -> Path:
    return data_home() / "serve.json"


def fingerprint() -> str:
    """Changes when the installed code does, so an upgrade never talks to an old server."""
    pkg = Path(__file__).parent
    return str(max(p.stat().st_mtime_ns for p in pkg.rglob("*.py")))


def enabled() -> bool:
    return os.environ.get("INVENTIO_SERVE", "1") != "0"


def _send(state: dict, msg: dict, timeout: float | None) -> dict:
    with socket.create_connection(("127.0.0.1", state["port"]), timeout=5) as s:
        s.settimeout(timeout)
        s.sendall(json.dumps({**msg, "token": state["token"]}).encode() + b"\n")
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(1 << 20)
            if not chunk:
                raise ConnectionError("server closed the connection")
            buf += chunk
    return json.loads(buf)


def _live() -> dict | None:
    """The running server's state, None when there is none or it is not answering."""
    try:
        state = json.loads(state_file().read_text())
        if _send(state, {"op": "ping"}, 5).get("ok"):
            return state
    except (OSError, ValueError, KeyError):
        pass
    return None


def _start() -> dict | None:
    log = data_home() / "serve.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    kw = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
    with open(log, "ab") as out:
        proc = subprocess.Popen([sys.executable, "-m", "inventio.cli", "serve"], stdin=subprocess.DEVNULL,
                                stdout=out, stderr=out, **kw)
    print("inventio: starting the model server (once; the first run also downloads a model)",
          file=sys.stderr)
    # A first download takes minutes, so the wait is generous — but never unbounded: a server that never
    # answers must end in this process doing the work, not in a query that hangs with nothing to show.
    deadline = time.time() + float(os.environ.get("INVENTIO_SERVE_START", 900))
    while time.time() < deadline:
        state = _live()
        if state and (state.get("fingerprint") == fingerprint() or state.get("pid") == proc.pid):
            return state   # whatever started it: a server running this code is the one to hand work to
        if proc.poll() is not None:
            print(f"inventio: the model server stopped (see {log}); running here instead", file=sys.stderr)
            return None
        time.sleep(0.2)
    print(f"inventio: the model server did not answer in {deadline - time.time():.0f}s (see {log}); "
          f"running here instead", file=sys.stderr)
    return None


def forward(argv: list[str]) -> int | None:
    """Run `inventio <argv>` in the server; its exit code, or None to run it here."""
    state = _live()
    if state and state.get("fingerprint") != fingerprint():
        with contextlib.suppress(OSError, ValueError):
            _send(state, {"op": "stop"}, 5)
        state = None
    state = state or _start()
    if not state:
        return None
    env = {k: v for k, v in os.environ.items() if k.startswith(ENV)}
    timeout = float(os.environ.get("INVENTIO_SERVE_TIMEOUT", 3600))   # a first query loads or downloads the model
    try:
        res = _send(state, {"op": "run", "argv": argv, "cwd": os.getcwd(), "env": env}, timeout)
    except (OSError, ValueError) as e:
        print(f"inventio: the model server did not answer ({e}); running here instead", file=sys.stderr)
        return None
    sys.stdout.write(res["out"])
    sys.stderr.write(res["err"])
    return res["code"]


@contextlib.contextmanager
def _as_client(cwd: str, env: dict):
    """The client's directory and INVENTIO_*/HF_*/TYPESAFE_* variables, for one request."""
    old_cwd, old_env = os.getcwd(), {k: v for k, v in os.environ.items() if k.startswith(ENV)}
    os.chdir(cwd)
    for k in old_env:
        os.environ.pop(k)
    os.environ.update(env)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        for k in env:
            os.environ.pop(k, None)
        os.environ.update(old_env)


def serve(idle: float) -> int:
    from . import cli, rankers

    loaded = {}
    make = rankers.make_ranker

    def cached(name):  # one loaded model per checkpoint and device
        if name != "dispositio":
            return make(name)
        from .systemone import run_id

        key = (run_id(), os.environ.get("INVENTIO_DEVICE"))
        if key not in loaded:
            loaded[key] = make(name)
        return loaded[key]

    opened = []  # the map connections one request opens, closed once it is answered
    connect = cli.connect

    def tracked(*a, **k):
        opened.append(connect(*a, **k))
        return opened[-1]

    cli.make_ranker, cli.connect = cached, tracked
    token = secrets.token_hex(16)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    srv.settimeout(idle)
    # the ranker object only: dispositio reads its checkpoint on the first query (a first run downloads it), and
    # a server that loaded before announcing would leave the client's ping unanswered for that long
    cached(rankers.default_ranker())
    path = state_file()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"port": srv.getsockname()[1], "token": token, "pid": os.getpid(),
                               "fingerprint": fingerprint()}))
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    print(f"serving on 127.0.0.1:{srv.getsockname()[1]}, pid {os.getpid()}, exits after {idle:.0f}s idle",
          flush=True)
    # A query runs on a worker thread, one at a time (stdout, cwd and the environment are the process's), so
    # the accept loop keeps answering pings: a busy server that missed the client's 5 s ping looked dead, and
    # the next query started a second server with a second copy of the model.
    busy = threading.Lock()

    def run(conn, msg):
        with conn, busy:
            out, err = io.StringIO(), io.StringIO()
            with _as_client(msg["cwd"], msg["env"]), contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(err):
                try:
                    code = cli.main([*msg["argv"], "--here"])
                except SystemExit as e:  # argparse errors and -h
                    code = e.code if isinstance(e.code, int) else 2
                except Exception as e:  # a failing query must not take the server down
                    print(f"{type(e).__name__}: {e}", file=sys.stderr)
                    code = 1
                finally:
                    while opened:
                        opened.pop().close()
            with contextlib.suppress(OSError):
                conn.sendall(json.dumps({"code": code, "out": out.getvalue(), "err": err.getvalue()}).encode()
                             + b"\n")

    try:
        while True:
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                if busy.locked():   # a long query is not idleness
                    continue
                return 0
            conn.settimeout(30)
            try:
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = conn.recv(1 << 16)
                    if not chunk:
                        break
                    buf += chunk
                msg = json.loads(buf)
            except (OSError, ValueError):
                conn.close()
                continue
            if not secrets.compare_digest(str(msg.get("token", "")), token):
                conn.close()
                continue
            if msg["op"] in ("stop", "ping"):
                with conn:
                    conn.sendall(b'{"ok": true}\n')
                if msg["op"] == "stop":
                    return 0
                continue
            conn.settimeout(None)
            threading.Thread(target=run, args=(conn, msg), daemon=True).start()
    finally:
        srv.close()
        with contextlib.suppress(OSError, ValueError):
            if json.loads(path.read_text()).get("pid") == os.getpid():
                path.unlink()


def stop() -> bool:
    """Stop the background server and the HTTP server, whichever run (both hold weights `update` replaces);
    whether one did."""
    stopped = False
    state = _live()
    if state:
        _send(state, {"op": "stop"}, 5)
        stopped = True
    with contextlib.suppress(OSError, ValueError, KeyError):
        os.kill(json.loads(http_state_file().read_text())["pid"], signal.SIGTERM)
        stopped = True
    return stopped


# --- `inventio serve --http PORT`: the answers of `query --json` over HTTP on loopback, for a program that asks
# many questions (a portal, an agent harness). It runs until stopped and loads the model at start, so no
# question pays for PyTorch and the checkpoint. The map is opened per request: a sync or init is seen by the
# next question, with nothing to reload. Requests run on their own threads; the model reads one state at a
# time (systemone.Model's lock), BM25 and the rest run side by side. A request carries the token written, with
# the port, to `serve-http.json` (readable by its owner alone).

def http_state_file() -> Path:
    return data_home() / "serve-http.json"


QUERY_FIELDS = {"text": str, "where": str, "source": list, "k": int, "pool": int, "ranker": str,
                "no_links": bool, "no_symbols": bool, "neighbours": bool, "types": bool, "facts": bool}


def query_argv(body: dict) -> list[str]:
    """A request body as the `inventio query --json` command line it stands for. ValueError on a field
    that is unknown or of the wrong type: a typo must not be answered with the defaults."""
    unknown = set(body) - set(QUERY_FIELDS)
    if unknown:
        raise ValueError(f"unknown field(s): {', '.join(sorted(unknown))}; known: {', '.join(QUERY_FIELDS)}")
    for key, kind in QUERY_FIELDS.items():
        if key in body and (not isinstance(body[key], kind) or kind is int and isinstance(body[key], bool)):
            raise ValueError(f"{key} must be {kind.__name__}")
    text = " ".join(body.get("text", "").split())
    if not text:
        raise ValueError("text is empty")
    # a leading space keeps a question that starts with '-' a positional, never a flag
    argv = ["query", " " + text, "--json", "--here"]
    if "where" in body:
        argv += ["-w", body["where"]]
    for name in body.get("source", []):
        argv.append(f"--source={name}")   # one token: a name like '-h' stays a value
    from .rankers import RANKERS

    if body.get("ranker", "dispositio") not in RANKERS:
        raise ValueError(f"ranker must be one of {', '.join(RANKERS)}")
    for key, flag in (("k", "-k"), ("pool", "--pool"), ("ranker", "--ranker")):
        if key in body:
            argv += [flag, str(body[key])]
    for key in ("no_links", "no_symbols", "neighbours", "types", "facts"):
        if body.get(key):
            argv.append("--" + key.replace("_", "-"))
    return argv


def serve_http(port: int, db: str | None = None) -> int:
    """`db`: the map `inventio --db` named when the server started; every question reads that one."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from . import cli, rankers

    token = secrets.token_hex(16)
    status = {"model": "loading", "run": None, "device": None, "error": None}
    loaded = threading.Event()   # the start-up load is over: questions wait for it rather than load a second copy

    def warm():
        try:
            if rankers.default_ranker() != "dispositio":
                status["model"] = "none"
            else:
                info = rankers.SystemOneRanker()._served()
                status.update(model="ready", run=info["run"], device=info.get("device"))
        except Exception as e:   # questions are still answered (in BM25 order, as `query` would); /health says why
            status.update(model="error", error=f"{type(e).__name__}: {e}")
        finally:
            loaded.set()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            sys.stderr.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {fmt % args}\n")

        def _reply(self, code: int, payload):
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            got = self.headers.get("authorization", "").removeprefix("Bearer ").strip()
            if secrets.compare_digest(got, token):
                return True
            self._reply(401, {"error": "missing or wrong token (see serve-http.json)"})
            return False

        def do_GET(self):
            if self.path != "/health":
                return self._reply(404, {"error": "GET /health or POST /query"})
            if self._authorized():
                self._reply(200, {"ok": True, "pid": os.getpid(), **status})

        def do_POST(self):
            if self.path != "/query":
                return self._reply(404, {"error": "GET /health or POST /query"})
            if not self._authorized():
                return
            t = time.perf_counter()
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("the body must be a JSON object")
                args = cli.parse((["--db", db] if db else []) + query_argv(body))
            except (ValueError, SystemExit) as e:   # argparse exits on a value it refuses (an unknown ranker)
                return self._reply(400, {"error": str(e) if isinstance(e, ValueError) else "bad argument"})
            loaded.wait()
            try:
                con = cli._db(args)
                try:
                    scope = cli._scope(con, args)
                    hits, notes = cli.ranked(con, args, scope)
                    rows = cli.json_rows(hits)
                finally:
                    con.close()
            except ValueError as e:   # a -w filter the map cannot read
                return self._reply(400, {"error": str(e)})
            except cli.QueryFailed as e:
                return self._reply(422 if e.code == 2 else 503, {"error": e.message, "code": e.code})
            except Exception as e:   # a failing question must not take the server down
                return self._reply(500, {"error": f"{type(e).__name__}: {e}"})
            for note in notes:
                sys.stderr.write(note + "\n")
            self.log_message("query: %d hits in %.0f ms", len(rows), (time.perf_counter() - t) * 1000)
            self._reply(200, rows)

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = False   # server_close waits for the questions in flight
    path = http_state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:   # never readable by others
        json.dump({"url": f"http://127.0.0.1:{srv.server_port}", "token": token, "pid": os.getpid()}, f)
    os.replace(tmp, path)
    print(f"serving HTTP on 127.0.0.1:{srv.server_port}, pid {os.getpid()}, until stopped", flush=True)
    threading.Thread(target=warm, daemon=True).start()
    # `inventio serve --stop`, `inventio update` (new weights) or a supervisor sends SIGTERM
    if threading.current_thread() is threading.main_thread():   # signals reach the main thread only
        signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=srv.shutdown, daemon=True).start())
    try:
        srv.serve_forever()
    finally:
        srv.server_close()
        with contextlib.suppress(OSError, ValueError):
            if json.loads(path.read_text()).get("pid") == os.getpid():
                path.unlink()
    return 0
