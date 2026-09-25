"""A background process that keeps dispositio loaded, so a query pays only for reading its
candidates. Importing PyTorch and loading the model take about 5 s on a laptop; reading 15
candidates about 1.5 s. The first `inventio query` with a local ranker starts it, later ones hand
it their command line, and it exits after INVENTIO_SERVE_IDLE seconds (default 900) without a
request. INVENTIO_SERVE=0 runs every query in its own process, as before.

It listens on 127.0.0.1 only and answers a request only when it carries the token written, with
the port, to `serve.json` in the data directory (readable by its owner alone). A server started
by an older copy of the code is stopped and replaced.
"""

import contextlib
import io
import json
import os
import secrets
import socket
import subprocess
import sys
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
    print("inventio: starting the model server (once; the first run also downloads dispositio)",
          file=sys.stderr)
    while proc.poll() is None:  # no deadline: a first download takes minutes
        state = _live()
        if state and state.get("pid") == proc.pid:
            return state
        time.sleep(0.2)
    print(f"inventio: the model server stopped (see {log}); running here instead", file=sys.stderr)
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
    try:
        res = _send(state, {"op": "run", "argv": argv, "cwd": os.getcwd(), "env": env}, None)
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
        if name not in ("dispositio", "laya"):
            return make(name)
        key = (rankers.checkpoint(name), os.environ.get("INVENTIO_DEVICE"))
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
    cached(rankers.default_ranker())  # load before announcing, so the first query finds it warm
    path = state_file()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"port": srv.getsockname()[1], "token": token, "pid": os.getpid(),
                               "fingerprint": fingerprint()}))
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    print(f"serving on 127.0.0.1:{srv.getsockname()[1]}, pid {os.getpid()}, exits after {idle:.0f}s idle",
          flush=True)
    try:
        while True:
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                return 0
            with conn:
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
                    continue
                if not secrets.compare_digest(str(msg.get("token", "")), token):
                    continue
                if msg["op"] in ("stop", "ping"):
                    conn.sendall(b'{"ok": true}\n')
                    if msg["op"] == "stop":
                        return 0
                    continue
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
                conn.sendall(json.dumps({"code": code, "out": out.getvalue(), "err": err.getvalue()}).encode()
                             + b"\n")
    finally:
        srv.close()
        with contextlib.suppress(OSError, ValueError):
            if json.loads(path.read_text()).get("pid") == os.getpid():
                path.unlink()


def stop() -> bool:
    state = _live()
    if not state:
        return False
    _send(state, {"op": "stop"}, 5)
    return True
