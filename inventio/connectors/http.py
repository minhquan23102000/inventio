"""JSON over HTTPS for connectors: standard library only, retries where the server asks for it."""

import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request

RETRY = {429, 500, 502, 503, 504}
ATTEMPTS = 5


class RemoteError(RuntimeError):
    """A remote source refused or failed; the message is meant for the person at the terminal."""


class Client:
    def __init__(self, base: str, headers: dict[str, str]):
        self.base = base.rstrip("/")
        self.headers = {"Accept": "application/json", **headers}

    def get(self, path: str, params: dict | None = None) -> dict:
        """`path` is relative to the base (or a full URL, as paging links sometimes are)."""
        return json.loads(self._send(path, params))

    def post(self, path: str, body: dict) -> dict:
        """A JSON body sent, a JSON answer read (Notion's search, GraphQL APIs)."""
        return json.loads(self._send(path, None, json.dumps(body).encode()))

    def raw(self, path: str, params: dict | None = None) -> bytes:
        """The answer's bytes as sent: a file download or an export that is not JSON."""
        return self._send(path, params)

    def _send(self, path: str, params: dict | None, body: bytes | None = None) -> bytes:
        url = path if path.startswith("http") else self.base + path
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params, doseq=True)
        headers = {**self.headers, "Content-Type": "application/json"} if body is not None else self.headers
        for attempt in range(ATTEMPTS):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, data=body, headers=headers), timeout=60) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code in RETRY and attempt < ATTEMPTS - 1:
                    time.sleep(float(e.headers.get("Retry-After") or 2 ** attempt))
                    continue
                raise RemoteError(f"{e.code} {e.reason} from {url.split('?')[0]}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead) as e:
                # a read that times out or a body cut short mid-transfer is as passing as a refused connection:
                # a 3,352-PR fetch died on each at ~3,000 without them
                if attempt < ATTEMPTS - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RemoteError(f"cannot reach {url.split('?')[0]}: {getattr(e, 'reason', e)}") from None
        raise AssertionError("unreachable")
