"""JSON over HTTPS for connectors: standard library only, retries where the server asks for it."""

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
        url = path if path.startswith("http") else self.base + path
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params, doseq=True)
        for attempt in range(ATTEMPTS):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=self.headers), timeout=60) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code in RETRY and attempt < ATTEMPTS - 1:
                    time.sleep(float(e.headers.get("Retry-After") or 2 ** attempt))
                    continue
                raise RemoteError(f"{e.code} {e.reason} from {url.split('?')[0]}") from None
            except urllib.error.URLError as e:
                if attempt < ATTEMPTS - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RemoteError(f"cannot reach {url.split('?')[0]}: {e.reason}") from None
        raise AssertionError("unreachable")
