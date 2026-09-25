"""Who a remote source signs in as, asked once and kept in the operating system's keychain
(Windows Credential Manager, macOS Keychain, Secret Service on Linux) through `keyring`.

A secret is kept per place, not per source: an Atlassian site (`atlassian:<host>`), a database
login (`sql:<URL without password>`). Two spaces of one site share one login; two sites with
different accounts stay apart. The map and the mirrors never hold a secret.

Where a secret is looked for, first found wins:
- the environment (ATLASSIAN_EMAIL + ATLASSIAN_API_TOKEN, INVENTIO_SQL_PASSWORD, PGPASSWORD ...),
  for CI and agents that are handed their credentials;
- the keychain, filled by `inventio login <url>`;
- for GitHub, the GitHub CLI (`gh auth token`): Inventio keeps no GitHub token of its own.
"""

import json
import os

from .connectors.http import RemoteError

SERVICE = "inventio"


class Missing(RemoteError):
    """No credential for this place; `inventio login <url>` would ask for one."""

    def __init__(self, url: str, what: str):
        self.url = url
        super().__init__(f"no {what} for {url}; run `inventio login {url}` (or set it in the environment)")


def _keyring():
    """The keyring module, or None where no backend can keep a secret (a headless Linux box)."""
    import keyring
    from keyring.backends import fail

    return None if isinstance(keyring.get_keyring(), fail.Keyring) else keyring


def get(key: str) -> dict | None:
    kr = _keyring()
    if kr is None:
        return None
    try:
        raw = kr.get_password(SERVICE, key)
    except Exception:  # a locked or broken keychain reads as "nothing kept"; the caller says what is missing
        return None
    return json.loads(raw) if raw else None


def put(key: str, secret: dict) -> None:
    kr = _keyring()
    if kr is None:
        raise RemoteError("no keychain on this machine (keyring found no backend); "
                          "set the credential in the environment instead")
    kr.set_password(SERVICE, key, json.dumps(secret))


def delete(key: str) -> bool:
    kr = _keyring()
    if kr is None or kr.get_password(SERVICE, key) is None:
        return False
    kr.delete_password(SERVICE, key)
    return True


# ------------------------------------------------------------------------ per kind of place


def atlassian_key(host: str) -> str:
    return f"atlassian:{host}"


def atlassian(site_url: str) -> tuple[str, str]:
    """(email, API token) for an Atlassian site."""
    email, token = os.environ.get("ATLASSIAN_EMAIL"), os.environ.get("ATLASSIAN_API_TOKEN")
    if email and token:
        return email, token
    kept = get(atlassian_key(site_url.split("://", 1)[-1]))
    if kept:
        return kept["email"], kept["token"]
    raise Missing(site_url, "Atlassian email and API token")


def sql_key(origin: str) -> str:
    return f"sql:{origin}"


def sql_password(origin: str) -> str | None:
    """The password of a database login; None lets the driver find its own (PGPASSWORD, a
    .pgpass file, a socket that needs none)."""
    return os.environ.get("INVENTIO_SQL_PASSWORD") or (get(sql_key(origin)) or {}).get("password")


def status(kind: str, origin: str) -> str:
    """Where a source's credential comes from: env, keyring, gh, or missing; `-` when the
    kind needs none of Inventio's (a directory; Kafka and S3 read their client's own settings)."""
    if kind in ("confluence", "jira"):
        if os.environ.get("ATLASSIAN_EMAIL") and os.environ.get("ATLASSIAN_API_TOKEN"):
            return "env"
        host = origin.split("://", 1)[-1].split("/", 1)[0]
        return "keyring" if get(atlassian_key(host)) else "missing"
    if kind == "sql":
        if os.environ.get("INVENTIO_SQL_PASSWORD"):
            return "env"
        return "keyring" if get(sql_key(origin)) else "driver"  # the driver's own: PGPASSWORD, .pgpass, none
    if kind == "github":
        from .connectors import github

        return "gh" if github.token(github.host_of(origin), quiet=True) else "missing"
    return "-"


# ------------------------------------------------------------------------ login and logout


def _place(url: str) -> tuple[str, str]:
    """(kind of place, where): ("atlassian", site URL), ("sql", origin), ("github", host)."""
    from .connectors import atlassian as site, github, sql

    s = site.site(url)
    if s:
        return "atlassian", s
    o = sql.origin(url)
    if o:
        return "sql", o
    if github.origin(url):
        return "github", github.host_of(url)
    raise RemoteError(f"not a URL that signs in: {url} (an Atlassian site, a database URL, or GitHub)")


def login(url: str, ask=input, ask_secret=None) -> str:
    """Ask for the credential of the place a URL names, try it once against the real service,
    and keep it in the keychain. Returns what to tell the person."""
    import getpass

    try:
        return _login(url, ask, ask_secret or getpass.getpass)
    except (EOFError, KeyboardInterrupt):  # no one answered the prompt
        raise RemoteError(f"login to {url} cancelled; nothing was kept") from None


def _login(url: str, ask, ask_secret) -> str:
    kind, where = _place(url)
    if kind == "github":
        from .connectors import github

        if github.token(where, quiet=True):
            return f"GitHub on {where} signs in through the GitHub CLI, which is signed in; nothing to keep"
        raise RemoteError(f"GitHub signs in through the GitHub CLI: gh auth login --hostname {where}")
    if kind == "atlassian":
        from .connectors import atlassian as site

        email = ask(f"email for {where}: ").strip()
        token = ask_secret("API token (https://id.atlassian.com/manage-profile/security/api-tokens): ").strip()
        try:
            name = site.whoami(site.signed(where, email, token))
        except RemoteError as e:
            if str(e).startswith(("401", "403")):
                raise RemoteError(f"{where} refused that email and token ({str(e).split(' from ')[0]}); "
                                  "nothing was kept") from None
            raise
        put(atlassian_key(where.split("://", 1)[-1]), {"email": email, "token": token})
        return f"signed in to {where} as {name}; kept in the keychain"
    from .connectors import sql

    password = sql._PASSWORDS.get(where) or ask_secret(f"password for {where}: ")
    sql.check(where, password)
    put(sql_key(where), {"password": password})
    return f"connected to {where}; password kept in the keychain"


def logout(url: str) -> str:
    kind, where = _place(url)
    if kind == "github":
        return f"Inventio keeps no GitHub token; `gh auth logout --hostname {where}` signs the CLI out"
    key = atlassian_key(where.split("://", 1)[-1]) if kind == "atlassian" else sql_key(where)
    return f"removed the login for {where}" if delete(key) else f"no login kept for {where}"
