"""Atlassian Cloud, shared by the Confluence and Jira connectors: which site a URL belongs to,
and a client that signs in as the person running Inventio.

The email and API token (made at id.atlassian.com) come from the environment (ATLASSIAN_EMAIL,
ATLASSIAN_API_TOKEN) or from the keychain, where `inventio login <site>` keeps them per site
(credentials.py). Every request runs as that person, so a mirror holds only what they may read.
ATLASSIAN_BASE_URL names a site that is not on atlassian.net."""

import base64
import os
import urllib.parse

from .http import Client, RemoteError


def site(url: str) -> str | None:
    """`https://<host>` when the URL is on an Atlassian site, else None."""
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        return None
    own = urllib.parse.urlparse(os.environ.get("ATLASSIAN_BASE_URL", "")).hostname
    if u.hostname.endswith(".atlassian.net") or u.hostname == own:
        return f"https://{u.hostname}"
    return None


_checked: set[str] = set()  # sites whose login was tried in this process


def client(site_url: str) -> Client:
    """A client signed in as the person, tried once per process. Atlassian answers a refused
    token on most endpoints as an anonymous visitor who may see nothing, so an expired token
    would read as an empty space, and a sync would then delete the whole mirror."""
    from ..credentials import atlassian

    api = signed(site_url, *atlassian(site_url))
    if site_url not in _checked:
        try:
            whoami(api)
        except RemoteError as e:
            if str(e).startswith(("401", "403")):
                raise RemoteError(f"{site_url} refused the Atlassian login ({str(e).split(' from ')[0]}): the token "
                                  f"may have expired or been revoked; `inventio login {site_url}` keeps a new one") from None
            raise
        _checked.add(site_url)
    return api


def signed(site_url: str, email: str, token: str) -> Client:
    auth = base64.b64encode(f"{email}:{token}".encode()).decode()
    return Client(site_url, {"Authorization": f"Basic {auth}"})


def whoami(api: Client) -> str:
    """The signed-in person's name: Jira's account endpoint, or Confluence's on a wiki-only site."""
    try:
        me = api.get("/rest/api/3/myself")
    except RemoteError as e:
        if not str(e).startswith("404"):
            raise
        me = api.get("/wiki/rest/api/user/current")
    return me.get("displayName") or me.get("publicName") or me.get("emailAddress") or "?"
