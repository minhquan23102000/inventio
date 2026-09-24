"""Atlassian Cloud, shared by the Confluence and Jira connectors: which site a URL belongs to,
and a client that signs in as the person running Inventio.

Credentials come from the environment: ATLASSIAN_EMAIL and ATLASSIAN_API_TOKEN (a token made at
id.atlassian.com). Every request runs as that person, so a mirror holds only what they may read.
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


def client(site_url: str) -> Client:
    email, token = os.environ.get("ATLASSIAN_EMAIL"), os.environ.get("ATLASSIAN_API_TOKEN")
    if not email or not token:
        raise RemoteError("set ATLASSIAN_EMAIL and ATLASSIAN_API_TOKEN (a token from "
                          "https://id.atlassian.com/manage-profile/security/api-tokens)")
    auth = base64.b64encode(f"{email}:{token}".encode()).decode()
    return Client(site_url, {"Authorization": f"Basic {auth}"})
