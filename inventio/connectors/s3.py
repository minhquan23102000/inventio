"""Datasets in S3 (or any S3-compatible store): one card per dataset (schema.py).

    inventio init s3://lake/warehouse/                                  -> source s3-lake

A dataset is a folder of data files (Parquet, CSV, JSON Lines); Hive-style folders below it
(`dt=2026-09-24/`) are its partitions, not datasets of their own. Its schema is read by DuckDB
from the newest file: Parquet from the footer, CSV and JSON from a sample, never a full scan.
Credentials and endpoint come from the usual AWS_* variables (AWS_ENDPOINT_URL for MinIO and
the like). Needs boto3 and duckdb (`pip install "inventio[data]"`)."""

import hashlib
import json
import posixpath

from . import mirror
from .markdown import segment

KIND = "s3"
DOC_TYPE = "Dataset"


def origin(url: str, query: str | None = None) -> str | None:
    if not url.startswith("s3://"):
        return None
    if query:
        raise ValueError("--jql applies to a Jira URL")
    bucket, _, prefix = url[5:].partition("/")
    return f"s3://{bucket}/{prefix.strip('/') + '/' if prefix.strip('/') else ''}"


def locate(url: str):
    return None


def heading_url(item: dict, heads: list[str]) -> str | None:
    return None


def fetch_url(url: str):
    return None


def dataset_of(key: str) -> tuple[str, list[str]] | None:
    """(dataset, partition keys) of an object key; None for a file that is not data."""
    from ..schema import DATA_FORMATS

    if posixpath.splitext(key)[1].lower() not in DATA_FORMATS:
        return None
    dirs = key.split("/")[:-1]
    cut = next((i for i, d in enumerate(dirs) if "=" in d), len(dirs))
    return ("/".join(dirs[:cut]) or key), [d.split("=", 1)[0] for d in dirs[cut:] if "=" in d]


class Remote:
    FORMAT = 1  # raise when the Markdown written changes, so every mirror is written again

    def __init__(self, origin_url: str):
        self.origin = origin_url
        self.bucket, _, self.prefix = origin_url[5:].partition("/")
        self.name = f"s3-{self.bucket}"
        self.docs: dict[str, str] = {}

    def listing(self) -> dict[str, mirror.Entry]:
        from .http import RemoteError

        try:
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError

            from ..schema import DATA_FORMATS, NAME, card, describe, duck
        except ImportError:
            raise RemoteError("reading S3 datasets needs boto3 and duckdb: pip install \"inventio[data]\"") from None
        groups: dict[str, dict] = {}
        try:
            pages = boto3.client("s3").get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=self.prefix)
            for page in pages:
                for obj in page.get("Contents", []):
                    hit = dataset_of(obj["Key"][len(self.prefix):])
                    if hit is None:
                        continue
                    g = groups.setdefault(hit[0], {"keys": [], "newest": obj})
                    if len(hit[1]) > len(g["keys"]):  # the deepest partition path names every key
                        g["keys"] = hit[1]
                    if obj["LastModified"] > g["newest"]["LastModified"]:
                        g["newest"] = obj
        except (BotoCoreError, ClientError) as e:
            raise RemoteError(f"cannot list {self.origin}: {e}") from None
        con = duck()
        out = {}
        for ds, g in sorted(groups.items()):
            key = g["newest"]["Key"]
            fmt = DATA_FORMATS[posixpath.splitext(key)[1].lower()]
            try:
                cols, _ = describe(con, f"s3://{self.bucket}/{key}", fmt, hive=bool(g["keys"]))
            except Exception as e:  # noqa: BLE001 - one unreadable file must not hide the others
                cols = [["(unreadable)", str(e).splitlines()[0][:200]]]
            where = f"s3://{self.bucket}/{self.prefix}{ds}" + ("/" if ds != key[len(self.prefix):] else "")
            about = [f"{fmt.upper() if fmt != 'json' else 'JSON Lines'} dataset in S3 at `{where}`."]
            if g["keys"]:
                about.append(f"Partitioned by `{', '.join(g['keys'])}` (Hive-style folders).")
            about.append(f"Schema read from `{key}`.")
            leaf = ds.rstrip("/").split("/")[-1].split(".")[0]
            self.docs[ds] = card(where, about, ["Column", "Type"], cols, defines=[leaf] if NAME.match(leaf) else [])
            version = hashlib.sha1(json.dumps([fmt, g["keys"], cols]).encode()).hexdigest()
            path = "/".join(segment(p) for p in ds.split("/")) + ".md"
            out[ds] = mirror.Entry(version, path, "")
        return out

    def fetch(self, ids: list[str]):
        for i in ids:
            yield i, mirror.Doc(self.docs[i])
