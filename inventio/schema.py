"""Where data lives, described by its schema and never by its rows.

A table, a Kafka topic, a dataset in S3 or a data file on disk becomes one card: a Markdown page
with its name as the title, where to connect, and its columns. Cards are indexed like any
Markdown (so BM25 finds a table by its column comments), their document type is `Dataset`, and
the facts step files them under the category `Reference` without asking a model. A card names
what it defines in a first-lines comment, `<!-- defines: shop.orders orders -->`, so
code, SQL, runbooks and tickets that name the table link to it (`mentions`); a foreign key is
written as a Markdown link to the other table's card (`citation`).

Files are read with DuckDB (optional: `pip install duckdb`): Parquet from its footer, CSV and
JSON Lines from a sample, local or in S3."""

import os
import re
from pathlib import Path

DATA_FORMATS = {".parquet": "parquet", ".csv": "csv", ".tsv": "csv", ".jsonl": "json", ".ndjson": "json"}
DEFINES = re.compile(r"^<!-- defines: (.+?) -->$", re.M)
NAME = re.compile(r"^[A-Za-z_][\w\-]{2,}$")  # a file stem worth defining: daily_orders, not part-0 or a hash


def card(title: str, about: list[str], header: list[str], rows: list[list[str]], more: list[str] = (),
         defines: list[str] = ()) -> str:
    """The Markdown of one card. One chunk when it fits; a wide table is cut by `table`, each
    piece repeating the header."""
    from .connectors.markdown import safe, table, tidy

    names = " ".join(dict.fromkeys(d for d in defines if d))
    head = [f"# {title}", f"<!-- defines: {names} -->" if names else "", ""]
    return tidy([*head, *(safe(l) for l in about), "", *table(header, rows), *more])


def defined_names(text: str) -> set[str]:
    m = DEFINES.search(text[:2000])
    return set(m.group(1).split()) if m else set()


# ------------------------------------------------------------------------------------ data files

def duck():
    """A DuckDB connection; with AWS_* in the environment it reads s3:// the way boto3 does
    (AWS_ENDPOINT_URL for MinIO and other S3-compatible stores)."""
    import duckdb

    con = duckdb.connect()
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        con.execute("INSTALL httpfs; LOAD httpfs")
        opts = {"KEY_ID": os.environ["AWS_ACCESS_KEY_ID"], "SECRET": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
                "REGION": os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"}
        if os.environ.get("AWS_SESSION_TOKEN"):
            opts["SESSION_TOKEN"] = os.environ["AWS_SESSION_TOKEN"]
        endpoint = os.environ.get("AWS_ENDPOINT_URL")
        if endpoint:
            opts["ENDPOINT"] = endpoint.split("://", 1)[-1].rstrip("/")
            opts["URL_STYLE"] = "path"
            opts["USE_SSL"] = str(endpoint.startswith("https")).lower()
        body = ", ".join(f"{k} {v}" if k == "USE_SSL" else f"{k} '{v}'" for k, v in opts.items())
        con.execute(f"CREATE OR REPLACE SECRET inventio_s3 (TYPE s3, {body})")
    return con


def describe(con, location: str, fmt: str, hive: bool = False) -> tuple[list[list[str]], int | None]:
    """(columns as [name, type], rows or None) of a data file; Parquet's row count is in its
    footer, a CSV's would need a full scan and is left out."""
    reader = {"parquet": "read_parquet", "csv": "read_csv", "json": "read_json_auto"}[fmt]
    args = ", hive_partitioning = true" if hive and fmt == "parquet" else ""
    cols = [[r[0], r[1]] for r in con.execute(f"DESCRIBE SELECT * FROM {reader}(?{args})", [location]).fetchall()]
    rows = None
    if fmt == "parquet":
        rows = con.execute("SELECT sum(num_rows) FROM parquet_file_metadata(?)", [location]).fetchone()[0]
    return cols, rows


def file_card(path: Path, rel: str) -> str | None:
    """The card of a data file on disk, or None when DuckDB is missing or cannot read it."""
    fmt = DATA_FORMATS.get(path.suffix.lower())
    try:
        con = duck()
        cols, rows = describe(con, str(path), fmt)
    except Exception:  # noqa: BLE001 - no duckdb, or a file it cannot read: left out of the map like a binary
        return None
    size = path.stat().st_size
    about = [f"{fmt.upper() if fmt != 'json' else 'JSON Lines'} file on disk, {_bytes(size)}"
             + (f", {rows:,} rows." if rows is not None else ".")]
    stem = path.name.split(".")[0]
    return card(path.name, about, ["Column", "Type"], cols, defines=[stem] if NAME.match(stem) else [])


def _bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
