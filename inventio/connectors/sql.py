"""A SQL database's schema: one card per table and view (schema.py), read through SQLAlchemy's
inspector, so Postgres, MySQL, SQL Server, Oracle, SQLite and the rest share one path.

    inventio init postgresql://reader@db.internal:5432/core      -> source postgresql-core

Only the catalog is read: columns, keys, indexes, comments, view definitions; never a row. The
password is taken from the URL for that run or from the driver's own variable (PGPASSWORD), or
INVENTIO_SQL_PASSWORD; it is never written to the map. A foreign key becomes a link from the
card of the table that holds it to the card of the table it references."""

import hashlib
import os
import posixpath

from . import mirror
from .markdown import fence, segment

KIND = "sql"
DOC_TYPE = "Dataset"
BACKENDS = {"postgresql", "postgres", "mysql", "mariadb", "mssql", "oracle", "sqlite", "redshift", "snowflake"}
SYSTEM_SCHEMAS = {"information_schema", "pg_catalog", "pg_toast", "mysql", "performance_schema", "sys"}
_PASSWORDS: dict[str, str] = {}  # origin -> password given on the command line, for this process only


def _url(text: str):
    try:
        from sqlalchemy.engine import make_url
    except ImportError:
        raise ValueError("reading a database schema needs SQLAlchemy and the database's driver: "
                         "pip install \"inventio[data]\"") from None
    return make_url(text)


def origin(url: str, query: str | None = None) -> str | None:
    if "://" not in url or url.split("://", 1)[0].split("+")[0].lower() not in BACKENDS:
        return None
    if query:
        raise ValueError("--jql applies to a Jira URL")
    u = _url(url)
    if u.drivername == "postgres":
        u = u.set(drivername="postgresql")
    clean = u._replace(password=None).render_as_string(hide_password=False)  # URL.set ignores None
    if u.password:
        _PASSWORDS[clean] = u.password
    return clean


def locate(url: str):
    return None


def heading_url(item: dict, heads: list[str]) -> str | None:
    return None


def fetch_url(url: str):
    return None


class Remote:
    FORMAT = 1  # raise when the Markdown written changes, so every mirror is written again

    def __init__(self, origin_url: str):
        self.origin = origin_url
        u = _url(origin_url)
        db = posixpath.basename((u.database or "").replace("\\", "/")).split(".")[0]
        self.name = f"{u.get_backend_name()}-{db or u.host or 'db'}"
        password = _PASSWORDS.get(origin_url) or os.environ.get("INVENTIO_SQL_PASSWORD")
        self.url = u.set(password=password) if password else u
        self.docs: dict[str, str] = {}

    def listing(self) -> dict[str, mirror.Entry]:
        from sqlalchemy import create_engine, inspect
        from sqlalchemy.exc import SQLAlchemyError

        from .http import RemoteError

        try:
            engine = create_engine(self.url)
            with engine.connect() as conn:
                tables = _catalog(inspect(conn))
        except SQLAlchemyError as e:
            raise RemoteError(f"cannot read the schema of {self.origin}: {str(e).splitlines()[0]}") from None
        finally:
            if "engine" in locals():
                engine.dispose()
        out = {}
        for tid, t in tables.items():
            text = _card(self.origin, t, tables)
            self.docs[tid] = text
            out[tid] = mirror.Entry(hashlib.sha1(text.encode()).hexdigest(), t["path"], "")
        return out

    def fetch(self, ids: list[str]):
        for i in ids:
            yield i, mirror.Doc(self.docs[i])


def _catalog(insp) -> dict[str, dict]:
    """Every table and view outside the system schemas, with what the card needs."""
    default = insp.default_schema_name
    tables: dict[str, dict] = {}
    for schema in insp.get_schema_names():
        if schema in SYSTEM_SCHEMAS or schema.startswith("pg_"):
            continue
        views = set(insp.get_view_names(schema=schema))
        for name in [*insp.get_table_names(schema=schema), *sorted(views)]:
            t = {"schema": schema, "name": name, "view": name in views, "default": schema == default,
                 "path": f"{segment(schema)}/{segment(name)}.md",
                 "columns": insp.get_columns(name, schema=schema)}
            t["pk"] = (insp.get_pk_constraint(name, schema=schema) or {}).get("constrained_columns") or []
            t["fks"] = insp.get_foreign_keys(name, schema=schema)
            t["indexes"] = insp.get_indexes(name, schema=schema)
            t["comment"] = _optional(lambda: insp.get_table_comment(name, schema=schema).get("text"))
            t["definition"] = _optional(lambda: insp.get_view_definition(name, schema=schema)) if t["view"] else None
            tables[f"{schema}.{name}"] = t
    for t in tables.values():  # the referenced side, and the schema a foreign key omits (the default one)
        for fk in t["fks"]:
            fk["referred_schema"] = fk.get("referred_schema") or default
            parent = tables.get(f"{fk['referred_schema']}.{fk['referred_table']}")
            if parent is not None:
                parent.setdefault("referenced_by", []).append((t, fk))
    return tables


def _optional(get):
    try:
        return get()
    except NotImplementedError:  # a dialect without table comments (SQLite)
        return None


def _title(t: dict) -> str:
    return t["name"] if t["default"] else f"{t['schema']}.{t['name']}"


def _link(frm: dict, to: dict) -> str:
    rel = posixpath.relpath(to["path"], posixpath.dirname(frm["path"]))
    return f"[{_title(to)}]({rel})"


def _card(origin: str, t: dict, tables: dict) -> str:
    from ..schema import card

    about = [f"{'View' if t['view'] else 'Table'} in `{origin}`, schema `{t['schema']}`."]
    if t["comment"]:
        about += ["", t["comment"]]
    rows = [[c["name"], _type(c["type"]), "yes" if c.get("nullable", True) else "no",
             str(c["default"]) if c.get("default") is not None else "", c.get("comment") or ""]
            for c in t["columns"]]
    more: list[str] = []
    if t["pk"]:
        more += [f"**Primary key:** `{', '.join(t['pk'])}`", ""]
    if t["fks"]:
        more.append("**References:**")
        for fk in t["fks"]:
            parent = tables.get(f"{fk['referred_schema']}.{fk['referred_table']}")
            target = _link(t, parent) if parent else f"`{fk['referred_schema']}.{fk['referred_table']}`"
            more.append(f"- `{', '.join(fk['constrained_columns'])}` → {target} (`{', '.join(fk['referred_columns'])}`)")
        more.append("")
    if t.get("referenced_by"):
        more.append("**Referenced by:**")
        more += [f"- {_link(t, child)} (`{', '.join(fk['constrained_columns'])}`)" for child, fk in t["referenced_by"]]
        more.append("")
    if t["indexes"]:
        more.append("**Indexes:**")
        more += [f"- `{i['name']}` on `{', '.join(c for c in i['column_names'] if c)}`" + (" (unique)" if i.get("unique") else "")
                 for i in t["indexes"]]
        more.append("")
    if t["definition"]:
        more += ["**Definition:**", *fence(t["definition"].strip(), "sql")]
    header = ["Column", "Type", "Null", "Default", "Comment"]
    return card(_title(t), about, header, rows, more, defines=[f"{t['schema']}.{t['name']}", t["name"]])


def _type(t) -> str:
    """A column type as the database names it; SQLAlchemy prints timestamptz as plain TIMESTAMP."""
    s = str(t)
    return f"{s} WITH TIME ZONE" if getattr(t, "timezone", False) and "ZONE" not in s else s
