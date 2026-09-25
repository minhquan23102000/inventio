"""A query's pre-filter: the part of the map a search may look at, decided before BM25.

    -w "kind:jira -status:Done updated:>=-90d"
    -w "path:Ops/* type:Article,Dataset"

Written the way GitHub's search box is: `key:value` terms separated by spaces, all of which must
hold; `a,b` is either value; a leading `-` negates a term; `>=`, `>`, `<=`, `<` after the colon
compare (dates are ISO, `-90d` and `-2w` count back from today); `*` and `?` are wildcards;
quotes hold a value with spaces (`assignee:"Nguyen An"`). Values match without regard to case.

Keys every map has: source, kind (dir, confluence, jira, github, sql, kafka, s3), type (the
document type), lang, path, category (the kept content category, after `facts`). Any other key
is a field a connector wrote for its items (file_meta): Jira status, assignee, labels, updated;
Confluence labels, author, updated; GitHub state, author, labels, updated. A chunk whose file has
no such field does not match a term on it, and does match its negation.

Every path into the pool (BM25, named files and definitions, predicted types, categories,
neighbours, links) is narrowed by the same scope, so no candidate outside it is ever ranked.
"""

import datetime as _dt
import re
import shlex
from dataclasses import dataclass, field

BUILTIN = {"source": "s.name", "kind": "s.kind", "type": "f.type", "lang": "f.lang", "path": "f.path",
           "category": None}
OPS = (">=", "<=", ">", "<")
KEY = re.compile(r"^[a-z_][\w.\-]*$")
RELATIVE = re.compile(r"^-(\d+)([dw])$")


@dataclass
class Term:
    key: str
    values: list[str]
    op: str = "="
    negate: bool = False


@dataclass
class Scope:
    terms: list[Term] = field(default_factory=list)
    text: str = ""

    @classmethod
    def parse(cls, text: str | None, sources: list[str] | None = None) -> "Scope":
        """`sources` (from --source) is the term `source:a,b`."""
        terms = []
        try:
            tokens = shlex.split(text or "")
        except ValueError as e:
            raise ValueError(f"filter {text!r}: {e}") from None
        for tok in tokens:
            negate = tok.startswith("-") and ":" in tok
            key, sep, value = tok[negate:].partition(":")
            key = key.lower()
            if not sep or not KEY.match(key) or value == "":
                raise ValueError(f"not a filter term: {tok!r} (write key:value, e.g. status:Done)")
            op = next((o for o in OPS if value.startswith(o)), "=")
            value = value[len(op):] if op != "=" else value
            values = [v for v in value.split(",") if v] if op == "=" else [_when(value)]
            if not values:
                raise ValueError(f"not a filter term: {tok!r}")
            if op != "=" and key in BUILTIN:
                raise ValueError(f"{key} is compared by value, not by {op}")
            terms.append(Term(key, values, op, negate))
        if sources:
            terms.append(Term("source", list(sources)))
        return cls(terms, " ".join(filter(None, [text or "", f"source:{','.join(sources)}" if sources else ""])))

    def __bool__(self) -> bool:
        return bool(self.terms)

    @classmethod
    def only(cls, sources: list[str]) -> "Scope":
        """The scope of some sources, as `--source` gives it."""
        return cls.parse(None, sources)


    def check(self, con) -> None:
        """Every key names something this map holds; a misspelt key would silently match nothing."""
        known = set(BUILTIN) | {r[0] for r in con.execute("SELECT DISTINCT key FROM file_meta")}
        unknown = sorted({t.key for t in self.terms} - known)
        if unknown:
            raise ValueError(f"no filter key {', '.join(unknown)} in this map; keys: {', '.join(sorted(known))}")

    def sql(self, chunks: bool = True) -> tuple[str, list]:
        """` AND ...` over the aliases s (sources), f (files) and, when `chunks`, c (chunks)."""
        where, args = "", []
        for t in self.terms:
            clause, a = _clause(t, chunks)
            where += f" AND {'NOT ' if t.negate else ''}({clause})"
            args += a
        return where, args


def where(scope: "Scope | None", chunks: bool = True) -> tuple[str, list]:
    return scope.sql(chunks) if scope else ("", [])


def count(con, scope: "Scope | None") -> int:
    """Chunks inside the scope."""
    w, a = where(scope)
    return con.execute("SELECT count(*) FROM chunks c JOIN files f ON f.id = c.file_id "
                       f"JOIN sources s ON s.id = f.source_id WHERE 1 = 1{w}", a).fetchone()[0]


def _when(value: str) -> str:
    m = RELATIVE.match(value)
    if not m:
        return value
    days = int(m.group(1)) * (7 if m.group(2) == "w" else 1)
    return (_dt.date.today() - _dt.timedelta(days=days)).isoformat()


def _match(col: str, values: list[str]) -> tuple[str, list]:
    """`col` equal to (or, with wildcards, like) any of the values, ignoring case."""
    parts, args = [], []
    for v in values:
        parts.append(f"lower({col}) GLOB lower(?)" if any(ch in v for ch in "*?[") else f"lower({col}) = lower(?)")
        args.append(v)
    return " OR ".join(parts), args


def _clause(t: Term, chunks: bool) -> tuple[str, list]:
    if t.key == "category":
        m, a = _match("cc.category", t.values)
        kept = f"SELECT cc.chunk_id FROM chunk_categories cc WHERE cc.kept = 1 AND ({m})"
        if chunks:
            return f"c.id IN ({kept})", a
        return f"f.id IN (SELECT c2.file_id FROM chunks c2 WHERE c2.id IN ({kept}))", a
    if t.key in BUILTIN:
        return _match(f"coalesce({BUILTIN[t.key]}, '')", t.values)
    if t.op == "=":
        m, a = _match("fm.value", t.values)
    else:
        m, a = f"fm.value {t.op} ?", list(t.values)
    return f"f.id IN (SELECT fm.file_id FROM file_meta fm WHERE fm.key = ? AND ({m}))", [t.key, *a]
