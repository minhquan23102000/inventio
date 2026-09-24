"""Writing the Markdown a mirror holds, the same for every connector.

The files are read by Inventio's own Markdown chunker (ingest.chunk_markdown): a `#` heading
starts a chunk, a link `[text](relative/path.md#anchor)` becomes a `citation`, and a chunk is at
most ingest.MAX_CHARS. What is written here is shaped for that reader."""

import re
import unicodedata

from ..ingest import FENCE, HEADING, MAX_CHARS

UNSAFE = re.compile(r"[^\w.\-]+")  # no spaces or brackets: the link syntax the chunker reads allows neither
RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def segment(title: str, limit: int = 60) -> str:
    """One path segment from a title, legal on every file system; letters of any script kept."""
    s = UNSAFE.sub("-", unicodedata.normalize("NFC", title))
    s = re.sub(r"[-.]*-[-.]*", "-", s).strip("-.")[:limit].rstrip("-.") or "untitled"
    return "_" + s if s.split(".")[0].lower() in RESERVED else s


def safe(line: str) -> str:
    """A line of body text that the chunker must not read as a heading or a code fence."""
    return "\\" + line if HEADING.match(line) or FENCE.match(line) else line


def fence(code: str, lang: str = "") -> list[str]:
    ticks = "`" * max(3, max((len(m) for m in re.findall(r"`+", code)), default=0) + 1)
    return [f"{ticks}{lang}", *code.rstrip("\n").split("\n"), ticks, ""]


def cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    """A pipe table, cut into several when long, each repeating the header: a chunk that holds
    the middle of a table still says what its columns mean."""
    width = max([len(header)] + [len(r) for r in rows])
    line = lambda r: "| " + " | ".join(cell(c) for c in r + [""] * (width - len(r))) + " |"  # noqa: E731
    head = [line(header), "|" + "---|" * width]
    out, size = list(head), 0
    for r in rows:
        s = line(r)
        if size and size + len(s) > MAX_CHARS // 2:
            out += ["", *head]
            size = 0
        out.append(s)
        size += len(s) + 1
    return out + [""]


def tidy(lines: list[str]) -> str:
    """Join, strip trailing spaces, keep at most one blank line in a row outside code fences."""
    out: list[str] = []
    in_fence = False
    for l in lines:
        l = l.rstrip()
        if FENCE.match(l):
            in_fence = not in_fence
        if l or in_fence or (out and out[-1]):
            out.append(l)
    return "\n".join(out).strip("\n") + "\n"
