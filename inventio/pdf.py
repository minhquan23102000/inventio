"""PDF files as the map holds them: the text of each page under a `## Page N` heading, so a result
says which page it came from and `read` shows the words the search matched.

The text comes from the PDF's own text layer (pypdf). A scanned PDF has none and is left out of
the map, as a binary file is. Columns and tables come out in the order the PDF stores its
text, which is usually reading order and sometimes not."""

import io
import logging
import re
import sys

from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)  # font and encoding notes, not the reader's concern
SPACES = re.compile(r"[ \t\u00a0]+")


def _skip(name: str, why: str) -> None:
    print(f"inventio: {name}.pdf left out: {why}", file=sys.stderr)


def pdf_text(data: bytes, name: str) -> str | None:
    """Markdown of a PDF's text, a heading per page; None when nothing can be read from it."""
    from .connectors.markdown import safe

    try:
        pdf = PdfReader(io.BytesIO(data))
        if pdf.is_encrypted and not pdf.decrypt(""):  # many are "encrypted" with an empty password only
            _skip(name, "it needs a password")
            return None
        pages = []
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            lines = [safe(SPACES.sub(" ", l).strip()) for l in text.replace("\r", "\n").split("\n")]
            body = "\n".join(lines).strip()
            if body:
                pages.append(f"## Page {i}\n\n{body}\n")
    except Exception as e:  # noqa: BLE001 - a damaged PDF is left out like a binary, with a word
        _skip(name, f"it cannot be read ({type(e).__name__})")
        return None
    if not pages:
        _skip(name, "it has no text layer (a scan?)")
        return None
    # the file name, not the PDF's Title field: that is often "about:blank" (a page printed from a
    # browser) or "Microsoft Word - draft.docx"
    return f"# {name}\n\n" + "\n".join(pages)
