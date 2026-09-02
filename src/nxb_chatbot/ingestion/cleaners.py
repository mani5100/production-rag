"""
ingestion/cleaners.py

Content normalization for chunks extracted by PyMuPDF4LLMLoader.

PyMuPDF4LLMLoader outputs markdown (bold/headers), and some source
PDFs use fonts with broken ToUnicode CMaps that surface as U+FFFD
('\ufffd') in extracted text. Neither of these is fit to feed
directly into embeddings or a cross-encoder reranker, so every chunk
should be passed through the relevant cleaner(s) here before being
stored in Qdrant.

Two entry points, used depending on segment type (see splitters.py):
    - clean_prose(text)   -> for regular text segments
    - flatten_table(text) -> for markdown pipe-table segments

Both internally call repair_unicode() first.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unicode repair
# ---------------------------------------------------------------------------

_REPLACEMENT_CHAR_RE = re.compile(r"\ufffd")


def repair_unicode(text: str) -> str:
    """
    Removes the Unicode replacement character (U+FFFD / '�') produced
    by broken ToUnicode CMaps in certain subset/embedded PDF fonts
    (confirmed via diagnostic: font 'AECHEL+EurostileCandyLTPro-SmBd'
    on header text, WinAnsiEncoding, extracts as raw \\x01 control
    bytes which PyMuPDF4LLM sanitizes to U+FFFD).

    We deliberately do NOT try to guess the original punctuation
    (hyphen vs parenthesis vs other) since different fonts on the
    same page mangle different glyphs into the same placeholder, and
    a wrong guess is worse than no guess. Stripping to whitespace
    keeps the surrounding words/numbers intact and readable, which
    is what matters for embedding and reranking.
    """
    if not text or "\ufffd" not in text:
        return text

    cleaned = _REPLACEMENT_CHAR_RE.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ---------------------------------------------------------------------------
# Prose cleaning
# ---------------------------------------------------------------------------

# Markdown headers: "#", "##", "###" ... at line start
_MD_HEADER_RE = re.compile(r"^#{1,6}\s*", flags=re.MULTILINE)

# Markdown bold/italic markers: **text** or *text* or __text__ or _text_
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_MD_UNDERSCORE_BOLD_RE = re.compile(r"__(.+?)__")

# HTML tags we expect from PyMuPDF4LLM output: <mark>, <u>, <br>, etc.
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")

# Collapse 3+ blank lines down to 2, and trim trailing whitespace per line
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+\n")


def clean_prose(text: str) -> str:
    """
    Strip markdown emphasis and HTML tags from a prose text block,
    while preserving the actual words, list markers ("-", "1.", etc.)
    and paragraph structure.

    Applied to non-table segments before chunking.
    """
    if not text:
        return text

    cleaned = repair_unicode(text)

    # Strip HTML tags first (e.g. <mark>1. Severe pain</mark> -> 1. Severe pain)
    cleaned = _HTML_TAG_RE.sub("", cleaned)

    # Strip markdown headers ("## Title" -> "Title")
    cleaned = _MD_HEADER_RE.sub("", cleaned)

    # Strip bold/italic markers, keep inner text
    cleaned = _MD_BOLD_RE.sub(r"\1", cleaned)
    cleaned = _MD_UNDERSCORE_BOLD_RE.sub(r"\1", cleaned)
    cleaned = _MD_ITALIC_RE.sub(r"\1", cleaned)

    # Normalize whitespace left behind by tag/marker removal
    cleaned = _MULTI_BLANK_RE.sub("\n\n", cleaned)
    cleaned = _TRAILING_WS_RE.sub("\n", cleaned)
    cleaned = cleaned.strip()

    return cleaned


# ---------------------------------------------------------------------------
# Table flattening
# ---------------------------------------------------------------------------

_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def flatten_table(text: str) -> str:
    """
    Convert a raw markdown pipe-table block into readable sentences,
    one per data row, using the header row as field labels.

    Falls back to a best-effort tag/pipe strip if the block doesn't
    parse as a clean table (e.g. merged/irregular rows, section
    headers embedded inside the table body).
    """
    if not text:
        return text

    text = repair_unicode(text)
    rows = [line for line in text.split("\n") if line.strip()]

    parsed_rows: list[list[str]] = []
    for row in rows:
        if _TABLE_SEPARATOR_RE.match(row):
            continue  # skip the |---|---| separator row

        cells = row.split("|")
        if cells and cells[0].strip() == "":
            cells = cells[1:]
        if cells and cells[-1].strip() == "":
            cells = cells[:-1]

        cleaned_cells = []
        for cell in cells:
            cell = repair_unicode(cell)
            cell = _HTML_TAG_RE.sub(", ", cell)  # <br> -> ", " inside a cell
            cell = _MD_BOLD_RE.sub(r"\1", cell)
            cell = _MD_UNDERSCORE_BOLD_RE.sub(r"\1", cell)
            cell = re.sub(r"\s+", " ", cell).strip(" ,")
            cleaned_cells.append(cell)

        if any(cleaned_cells):
            parsed_rows.append(cleaned_cells)

    if len(parsed_rows) < 2:
        # Not enough structure to treat as header + data; fall back to
        # a plain strip so we at least remove tags/pipes.
        logger.warning(
            "flatten_table: could not parse table structure, using fallback strip"
        )
        fallback = _HTML_TAG_RE.sub(" ", text)
        fallback = fallback.replace("|", " ")
        fallback = re.sub(r"\s+", " ", fallback).strip()
        return fallback

    header = parsed_rows[0]
    data_rows = parsed_rows[1:]

    sentences = []
    for row in data_rows:
        parts = []
        for label, value in zip(header, row):
            label = label.strip()
            value = value.strip()
            if not value:
                continue
            if label:
                parts.append(f"{label}: {value}")
            else:
                parts.append(value)
        if parts:
            sentences.append(". ".join(parts) + ".")

    return "\n".join(sentences)


# ---------------------------------------------------------------------------
# Boilerplate detection (used by splitters.py for the min-content filter)
# ---------------------------------------------------------------------------


def is_boilerplate(text: str, min_chars: int = 40) -> bool:
    """
    Flags near-empty chunks (e.g. a lone page header like
    "Nextbridge (Private) Limited" repeated across many pages)
    that are too short to carry standalone meaning and tend to
    crowd out genuinely relevant chunks during reranking.

    Applied AFTER cleaning, so length reflects real content,
    not markdown/HTML padding.
    """
    if not text:
        return True
    return len(text.strip()) < min_chars