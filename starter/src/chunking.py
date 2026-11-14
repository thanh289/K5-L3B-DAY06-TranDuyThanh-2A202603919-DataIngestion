"""Step 5: deterministic, page-scoped chunking

A: fixed-size character windows with overlap.
B/C: paragraph-aware grouping with a maximum character length.

B and C must use IDENTICAL chunks; filtering is added only at retrieval time.
"""

import re

from .models import Chunk, DocumentPage


def _chunk(
    page: DocumentPage,
    text: str,
    index: int,
    method: str,
    settings: str,
    **extra_metadata: int | str,
) -> Chunk:
    """Preserve source provenance; never mutate metadata on the input page."""
    metadata = dict(page.metadata)
    metadata.update(
        {
            "chunk_method": method,
            "chunk_index": index,
            **extra_metadata,
        }
    )
    return Chunk(
        chunk_id=f"{page.document_id}:p{page.page:03d}:{method}:{settings}:{index:04d}",
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=metadata,
    )


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Use sliding character windows; final window is not padded or duplicated.

    Overlap is the count of characters repeated at the boundary of adjacent
    chunks from the same page. Each page is processed independently.
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    text = page.text or ""
    if not text.strip():
        return []

    result: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        result.append(
            _chunk(
                page, text[start:end], len(result), "fixed", f"{size}-{overlap}",
                char_start=start, char_end=end, chunk_size=size, overlap=overlap,
            )
        )
        if end == len(text):
            break
        start = end - overlap
    return result


def _starts_new_unit(line: str, previous: str) -> bool:
    """
    pypdf frequently emits a newline on EVERY visual line rather than on
    paragraph boundaries. Treat explicit bullets, headings, and likely sentence
    endings as natural breaks, rather than treating every PDF line as a paragraph.

    Đầu vào gồm dòng hiện tại và dòng trước đó. 
    Hàm trả về True nếu phát hiện dấu hiệu nên bắt đầu một đơn vị văn bản mới.
    """
    if re.match(r"^[\u2022\u25cf\u25aa\u25e6]\s*", line):
        return True
    if line.isupper() and len(line) <= 100 and any(c.isalpha() for c in line):
        return True
    if re.match(r"^(?:FIGURE|TABLE|BOX)\s+\d", line, flags=re.I):
        return True
    # A full stop at the end of a PDF text line is a useful, but imperfect,
    # hint that the following capitalized line begins a new sentence/paragraph.
    return bool(
        re.search(r"[.!?][\"'\u201d\u2019)?]*$", previous)
        and re.match(r"^[\u2022\u25cf\u25aa\u25e6A-Z]", line)
    )


def _paragraph_units(text: str) -> list[str]:
    """Gom các dòng thành đơn vị văn bản"""
    units: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            units.append(" ".join(current))
            current.clear()

    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            flush()
            continue
        if current and _starts_new_unit(line, current[-1]):
            flush()
        current.append(line)
    flush()
    return units


def _split_oversize(text: str, limit: int) -> list[str]:
    """Split a long paragraph at sentence/word boundaries where possible."""
    remaining = text
    pieces: list[str] = []
    while len(remaining) > limit:
        prefix = remaining[:limit + 1]
        sentence_cuts = [m.end() for m in re.finditer(r"[.!?][\"'\u201d\u2019)?]*\s+", prefix)]
        # Avoid extremely tiny chunks when a natural boundary occurs very early.
        cut = max((c for c in sentence_cuts if c >= limit // 2), default=0)
        if not cut:
            cut = prefix.rfind(" ", 0, limit + 1)
        if cut <= 0:
            cut = limit  # Rare single token longer than the limit.
        piece = remaining[:cut].strip()
        if piece:
            pieces.append(piece)
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group paragraph/sentence units up to max_size without overlap.

    Paragraph boundaries in a PDF are inferred, not guaranteed; the output
    preserves the textual order, source page and verified section metadata.

    DocumentPage -> _paragraph_units -> _split_oversize -> _chunk
    """
    if max_size < 1:
        raise ValueError("max_size must be >= 1")
    if not (page.text or "").strip():
        return []

    units = _paragraph_units(page.text)
    parts: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            parts.append(current)
            current = ""

    for unit in units:
        if len(unit) > max_size:
            flush()
            parts.extend(_split_oversize(unit, max_size))
        elif not current:
            current = unit
        elif len(current) + 1 + len(unit) <= max_size:
            current += " " + unit
        else:
            flush()
            current = unit
    flush()

    return [
        _chunk(
            page, part, index, "paragraph", str(max_size), max_size=max_size
        )
        for index, part in enumerate(parts)
    ]
