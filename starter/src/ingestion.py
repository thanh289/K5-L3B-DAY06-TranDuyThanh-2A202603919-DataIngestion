"""Read and normalize source data. TODO: implement functions."""
import json
import logging
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage

logger = logging.getLogger(__name__)


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """TODO: extract each physical page; keep its page number and source.

    Detect and record pages with no text. OCR, when needed, is a separate step.
    """
    """Read every physical PDF page; numbers are 1-based (not book labels)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    for page_number, pdf_page in enumerate(reader.pages, start=1):
        extraction_error = None
        try:
            text = normalize_text(pdf_page.extract_text() or "")
        except Exception as exc:
            extraction_error = str(exc)
            text = ""
            logger.warning("Failed to read PDF page %d: %s", page_number, exc)

        if not text:
            logger.warning("No extractable text on physical PDF page %d", page_number)

        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata={
                    "is_empty": not bool(text),
                    "extraction_error": extraction_error,
                    "char_count": len(text),
                },
            )
        )
    return pages


def normalize_text(text: str) -> str:
    """TODO: normalize Unicode and whitespace without discarding useful structure."""
    """Normalize Unicode/spacing while keeping line breaks for later chunking."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u00ad", "")  # PDF soft hyphen
    text = text.replace("\u200b", "")  # zero-width space
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)  # whitespace except newlines
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_corpus_from_manifest(manifest_path: Path) -> list[DocumentPage]:
    """Use the verified 37-page scope and section labels in data/manifest.json."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_root = manifest_path.parent.parent  # repo/data/manifest.json -> repo/

    source = manifest["source_path_or_url"]
    pdf_path = repo_root / source
    selected = {entry["pdf_page"]: entry for entry in manifest["pages"]}
    if len(selected) != len(manifest["pages"]):
        raise ValueError("Duplicate physical page in manifest")

    all_pages = load_pdf(pdf_path, manifest["source_id"], source)
    if any(page_no < 1 or page_no > len(all_pages) for page_no in selected):
        raise ValueError("A selected page is outside the PDF page range")

    corpus: list[DocumentPage] = []
    for page in all_pages:
        if page.page not in selected:
            continue
        entry = selected[page.page]
        if not page.text:
            raise ValueError(f"Selected page {page.page} has no extractable text")
        page.metadata.update(
            {
                "page_id": entry["page_id"],
                "printed_page": entry["printed_page"],
                "section": entry["section"],
                "section_title": entry["section_title"],
            }
        )
        corpus.append(page)

    expected = manifest["corpus_scope"]["content_page_count"]
    if len(corpus) != expected:
        raise ValueError(f"Expected {expected} content pages, got {len(corpus)}")
    return corpus