"""Run from starter/: python -m scripts.check_chunking"""

from collections import Counter
from pathlib import Path
from statistics import mean

from src.chunking import by_paragraph, fixed_size
from src.ingestion import load_corpus_from_manifest

FIXED_SIZE = 500
FIXED_OVERLAP = 50
PARAGRAPH_MAX_SIZE = 500


def main() -> None:
    manifest_path = Path(__file__).resolve().parents[2] / "data" / "manifest.json"
    pages = load_corpus_from_manifest(manifest_path)
    chunks_a = [c for p in pages for c in fixed_size(p, FIXED_SIZE, FIXED_OVERLAP)]
    chunks_b = [c for p in pages for c in by_paragraph(p, PARAGRAPH_MAX_SIZE)]

    assert len(pages) == 37
    assert chunks_a and chunks_b
    assert len({c.chunk_id for c in chunks_a}) == len(chunks_a)
    assert len({c.chunk_id for c in chunks_b}) == len(chunks_b)
    for chunks, max_len in ((chunks_a, FIXED_SIZE), (chunks_b, PARAGRAPH_MAX_SIZE)):
        assert all(c.text and len(c.text) <= max_len for c in chunks)
        assert all(c.document_id == "digital_progress_2023" for c in chunks)
        assert all(c.source.endswith("digital_progress_2023.pdf") for c in chunks)
        assert all(c.metadata.get("section") in {"chapter_1", "chapter_2"} for c in chunks)
        assert all(c.metadata.get("page_id") for c in chunks)

    for page in pages:
        page_a = [c for c in chunks_a if c.page == page.page]
        page_b = [c for c in chunks_b if c.page == page.page]
        assert page_a and page_b
        # Fixed windows reassemble exactly when overlapping prefixes are omitted.
        restored_a = page_a[0].text + "".join(
            chunk.text[FIXED_OVERLAP:] for chunk in page_a[1:]
        )
        assert restored_a == page.text, f"Fixed-size content mismatch on PDF page {page.page}"
        # Paragraph-aware extraction rejoins visual line breaks, not content.
        canonical = " ".join(page.text.split())
        restored_b = " ".join(c.text for c in page_b)
        assert restored_b == canonical, f"Paragraph content mismatch on PDF page {page.page}"

    print("Chunking OK")
    print(f"Content pages: {len(pages)}")
    for name, chunks in (("A (fixed, size=500, overlap=50)", chunks_a),
                         ("B (paragraph, max_size=500)", chunks_b)):
        lengths = [len(c.text) for c in chunks]
        sections = Counter(c.metadata["section"] for c in chunks)
        print(f"{name}: {len(chunks)} chunks; min/avg/max chars "
              f"{min(lengths)}/{mean(lengths):.1f}/{max(lengths)}; sections={dict(sections)}")
    print("C: same chunks as B (metadata filter is applied during retrieval, not here)")
    print("Example A:", chunks_a[0].chunk_id, repr(chunks_a[0].text[:120]))
    print("Example B:", chunks_b[0].chunk_id, repr(chunks_b[0].text[:120]))
    print("Example metadata:", chunks_b[0].metadata)


if __name__ == "__main__":
    main()
