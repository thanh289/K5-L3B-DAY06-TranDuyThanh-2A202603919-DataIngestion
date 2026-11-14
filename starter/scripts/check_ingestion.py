"""Run from starter/: python -m scripts.check_ingestion"""

from collections import Counter
from pathlib import Path

from src.ingestion import load_corpus_from_manifest


def main() -> None:
    manifest_path = Path(__file__).resolve().parents[2] / "data" / "manifest.json"
    pages = load_corpus_from_manifest(manifest_path)

    section_counts = Counter(page.metadata["section"] for page in pages)
    print("Ingestion OK")
    print(f"Selected content pages: {len(pages)}")
    print(f"Chapters: {dict(section_counts)}")
    print(f"Physical PDF page range: {pages[0].page} ... {pages[-1].page}")
    print(f"Normalized text characters (pypdf, including line breaks): {sum(len(p.text) for p in pages):,}")
    print(f"Pages with empty text: {sum(not p.text for p in pages)}")
    print("Example metadata:", pages[0].metadata)
    print("Example text:", pages[0].text[:160].replace("\n", " "))
    assert len(pages) == 37
    assert section_counts == {"chapter_1": 17, "chapter_2": 20}


if __name__ == "__main__":
    main()
