"""Offline Step 7 checks: strict scoring and the 20-question gold validation.

Run from starter/: python -m scripts.check_step7
No database/API calls are made.
"""

from src.evaluation import evaluate_query, summarize
from src.run import main as run_main


def _hit(page: int, text: str) -> dict:
    return {
        "chunk_id": f"fake:p{page}", "document_id": "test_doc", "source": "test.pdf",
        "page": page, "section": "chapter_1", "text": text,
        "score": 0.9, "metadata": {"section": "chapter_1"},
    }


def main() -> None:
    question = {
        "id": "fake-q", "split": "dev", "question": "Question?", "type": "direct",
        "expected_answer": "complete answer", "gold_source_id": "test_doc",
        "gold_page": 29, "evidence_quote": "The full evidence is here.",
    }
    good = evaluate_query(question, [_hit(29, "The full evidence is here.")])
    wrong_page = evaluate_query(question, [_hit(30, "The full evidence is here.")])
    partial = evaluate_query(question, [_hit(29, "The full evidence")])
    assert good["hit_at_1"] == good["evidence_hit_at_3"] == 1
    assert wrong_page["hit_at_1"] == wrong_page["evidence_hit_at_3"] == 0
    assert partial["gold_page_at_3_diagnostic"] == 1
    assert partial["evidence_hit_at_3"] == 0
    summary = summarize([{**good, "strategy": s} for s in ("A", "B", "C")])
    assert all(summary[s]["dev"]["hit_at_1"]["correct"] == 1 for s in ("A", "B", "C"))
    print("Strict evaluation unit checks OK")
    run_main(["--validate-only"])


if __name__ == "__main__":
    main()
