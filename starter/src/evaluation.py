"""Step 7: strictly scored retrieval evidence and complete per-query traces.

This is an AUTOMATIC retrieval metric, not an LLM answer grade and not the
manual coherence/grounding evaluation required later in Step 8.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def _canonical(text: str) -> str:
    """Normalize the extraction's presentation differences for evidence matching.

    We do not paraphrase, infer, stem, or substitute gold evidence. Whitespace
    and unicode formatting only are ignored (as specified for the benchmark).
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u00ad", "").replace("\u200b", "")
    return re.sub(r"\s+", " ", text).strip().casefold()


def _gold_candidates(question: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept the starter's single gold and optional documented alternatives.

    Optional `gold_evidences` entries, when authored and verified BEFORE the
    experiment, can add legitimate alternative source/page/evidence tuples.
    They are never derived from search results.
    """
    candidates = [{
        "document_id": question["gold_source_id"],
        "page": question["gold_page"],
        "evidence_quote": question["evidence_quote"],
    }]
    for candidate in question.get("gold_evidences", []):
        candidates.append({
            "document_id": candidate["document_id"],
            "page": candidate["page"],
            "evidence_quote": candidate["evidence_quote"],
        })
    return candidates


def evaluate_query(
    question: dict[str, Any], hits: list[dict[str, Any]]
) -> dict[str, Any]:
    """Grade top-3: exact gold source/page AND full evidence in one chunk.

    Alternatives are accepted when predeclared. A human may later adjudicate
    semantically equivalent evidence; these exact-string metrics remain a
    reproducible, conservative automated baseline.
    """
    if len(hits) > 3:
        raise ValueError("evaluate_query expects at most top-3 retrieval hits")

    candidates = _gold_candidates(question)
    if not candidates or any(not _canonical(c["evidence_quote"]) for c in candidates):
        raise ValueError("Every gold candidate must contain a non-empty evidence quote")

    marked_hits: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        same_gold_page = any(
            hit.get("document_id") == gold["document_id"]
            and hit.get("page") == gold["page"]
            for gold in candidates
        )
        matched = next((
            gold for gold in candidates
            if hit.get("document_id") == gold["document_id"]
            and hit.get("page") == gold["page"]
            and _canonical(gold["evidence_quote"]) in _canonical(hit.get("text", ""))
        ), None)
        # Copy all fields for a complete, auditable retrieval trace including
        # text, score and metadata. Database hits are JSON-compatible objects.
        marked_hits.append({
            **hit,
            "rank": rank,
            "gold_page_match": same_gold_page,
            "full_gold_evidence_match": matched is not None,
            "is_strict_hit": matched is not None,
            "matched_gold_evidence": matched,
        })

    return {
        "question_id": question["id"],
        "split": question["split"],
        "question_type": question.get("type"),
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "gold_source_id": question["gold_source_id"],
        "gold_page": question["gold_page"],
        "evidence_quote": question["evidence_quote"],
        "gold_evidences": candidates,
        "declared_filter": question.get("filter") or {},
        "hits": marked_hits,
        "hit_at_1": int(bool(marked_hits and marked_hits[0]["is_strict_hit"])),
        "evidence_hit_at_3": int(any(hit["is_strict_hit"] for hit in marked_hits)),
        "gold_page_at_3_diagnostic": int(any(hit["gold_page_match"] for hit in marked_hits)),
        "manual_grounding_0_2": None,
        "manual_grounding_reason": None,
    }


def _metrics(traces: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(traces)
    hit1 = sum(t["hit_at_1"] for t in traces)
    hit3 = sum(t["evidence_hit_at_3"] for t in traces)
    page3 = sum(t["gold_page_at_3_diagnostic"] for t in traces)
    return {
        "questions": n,
        "hit_at_1": {"correct": hit1, "total": n,
                     "rate": round(hit1 / n, 4) if n else None},
        "evidence_hit_at_3": {"correct": hit3, "total": n,
                              "rate": round(hit3 / n, 4) if n else None},
        "gold_page_at_3_diagnostic": {"correct": page3, "total": n,
                                       "rate": round(page3 / n, 4) if n else None},
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize Hit@1/Hit@3 per strategy and split; never mix dev and test."""
    required = {"A", "B", "C"}
    seen: set[tuple[str, str]] = set()
    for trace in traces:
        key = (trace["strategy"], trace["question_id"])
        if key in seen:
            raise ValueError(f"Repeated trace for {key}")
        seen.add(key)
    output: dict[str, Any] = {}
    for strategy in sorted(required):
        selected = [t for t in traces if t["strategy"] == strategy]
        output[strategy] = {
            "dev": _metrics([t for t in selected if t["split"] == "dev"]),
            "test": _metrics([t for t in selected if t["split"] == "test"]),
            "all": _metrics(selected),
        }
        # C's filter compliance is meaningful only for questions where the
        # benchmark specified an input filter. Do not claim 100% for N/A.
        filtered = [t for t in selected if t.get("applied_filter")]
        output[strategy]["filter_compliance"] = (
            {"compliant": sum(t["filter_compliant"] is True for t in filtered),
             "total": len(filtered),
             "rate": round(sum(t["filter_compliant"] is True for t in filtered) / len(filtered), 4)}
            if filtered else {"compliant": 0, "total": 0, "rate": None}
        )
    return output
    