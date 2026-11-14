"""Step 7: run one reproducible A/B/C retrieval benchmark on the stored vectors.

Run from starter/: python -m src.run
Offline validation: python -m src.run --validate-only

This does NOT re-embed the document corpus. It embeds the 20 queries once,
reusing each query vector for A, B, C, and leaves the 5-question test split
unchanged. No answer generation or OCR is involved.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .evaluation import _canonical, evaluate_query, summarize
from .ingestion import load_corpus_from_manifest
from .models import Chunk, DocumentPage

STARTER = Path(__file__).resolve().parents[1]
REPO = STARTER.parent
MANIFEST = REPO / "data" / "manifest.json"
BENCHMARK = REPO / "benchmark" / "questions.json"
RESULTS = STARTER / "results"


def read_questions(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    questions = raw.get("questions") if isinstance(raw, dict) else raw
    if not isinstance(questions, list):
        raise ValueError("benchmark/questions.json must be a list or {'questions': [...]} ")
    return questions


def validate_questions(
    questions: list[dict[str, Any]], pages: list[DocumentPage]
) -> dict[str, int]:
    """Validate benchmark BEFORE retrieval and without using gold to filter."""
    if len(questions) < 20:
        raise ValueError("Benchmark needs at least 20 questions")
    counts = Counter(q.get("split") for q in questions)
    if counts["dev"] < 15 or counts["test"] < 5 or sum(counts.values()) != len(questions):
        raise ValueError(f"Expected at least 15 dev / 5 test questions, got {dict(counts)}")
    ids = [q.get("id") for q in questions]
    if not all(isinstance(qid, str) and qid for qid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Missing or duplicate benchmark question IDs")

    by_page = {(p.document_id, p.page): p for p in pages}
    known_sections = {p.metadata.get("section") for p in pages}
    filtered_count = filtered_test = 0
    for q in questions:
        for field in ("question", "expected_answer", "gold_source_id", "gold_page", "evidence_quote"):
            if field not in q or q[field] is None or q[field] == "":
                raise ValueError(f"{q['id']}: required field {field} is missing")
        if not isinstance(q["gold_page"], int):
            raise ValueError(f"{q['id']}: gold_page must be a physical PDF page integer")
        gold = by_page.get((q["gold_source_id"], q["gold_page"]))
        if gold is None:
            raise ValueError(f"{q['id']}: gold PDF page outside the selected corpus")
        if _canonical(q["evidence_quote"]) not in _canonical(gold.text):
            raise ValueError(
                f"{q['id']}: gold evidence NOT found on PDF page {q['gold_page']} "
                "after text normalization. Fix gold BEFORE running the benchmark."
            )
        for alt in q.get("gold_evidences", []):
            alt_page = by_page.get((alt["document_id"], alt["page"]))
            if alt_page is None or _canonical(alt["evidence_quote"]) not in _canonical(alt_page.text):
                raise ValueError(f"{q['id']}: invalid predeclared alternative gold evidence")

        filt = q.get("filter") or {}
        if not isinstance(filt, dict):
            raise ValueError(f"{q['id']}: filter must be an object")
        # Current report uses one trusted section field, with assignments
        # in the manifest. Never infer filter from gold page/source.
        if set(filt) - {"section"}:
            raise ValueError(f"{q['id']}: only the declared section filter is supported")
        if filt:
            if filt["section"] not in known_sections:
                raise ValueError(f"{q['id']}: unknown section filter: {filt['section']}")
            filtered_count += 1
            filtered_test += int(q["split"] == "test")
    if filtered_count < 3 or filtered_test < 1:
        raise ValueError("Benchmark needs 3 filtered questions, including 1 test")
    return {"total": len(questions), "dev": counts["dev"], "test": counts["test"],
            "with_filter": filtered_count, "test_with_filter": filtered_test}


def make_chunks(pages: list[DocumentPage]) -> dict[str, list[Chunk]]:
    return {
        "A": [c for p in pages for c in fixed_size(p, size=500, overlap=50)],
        "B": [c for p in pages for c in by_paragraph(p, max_size=500)],
    }


def chunk_statistics(chunks: list[Chunk]) -> dict[str, Any]:
    sizes = [len(c.text) for c in chunks]
    return {
        "count": len(chunks),
        "min_chars": min(sizes) if sizes else 0,
        "avg_chars": round(sum(sizes) / len(sizes), 2) if sizes else 0,
        "max_chars": max(sizes) if sizes else 0,
        "total_chunk_characters_including_overlap": sum(sizes),
        "sections": dict(sorted(Counter(c.metadata.get("section") for c in chunks).items())),
    }


def evidence_chunk_coverage(
    questions: list[dict[str, Any]], chunks_by_strategy: dict[str, list[Chunk]]
) -> dict[str, Any]:
    """Pre-retrieval diagnostic: whether ANY chunk can hold the whole gold."""
    out: dict[str, Any] = {}
    for strategy, chunks in chunks_by_strategy.items():
        index: dict[tuple[str, int], list[str]] = {}
        for chunk in chunks:
            index.setdefault((chunk.document_id, chunk.page), []).append(_canonical(chunk.text))
        missing = []
        for q in questions:
            gold_candidates = [{"document_id": q["gold_source_id"], "page": q["gold_page"],
                                "evidence_quote": q["evidence_quote"]}] + q.get("gold_evidences", [])
            matched = any(
                _canonical(gold["evidence_quote"]) in chunk_text
                for gold in gold_candidates
                for chunk_text in index.get((gold["document_id"], gold["page"]), [])
            )
            if not matched:
                missing.append(q["id"])
        out[strategy] = {
            "questions_with_full_evidence_somewhere_in_index": len(questions) - len(missing),
            "total_questions": len(questions), "not_in_any_single_chunk": missing,
        }
    out["C"] = {**out["B"]}  # B chunks are exactly C chunks
    return out


def check_index(store: Any, chunks: dict[str, list[Chunk]]) -> None:
    """Verify both count and indexed text against the actual local corpus."""
    for strategy, expected in chunks.items():
        stored_count = store.count(strategy)
        if stored_count != len(expected):
            raise RuntimeError(
                f"{strategy}: database has {stored_count} rows, but corpus has "
                f"{len(expected)} chunks. Run python -m scripts.check_step6 again."
            )
        cached = store.get_cached_embeddings(strategy, expected)
        missing = sum(vector is None for vector in cached)
        if missing:
            raise RuntimeError(
                f"{strategy}: {missing} stored chunk IDs or texts do not match the current "
                "corpus. Run python -m scripts.check_step6 again."
            )
    if store.count("C") != store.count("B"):
        raise RuntimeError("C must alias B embeddings")


def _print_one(label: str, group: dict[str, Any]) -> None:
    h1 = group["hit_at_1"]
    h3 = group["evidence_hit_at_3"]
    print(f"  {label}: Hit@1 {h1['correct']}/{h1['total']} "
          f"({h1['rate']:.1%}), Evidence Hit@3 {h3['correct']}/{h3['total']} "
          f"({h3['rate']:.1%})")


def _write_experiment_log(summary: dict[str, Any], model_name: str) -> Path:
    path = REPO / "artifacts" / "experiment_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["strategy", "chunking", "chunk_size", "overlap", "metadata", "filter",
              "vectorizer", "hypothesis", "dev_hit1", "dev_hit3", "test_hit1",
              "test_hit3", "notes"]

    # Preserve an existing hand-edited experiment log before overwriting it.
    if path.exists():
        old_text = path.read_text(encoding="utf-8-sig")
        try:
            existing_rows = list(csv.DictReader(old_text.splitlines()))
        except csv.Error:
            existing_rows = []
        if any(any(row.get(key, "").strip() for key in fields[1:] if key != "strategy")
               for row in existing_rows):
            backup = path.with_name("experiment_log_before_step7.csv")
            if not backup.exists():
                backup.write_text(old_text, encoding="utf-8")

    descriptions = {
        "A": ("fixed_characters", 500, 50, "none",
              "Baseline: fixed 500 characters with 50 overlap"),
        "B": ("paragraph_heuristic", 500, 0, "none",
              "Paragraph grouping may preserve complete evidence better than A"),
        "C": ("same_as_B", 500, 0, "section if question provides it",
              "Section filter may improve B on scoped questions"),
    }
    def metric(strategy: str, split: str, key: str) -> str:
        m = summary[strategy][split][key]
        return f"{m['correct']}/{m['total']} ({m['rate']:.1%})"

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for strategy in ("A", "B", "C"):
            chunk_method, size, overlap, filt, hypothesis = descriptions[strategy]
            writer.writerow({
                "strategy": strategy, "chunking": chunk_method,
                "chunk_size": size, "overlap": overlap,
                "metadata": "document_id,source,page,section",
                "filter": filt, "vectorizer": model_name,
                "hypothesis": hypothesis,
                "dev_hit1": metric(strategy, "dev", "hit_at_1"),
                "dev_hit3": metric(strategy, "dev", "evidence_hit_at_3"),
                "test_hit1": metric(strategy, "test", "hit_at_1"),
                "test_hit3": metric(strategy, "test", "evidence_hit_at_3"),
                "notes": "Automatic strict evidence-string matching; manual audits pending Step 8; exact cosine",
            })
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Day 06 benchmark: A/B/C top-3 retrieval")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate corpus, gold evidence and chunks; no OpenAI/DB")
    args = parser.parse_args(argv)
    started = time.perf_counter()

    if not MANIFEST.exists() or not BENCHMARK.exists():
        raise FileNotFoundError(f"Expected {MANIFEST} and {BENCHMARK}")
    pages = load_corpus_from_manifest(MANIFEST)
    if len(pages) < 30:
        raise ValueError(f"At least 30 content pages are required, got {len(pages)}")
    questions = read_questions(BENCHMARK)
    info = validate_questions(questions, pages)
    chunks = make_chunks(pages)
    coverage = evidence_chunk_coverage(questions, chunks)
    stats = {name: chunk_statistics(data) for name, data in chunks.items()}
    stats["C"] = {**stats["B"], "reuses_B_index": True}

    print(f"Corpus: {len(pages)} pages | A={len(chunks['A'])} chunks | "
          f"B={len(chunks['B'])} chunks | C reuses B")
    print(f"Benchmark: {info['total']} questions ({info['dev']} dev, {info['test']} test), "
          f"{info['with_filter']} predeclared filters")
    for strategy in ("A", "B"):
        c = coverage[strategy]
        print(f"{strategy} full-evidence single-chunk coverage: "
              f"{c['questions_with_full_evidence_somewhere_in_index']}/{c['total_questions']} "
              f"(not covered: {c['not_in_any_single_chunk']})")
    if args.validate_only:
        print("Step 7 validation OK. No OpenAI API or database requests.")
        return

    load_dotenv(STARTER / ".env")
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is missing from starter/.env")
    from .embeddings import MyEmbedder
    from .store import PgVectorStore

    model_name = os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
    embedder = MyEmbedder(model_name)
    store = PgVectorStore(db_url, embedder.dimensions, model_name=embedder.model_name)
    store.create_schema()
    check_index(store, chunks)

    # Embed 20 questions in a SINGLE batch (not 60 times for A/B/C).
    embed_start = time.perf_counter()
    vectors = embedder.embed_documents([q["question"] for q in questions])
    embedding_seconds = round(time.perf_counter() - embed_start, 3)
    if len(vectors) != len(questions):
        raise RuntimeError("Embedding result count does not match benchmark count")

    traces: list[dict[str, Any]] = []
    search_seconds = {"A": 0.0, "B": 0.0, "C": 0.0}
    for question, query_vector in zip(questions, vectors):
        unfiltered_b: list[dict[str, Any]] | None = None
        for strategy in ("A", "B", "C"):
            # Critical experimental constraint: only C gets the predeclared
            # filter; NEVER infer it from gold_page or expected_answer.
            declared_filter = question.get("filter") or {}
            applied_filter = declared_filter if strategy == "C" else {}
            t0 = time.perf_counter()
            hits = store.search(strategy, query_vector, top_k=3,
                                filters=applied_filter or None)
            seconds = time.perf_counter() - t0
            if strategy == "B":
                unfiltered_b = hits
            if strategy == "C" and not applied_filter:
                # Run a real SQL search for both strategies (fair timing),
                # but results must be the same on an unfiltered question.
                if unfiltered_b is None:
                    raise RuntimeError("B results must be computed before C")
                b_ids = [hit["chunk_id"] for hit in unfiltered_b]
                c_ids = [hit["chunk_id"] for hit in hits]
                if b_ids != c_ids:
                    raise RuntimeError(f"B/C should return identical unfiltered results for {question['id']}")
            search_seconds[strategy] += seconds
            if len(hits) != 3:
                raise RuntimeError(f"{strategy}/{question['id']}: expected top 3, got {len(hits)}")
            if applied_filter and any(
                any(hit.get(key) != value for key, value in applied_filter.items())
                for hit in hits
            ):
                raise RuntimeError(f"{strategy}/{question['id']}: SQL filter was not respected")

            trace = evaluate_query(question, hits)
            trace.update({
                "strategy": strategy,
                "stored_vector_strategy": "B" if strategy == "C" else strategy,
                "applied_filter": applied_filter,
                "filter_compliant": (
                    all(all(hit.get(key) == value for key, value in applied_filter.items())
                        for hit in hits) if applied_filter else None
                ),
                "query_search_seconds": round(seconds, 5),
            })
            traces.append(trace)

    if len(traces) != len(questions) * 3:
        raise RuntimeError("Incomplete benchmark traces")
    metrics = summarize(traces)
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = {
        "generated_at_utc": generated,
        "corpus_source": str(MANIFEST.relative_to(REPO)),
        "benchmark_source": str(BENCHMARK.relative_to(REPO)),
        "model": embedder.model_name,
        "vector_dimensions": embedder.dimensions,
        "vector_store": "PostgreSQL + pgvector; exact cosine distance",
        "top_k": 3, "selected_content_pages": len(pages),
        "benchmark_validation": info,
        "chunk_stats": stats,
        "full_gold_evidence_chunk_coverage_before_retrieval": coverage,
        "embedding_api_calls_this_run": embedder.api_calls,
        "embedding_query_tokens_this_run": embedder.tokens_used,
        "query_embedding_elapsed_seconds": embedding_seconds,
        "search_elapsed_seconds_by_strategy": {k: round(v, 3) for k, v in search_seconds.items()},
        "total_elapsed_seconds": round(time.perf_counter() - started, 3),
        "document_embeddings_new_this_run": 0,
        "estimated_usd_cost": None,
        "cost_note": "Record actual query tokens; compute USD separately using provider pricing at run time",
        "metric_scope": "Strict auto evidence match: correct document/page and entire normalized gold quote in ONE chunk. Manual equivalent-evidence adjudication pending.",
        "manual_coherence_audit": "pending Step 8",
        "manual_metadata_audit": "pending Step 8",
        "manual_grounding_audit": "pending Step 8",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    traces_path = RESULTS / "step7_traces.json"
    summary_path = RESULTS / "step7_summary.json"
    jsonl_path = RESULTS / "step7_traces.jsonl"
    traces_path.write_text(json.dumps({"run": meta, "traces": traces}, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    summary_path.write_text(json.dumps({"run": meta, "metrics": metrics}, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
    log_path = _write_experiment_log(metrics, embedder.model_name)

    print("\n=== Automatic retrieval benchmark (strict) ===")
    for strategy in ("A", "B", "C"):
        print(strategy)
        _print_one("dev ", metrics[strategy]["dev"])
        _print_one("test", metrics[strategy]["test"])
        fc = metrics[strategy]["filter_compliance"]
        if fc["total"]:
            print(f"  Filter compliance: {fc['compliant']}/{fc['total']}")
        else:
            print("  Filter compliance: N/A (no filter applied)")
    print(f"\nQuery embedding: {embedder.api_calls} API calls; {embedder.tokens_used} tokens")
    print(f"Traces: {traces_path.relative_to(REPO)}")
    print(f"Summary: {summary_path.relative_to(REPO)}")
    print(f"Experiment log: {log_path.relative_to(REPO)}")
    print("Step 7 OK — manual 5-criteria assessment is still required in Step 8.")


if __name__ == "__main__":
    main()
