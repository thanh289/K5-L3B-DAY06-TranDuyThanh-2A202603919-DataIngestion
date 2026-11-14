"""Step 6 smoke test: PDF -> chunks -> OpenAI embeddings -> pgvector -> retrieval.

Run from starter/: python -m scripts.check_step6
  Add --dry-run to check only the corpus/chunk counts (NO API/DB access).
  Add --search-only to skip re-ingestion (still embeds one test query).
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from src.chunking import by_paragraph, fixed_size
from src.ingestion import load_corpus_from_manifest

STARTER = Path(__file__).resolve().parents[1]
REPO = STARTER.parent


def _sync_one_strategy(store, embedder, strategy, chunks):
    cached = store.get_cached_embeddings(strategy, chunks)
    missing_indices = [i for i, vector in enumerate(cached) if vector is None]
    if missing_indices:
        print(f"{strategy}: requesting {len(missing_indices)} REAL document embeddings...")
        new_vectors = embedder.embed_documents([chunks[i].text for i in missing_indices])
        for index, vector in zip(missing_indices, new_vectors):
            cached[index] = vector
    # At this point there are no Nones: all chunk vectors were cached or made.
    store.upsert(strategy, chunks, cached)
    actual = store.count(strategy)
    if actual != len(chunks):
        raise RuntimeError(f"{strategy}: expected {len(chunks)} rows, got {actual}")
    print(f"{strategy}: {actual} rows; reused {len(chunks)-len(missing_indices)} cached; "
          f"embedded {len(missing_indices)} new")
    return {"chunks": actual, "reused": len(chunks)-len(missing_indices),
            "new_embeddings": len(missing_indices)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="No API and no database")
    parser.add_argument("--search-only", action="store_true", help="Query existing vectors only")
    args = parser.parse_args()
    if args.dry_run and args.search_only:
        parser.error("Choose only one mode")
    start = time.perf_counter()

    manifest = REPO / "data" / "manifest.json"
    pages = load_corpus_from_manifest(manifest)
    if len(pages) < 30:
        raise RuntimeError(f"Only {len(pages)} content pages; at least 30 required")
    chunks_a = [chunk for page in pages for chunk in fixed_size(page, 500, 50)]
    chunks_b = [chunk for page in pages for chunk in by_paragraph(page, 500)]
    print(f"PDF pages: {len(pages)} | A: {len(chunks_a)} chunks | "
          f"B: {len(chunks_b)} chunks | C: B + metadata filter")

    if args.dry_run:
        print("Dry-run OK. No database or OpenAI API calls were made.")
        return

    load_dotenv(STARTER / ".env")
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is missing from starter/.env")

    from src.embeddings import MyEmbedder
    from src.store import PgVectorStore

    model_name = (os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small")
    embedder = MyEmbedder(model_name)
    store = PgVectorStore(db_url, embedder.dimensions, model_name=embedder.model_name)
    store.create_schema()

    ingested = {}
    if not args.search_only:
        ingested["A"] = _sync_one_strategy(store, embedder, "A", chunks_a)
        ingested["B"] = _sync_one_strategy(store, embedder, "B", chunks_b)
    else:
        print("Skipping ingestion: querying existing database rows")
    count_a, count_b = store.count("A"), store.count("B")
    if count_a != len(chunks_a) or count_b != len(chunks_b):
        raise RuntimeError("Stored A/B counts do not match current corpus. Run without --search-only")
    assert store.count("C") == count_b

    question = "Theo chương Digital Adoption, năm 2022 có bao nhiêu người dùng Internet toàn cầu?"
    query_embedding = embedder.embed_query(question)
    results = {
        "A": store.search("A", query_embedding, top_k=3),
        "B": store.search("B", query_embedding, top_k=3),
        "C": store.search("C", query_embedding, top_k=3, filters={"section": "chapter_1"}),
    }
    if not all(results.values()):
        raise RuntimeError("At least one strategy returned no hits")
    if not all(hit["section"] == "chapter_1" for hit in results["C"]):
        raise RuntimeError("C search returned a result outside the requested section")

    for strategy, hits in results.items():
        print(f"\n{strategy} — top {len(hits)}:")
        for rank, hit in enumerate(hits, 1):
            preview = " ".join(hit["text"].split())[:95]
            print(f"  {rank}. p{hit['page']}, {hit['section']}, "
                  f"cosine={hit['score']:.4f}, id={hit['chunk_id']}")
            print(f"     {preview}...")

    summary = {
        "status": "Step 6 OK",
        "model": embedder.model_name,
        "dimensions": embedder.dimensions,
        "search_type": "exact_cosine",
        "content_pages": len(pages),
        "chunks_stored": {"A": count_a, "B": count_b, "C": "reuses B"},
        "ingestion": ingested,
        "openai_api_calls_this_run": embedder.api_calls,
        "openai_prompt_tokens_this_run": embedder.tokens_used,
        "elapsed_seconds": round(time.perf_counter()-start, 2),
        "smoke_question": question,
        "smoke_filter_for_c": {"section": "chapter_1"},
        "top3": results,
    }
    result_path = STARTER / "results" / "step6_smoke_test.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nStep 6 OK | result: {result_path.relative_to(REPO)}")
    print(f"API calls this run: {embedder.api_calls} | tokens: {embedder.tokens_used}")


if __name__ == "__main__":
    main()
