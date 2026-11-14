"""Persistent PostgreSQL + pgvector store (exact cosine retrieval).

Only A and B are persisted. C deliberately aliases B and applies SQL metadata
filters at query time, so B and C have identical chunks/embeddings.
"""

import json
import math
from collections import defaultdict
from typing import Any

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from .models import Chunk


_TABLE = "day06_chunks"
_ALLOWED_FILTERS = {"section", "document_id", "source", "page"}


def _vector_literal(values: list[float], dimensions: int) -> str:
    if len(values) != dimensions:
        raise ValueError(f"Vector has {len(values)} dimensions, expected {dimensions}")
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
        raise ValueError("Vector contains invalid or non-finite values")
    if not any(x != 0 for x in values):
        raise ValueError("Zero vectors are not valid for cosine search")
    return "[" + ",".join(format(float(x), ".9g") for x in values) + "]"


class PgVectorStore:
    """pgvector-backed chunk table scoped by (strategy, model, chunk ID)."""

    def __init__(
        self,
        database_url: str,
        dimensions: int,
        model_name: str = "text-embedding-3-small",
    ) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL is missing")
        if not isinstance(dimensions, int) or not 1 <= dimensions <= 16000:
            raise ValueError("Invalid embedding dimensions")
        if not model_name or not isinstance(model_name, str):
            raise ValueError("Embedding model name is required")
        self.database_url = database_url
        self.dimensions = dimensions
        self.model_name = model_name

    @staticmethod
    def _stored_strategy(strategy: str) -> str:
        key = strategy.upper()
        if key == "C":
            return "B"  # never create independent C chunks
        if key not in {"A", "B"}:
            raise ValueError("strategy must be A, B, or C")
        return key

    def _connect(self):
        return psycopg.connect(self.database_url, connect_timeout=10)

    def create_schema(self) -> None:
        """Create one persistent table; fail on incompatible vector dimensions."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    sql.SQL("""
                        CREATE TABLE IF NOT EXISTS {} (
                            strategy TEXT NOT NULL CHECK (strategy IN ('A', 'B')),
                            model_name TEXT NOT NULL,
                            chunk_id TEXT NOT NULL,
                            document_id TEXT NOT NULL,
                            source TEXT NOT NULL,
                            page INTEGER NOT NULL CHECK (page > 0),
                            section TEXT,
                            text TEXT NOT NULL,
                            metadata JSONB NOT NULL,
                            embedding vector({}) NOT NULL,
                            PRIMARY KEY (strategy, model_name, chunk_id)
                        )
                    """).format(sql.Identifier(_TABLE), sql.SQL(str(self.dimensions)))
                )
                # Confirm any pre-existing table actually uses the requested
                # dimension; CREATE TABLE IF NOT EXISTS doesn't check schema.
                cur.execute(
                    """SELECT format_type(a.atttypid, a.atttypmod)
                       FROM pg_attribute AS a
                       WHERE a.attrelid = %s::regclass
                         AND a.attname = 'embedding' AND NOT a.attisdropped""",
                    (_TABLE,),
                )
                row = cur.fetchone()
                if not row or row[0] != f"vector({self.dimensions})":
                    raise RuntimeError(
                        f"Existing {_TABLE}.embedding has type {row}; "
                        f"expected vector({self.dimensions}). Do not mix model dimensions."
                    )
                cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS day06_chunks_section_idx "
                            "ON {} (strategy, model_name, section)").format(sql.Identifier(_TABLE))
                )
                # No ANN index: ORDER BY cosine distance performs an exact
                # search on this small corpus.

    def count(self, strategy: str) -> int:
        stored = self._stored_strategy(strategy)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT COUNT(*) FROM {} WHERE strategy=%s AND model_name=%s")
                    .format(sql.Identifier(_TABLE)),
                    (stored, self.model_name),
                )
                return int(cur.fetchone()[0])

    def get_cached_embeddings(
        self, strategy: str, chunks: list[Chunk]
    ) -> list[list[float] | None]:
        """Reuse embeddings only if chunk_id AND text AND model all match.

        Returning None means the caller must call the real embedding API.
        This avoids re-billing for unchanged chunks on a rerun.
        """
        stored = self._stored_strategy(strategy)
        if not chunks:
            return []
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate chunk_id in the given input")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("""SELECT chunk_id, text, embedding::text
                               FROM {} WHERE strategy=%s AND model_name=%s
                               AND chunk_id = ANY(%s::text[])""").format(sql.Identifier(_TABLE)),
                    (stored, self.model_name, ids),
                )
                cached = {chunk_id: (text, vector_text)
                          for chunk_id, text, vector_text in cur.fetchall()}

        result: list[list[float] | None] = []
        for chunk in chunks:
            value = cached.get(chunk.chunk_id)
            if value is not None and value[0] == chunk.text:
                vector = json.loads(value[1])
                if len(vector) != self.dimensions:
                    raise RuntimeError("Cached embedding dimensions do not match")
                result.append(vector)
            else:
                result.append(None)
        return result

    def upsert(
        self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        """Insert/update matching chunks; delete stale IDs per included document.

        Provide the COMPLETE current chunk set for each document/strategy,
        not just changed chunks. That way deletion safely removes obsolete IDs.
        """
        stored = self._stored_strategy(strategy)
        if strategy.upper() == "C":
            raise ValueError("Never ingest C: use B chunks and filter on retrieval")
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal lengths")
        if not chunks:
            raise ValueError("Cannot sync an empty corpus without document IDs")
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate chunk_id")

        rows = []
        ids_by_document: dict[str, list[str]] = defaultdict(list)
        for chunk, vector in zip(chunks, vectors):
            if not chunk.chunk_id or not chunk.document_id or not chunk.text.strip():
                raise ValueError("Chunk ID, document ID and text are required")
            if not chunk.source or not isinstance(chunk.page, int) or chunk.page < 1:
                raise ValueError("Chunk source and physical page are required")
            section = chunk.metadata.get("section")
            if not isinstance(section, str) or not section:
                raise ValueError("A trustworthy section metadata value is required")
            ids_by_document[chunk.document_id].append(chunk.chunk_id)
            rows.append((
                stored, self.model_name, chunk.chunk_id, chunk.document_id,
                chunk.source, chunk.page, section, chunk.text,
                Jsonb(chunk.metadata), _vector_literal(vector, self.dimensions),
            ))

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    sql.SQL("""
                        INSERT INTO {} (
                            strategy, model_name, chunk_id, document_id, source,
                            page, section, text, metadata, embedding
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
                        ON CONFLICT (strategy, model_name, chunk_id)
                        DO UPDATE SET
                            document_id = EXCLUDED.document_id,
                            source = EXCLUDED.source,
                            page = EXCLUDED.page,
                            section = EXCLUDED.section,
                            text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding
                    """).format(sql.Identifier(_TABLE)),
                    rows,
                )
                # Remove outdated IDs for each included document. Deletions
                # happen in the SAME transaction as the upsert, so failures
                # cannot leave a partially replaced document.
                for document_id, current_ids in ids_by_document.items():
                    cur.execute(
                        sql.SQL("""DELETE FROM {}
                                   WHERE strategy=%s AND model_name=%s
                                     AND document_id=%s
                                     AND NOT (chunk_id = ANY(%s::text[]))""")
                        .format(sql.Identifier(_TABLE)),
                        (stored, self.model_name, document_id, current_ids),
                    )

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Exact cosine top-k with safe, SQL-level metadata filtering.

        A and B should be called without filters in the benchmark; C with the
        question's pre-declared filter (if any). C ALWAYS searches B storage.
        """
        stored = self._stored_strategy(strategy)
        if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
            raise ValueError("top_k must be between 1 and 100")
        vector = _vector_literal(query_vector, self.dimensions)
        filters = filters or {}
        if not isinstance(filters, dict):
            raise ValueError("filters must be a dictionary")
        unsupported = set(filters) - _ALLOWED_FILTERS
        if unsupported:
            raise ValueError(f"Unsupported filter field(s): {sorted(unsupported)}")

        where = [sql.SQL("strategy = %s"), sql.SQL("model_name = %s")]
        params: list[Any] = [stored, self.model_name]
        for name, value in filters.items():
            if name == "page":
                if not isinstance(value, int) or value < 1:
                    raise ValueError("page filter must be a positive integer")
            elif not isinstance(value, str) or not value:
                raise ValueError(f"{name} filter must be a non-empty string")
            where.append(sql.SQL("{} = %s").format(sql.Identifier(name)))
            params.append(value)

        # Use cosine distance as the ORDER BY expression for exact KNN.
        query = sql.SQL("""
            SELECT chunk_id, document_id, source, page, section,
                   text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM {}
            WHERE {}
            ORDER BY embedding <=> %s::vector, chunk_id
            LIMIT %s
        """).format(sql.Identifier(_TABLE), sql.SQL(" AND ").join(where))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, [vector, *params, vector, top_k])
                records = cur.fetchall()
        return [
            {
                "chunk_id": row[0], "document_id": row[1],
                "source": row[2], "page": row[3], "section": row[4],
                "text": row[5], "metadata": row[6],
                "score": float(row[7]),
            }
            for row in records
        ]
