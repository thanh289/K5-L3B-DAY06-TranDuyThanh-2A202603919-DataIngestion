"""OpenAI embedding adapter
"""

import math
import os
from typing import Protocol

from openai import OpenAI


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, question: str) -> list[float]:
        ...


# Default dimensions when the OpenAI embeddings API is called without a custom
# `dimensions` parameter. Fix one model for ALL three experiment strategies.
_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class MyEmbedder:
    """Real OpenAI embeddings with order-preserving batching and validation."""

    def __init__(self, model_name: str) -> None:
        if model_name not in _MODEL_DIMENSIONS:
            raise ValueError(
                f"Unsupported embedding model {model_name!r}. "
                f"Choose one of: {', '.join(_MODEL_DIMENSIONS)}"
            )
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing from starter/.env")
        self.model_name = model_name
        self.dimensions = _MODEL_DIMENSIONS[model_name]
        self.client = OpenAI(api_key=api_key, timeout=90.0, max_retries=2)
        self.batch_size = 64
        self.api_calls = 0
        self.tokens_used = 0

    def _validate_embedding(self, vector: list[float]) -> list[float]:
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Expected {self.dimensions} dimensions from {self.model_name}, "
                f"received {len(vector)}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("OpenAI returned a non-finite embedding value")
        if not any(value != 0 for value in vector):
            raise ValueError("OpenAI returned a zero embedding vector")
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return a vector per input text, preserving the original text order."""
        if not texts:
            return []
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Embedding input contains an empty or non-string text")

        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            # Exceptions from the API are intentional: NEVER fall back to mock.
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
                encoding_format="float",
            )
            self.api_calls += 1
            if response.usage is not None:
                self.tokens_used += response.usage.total_tokens

            data = sorted(response.data, key=lambda item: item.index)
            if len(data) != len(batch) or [v.index for v in data] != list(range(len(batch))):
                raise RuntimeError("Unexpected embedding response count/order")
            for item in data:
                all_vectors.append(self._validate_embedding(item.embedding))
        return all_vectors

    def embed_query(self, question: str) -> list[float]:
        """Embed Vietnamese or English questions using the SAME document model."""
        return self.embed_documents([question])[0]
