"""Local embedding and exact in-memory vector search."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from contractguard.chunking import Chunk
from contractguard.retrieval.models import SearchResult

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def _load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def warm_up_model() -> None:
    """Load the shared embedding model before serving requests."""
    _load_model(MODEL_NAME)


def embed_chunks(chunks: Sequence[Chunk], model_name: str = MODEL_NAME) -> np.ndarray:
    """Embed chunk text with normalized local vectors."""
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)
    return _encode([chunk.text for chunk in chunks], model_name)


class InMemoryVectorIndex:
    """A naive, exact vector index kept entirely in process memory."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.chunks = list(chunks)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.model_name = model_name
        if len(self.chunks) != len(self.vectors):
            raise ValueError("chunks and vectors must have the same length")
        if self.vectors.ndim != 2:
            raise ValueError("vectors must be a two-dimensional array")

    @classmethod
    def from_chunks(
        cls, chunks: Sequence[Chunk], model_name: str = MODEL_NAME
    ) -> InMemoryVectorIndex:
        """Create an index by embedding the supplied chunks locally."""
        chunk_list = list(chunks)
        return cls(chunk_list, embed_chunks(chunk_list, model_name), model_name)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return up to *k* chunks ranked by descending cosine similarity."""
        if k <= 0:
            raise ValueError("k must be greater than 0")
        if not self.chunks:
            return []

        query_vector = _encode([QUERY_PREFIX + query], self.model_name)[0]
        scores = self.vectors @ query_vector
        indices = np.argsort(-scores, kind="stable")[:k]
        return [
            SearchResult(chunk=self.chunks[index], score=float(scores[index]))
            for index in indices
        ]

    def save(self, path: str | Path) -> None:
        """Persist vectors and chunk metadata without re-embedding them."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            vectors=self.vectors,
            chunk_json=np.asarray([chunk.model_dump_json() for chunk in self.chunks]),
            model_name=np.asarray(self.model_name),
        )

    @classmethod
    def load(cls, path: str | Path) -> InMemoryVectorIndex:
        """Restore a previously saved in-memory index."""
        with np.load(Path(path), allow_pickle=False) as saved:
            chunks = [
                Chunk.model_validate_json(chunk_json)
                for chunk_json in saved["chunk_json"].tolist()
            ]
            return cls(chunks, saved["vectors"], str(saved["model_name"].item()))


def _encode(texts: list[str], model_name: str) -> np.ndarray:
    model = _load_model(model_name)
    return np.asarray(
        model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
