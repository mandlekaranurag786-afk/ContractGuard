"""Sparse BM25 retrieval over a fixed set of document chunks."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from contractguard.chunking import Chunk
from contractguard.retrieval.models import SearchResult


def tokenize(text: str) -> list[str]:
    """Lowercase text and retain alphanumeric tokens only."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """Sparse keyword search over a fixed set of chunks, using BM25Okapi."""

    def __init__(self, chunks: list[Chunk], index: BM25Okapi) -> None:
        self.chunks = chunks
        self._index = index

    @classmethod
    def from_chunks(cls, chunks: list[Chunk]) -> BM25Index:
        """Tokenize chunks and build a BM25 index in the same chunk order."""
        chunk_list = list(chunks)
        tokenized_chunks = [tokenize(chunk.text) for chunk in chunk_list]
        return cls(chunk_list, BM25Okapi(tokenized_chunks))

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Return positive-scoring chunks in descending BM25 score order."""
        if k <= 0:
            return []

        scored_chunks = [
            (chunk, float(score))
            for chunk, score in zip(
                self.chunks, self._index.get_scores(tokenize(query))
            )
            if score > 0.0
        ]
        scored_chunks.sort(key=lambda item: (-item[1], item[0].chunk_id))
        return [
            SearchResult(chunk=chunk, score=score, bm25_score=score)
            for chunk, score in scored_chunks[:k]
        ]
