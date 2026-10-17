"""Rerank candidates from an existing retriever with a pair scorer."""

from __future__ import annotations

from typing import Protocol

from contractguard.retrieval.models import SearchResult


class PairScorer(Protocol):
    """Scores a query against a list of candidate texts."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one score for each candidate text."""


class RerankingRetriever:
    """Reranks candidates returned by a base retriever."""

    def __init__(self, base: object, scorer: PairScorer, pool_size: int = 20) -> None:
        self._base = base
        self._scorer = scorer
        self._pool_size = pool_size

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Return the top candidates ordered by the pair scorer."""
        candidate_pool = max(self._pool_size, k)
        candidates = self._base.search(query, k=candidate_pool)
        if not candidates:
            return []

        scores = self._scorer.score(query, [result.chunk.text for result in candidates])
        if len(scores) != len(candidates):
            raise ValueError("scorer must return one score per candidate")

        ranked = sorted(
            zip(candidates, scores, strict=True), key=lambda item: item[1], reverse=True
        )
        return [
            result.model_copy(update={"score": score, "rerank_score": score})
            for result, score in ranked[:k]
        ]
