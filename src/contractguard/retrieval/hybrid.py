"""Hybrid dense and sparse retrieval fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from contractguard.chunking import Chunk
from contractguard.retrieval.fusion import reciprocal_rank_fusion
from contractguard.retrieval.models import SearchResult


class HybridSearchIndex:
    """Combines dense and sparse search over chunks, fused via RRF."""

    def __init__(
        self,
        chunks: list[Chunk],
        dense_index: object,
        bm25_index: object,
    ) -> None:
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self._dense_index = dense_index
        self._bm25_index = bm25_index

    @classmethod
    def from_chunks(cls, chunks: list[Chunk]) -> HybridSearchIndex:
        """Build dense and sparse indexes over the same chunks."""
        from contractguard.retrieval.bm25 import BM25Index
        from contractguard.retrieval.vector import InMemoryVectorIndex

        chunk_list = list(chunks)
        return cls(
            chunk_list,
            InMemoryVectorIndex.from_chunks(chunk_list),
            BM25Index.from_chunks(chunk_list),
        )

    def search(self, query: str, k: int = 10, rrf_k: int = 60) -> list[SearchResult]:
        """Return dense and sparse candidates fused with Reciprocal Rank Fusion."""
        candidate_pool = max(20, k)
        dense_results = self._dense_index.search(query, k=candidate_pool)
        bm25_results = self._bm25_index.search(query, k=candidate_pool)

        dense_scores = {result.chunk.chunk_id: result.score for result in dense_results}
        bm25_scores = {result.chunk.chunk_id: result.score for result in bm25_results}
        dense_ids = [result.chunk.chunk_id for result in dense_results]
        bm25_ids = [result.chunk.chunk_id for result in bm25_results]
        fused_results = reciprocal_rank_fusion([dense_ids, bm25_ids], k=rrf_k)

        return [
            SearchResult(
                chunk=self._chunks_by_id[chunk_id],
                score=fused_score,
                rrf_score=fused_score,
                dense_score=dense_scores.get(chunk_id),
                bm25_score=bm25_scores.get(chunk_id),
            )
            for chunk_id, fused_score in fused_results[:k]
        ]
