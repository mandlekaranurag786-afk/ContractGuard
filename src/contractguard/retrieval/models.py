"""Models returned by vector retrieval."""

from pydantic import BaseModel

from contractguard.chunking import Chunk


class SearchResult(BaseModel):
    """A chunk returned from search, with its score and the method
    that produced it. For dense-only search, `score` is cosine
    similarity and dense_score mirrors it. For hybrid search, `score`
    is the fused RRF score, rrf_score mirrors it, and dense_score/
    bm25_score hold the two raw inputs that went into the fusion."""

    chunk: Chunk
    score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
