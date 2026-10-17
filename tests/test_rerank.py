from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from contractguard.chunking import Chunk
from contractguard.retrieval.models import SearchResult
from contractguard.retrieval.rerank import RerankingRetriever


def _chunk(index: int) -> Chunk:
    text = f"Candidate text {index}"
    return Chunk(
        chunk_id=f"doc-1:{index}",
        doc_id="doc-1",
        text=text,
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(text),
    )


@dataclass
class _FakeBaseRetriever:
    results: list[SearchResult]
    calls: list[tuple[str, int]] = field(default_factory=list)

    def search(self, query: str, k: int) -> list[SearchResult]:
        self.calls.append((query, k))
        return self.results[:k]


@dataclass
class _FakePairScorer:
    scores: list[float]
    calls: list[tuple[str, list[str]]] = field(default_factory=list)

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, texts))
        return self.scores


def test_reranker_reorders_results_and_limits_them_to_k() -> None:
    candidates = [
        SearchResult(chunk=_chunk(index), score=float(index)) for index in range(3)
    ]
    base = _FakeBaseRetriever(candidates)
    scorer = _FakePairScorer([0.2, 0.9, 0.5])

    results = RerankingRetriever(base, scorer).search("raw query", k=2)

    assert [result.chunk.chunk_id for result in results] == ["doc-1:1", "doc-1:2"]
    assert len(results) == 2
    assert base.calls == [("raw query", 20)]
    assert scorer.calls == [
        ("raw query", ["Candidate text 0", "Candidate text 1", "Candidate text 2"])
    ]


def test_reranker_candidate_pool_scales_with_k() -> None:
    base = _FakeBaseRetriever([])
    scorer = _FakePairScorer([])
    reranker = RerankingRetriever(base, scorer)

    assert reranker.search("query", k=5) == []
    assert reranker.search("query", k=30) == []

    assert base.calls == [("query", 20), ("query", 30)]


def test_reranker_keeps_base_order_for_equal_scores() -> None:
    candidates = [
        SearchResult(chunk=_chunk(2), score=0.0),
        SearchResult(chunk=_chunk(0), score=0.0),
        SearchResult(chunk=_chunk(1), score=0.0),
    ]
    results = RerankingRetriever(
        _FakeBaseRetriever(candidates), _FakePairScorer([0.5, 0.5, 0.4])
    ).search("query", k=3)

    assert [result.chunk.chunk_id for result in results] == [
        "doc-1:2",
        "doc-1:0",
        "doc-1:1",
    ]


def test_reranker_skips_scorer_when_base_returns_no_candidates() -> None:
    scorer = _FakePairScorer([])

    assert RerankingRetriever(_FakeBaseRetriever([]), scorer).search("query") == []
    assert scorer.calls == []


def test_reranker_accepts_fewer_candidates_than_its_pool() -> None:
    candidates = [SearchResult(chunk=_chunk(index), score=0.0) for index in range(2)]

    results = RerankingRetriever(
        _FakeBaseRetriever(candidates), _FakePairScorer([0.4, 0.8])
    ).search("query", k=5)

    assert [result.chunk.chunk_id for result in results] == ["doc-1:1", "doc-1:0"]


def test_reranker_preserves_base_score_provenance() -> None:
    candidate = SearchResult(
        chunk=_chunk(0),
        score=0.4,
        dense_score=0.4,
        bm25_score=1.5,
        rrf_score=0.03,
    )

    result = RerankingRetriever(
        _FakeBaseRetriever([candidate]), _FakePairScorer([0.9])
    ).search("query")[0]

    assert result.score == 0.9
    assert result.rerank_score == 0.9
    assert result.dense_score == 0.4
    assert result.bm25_score == 1.5
    assert result.rrf_score == 0.03
    assert candidate.score == 0.4
    assert candidate.rerank_score is None


def test_reranker_rejects_wrong_number_of_scores() -> None:
    candidates = [SearchResult(chunk=_chunk(index), score=0.0) for index in range(2)]

    with pytest.raises(ValueError, match="one score per candidate"):
        RerankingRetriever(
            _FakeBaseRetriever(candidates), _FakePairScorer([0.7])
        ).search("query")
