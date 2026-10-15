from __future__ import annotations

from dataclasses import dataclass, field

from contractguard.chunking import Chunk
from contractguard.retrieval.hybrid import HybridSearchIndex
from contractguard.retrieval.models import SearchResult


def _chunk(index: int, text: str) -> Chunk:
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
class _RecordingSearchIndex:
    results: list[SearchResult]
    calls: list[tuple[str, int]] = field(default_factory=list)

    def search(self, query: str, k: int) -> list[SearchResult]:
        self.calls.append((query, k))
        return self.results[:k]


def test_keyword_match_is_fused_into_top_k_when_dense_search_misses_it() -> None:
    dense_first = _chunk(0, "General product specifications")
    dense_second = _chunk(1, "Invoice processing procedures")
    keyword_match = _chunk(2, "The precise indemnification keyword appears here")
    dense_index = _RecordingSearchIndex(
        [
            SearchResult(chunk=dense_first, score=0.95, dense_score=0.95),
            SearchResult(chunk=dense_second, score=0.85, dense_score=0.85),
            SearchResult(chunk=keyword_match, score=0.10, dense_score=0.10),
        ]
    )
    bm25_index = _RecordingSearchIndex(
        [SearchResult(chunk=keyword_match, score=12.0, bm25_score=12.0)]
    )
    index = HybridSearchIndex(
        [dense_first, dense_second, keyword_match], dense_index, bm25_index
    )

    dense_only_top_two = dense_index.search("indemnification", k=2)
    results = index.search("indemnification", k=2)

    assert keyword_match.chunk_id not in {
        result.chunk.chunk_id for result in dense_only_top_two
    }
    assert keyword_match.chunk_id in {result.chunk.chunk_id for result in results}


def test_candidate_pool_scales_above_its_floor() -> None:
    dense_index = _RecordingSearchIndex([])
    bm25_index = _RecordingSearchIndex([])
    index = HybridSearchIndex([], dense_index, bm25_index)

    assert index.search("term", k=25) == []
    assert dense_index.calls == [("term", 25)]
    assert bm25_index.calls == [("term", 25)]


def test_hybrid_result_scores_preserve_rrf_and_raw_score_provenance() -> None:
    both = _chunk(0, "Both search methods return this clause")
    dense_only = _chunk(1, "Only dense search returns this clause")
    bm25_only = _chunk(2, "Only sparse search returns this clause")
    dense_index = _RecordingSearchIndex(
        [
            SearchResult(chunk=both, score=0.9, dense_score=0.9),
            SearchResult(chunk=dense_only, score=0.8, dense_score=0.8),
        ]
    )
    bm25_index = _RecordingSearchIndex(
        [
            SearchResult(chunk=both, score=4.0, bm25_score=4.0),
            SearchResult(chunk=bm25_only, score=3.0, bm25_score=3.0),
        ]
    )
    index = HybridSearchIndex([both, dense_only, bm25_only], dense_index, bm25_index)

    results = {result.chunk.chunk_id: result for result in index.search("clause", k=3)}

    assert all(result.rrf_score == result.score for result in results.values())
    assert results[both.chunk_id].dense_score == 0.9
    assert results[both.chunk_id].bm25_score == 4.0
    assert results[dense_only.chunk_id].dense_score == 0.8
    assert results[dense_only.chunk_id].bm25_score is None
    assert results[bm25_only.chunk_id].dense_score is None
    assert results[bm25_only.chunk_id].bm25_score == 3.0
