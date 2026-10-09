from __future__ import annotations

import numpy as np

from contractguard.chunking import Chunk
from contractguard.retrieval import InMemoryVectorIndex, embed_chunks


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


def _topic_chunks() -> list[Chunk]:
    return [
        _chunk(0, "The liability cap is limited to fees paid under this agreement."),
        _chunk(1, "The renewal term automatically extends for twelve months."),
        _chunk(2, "Invoices are due thirty days after the invoice date."),
    ]


def test_embedding_the_same_chunk_twice_is_deterministic() -> None:
    chunk = _topic_chunks()[0]

    first = embed_chunks([chunk])
    second = embed_chunks([chunk])

    assert np.array_equal(first, second)


def test_search_results_are_descending_and_limited_by_k() -> None:
    index = InMemoryVectorIndex.from_chunks(_topic_chunks())

    results = index.search("contract terms", k=2)

    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_matching_query_ranks_its_topic_highly() -> None:
    index = InMemoryVectorIndex.from_chunks(_topic_chunks())

    results = index.search("What is the maximum liability cap?", k=3)

    assert results[0].chunk.chunk_id == "doc-1:0"


def test_saved_index_returns_identical_search_results(tmp_path) -> None:
    index = InMemoryVectorIndex.from_chunks(_topic_chunks())
    before = index.search("How long is the renewal term?", k=3)
    path = tmp_path / "embeddings.npz"
    index.save(path)

    after = InMemoryVectorIndex.load(path).search("How long is the renewal term?", k=3)

    assert [(result.chunk.chunk_id, result.score) for result in after] == [
        (result.chunk.chunk_id, result.score) for result in before
    ]
