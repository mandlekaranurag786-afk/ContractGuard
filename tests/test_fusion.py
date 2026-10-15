from __future__ import annotations

import pytest

from contractguard.retrieval.fusion import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_scores_match_hand_calculation() -> None:
    list_a = ["c1", "c2", "c3"]
    list_b = ["c3", "c1", "c2"]

    results = reciprocal_rank_fusion([list_a, list_b], k=60)

    expected = {
        "c1": 1 / 61 + 1 / 62,
        "c2": 1 / 62 + 1 / 63,
        "c3": 1 / 63 + 1 / 61,
    }
    assert [chunk_id for chunk_id, _ in results] == ["c1", "c3", "c2"]
    for chunk_id, score in results:
        assert score == pytest.approx(expected[chunk_id], abs=1e-6)


def test_reciprocal_rank_fusion_breaks_exact_ties_by_chunk_id() -> None:
    results = reciprocal_rank_fusion([["b"], ["a"]], k=60)

    assert results == [("a", 1 / 61), ("b", 1 / 61)]


def test_reciprocal_rank_fusion_handles_an_empty_and_a_non_empty_list() -> None:
    results = reciprocal_rank_fusion([[], ["c2", "c1"]], k=60)

    assert results == [("c2", 1 / 61), ("c1", 1 / 62)]


def test_reciprocal_rank_fusion_handles_all_empty_lists() -> None:
    assert reciprocal_rank_fusion([[], []]) == []
