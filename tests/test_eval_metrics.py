from __future__ import annotations

import pytest

from contractguard.evaluation.metrics import first_hit_rank, mrr_at_k, recall_at_k


def test_first_hit_rank_returns_earliest_qualifying_rank() -> None:
    overlaps = [0.6, 0.2, 0.3, 1.0]

    assert first_hit_rank(overlaps, 0.5) == 1
    assert first_hit_rank(overlaps, 0.9) == 4


def test_first_hit_rank_returns_none_without_a_qualifying_overlap() -> None:
    assert first_hit_rank([0.1, 0.2], 0.5) is None
    assert first_hit_rank([], 0.5) is None


def test_first_hit_rank_treats_threshold_as_inclusive() -> None:
    assert first_hit_rank([0.5], 0.5) == 1


def test_recall_at_k_counts_first_hits_at_or_before_k() -> None:
    first_hit_ranks = [1, 4, None, 2]

    assert recall_at_k(first_hit_ranks, k=1) == 0.25
    assert recall_at_k(first_hit_ranks, k=3) == 0.5


def test_recall_at_k_returns_zero_for_no_instances() -> None:
    assert recall_at_k([], k=5) == 0.0


def test_mrr_at_k_scores_hits_with_reciprocal_rank() -> None:
    ranks = [1, 2, None, 5]

    assert mrr_at_k(ranks, k=10) == pytest.approx(0.425)
    assert mrr_at_k(ranks, k=3) == pytest.approx(0.375)


def test_mrr_at_k_returns_zero_for_all_misses_and_empty_input() -> None:
    assert mrr_at_k([None, None], k=10) == pytest.approx(0.0)
    assert mrr_at_k([], k=5) == pytest.approx(0.0)


def test_mrr_at_k_counts_a_hit_exactly_at_k() -> None:
    assert mrr_at_k([5], k=5) == pytest.approx(1 / 5)
