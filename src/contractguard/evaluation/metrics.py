"""Pure metrics for ranked retrieval results."""


def first_hit_rank(rank_overlaps: list[float], threshold: float) -> int | None:
    """Return the first 1-indexed rank meeting an overlap threshold."""
    for rank, overlap in enumerate(rank_overlaps, start=1):
        if overlap >= threshold:
            return rank
    return None


def recall_at_k(first_hit_ranks: list[int | None], k: int) -> float:
    """Return the fraction of instances with a hit at or before rank *k*."""
    if not first_hit_ranks:
        return 0.0
    hits = sum(rank is not None and rank <= k for rank in first_hit_ranks)
    return hits / len(first_hit_ranks)


def mrr_at_k(first_hit_ranks: list[int | None], k: int) -> float:
    """Return the mean reciprocal rank for hits at or before rank *k*."""
    if not first_hit_ranks:
        return 0.0
    scores = [
        1 / rank if rank is not None and 1 <= rank <= k else 0.0
        for rank in first_hit_ranks
    ]
    return sum(scores) / len(scores)
