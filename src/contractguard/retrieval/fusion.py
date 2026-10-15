"""Rank-list fusion helpers."""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked chunk ID lists with Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
