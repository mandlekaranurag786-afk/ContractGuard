"""Document retrieval package."""

from contractguard.retrieval.models import SearchResult
from contractguard.retrieval.vector import (
    MODEL_NAME,
    InMemoryVectorIndex,
    embed_chunks,
)

__all__ = ["MODEL_NAME", "InMemoryVectorIndex", "SearchResult", "embed_chunks"]
