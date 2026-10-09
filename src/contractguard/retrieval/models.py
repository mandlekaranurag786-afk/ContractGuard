"""Models returned by vector retrieval."""

from pydantic import BaseModel

from contractguard.chunking import Chunk


class SearchResult(BaseModel):
    """A chunk returned from vector search with its cosine similarity score."""

    chunk: Chunk
    score: float
