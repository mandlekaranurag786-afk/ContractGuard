"""Document chunking package."""

from contractguard.chunking.fixed import chunk_document
from contractguard.chunking.models import Chunk

__all__ = ["Chunk", "chunk_document"]
