"""Chunk models."""

from pydantic import BaseModel


class Chunk(BaseModel):
    """A fixed window of normalized contract text."""

    chunk_id: str
    doc_id: str
    text: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
