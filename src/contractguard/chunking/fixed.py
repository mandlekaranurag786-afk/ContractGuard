"""Deterministic fixed-size document chunking."""

from __future__ import annotations

from contractguard.chunking.models import Chunk
from contractguard.ingestion import ParsedDocument


def chunk_document(
    parsed_document: ParsedDocument,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[Chunk]:
    """Split a parsed document into overlapping, page-aware text windows."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    source_text, page_for_character = _concatenate_pages(parsed_document)
    if not source_text or not source_text.strip():
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    window_start = 0
    window_index = 0
    while window_start < len(source_text):
        window_end = min(window_start + chunk_size, len(source_text))
        text = source_text[window_start:window_end]

        if text.strip():
            pages = page_for_character[window_start:window_end]
            # IDs use the full fixed-window position, so skipped windows leave gaps.
            chunks.append(
                Chunk(
                    chunk_id=f"{parsed_document.doc_id}:{window_index}",
                    doc_id=parsed_document.doc_id,
                    text=text,
                    page_start=min(pages),
                    page_end=max(pages),
                    char_start=window_start,
                    char_end=window_end,
                )
            )

        if window_end == len(source_text):
            break
        window_start += step
        window_index += 1

    return chunks


def _concatenate_pages(parsed_document: ParsedDocument) -> tuple[str, list[int]]:
    text_parts: list[str] = []
    page_for_character: list[int] = []

    for index, page in enumerate(parsed_document.pages):
        if index:
            text_parts.append("\n")
            page_for_character.append(parsed_document.pages[index - 1].page_number)
        text_parts.append(page.clean_text)
        page_for_character.extend([page.page_number] * len(page.clean_text))

    return "".join(text_parts), page_for_character
