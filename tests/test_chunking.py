from __future__ import annotations

import pytest

from contractguard.chunking import Chunk, chunk_document
from contractguard.ingestion import ParsedDocument, ParsedPage


def _document(*texts: str) -> ParsedDocument:
    return ParsedDocument(
        doc_id="doc-123",
        original_filename="contract.pdf",
        page_count=len(texts),
        pages=[
            ParsedPage(
                page_number=index,
                raw_text=text,
                clean_text=text,
                needs_ocr=False,
            )
            for index, text in enumerate(texts, start=1)
        ],
    )


def test_every_non_whitespace_source_character_is_covered() -> None:
    document = _document("abcdefghij" * 3)
    chunks = chunk_document(document, chunk_size=10, overlap=2)
    source = "\n".join(page.clean_text for page in document.pages)

    covered = {
        index for chunk in chunks for index in range(chunk.char_start, chunk.char_end)
    }

    assert all(
        character.isspace() or index in covered
        for index, character in enumerate(source)
    )


def test_mid_document_whitespace_gap_is_skipped_but_text_both_sides_is_covered() -> (
    None
):
    document = _document("abcdefghij", "          ", "klmnopqrst")
    chunks = chunk_document(document, chunk_size=10, overlap=0)
    source = "\n".join(page.clean_text for page in document.pages)

    covered = {
        index for chunk in chunks for index in range(chunk.char_start, chunk.char_end)
    }
    gap_start = len("abcdefghij\n")
    gap_end = gap_start + len("          ")

    assert all(
        character.isspace() or index in covered
        for index, character in enumerate(source)
    )
    assert all(
        not (chunk.char_start >= gap_start and chunk.char_end <= gap_end)
        for chunk in chunks
    )
    assert any(index < gap_start for index in covered)
    assert any(index >= gap_end for index in covered)


def test_consecutive_windows_overlap_by_configured_amount() -> None:
    chunks = chunk_document(_document("a" * 30), chunk_size=10, overlap=3)

    for first, second in zip(chunks, chunks[1:]):
        assert first.char_end - second.char_start == 3
        assert first.text[-3:] == second.text[:3]


def test_page_bounds_are_correct_across_a_page_boundary() -> None:
    document = _document("a" * 8, "b" * 8)

    chunks = chunk_document(document, chunk_size=10, overlap=0)

    spanning_chunk = next(
        chunk for chunk in chunks if chunk.page_start == 1 and chunk.page_end == 2
    )
    assert spanning_chunk.text == "a" * 8 + "\n" + "b"
    assert spanning_chunk.char_start == 0
    assert spanning_chunk.char_end == 10


def test_chunking_is_deterministic() -> None:
    document = _document("a" * 1400)

    first = chunk_document(document)
    second = chunk_document(document)

    assert [(chunk.chunk_id, chunk.text) for chunk in first] == [
        (chunk.chunk_id, chunk.text) for chunk in second
    ]


def test_document_smaller_than_chunk_size_returns_one_chunk() -> None:
    chunks = chunk_document(_document("short contract"), chunk_size=1000, overlap=150)

    assert len(chunks) == 1
    assert chunks[0].text == "short contract"


def test_chunk_size_and_overlap_are_validated() -> None:
    document = _document("text")

    with pytest.raises(ValueError, match="chunk_size"):
        chunk_document(document, chunk_size=0)
    with pytest.raises(ValueError, match="overlap"):
        chunk_document(document, chunk_size=10, overlap=-1)
    with pytest.raises(ValueError, match="overlap"):
        chunk_document(document, chunk_size=10, overlap=10)


def test_whitespace_only_document_returns_no_chunks() -> None:
    assert chunk_document(_document("   \n\t  ")) == []


def test_output_models_are_chunks() -> None:
    assert isinstance(chunk_document(_document("text"))[0], Chunk)
