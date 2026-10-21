import pytest

from contractguard.chunking import Chunk
from contractguard.extraction.checks import (
    build_context_spans,
    cited_chunk_correct,
    evidence_composite,
    evidence_in_context,
    evidence_level,
)
from contractguard.extraction.schemas import RenewalTermExtraction


def chunk(index, start, text):
    return Chunk(
        chunk_id=f"doc:{index}",
        doc_id="doc",
        text=text,
        char_start=start,
        char_end=start + len(text),
        page_start=1,
        page_end=1,
    )


def extraction(quote, cited="doc:0", found=True):
    return RenewalTermExtraction(
        found=found,
        evidence_quote=quote,
        evidence_chunk_id=cited,
        initial_term_months=None,
        auto_renews=None,
        renewal_term_months=None,
        notice_days=None,
    )


def test_disjoint_chunks_cannot_form_fake_quote():
    chunks = [chunk(0, 0, "hello "), chunk(1, 20, "world")]
    assert len(build_context_spans(chunks)) == 2
    assert not evidence_in_context(extraction("hello world"), chunks)
    assert cited_chunk_correct(extraction("hello world"), chunks) is None


@pytest.mark.parametrize("second_start,second_text", [(6, "world!"), (3, "lo world!")])
def test_touching_and_overlapping_chunks_merge_without_duplicate_text(
    second_start, second_text
):
    chunks = [chunk(1, second_start, second_text), chunk(0, 0, "hello world")]
    assert build_context_spans(chunks)[0].text == "hello world!"
    assert evidence_in_context(extraction("hello world!"), chunks)
    assert evidence_level(extraction("hello world!"), chunks) == "in_context"
    assert cited_chunk_correct(extraction("hello world!"), chunks) is True


def test_whitespace_and_raw_citation_ranges():
    chunks = [chunk(0, 0, "a \u200b b"), chunk(1, 5, " cdef")]
    assert evidence_in_context(extraction("ab"), chunks)
    assert cited_chunk_correct(extraction("ab"), chunks) is True
    assert cited_chunk_correct(extraction("ab", "doc:1"), chunks) is False
    assert cited_chunk_correct(extraction("absent"), chunks) is None


@pytest.mark.parametrize("ellipsis", ["...", "…"])
def test_composite_is_distinct_from_in_context(ellipsis):
    chunks = [chunk(0, 0, "hello"), chunk(1, 20, "world")]
    item = extraction(f"hello {ellipsis} world")
    assert evidence_composite(item, chunks)
    assert not evidence_in_context(item, chunks)
    assert evidence_level(item, chunks) == "composite"
    assert cited_chunk_correct(item, chunks) is None


def test_levels_strict_and_abstention():
    chunks = [chunk(0, 0, "hello")]
    assert evidence_level(extraction("hello"), chunks) == "strict"
    assert evidence_level(extraction("missing"), chunks) == "not_found"
    assert evidence_level(extraction(None, found=False), chunks) is None
