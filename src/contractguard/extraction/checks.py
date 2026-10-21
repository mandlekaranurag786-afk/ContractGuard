"""Consistency checks for permissive extraction outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass

from contractguard.chunking import Chunk
from contractguard.extraction.schemas import (
    LiabilityCapExtraction,
    RenewalTermExtraction,
)

Extraction = RenewalTermExtraction | LiabilityCapExtraction


def normalize(text: str) -> str:
    """Normalize whitespace and common typographic punctuation."""
    translation = str.maketrans(
        {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "-",
            "\u00ad": "",
            "\u200b": "",
            "\u200c": "",
            "\u200d": "",
            "\u2060": "",
            "\ufeff": "",
        }
    )
    return re.sub(r"\s+", " ", text.translate(translation)).strip()


def evidence_grounded(extraction: Extraction, chunks: list[Chunk]) -> bool:
    """Return whether the stated quote appears in its cited chunk."""
    if not extraction.evidence_quote or not extraction.evidence_chunk_id:
        return False
    quote = normalize(extraction.evidence_quote)
    if not quote:
        return False
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    chunk = chunk_by_id.get(extraction.evidence_chunk_id)
    if chunk is None:
        return False
    return quote in normalize(chunk.text)


def required_fields_present(extraction: Extraction) -> bool:
    """Return whether a found extraction includes evidence and a value."""
    if not extraction.found:
        return True
    if not extraction.evidence_quote or not extraction.evidence_chunk_id:
        return False
    if isinstance(extraction, RenewalTermExtraction):
        return any(
            value is not None
            for value in (
                extraction.initial_term_months,
                extraction.auto_renews,
                extraction.renewal_term_months,
                extraction.notice_days,
            )
        )
    return extraction.has_money_cap or extraction.has_damages_exclusion


def not_found_is_clean(extraction: Extraction) -> bool:
    """Return whether a not-found extraction contains no contradictory facts."""
    if extraction.found:
        return True
    if isinstance(extraction, RenewalTermExtraction):
        return (
            extraction.initial_term_months is None
            and extraction.auto_renews is None
            and extraction.renewal_term_months is None
            and extraction.notice_days is None
            and extraction.evidence_quote is None
            and extraction.evidence_chunk_id is None
        )
    return (
        not extraction.has_money_cap
        and extraction.cap_amount is None
        and extraction.currency is None
        and extraction.cap_formula is None
        and not extraction.has_damages_exclusion
        and extraction.evidence_quote is None
        and extraction.evidence_chunk_id is None
    )


@dataclass
class ContextSpan:
    doc_id: str
    char_start: int
    char_end: int
    text: str


def build_context_spans(chunks: list[Chunk]) -> list[ContextSpan]:
    """Merge only touching/overlapping ranges within the same document."""
    spans: list[ContextSpan] = []
    for chunk in sorted(chunks, key=lambda c: (c.doc_id, c.char_start)):
        if (
            spans
            and spans[-1].doc_id == chunk.doc_id
            and chunk.char_start <= spans[-1].char_end
        ):
            span = spans[-1]
            if chunk.char_end > span.char_end:
                span.text += chunk.text[span.char_end - chunk.char_start :]
                span.char_end = chunk.char_end
        else:
            spans.append(
                ContextSpan(chunk.doc_id, chunk.char_start, chunk.char_end, chunk.text)
            )
    return spans


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", normalize(text))


def evidence_in_context(extraction: Extraction, chunks: list[Chunk]) -> bool:
    quote = _compact(extraction.evidence_quote or "")
    return bool(quote) and any(
        quote in _compact(s.text) for s in build_context_spans(chunks)
    )


def evidence_composite(extraction: Extraction, chunks: list[Chunk]) -> bool:
    quote = extraction.evidence_quote or ""
    if not re.search(r"\.{3,}|\u2026", quote) or evidence_in_context(
        extraction, chunks
    ):
        return False
    parts = [_compact(p) for p in re.split(r"\.{3,}|\u2026", quote)]
    parts = [p for p in parts if p]
    spans = [_compact(s.text) for s in build_context_spans(chunks)]
    return bool(parts) and all(any(p in s for s in spans) for p in parts)


def cited_chunk_correct(extraction: Extraction, chunks: list[Chunk]) -> bool | None:
    """Accept any occurrence whose raw range overlaps the cited chunk."""
    quote = _compact(extraction.evidence_quote or "")
    if not quote:
        return None
    cited = next(
        (c for c in chunks if c.chunk_id == extraction.evidence_chunk_id), None
    )
    matched = False
    for span in build_context_spans(chunks):
        text = _compact(span.text)
        start = text.find(quote)
        if start < 0:
            continue
        matched = True
        # Normalization is character-wise after removing all whitespace.
        offsets = [
            span.char_start + i
            for i, char in enumerate(span.text)
            for _ in _compact(char)
        ]
        while start >= 0:
            end = start + len(quote)
            if (
                cited
                and cited.doc_id == span.doc_id
                and (
                    offsets[start] < cited.char_end
                    and offsets[end - 1] + 1 > cited.char_start
                )
            ):
                return True
            start = text.find(quote, start + 1)
    return False if matched else None


def evidence_level(extraction: Extraction, chunks: list[Chunk]) -> str | None:
    if not extraction.found:
        return None
    if evidence_grounded(extraction, chunks):
        return "strict"
    if evidence_in_context(extraction, chunks):
        return "in_context"
    if evidence_composite(extraction, chunks):
        return "composite"
    return "not_found"


def run_checks(
    extraction: Extraction, chunks: list[Chunk]
) -> dict[str, bool | str | None]:
    """Run all non-throwing consistency checks for an extraction."""
    return {
        "evidence_grounded": evidence_grounded(extraction, chunks),
        "required_fields_present": required_fields_present(extraction),
        "not_found_is_clean": not_found_is_clean(extraction),
        "evidence_in_context": evidence_in_context(extraction, chunks),
        "evidence_composite": evidence_composite(extraction, chunks),
        "cited_chunk_correct": cited_chunk_correct(extraction, chunks),
        "evidence_level": evidence_level(extraction, chunks),
    }
