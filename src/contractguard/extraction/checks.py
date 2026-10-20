"""Consistency checks for permissive extraction outputs."""

from __future__ import annotations

import re

from contractguard.chunking import Chunk
from contractguard.extraction.schemas import (
    LiabilityCapExtraction,
    RenewalTermExtraction,
)

Extraction = RenewalTermExtraction | LiabilityCapExtraction


def normalize(text: str) -> str:
    """Normalize whitespace and common typographic punctuation."""
    translation = str.maketrans({
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
    })
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


def run_checks(extraction: Extraction, chunks: list[Chunk]) -> dict[str, bool]:
    """Run all non-throwing consistency checks for an extraction."""
    return {
        "evidence_grounded": evidence_grounded(extraction, chunks),
        "required_fields_present": required_fields_present(extraction),
        "not_found_is_clean": not_found_is_clean(extraction),
    }
