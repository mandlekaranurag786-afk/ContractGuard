"""Permissive structured outputs for contract-clause extraction."""

from __future__ import annotations

from pydantic import BaseModel


class RenewalTermExtraction(BaseModel):
    """Extracted renewal-term facts and their supporting evidence."""

    found: bool
    initial_term_months: int | None
    auto_renews: bool | None
    renewal_term_months: int | None
    notice_days: int | None
    evidence_quote: str | None
    evidence_chunk_id: str | None


class LiabilityCapExtraction(BaseModel):
    """Extracted liability-cap facts and their supporting evidence."""

    found: bool
    has_money_cap: bool
    cap_amount: float | None
    currency: str | None
    cap_formula: str | None
    has_damages_exclusion: bool
    evidence_quote: str | None
    evidence_chunk_id: str | None


class ExtractionResult(BaseModel):
    """One extraction attempt, including consistency-check outcomes."""

    clause_type: str
    extraction: RenewalTermExtraction | LiabilityCapExtraction | None
    checks: dict[str, bool]
    attempts: int
    model: str | None
    cached: bool
    latency_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    error: str | None
