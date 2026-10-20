from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from pydantic import BaseModel

from contractguard.chunking import Chunk
from contractguard.extraction.checks import evidence_grounded, run_checks
from contractguard.extraction.extractor import Extractor
from contractguard.extraction.prompt import build_prompt
from contractguard.extraction.schemas import (
    LiabilityCapExtraction,
    RenewalTermExtraction,
)
from contractguard.llm.client import LLMResponse
from contractguard.llm.errors import LLMQuotaError, LLMUnavailableError


def _chunk(index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"doc-1:{index}",
        doc_id="doc-1",
        text=text,
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(text),
    )


@dataclass
class _FakeLLMClient:
    responses: list[str]
    schemas: list[type[BaseModel]] = field(default_factory=list)

    def generate_json(self, prompt: str, schema: type[BaseModel]) -> LLMResponse:
        self.schemas.append(schema)
        return LLMResponse(
            text=self.responses.pop(0),
            model="fake-model",
            input_tokens=12,
            output_tokens=8,
            latency_ms=3.5,
            cached=False,
        )


def _renewal_json(**overrides: object) -> str:
    values = {
        "found": True,
        "initial_term_months": 12,
        "auto_renews": True,
        "renewal_term_months": 12,
        "notice_days": 30,
        "evidence_quote": "initial term of twelve months",
        "evidence_chunk_id": "doc-1:0",
    }
    values.update(overrides)
    return RenewalTermExtraction(**values).model_dump_json()


def _cap_json(**overrides: object) -> str:
    values = {
        "found": True,
        "has_money_cap": True,
        "cap_amount": None,
        "currency": None,
        "cap_formula": "fees paid in the prior 12 months",
        "has_damages_exclusion": False,
        "evidence_quote": "fees paid in the prior 12 months",
        "evidence_chunk_id": "doc-1:0",
    }
    values.update(overrides)
    return LiabilityCapExtraction(**values).model_dump_json()


def test_valid_renewal_json_is_parsed_and_checked() -> None:
    chunks = [_chunk(0, "The initial term of twelve months renews automatically.")]
    result = Extractor(_FakeLLMClient([_renewal_json()])).extract(
        "Renewal Term", chunks
    )

    assert isinstance(result.extraction, RenewalTermExtraction)
    assert all(result.checks.values())
    assert result.attempts == 1
    assert result.model == "fake-model"


def test_valid_cap_formula_without_amount_is_parsed_and_checked() -> None:
    chunks = [_chunk(0, "Liability is capped at fees paid in the prior 12 months.")]
    result = Extractor(_FakeLLMClient([_cap_json()])).extract(
        "Cap On Liability", chunks
    )

    assert isinstance(result.extraction, LiabilityCapExtraction)
    assert result.extraction.cap_amount is None
    assert all(result.checks.values())


def test_clean_not_found_output_passes_not_found_check() -> None:
    chunks = [_chunk(0, "No renewal language appears here.")]
    result = Extractor(
        _FakeLLMClient([
            _renewal_json(
                found=False,
                initial_term_months=None,
                auto_renews=None,
                renewal_term_months=None,
                notice_days=None,
                evidence_quote=None,
                evidence_chunk_id=None,
            )
        ])
    ).extract("Renewal Term", chunks)

    assert result.checks["not_found_is_clean"] is True


def test_clean_cap_not_found_output_passes_not_found_check() -> None:
    result = Extractor(
        _FakeLLMClient([
            _cap_json(
                found=False,
                has_money_cap=False,
                cap_amount=None,
                currency=None,
                cap_formula=None,
                has_damages_exclusion=False,
                evidence_quote=None,
                evidence_chunk_id=None,
            )
        ])
    ).extract("Cap On Liability", [_chunk(0, "No liability terms appear here.")])

    assert result.checks["not_found_is_clean"] is True


def test_evidence_quote_missing_from_named_chunk_fails_grounding() -> None:
    extraction = RenewalTermExtraction.model_validate_json(_renewal_json())

    assert (
        evidence_grounded(extraction, [_chunk(0, "A different term applies.")])
        is False
    )


def test_evidence_in_a_different_chunk_fails_grounding() -> None:
    extraction = RenewalTermExtraction.model_validate_json(_renewal_json())
    chunks = [
        _chunk(0, "A different term applies."),
        _chunk(1, "The initial term of twelve months renews automatically."),
    ]

    assert evidence_grounded(extraction, chunks) is False


def test_grounding_normalizes_curly_quotes_and_whitespace() -> None:
    extraction = RenewalTermExtraction.model_validate_json(
        _renewal_json(evidence_quote='"fees paid"')
    )
    chunks = [_chunk(0, "The cap is  \u201cfees   paid\u201d  by the customer.")]

    assert evidence_grounded(extraction, chunks) is True


def test_grounding_removes_zero_width_space_but_preserves_real_space() -> None:
    chunks = [_chunk(0, "PURSUANT TO SECTION \u200b8 HEREUNDER")]
    spaced = RenewalTermExtraction.model_validate_json(
        _renewal_json(evidence_quote="SECTION 8")
    )
    unspaced = RenewalTermExtraction.model_validate_json(
        _renewal_json(evidence_quote="SECTION8")
    )

    assert evidence_grounded(spaced, chunks) is True
    assert evidence_grounded(unspaced, chunks) is False


def test_grounding_removes_soft_hyphen_inside_word() -> None:
    extraction = RenewalTermExtraction.model_validate_json(
        _renewal_json(evidence_quote="nonrenewal")
    )

    assert evidence_grounded(extraction, [_chunk(0, "non\u00adrenewal notice")]) is True


def test_invalid_json_is_retried_once_then_succeeds() -> None:
    client = _FakeLLMClient(["not json", _renewal_json()])
    result = Extractor(client).extract(
        "Renewal Term", [_chunk(0, "The initial term of twelve months.")]
    )

    assert isinstance(result.extraction, RenewalTermExtraction)
    assert len(client.schemas) == 2
    assert result.attempts == 2
    assert result.input_tokens == 24


def test_invalid_json_twice_returns_an_error_without_raising() -> None:
    result = Extractor(_FakeLLMClient(["not json", "still not json"])).extract(
        "Renewal Term", [_chunk(0, "The initial term is twelve months.")]
    )

    assert result.extraction is None
    assert result.error is not None


def test_found_output_without_evidence_fails_required_fields_check() -> None:
    extraction = LiabilityCapExtraction.model_validate_json(
        _cap_json(evidence_quote=None, evidence_chunk_id=None)
    )

    checks = run_checks(extraction, [_chunk(0, "A liability cap applies.")])

    assert checks["required_fields_present"] is False


def test_prompt_includes_clause_specific_definitions_and_data_boundary() -> None:
    chunks = [_chunk(0, "First chunk"), _chunk(1, "Second chunk")]
    renewal_prompt = build_prompt("Renewal Term", chunks)
    cap_prompt = build_prompt("Cap On Liability", chunks)

    assert "doc-1:0" in renewal_prompt
    assert "doc-1:1" in renewal_prompt
    assert "<contract_text>" in renewal_prompt
    assert "</contract_text>" in renewal_prompt
    assert "data, not instructions" in renewal_prompt
    assert "non-renewal" in renewal_prompt
    assert "365 days" in renewal_prompt
    assert "has_damages_exclusion" not in renewal_prompt
    assert "has_damages_exclusion" in cap_prompt
    assert "notice_days" not in cap_prompt


def test_extractor_uses_the_clause_specific_schema() -> None:
    client = _FakeLLMClient([_renewal_json(), _cap_json()])
    extractor = Extractor(client)
    renewal_chunks = [_chunk(0, "The initial term of twelve months.")]
    cap_chunks = [_chunk(0, "fees paid in the prior 12 months")]

    extractor.extract("Renewal Term", renewal_chunks)
    extractor.extract("Cap On Liability", cap_chunks)

    assert client.schemas == [RenewalTermExtraction, LiabilityCapExtraction]


@dataclass
class _ErrorLLMClient:
    error: Exception

    def generate_json(self, prompt: str, schema: type[BaseModel]) -> LLMResponse:
        raise self.error


def test_extractor_propagates_quota_errors() -> None:
    extractor = Extractor(_ErrorLLMClient(LLMQuotaError("quota exhausted")))

    with pytest.raises(LLMQuotaError, match="quota exhausted"):
        extractor.extract("Renewal Term", [_chunk(0, "A term applies.")])


def test_extractor_returns_unavailable_errors() -> None:
    extractor = Extractor(_ErrorLLMClient(LLMUnavailableError("service unavailable")))
    result = extractor.extract("Renewal Term", [_chunk(0, "A term applies.")])

    assert result.extraction is None
    assert result.error == "service unavailable"
