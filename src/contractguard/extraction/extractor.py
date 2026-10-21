"""LLM-backed, evidence-grounded clause extraction."""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from contractguard.chunking import Chunk
from contractguard.extraction.checks import run_checks
from contractguard.extraction.prompt import build_prompt
from contractguard.extraction.schemas import (
    ExtractionResult,
    LiabilityCapExtraction,
    RenewalTermExtraction,
)
from contractguard.llm.client import LLMClient, LLMResponse
from contractguard.llm.errors import LLMError, LLMQuotaError

SCHEMAS: dict[str, type[BaseModel]] = {
    "Renewal Term": RenewalTermExtraction,
    "Cap On Liability": LiabilityCapExtraction,
}


def _sum_optional_int(values: list[int | None]) -> int | None:
    """Sum values while retaining absent metadata as None."""
    non_none_values = [value for value in values if value is not None]
    return sum(non_none_values) if non_none_values else None


def _sum_optional_float(values: list[float | None]) -> float | None:
    """Sum values while retaining absent metadata as None."""
    non_none_values = [value for value in values if value is not None]
    return sum(non_none_values) if non_none_values else None


class Extractor:
    """Extracts a clause using a structured-output language-model client."""

    def __init__(self, client: LLMClient, prompt_version: str = "v1") -> None:
        self._client = client
        self._prompt_version = prompt_version

    def extract(self, clause_type: str, chunks: list[Chunk]) -> ExtractionResult:
        """Extract a clause, retrying once when the model returns invalid JSON."""
        schema = SCHEMAS[clause_type]
        prompt = build_prompt(clause_type, chunks, version=self._prompt_version)
        response: LLMResponse | None = None
        responses: list[LLMResponse] = []
        error: ValidationError | ValueError | None = None

        for _ in range(2):
            try:
                response = self._client.generate_json(prompt, schema)
            except LLMQuotaError:
                raise
            except LLMError as exc:
                return ExtractionResult(
                    clause_type=clause_type,
                    extraction=None,
                    checks={},
                    attempts=len(responses) + 1,
                    model=None,
                    cached=False,
                    latency_ms=None,
                    input_tokens=None,
                    output_tokens=None,
                    error=str(exc),
                )
            responses.append(response)
            try:
                extraction = schema.model_validate_json(response.text)
            except (ValidationError, ValueError) as exc:
                error = exc
                continue
            return ExtractionResult(
                clause_type=clause_type,
                extraction=extraction,
                checks=run_checks(extraction, chunks),
                attempts=len(responses),
                model=response.model,
                cached=response.cached,
                latency_ms=_sum_optional_float([item.latency_ms for item in responses]),
                input_tokens=_sum_optional_int(
                    [item.input_tokens for item in responses]
                ),
                output_tokens=_sum_optional_int(
                    [item.output_tokens for item in responses]
                ),
                error=None,
            )

        assert response is not None
        assert error is not None
        return ExtractionResult(
            clause_type=clause_type,
            extraction=None,
            checks={},
            attempts=len(responses),
            model=response.model,
            cached=response.cached,
            latency_ms=_sum_optional_float([item.latency_ms for item in responses]),
            input_tokens=_sum_optional_int([item.input_tokens for item in responses]),
            output_tokens=_sum_optional_int([item.output_tokens for item in responses]),
            error=str(error),
        )
