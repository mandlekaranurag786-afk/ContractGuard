from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from contractguard.extraction.schemas import RenewalTermExtraction
from contractguard.llm.cache import CachedLLMClient
from contractguard.llm.client import LLMResponse
from contractguard.llm.errors import LLMQuotaError, LLMUnavailableError
from contractguard.llm.fallback import FallbackLLMClient
from contractguard.llm.gemini import GeminiClient


def _renewal_json() -> str:
    return RenewalTermExtraction(
        found=False,
        initial_term_months=None,
        auto_renews=None,
        renewal_term_months=None,
        notice_days=None,
        evidence_quote=None,
        evidence_chunk_id=None,
    ).model_dump_json()


@dataclass
class _FakeLLMClient:
    responses: list[str]
    calls: int = 0

    def generate_json(self, prompt: str, schema: type[BaseModel]) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text=self.responses.pop(0),
            model="fake-model",
            input_tokens=4,
            output_tokens=2,
            latency_ms=1.5,
            cached=False,
        )


def test_cache_miss_saves_response_and_hit_avoids_inner_client(tmp_path) -> None:
    inner = _FakeLLMClient([_renewal_json()])
    client = CachedLLMClient(inner, "fake-model", tmp_path)

    miss = client.generate_json("prompt", RenewalTermExtraction)
    hit = client.generate_json("prompt", RenewalTermExtraction)

    assert miss.cached is False
    assert hit.cached is True
    assert hit.text == miss.text
    assert inner.calls == 1
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_cache_does_not_save_invalid_json(tmp_path) -> None:
    inner = _FakeLLMClient(["not json"])
    response = CachedLLMClient(inner, "fake-model", tmp_path).generate_json(
        "prompt", RenewalTermExtraction
    )

    assert response.text == "not json"
    assert not list(tmp_path.glob("*.json"))


@dataclass
class _QuotaClient:
    response: LLMResponse | None = None
    calls: int = 0

    def generate_json(self, prompt: str, schema: type[BaseModel]) -> LLMResponse:
        self.calls += 1
        if self.response is None:
            raise LLMQuotaError("quota exhausted")
        return self.response


def test_fallback_uses_next_client_after_quota_error() -> None:
    second_response = _FakeLLMClient([_renewal_json()]).generate_json(
        "prompt", RenewalTermExtraction
    )
    first = _QuotaClient()
    second = _QuotaClient(second_response)

    result = FallbackLLMClient([first, second]).generate_json(
        "prompt", RenewalTermExtraction
    )

    assert result is second_response
    assert first.calls == 1
    assert second.calls == 1


def test_fallback_raises_last_quota_error_when_all_clients_fail() -> None:
    with pytest.raises(LLMQuotaError, match="quota exhausted"):
        FallbackLLMClient([_QuotaClient(), _QuotaClient()]).generate_json(
            "prompt", RenewalTermExtraction
        )


@dataclass
class _FakeModels:
    outcomes: list[object]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def generate_content(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@dataclass
class _FakeSDKClient:
    models: _FakeModels


class _APIError(Exception):
    def __init__(self, code: int, message: str = "") -> None:
        super().__init__(message)
        self.code = code


def _sdk_response() -> object:
    return SimpleNamespace(
        text=_renewal_json(),
        usage_metadata=SimpleNamespace(prompt_token_count=11, candidates_token_count=7),
    )


def test_gemini_retries_rate_limits_then_returns_response(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    sleeps: list[float] = []
    monkeypatch.setattr("contractguard.llm.gemini.time.sleep", sleeps.append)
    sdk = _FakeSDKClient(_FakeModels([_APIError(429), _APIError(429), _sdk_response()]))

    response = GeminiClient(min_interval_s=0, sdk_client=sdk).generate_json(
        "prompt", RenewalTermExtraction
    )

    assert response.text == _renewal_json()
    assert response.input_tokens == 11
    assert response.output_tokens == 7
    assert len(sdk.models.calls) == 3
    assert sleeps == [5, 15]
    assert (
        sdk.models.calls[0]["config"].response_schema is RenewalTermExtraction
    )


def test_gemini_latency_excludes_backoff_waits(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("contractguard.llm.gemini.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "contractguard.llm.gemini.time.perf_counter",
        iter([0.0, 10.0, 10.123]).__next__,
    )
    sdk = _FakeSDKClient(_FakeModels([_APIError(500), _sdk_response()]))

    response = GeminiClient(min_interval_s=0, sdk_client=sdk).generate_json(
        "prompt", RenewalTermExtraction
    )

    assert response.latency_ms == pytest.approx(123)


@pytest.mark.parametrize("error", [ConnectionError(), TimeoutError()])
def test_gemini_retries_connection_and_timeout_errors(monkeypatch, error) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("contractguard.llm.gemini.time.sleep", lambda _: None)
    sdk = _FakeSDKClient(_FakeModels([error] * 4))

    with pytest.raises(LLMUnavailableError, match="remained unavailable"):
        GeminiClient(min_interval_s=0, sdk_client=sdk).generate_json(
            "prompt", RenewalTermExtraction
        )

    assert len(sdk.models.calls) == 4


def test_gemini_raises_quota_error_after_all_rate_limit_retries(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("contractguard.llm.gemini.time.sleep", lambda _: None)
    sdk = _FakeSDKClient(_FakeModels([_APIError(429)] * 4))

    with pytest.raises(LLMQuotaError, match="rate or quota"):
        GeminiClient(min_interval_s=0, sdk_client=sdk).generate_json(
            "prompt", RenewalTermExtraction
        )


def test_gemini_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(
        ValueError, match="GEMINI_API_KEY environment variable is required"
    ):
        GeminiClient(sdk_client=object())
