"""Google Gemini implementation of the structured LLM client."""

from __future__ import annotations

import os
import time
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel

from contractguard.llm.client import LLMResponse
from contractguard.llm.errors import LLMQuotaError, LLMUnavailableError

BACKOFF_SECONDS = (5, 15, 45)


class GeminiClient:
    """Generate structured JSON through Gemini with pacing and retries."""

    def __init__(
        self,
        model: str = "gemini-3.5-flash-lite",
        min_interval_s: float = 4.0,
        max_retries: int = 3,
        sdk_client: Any | None = None,
    ) -> None:
        try:
            api_key = os.environ["GEMINI_API_KEY"]
        except KeyError as exc:
            raise ValueError("GEMINI_API_KEY environment variable is required") from exc

        self.model = model
        self._min_interval_s = min_interval_s
        self._max_retries = max_retries
        self._sdk_client = sdk_client or genai.Client(api_key=api_key)
        self._last_request_started_at: float | None = None

    def generate_json(self, prompt: str, schema: type[BaseModel]) -> LLMResponse:
        """Generate JSON matching *schema*, retrying transient API failures."""
        response, latency_ms = self._generate_with_retries(prompt, schema)
        return LLMResponse(
            text=response.text,
            model=self.model,
            input_tokens=_usage_value(response, "prompt_token_count"),
            output_tokens=_usage_value(response, "candidates_token_count"),
            latency_ms=latency_ms,
            cached=False,
        )

    def _generate_with_retries(
        self, prompt: str, schema: type[BaseModel]
    ) -> tuple[Any, float]:
        for attempt in range(self._max_retries + 1):
            self._pace_request()
            try:
                started_at = time.perf_counter()
                response = self._sdk_client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0,
                    ),
                )
                return response, (time.perf_counter() - started_at) * 1_000
            except Exception as exc:
                if not _is_retryable(exc):
                    raise LLMUnavailableError("Gemini API request failed") from exc
                if attempt == self._max_retries:
                    if _is_rate_limit(exc):
                        raise LLMQuotaError(
                            "Gemini rate or quota limit exhausted"
                        ) from exc
                    raise LLMUnavailableError(
                        "Gemini API remained unavailable"
                    ) from exc
                time.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])

        raise AssertionError("unreachable")

    def _pace_request(self) -> None:
        now = time.monotonic()
        if self._last_request_started_at is not None:
            elapsed = now - self._last_request_started_at
            if elapsed < self._min_interval_s:
                time.sleep(self._min_interval_s - elapsed)
        self._last_request_started_at = time.monotonic()


def _usage_value(response: Any, attribute: str) -> int | None:
    usage = getattr(response, "usage_metadata", None)
    value = getattr(usage, attribute, None)
    return value if isinstance(value, int) else None


def _is_rate_limit(exc: Exception) -> bool:
    return getattr(exc, "code", None) == 429 or "resource_exhausted" in str(exc).lower()


def _is_retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    return (
        _is_rate_limit(exc)
        or isinstance(code, int) and 500 <= code < 600
        or code is None and isinstance(exc, (ConnectionError, TimeoutError))
    )
