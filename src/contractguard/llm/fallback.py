"""Fallback routing for rate-limited LLM clients."""

from __future__ import annotations

from pydantic import BaseModel

from contractguard.llm.client import LLMClient, LLMResponse
from contractguard.llm.errors import LLMQuotaError


class FallbackLLMClient:
    """Try successive clients only when the current client exhausts quota."""

    def __init__(self, clients: list[LLMClient]) -> None:
        if not clients:
            raise ValueError("FallbackLLMClient requires at least one client")
        self._clients = clients

    def generate_json(self, prompt: str, schema: type[BaseModel]) -> LLMResponse:
        """Generate through the first client that has available quota."""
        last_error: LLMQuotaError | None = None
        for client in self._clients:
            try:
                return client.generate_json(prompt, schema)
            except LLMQuotaError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error
