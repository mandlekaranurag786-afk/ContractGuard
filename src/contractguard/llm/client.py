"""Language-model client interface for structured extraction."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class LLMResponse(BaseModel):
    """A structured-generation response and its execution metadata."""

    text: str
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float | None
    cached: bool


class LLMClient(Protocol):
    """Generates JSON that conforms to a requested Pydantic schema."""

    def generate_json(self, prompt: str, schema: type[BaseModel]) -> LLMResponse:
        """Generate one JSON response for *prompt*."""
