"""Disk cache for valid structured LLM responses."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from contractguard.llm.client import LLMClient, LLMResponse

DEFAULT_CACHE_DIR = Path("evals/llm_cache")


class CachedLLMClient:
    """Cache valid JSON responses from another structured LLM client."""

    def __init__(
        self, inner: LLMClient, model: str, cache_dir: Path = DEFAULT_CACHE_DIR
    ) -> None:
        self._inner = inner
        self._cache_dir = cache_dir
        self._model = model

    def generate_json(self, prompt: str, schema: type[BaseModel]) -> LLMResponse:
        """Return a cached valid response, or obtain and persist a fresh one."""
        cache_path = self._cache_dir / f"{self._cache_key(prompt, schema)}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return LLMResponse(
                text=cached["response_text"],
                model=cached["model"],
                input_tokens=cached["input_tokens"],
                output_tokens=cached["output_tokens"],
                latency_ms=cached["latency_ms"],
                cached=True,
            )

        response = self._inner.generate_json(prompt, schema)
        try:
            schema.model_validate_json(response.text)
        except (ValidationError, ValueError):
            return response

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "model": response.model,
                    "response_text": response.text,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "latency_ms": response.latency_ms,
                    "created_at": datetime.now(UTC).isoformat(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return response

    def _cache_key(self, prompt: str, schema: type[BaseModel]) -> str:
        schema_json = json.dumps(
            schema.model_json_schema(), sort_keys=True, separators=(",", ":")
        )
        key_material = "\n".join((self._model, schema.__name__, schema_json, prompt))
        return hashlib.sha256(key_material.encode("utf-8")).hexdigest()
