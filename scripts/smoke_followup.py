"""Diagnose one field-generation attempt; --dry-run prints schemas only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contractguard.extraction.schemas import LiabilityCapExtraction
from contractguard.followup.extract import FIELD_SCHEMAS, FieldExtractor
from contractguard.llm.cache import CachedLLMClient
from contractguard.llm.errors import exception_details

CLAUSE = "Cap On Liability"
FIELD = "has_damages_exclusion"


def print_schemas() -> None:
    # GeminiClient passes these classes as response_schema. Print each class's
    # unmodified JSON schema identically; SDK wire conversion happens afterward.
    for name, schema in (
        ("FOLLOW-UP", FIELD_SCHEMAS[CLAUSE, FIELD]),
        ("MAIN", LiabilityCapExtraction),
    ):
        print(f"{name}: response_schema={schema.__name__}")
        print(json.dumps(schema.model_json_schema(), indent=2))


class OneCallClient:
    """No second generation, even if FieldExtractor would retry invalid JSON."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def generate_json(self, prompt, schema):
        if self.calls:
            raise RuntimeError("Smoke test permits only one generation attempt")
        self.calls += 1
        response = self.inner.generate_json(prompt, schema)
        # Let validation failure escape immediately, preserving its exception type
        # and full message rather than allowing another live generation.
        schema.model_validate_json(response.text)
        return response


def load_chunks():
    # Keep embedding loading offline, even when the explicit Gemini call is live.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from contractguard.pipeline import build_retriever, chunk_document, load_document

    paths = sorted(Path("data/contracts").glob("XpresspaGroupInc*"))
    paths = [p for p in paths if p.suffix.lower() == ".pdf"]
    if len(paths) != 1:
        raise ValueError(f"Expected one Xpresspa PDF; found {len(paths)}")
    chunks = chunk_document(load_document(paths[0]))
    build_retriever(chunks)
    selected = []
    for suffix in (":44", ":45"):
        matches = [c for c in chunks if c.chunk_id.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one chunk ending {suffix}; found {len(matches)}"
            )
        selected.extend(matches)
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print_schemas()
    if args.dry_run:
        print("Dry run: no document/model loading, client creation, or API calls.")
        return 0
    try:
        chunks = load_chunks()
        print("Chunks:", [c.chunk_id for c in chunks])
        from google import genai
        from google.genai import types

        from contractguard.llm.gemini import GeminiClient

        with genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)
            ),
        ) as sdk:
            gemini = GeminiClient(max_retries=0, sdk_client=sdk)
            with TemporaryDirectory(
                prefix="contractguard-followup-smoke-"
            ) as directory:
                client = OneCallClient(
                    CachedLLMClient(
                        gemini, model=gemini.model, cache_dir=Path(directory)
                    )
                )
                result = FieldExtractor(client).extract(CLAUSE, FIELD, chunks)
        if result.error:
            print("FAILED (exception types and full cause-chain messages):")
            print(result.error)
            return 1
        print("Evidence:")
        print(result.evidence.model_dump_json(indent=2))
        return 0
    except Exception as exc:
        print("FAILED (exception types and full cause-chain messages):")
        print(exception_details(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
