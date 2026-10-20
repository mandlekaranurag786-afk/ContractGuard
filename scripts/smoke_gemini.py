"""Run a cached Gemini extraction twice against the Xpresspa liability clause.

Usage:
    uv run --env-file .env python scripts/smoke_gemini.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contractguard.extraction.extractor import Extractor
from contractguard.llm.cache import DEFAULT_CACHE_DIR, CachedLLMClient
from contractguard.llm.gemini import GeminiClient
from contractguard.retrieval.hybrid import HybridSearchIndex
from evals.eval_retrieval import (  # noqa: E402
    PARSED_DIR,
    build_filename_to_docid_map,
    chunk_document,
    load_parsed_document,
    normalize_filename,
)


def _load_xpresspa_chunks():
    filename_to_docid = build_filename_to_docid_map(PARSED_DIR)
    normalized_prefix = normalize_filename("XpresspaGroupInc")
    matches = [
        (filename, doc_id)
        for filename, doc_id in filename_to_docid.items()
        if normalize_filename(filename).startswith(normalized_prefix)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Xpresspa contract, found {len(matches)}")
    _, doc_id = matches[0]
    return chunk_document(load_parsed_document(PARSED_DIR / f"{doc_id}.json"))


def main() -> None:
    chunks = _load_xpresspa_chunks()
    retrieved_chunks = [
        result.chunk
        for result in HybridSearchIndex.from_chunks(chunks).search(
            "what is the liability cap", k=5
        )
    ]
    gemini = GeminiClient()
    extractor = Extractor(
        CachedLLMClient(gemini, cache_dir=DEFAULT_CACHE_DIR, model=gemini.model)
    )

    for run_number in range(1, 3):
        result = extractor.extract("Cap On Liability", retrieved_chunks)
        print(f"Run {run_number}")
        print(
            "extraction:",
            result.extraction.model_dump_json(indent=2)
            if result.extraction is not None
            else None,
        )
        print("checks:", result.checks)
        print("cached:", result.cached)
        print("tokens:", {"input": result.input_tokens, "output": result.output_tokens})
        print("latency_ms:", result.latency_ms)
        print("attempts:", result.attempts)
        print("error:", result.error)
        print()


if __name__ == "__main__":
    main()
