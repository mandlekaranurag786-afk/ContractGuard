"""Evaluate hand-labelled extractions with hybrid top-5 retrieval.

Usage:
    uv run --env-file .env python evals/eval_extraction.py --tag live
    uv run python evals/eval_extraction.py --fake --tag fake

Fake mode uses cached local embeddings, never Gemini, and a temporary LLM
cache directory. Results include the model/mode to distinguish dry runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

# Hugging Face reads offline flags at import time, before main() runs.
if __name__ == "__main__" and any(a in sys.argv for a in ("--fake", "--cache-only")):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pydantic import BaseModel

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contractguard.evaluation.extraction_metrics import (
    FIELD_KINDS,
    parse_label,
    score_row,
)
from contractguard.extraction.extractor import Extractor
from contractguard.llm.cache import DEFAULT_CACHE_DIR, CachedLLMClient
from contractguard.llm.client import LLMClient, LLMResponse
from contractguard.llm.errors import LLMQuotaError
from contractguard.llm.gemini import GeminiClient
from contractguard.pipeline import (
    CLAUSE_QUERIES,
    build_filename_to_docid_map,
    chunk_document,
    load_parsed_document,
    normalize_filename,
)
from contractguard.retrieval.hybrid import HybridSearchIndex
from evals.eval_retrieval import (  # noqa: E402
    KNOWN_BAD_LABELS,
    MASTER_CLAUSES_CSV,
    PARSED_DIR,
    best_overlap_against_spans,
    first_hit_rank,
    load_cuad_answers,
)

LABELS = Path("evals/extraction_labels.csv")
RESULTS_OUT = Path("evals/extraction_results.csv")
SUMMARY_OUT = Path("evals/extraction_summary.json")
TOP_K = 5
FIELDS = ["found", *(f for fields in FIELD_KINDS.values() for f in fields)]
PROMPT_VERSION = "v1"


class CacheOnlyClient:
    """Fail outside the LLMError hierarchy so Extractor cannot hide a miss."""

    model = "gemini-3.5-flash-lite"

    def generate_json(self, prompt: str, schema: type[BaseModel]) -> LLMResponse:
        raise RuntimeError("CACHE MISS: --cache-only forbids generating a response")


class FakeLLMClient:
    """Return a clean abstention for either extraction schema, without I/O."""

    model = "fake-always-not-found"

    def generate_json(self, prompt: str, schema: type[BaseModel]) -> LLMResponse:
        values = dict.fromkeys(schema.model_fields)
        for name in ("found", "has_money_cap", "has_damages_exclusion"):
            if name in values:
                values[name] = False
        return LLMResponse(
            text=schema.model_validate(values).model_dump_json(),
            model=self.model,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
            cached=False,
        )


class CallRecorder:
    """Record uncached generation calls, including extraction repair attempts.

    Place inside CachedLLMClient so cache hits never contribute to cost.
    Failed calls are counted; unavailable token/latency metadata is omitted.
    Provider-internal HTTP retries are not separately exposed by LLMClient.
    """

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.calls = 0
        self.responses: list[LLMResponse] = []

    def generate_json(self, prompt: str, schema: type[BaseModel]) -> LLMResponse:
        self.calls += 1
        response = self.inner.generate_json(prompt, schema)
        self.responses.append(response)
        return response

    def cost(self) -> dict:
        latencies = sorted(
            r.latency_ms for r in self.responses if r.latency_ms is not None
        )
        return {
            "uncached_calls": self.calls,
            "total_input_tokens": sum(r.input_tokens or 0 for r in self.responses),
            "total_output_tokens": sum(r.output_tokens or 0 for r in self.responses),
            "latency_median_ms": statistics.median(latencies) if latencies else None,
            "latency_p95_ms": (
                latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None
            ),
            "calls_with_metadata": len(self.responses),
        }


def make_result(label: dict, result, retrieval_hit: bool | None) -> dict:
    """Flatten one extraction, labels, scores and attribution for reporting."""
    scores = score_row(label["clause_type"], label, result.extraction)
    predicted = result.extraction.model_dump() if result.extraction else {}
    attribution = {
        key.removesuffix("_correct"): (
            "retrieval_miss" if retrieval_hit is False else "extraction_error"
        )
        for key, value in scores.items()
        if key.endswith("_correct") and not value
    }
    return {
        "contract": label["contract"],
        "clause_type": label["clause_type"],
        "is_abstention_case": parse_label(label["is_abstention_case"], bool),
        "retrieval_hit": retrieval_hit,
        **{f"label_{f}": label.get(f, "") for f in FIELDS},
        **{f"predicted_{f}": predicted.get(f) for f in FIELDS},
        **scores,
        "error_attribution": attribution,
        "checks": result.checks,
        "evidence_quote": predicted.get("evidence_quote"),
        "evidence_chunk_id": predicted.get("evidence_chunk_id"),
        "cached": result.cached,
        "attempts": result.attempts,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_ms": result.latency_ms,
        "error": result.error,
    }


def ratio(values: list[bool]) -> str:
    return f"{sum(values)}/{len(values)}"


def summarize(rows: list[dict], model: str, recorder: CallRecorder) -> dict:
    """Summarize completed instances; all zero denominators remain explicit."""
    clauses = {}
    for clause, fields in FIELD_KINDS.items():
        selected = [r for r in rows if r["clause_type"] == clause]
        full = [
            all(v for k, v in r.items() if k.endswith("_correct")) for r in selected
        ]
        clauses[clause] = {
            "rows": len(selected),
            "found_accuracy": ratio([r["found_correct"] for r in selected]),
            "fields": {
                field: ratio(
                    [r[f"{field}_correct"] for r in selected if f"{field}_correct" in r]
                )
                for field in fields
            },
            "fully_correct": ratio(full),
            "fully_correct_percent": 100 * sum(full) / len(full) if full else None,
        }
    found = [r for r in rows if r["predicted_found"] is True]
    return {
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "labels": str(LABELS),
        "top_k": TOP_K,
        "date": datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat(),
        "completed_rows": len(rows),
        "per_clause_type": clauses,
        "abstention": ratio(
            [r["found_correct"] for r in rows if r["is_abstention_case"]]
        ),
        "faithfulness": {
            "found_predictions": len(found),
            "levels": {
                level: sum(r["checks"].get("evidence_level") == level for r in found)
                for level in ("strict", "in_context", "composite", "not_found")
            },
            "cited_chunk_correct": ratio(
                [
                    r["checks"]["cited_chunk_correct"]
                    for r in found
                    if r["checks"].get("cited_chunk_correct") is not None
                ]
            ),
            "not_found_rows": [
                {k: r[k] for k in ("contract", "clause_type", "evidence_quote")}
                for r in found
                if r["checks"].get("evidence_level") == "not_found"
            ],
            "grounded": ratio(
                [r["checks"].get("evidence_grounded", False) for r in found]
            ),
            "ungrounded_rows": [
                {k: r[k] for k in ("contract", "clause_type", "evidence_quote")}
                for r in found
                if not r["checks"].get("evidence_grounded", False)
            ],
        },
        "check_pass_rates": {
            check: ratio([r["checks"].get(check, False) for r in rows])
            for check in ("required_fields_present", "not_found_is_clean")
        },
        "error_split": {
            cause: sum(
                value == cause
                for r in rows
                for value in r["error_attribution"].values()
            )
            for cause in ("retrieval_miss", "extraction_error")
        },
        "extra_values": [
            {
                "contract": r["contract"],
                "clause_type": r["clause_type"],
                "field": field,
                "predicted_value": r[f"predicted_{field}"],
            }
            for r in rows
            for field in r["extra_values"]
        ],
        "cost": recorder.cost(),
        "errors": [
            {k: r[k] for k in ("contract", "clause_type", "error")}
            for r in rows
            if r["error"]
        ],
    }


def write_results(rows: list[dict], summary: dict) -> None:
    """Write even an empty partial run, with stable CSV columns."""
    columns = [
        "contract",
        "clause_type",
        "is_abstention_case",
        "retrieval_hit",
        *(f"label_{f}" for f in FIELDS),
        *(f"predicted_{f}" for f in FIELDS),
        *(f"{f}_correct" for f in FIELDS),
        "extra_values",
        "error_attribution",
        "checks",
        "evidence_quote",
        "evidence_chunk_id",
        "cached",
        "attempts",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "error",
        "model",
        "fake",
        "prompt_version",
        "cache_dir",
        "retrieved_chunk_ids",
    ]
    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{
                        k: json.dumps(v) if isinstance(v, (dict, list)) else v
                        for k, v in row.items()
                    },
                    "model": summary["model"],
                    "fake": summary["fake"],
                    "prompt_version": summary["prompt_version"],
                    "cache_dir": summary["cache_dir"],
                }
            )
    rendered = json.dumps(summary, indent=2, allow_nan=False)
    SUMMARY_OUT.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


def run_eval(
    client: LLMClient,
    model: str,
    *,
    fake: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> int:
    """Keep fake responses in an automatically cleaned temporary directory."""
    if fake:
        with TemporaryDirectory(prefix="contractguard-fake-") as cache_dir:
            return _run_eval(client, model, fake=True, cache_dir=Path(cache_dir))
    return _run_eval(client, model, cache_dir=cache_dir)


def _run_eval(
    client: LLMClient,
    model: str,
    *,
    fake: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> int:
    """Run every labelled row; preserve completed rows on quota exhaustion."""
    with LABELS.open(newline="", encoding="utf-8") as handle:
        labels = list(csv.DictReader(handle))
    docids = {
        normalize_filename(filename): docid
        for filename, docid in build_filename_to_docid_map(PARSED_DIR).items()
    }
    answers = load_cuad_answers(MASTER_CLAUSES_CSV)
    recorder = CallRecorder(client)
    extractor = Extractor(
        CachedLLMClient(recorder, model=model, cache_dir=cache_dir),
        prompt_version=PROMPT_VERSION,
    )
    indexes = {}
    rows = []
    quota_error = None
    for number, label in enumerate(labels, 1):
        filename, clause = label["contract"], label["clause_type"]
        print(f"[{number}/{len(labels)}] {filename} | {clause}", flush=True)
        key = normalize_filename(filename)
        if key not in indexes:
            chunks = chunk_document(
                load_parsed_document(PARSED_DIR / f"{docids[key]}.json")
            )
            indexes[key] = HybridSearchIndex.from_chunks(chunks)
        retrieved = [
            result.chunk
            for result in indexes[key].search(CLAUSE_QUERIES[clause], k=TOP_K)
        ]
        abstention = parse_label(label["is_abstention_case"], bool)
        retrieval_hit = None
        if not abstention and (key, clause) not in KNOWN_BAD_LABELS:
            overlaps = [
                best_overlap_against_spans(chunk.text, answers[key][clause])
                for chunk in retrieved
            ]
            retrieval_hit = first_hit_rank(overlaps, 0.5) is not None
        try:
            result = extractor.extract(clause, retrieved)
        except LLMQuotaError as exc:
            quota_error = {
                "contract": filename,
                "clause_type": clause,
                "error": str(exc),
            }
            print(
                f"Quota exhausted after {len(rows)}/{len(labels)} rows: {exc}",
                flush=True,
            )
            break
        rows.append(make_result(label, result, retrieval_hit))
        rows[-1]["retrieved_chunk_ids"] = [chunk.chunk_id for chunk in retrieved]
        print(f"  cached={result.cached} error={result.error}", flush=True)
    summary = summarize(rows, model, recorder)
    summary.update(fake=fake, total_rows=len(labels), complete=quota_error is None)
    summary["cache_dir"] = str(cache_dir)
    summary["retrieved_chunk_ids"] = [
        {
            "contract": row["contract"],
            "clause_type": row["clause_type"],
            "chunk_ids": row["retrieved_chunk_ids"],
        }
        for row in rows
    ]
    if isinstance(client, CacheOnlyClient):
        summary.update(cache_only=True, cache_misses=recorder.calls)
    if quota_error:
        summary["errors"].append(quota_error)
    write_results(rows, summary)
    return 1 if quota_error else 0


def main() -> int:
    global LABELS, RESULTS_OUT, SUMMARY_OUT, PROMPT_VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fake", action="store_true", help="Offline always-abstain LLM"
    )
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--tag", required=True, help="Versioned output suffix")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--prompt-version", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("--tag must contain only letters, digits, underscores or hyphens")
    if args.fake and args.cache_only:
        parser.error("--fake and --cache-only are mutually exclusive")
    LABELS = args.labels
    PROMPT_VERSION = args.prompt_version
    RESULTS_OUT = Path(f"evals/extraction_results_{args.tag}.csv")
    SUMMARY_OUT = Path(f"evals/extraction_summary_{args.tag}.json")
    if args.fake:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        client = FakeLLMClient()
    elif args.cache_only:
        client = CacheOnlyClient()
    else:
        client = GeminiClient()
    return run_eval(client, client.model, fake=args.fake, cache_dir=args.cache_dir)


if __name__ == "__main__":
    raise SystemExit(main())
