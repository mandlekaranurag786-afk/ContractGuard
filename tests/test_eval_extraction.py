from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from contractguard.extraction.schemas import ExtractionResult, RenewalTermExtraction
from contractguard.llm.client import LLMResponse
from contractguard.llm.errors import LLMQuotaError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals import eval_extraction as evaluation  # noqa: E402


def test_cache_only_miss_fails_without_api(tmp_path):
    client = evaluation.CachedLLMClient(evaluation.CacheOnlyClient(), "test", tmp_path)
    with pytest.raises(RuntimeError, match="CACHE MISS"):
        evaluation.Extractor(client).extract("Renewal Term", [])
    assert not list(tmp_path.iterdir())


def test_summary_counts_wrong_fields_and_ungrounded_predictions():
    label = {
        "contract": "sample",
        "clause_type": "Renewal Term",
        "is_abstention_case": "False",
        "found": "True",
        "notice_days": "30",
    }
    extraction = RenewalTermExtraction(
        found=True,
        initial_term_months=12,
        auto_renews=None,
        renewal_term_months=None,
        notice_days=60,
        evidence_quote="invented",
        evidence_chunk_id="missing",
    )
    result = ExtractionResult(
        clause_type="Renewal Term",
        extraction=extraction,
        checks={
            "evidence_grounded": False,
            "required_fields_present": True,
            "not_found_is_clean": True,
        },
        attempts=1,
        model="fake",
        cached=True,
        latency_ms=100,
        input_tokens=10,
        output_tokens=20,
        error=None,
    )
    rows = [evaluation.make_result(label, result, hit) for hit in (False, True)]
    recorder = evaluation.CallRecorder(evaluation.FakeLLMClient())
    summary = evaluation.summarize(rows, "fake", recorder)
    assert summary["error_split"] == {"retrieval_miss": 1, "extraction_error": 1}
    assert summary["faithfulness"]["grounded"] == "0/2"
    assert len(summary["faithfulness"]["ungrounded_rows"]) == 2
    assert summary["extra_values"][0]["predicted_value"] == 12
    assert summary["cost"]["uncached_calls"] == 0
    assert summary["cost"]["total_input_tokens"] == 0
    assert summary["cost"]["latency_p95_ms"] is None


def test_cost_counts_each_uncached_generation_and_uses_nearest_rank_p95():
    recorder = evaluation.CallRecorder(evaluation.FakeLLMClient())
    recorder.calls = 3
    recorder.responses = [
        LLMResponse(
            text="{}",
            model="fake",
            input_tokens=2,
            output_tokens=1,
            latency_ms=latency,
            cached=False,
        )
        for latency in (100, 200, 900)
    ]
    assert recorder.cost() == {
        "uncached_calls": 3,
        "total_input_tokens": 6,
        "total_output_tokens": 3,
        "latency_median_ms": 200,
        "latency_p95_ms": 900,
        "calls_with_metadata": 3,
    }


@pytest.mark.parametrize("done", [0, 1])
def test_quota_writes_partial_results_and_returns_nonzero(tmp_path, monkeypatch, done):
    label_path = tmp_path / "labels.csv"
    labels = [
        {
            "contract": "one.pdf",
            "clause_type": clause,
            "found": "False",
            "is_abstention_case": "True",
        }
        for clause in ("Renewal Term", "Cap On Liability")
    ]
    with label_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=labels[0])
        writer.writeheader()
        writer.writerows(labels)
    monkeypatch.setattr(evaluation, "LABELS", label_path)
    monkeypatch.setattr(evaluation, "RESULTS_OUT", tmp_path / "results.csv")
    monkeypatch.setattr(evaluation, "SUMMARY_OUT", tmp_path / "summary.json")
    monkeypatch.setattr(
        evaluation, "build_filename_to_docid_map", lambda _: {"one.pdf": "1"}
    )
    monkeypatch.setattr(evaluation, "load_cuad_answers", lambda _: {})
    monkeypatch.setattr(evaluation, "load_parsed_document", lambda _: None)
    monkeypatch.setattr(evaluation, "chunk_document", lambda _: [])

    class Index:
        def search(self, query, k):
            assert k == 5
            return []

    monkeypatch.setattr(evaluation.HybridSearchIndex, "from_chunks", lambda _: Index())
    original_cache = evaluation.CachedLLMClient
    monkeypatch.setattr(
        evaluation,
        "CachedLLMClient",
        lambda inner, model, cache_dir: original_cache(
            inner, model=model, cache_dir=cache_dir
        ),
    )

    class QuotaClient(evaluation.FakeLLMClient):
        calls = 0

        def generate_json(self, prompt, schema):
            self.calls += 1
            if self.calls > done:
                raise LLMQuotaError("quota exhausted")
            return super().generate_json(prompt, schema)

    assert (
        evaluation.run_eval(QuotaClient(), "quota-test", cache_dir=tmp_path / "cache")
        == 1
    )
    summary = json.loads(evaluation.SUMMARY_OUT.read_text())
    assert summary["completed_rows"] == done
    assert summary["complete"] is False
    assert summary["errors"][0]["error"] == "quota exhausted"
    with evaluation.RESULTS_OUT.open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == done

    # A rerun reuses completed rows' responses and generates only missing ones.
    assert (
        evaluation.run_eval(
            evaluation.FakeLLMClient(), "quota-test", cache_dir=tmp_path / "cache"
        )
        == 0
    )
    resumed = json.loads(evaluation.SUMMARY_OUT.read_text())
    assert resumed["complete"] is True
    assert resumed["completed_rows"] == 2
    assert resumed["cost"]["uncached_calls"] == 2 - done
    assert resumed["cache_dir"] == str(tmp_path / "cache")
    assert len(list((tmp_path / "cache").glob("*.json"))) == 2
    assert resumed["retrieved_chunk_ids"][0]["chunk_ids"] == []


def test_cache_dir_cli_is_forwarded(tmp_path, monkeypatch):
    target = tmp_path / "repeat-cache"
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval", "--tag", "test", "--cache-only", "--cache-dir", str(target)],
    )
    # Restore global output settings after main() so other tests remain isolated.
    for name in ("LABELS", "RESULTS_OUT", "SUMMARY_OUT", "PROMPT_VERSION"):
        monkeypatch.setattr(evaluation, name, getattr(evaluation, name))

    def run(client, model, *, fake, cache_dir):
        assert cache_dir == target
        assert not fake
        return 0

    monkeypatch.setattr(evaluation, "run_eval", run)
    assert evaluation.main() == 0


def test_fake_cache_is_temporary_and_cleaned_even_on_failure(monkeypatch):
    paths = []

    def fake_run(client, model, *, fake, cache_dir):
        assert fake
        assert cache_dir != evaluation.DEFAULT_CACHE_DIR
        paths.append(cache_dir)
        cached = evaluation.CachedLLMClient(client, model, cache_dir=cache_dir)
        assert not cached.generate_json("prompt", RenewalTermExtraction).cached
        assert cached.generate_json("prompt", RenewalTermExtraction).cached
        assert len(list(cache_dir.glob("*.json"))) == 1
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(evaluation, "_run_eval", fake_run)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="simulated failure"):
            evaluation.run_eval(evaluation.FakeLLMClient(), "fake", fake=True)
        assert not paths[-1].exists()
    assert paths[0] != paths[1]
