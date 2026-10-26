"""Offline batch aggregation, fail-closed caching, and comparison tests."""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contractguard.checker import ContractChecker
from contractguard.extraction.extractor import Extractor
from contractguard.extraction.prompt import PRODUCT_PROMPT_VERSION
from contractguard.ingestion import ParsedDocument, ParsedPage
from contractguard.llm.cache import CachedLLMClient
from contractguard.report import ContractReport, Cost, overall_status
from contractguard.rules.models import Decision, Rule, RuleResult, TrustPolicy
from evals.run_checker_batch import (
    CacheOnlyClient,
    compare_offline,
    run_batch,
    summarize,
)


def report(decision, name="fake.pdf", reason="compared"):
    result = RuleResult(
        rule_id="cap",
        clause_type="Cap On Liability",
        policy=TrustPolicy.KEYWORD,
        decision=decision,
        reason_code=reason,
        reason="test",
        field="cap_amount",
        value=100,
        field_status="supported",
        condition_field=None,
        condition_value=None,
        condition_status=None,
        currency_status=None,
        evidence_strength=None if decision == Decision.NEEDS_REVIEW else "proof",
        confidence_level="HIGH",
        evidence_quote="cap",
        evidence_chunk_id="doc:0",
    )
    status = overall_status([result])
    return ContractReport(
        contract=name,
        policy_path="fake.json",
        trust_policy=TrustPolicy.KEYWORD,
        model="fake",
        prompt_version=PRODUCT_PROMPT_VERSION,
        overall_status=status,
        overall_reason=status.reason,
        counts={decision: 1},
        rule_results=[result],
        clauses=[],
        cost=Cost(
            llm_calls=0,
            cache_hits=2,
            input_tokens=0,
            output_tokens=0,
            cached_input_tokens=10,
            cached_output_tokens=5,
            elapsed_ms=10,
        ),
    )


def test_summary_counts_and_review_rate():
    text = summarize([report(d, f"{d}.pdf") for d in Decision], 42)
    assert "Decision counts: PASS=1, FAIL=1, NEEDS_REVIEW=1, NOT_APPLICABLE=1" in text
    assert "Review rate: 1/4 (25.00%)" in text
    assert "NO_APPLICABLE_RULES=1" in text
    assert "Total llm_calls: 0" in text
    assert "Total cache_hits: 8" in text
    assert "Model load time (separate): 42.00 ms" in text


def test_cache_miss_stops_batch(tmp_path):
    loaded = []

    def loader(path):
        loaded.append(path)
        return ParsedDocument(
            doc_id="fake",
            original_filename=path.name,
            page_count=1,
            pages=[
                ParsedPage(
                    page_number=1, raw_text="test", clean_text="test", needs_ocr=False
                )
            ],
        )

    class Retriever:
        def search(self, query, k):
            return []

    inner = CacheOnlyClient()
    checker = ContractChecker(
        Extractor(
            CachedLLMClient(inner, model=inner.model, cache_dir=tmp_path / "cache"),
            prompt_version=PRODUCT_PROMPT_VERSION,
        ),
        document_loader=loader,
        retriever_builder=lambda _: Retriever(),
    )
    rule = Rule(
        id="cap",
        description="cap",
        clause_type="Cap On Liability",
        field="cap_amount",
        operator=">=",
        threshold=100,
    )
    with pytest.raises(RuntimeError, match="CACHE MISS"):
        run_batch(
            checker,
            [Path("a.pdf"), Path("b.pdf")],
            [rule],
            Path("fake.json"),
            tmp_path / "reports",
        )
    assert loaded == [Path("a.pdf")]
    assert not list((tmp_path / "reports").glob("*.json"))
    assert not (tmp_path / "cache").exists()


def test_comparison_reports_allowed_reason_and_real_difference(tmp_path):
    reports = [
        report(Decision.NEEDS_REVIEW, reason="cap_is_formula"),
        report(Decision.PASS, "other.pdf"),
    ]
    path = tmp_path / "offline.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "contract",
                "rule_id",
                "decision",
                "reason_code",
                "evidence_strength",
            ],
        )
        writer.writeheader()
        for p in reports:
            writer.writerow(
                dict(
                    contract=p.contract,
                    rule_id="cap",
                    decision="NEEDS_REVIEW",
                    reason_code="value_absence_claim",
                    evidence_strength="",
                )
            )
    text = compare_offline(reports, path)
    assert "Rows compared: 2" in text
    assert "Allowed cap_is_formula changes: 1" in text
    assert "Decision differences: 1" in text
    assert "Unexpected differences: 1" in text
    assert "other.pdf | cap" in text
