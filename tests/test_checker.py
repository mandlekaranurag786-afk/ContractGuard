"""Pure orchestration tests: fake loading, retrieval and extraction."""

from dataclasses import dataclass, field

import pytest

from contractguard.checker import ContractChecker
from contractguard.extraction.extractor import Extractor
from contractguard.extraction.schemas import ExtractionResult, RenewalTermExtraction
from contractguard.ingestion import ParsedDocument, ParsedPage
from contractguard.llm.errors import LLMQuotaError
from contractguard.pipeline import CLAUSE_QUERIES
from contractguard.report import ContractReport, OverallStatus, overall_status
from contractguard.retrieval.models import SearchResult
from contractguard.rules.models import Decision, Rule, load_rules


def rules():
    return [
        Rule(
            id=str(n),
            description="notice",
            clause_type="Renewal Term",
            field="notice_days",
            operator=">=",
            threshold=n,
        )
        for n in (30, 90)
    ]


def document(_):
    return ParsedDocument(
        doc_id="doc",
        original_filename="fake.pdf",
        page_count=1,
        pages=[
            ParsedPage(
                page_number=1,
                raw_text="60 days prior notice",
                clean_text="60 days prior notice",
                needs_ocr=False,
            )
        ],
    )


@dataclass
class FakeExtractor:
    prompt_version: str = "v2"
    calls: list = field(default_factory=list)
    cached: bool = False

    def extract(self, clause_type, chunks):
        self.calls.append((clause_type, chunks))
        return ExtractionResult(
            clause_type=clause_type,
            extraction=RenewalTermExtraction(
                found=True,
                notice_days=60,
                initial_term_months=None,
                renewal_term_months=None,
                auto_renews=None,
                evidence_quote="60 days prior notice",
                evidence_chunk_id="doc:0",
            ),
            checks={"evidence_level": "strict"},
            attempts=1,
            model="fake",
            cached=self.cached,
            latency_ms=5,
            input_tokens=10,
            output_tokens=2,
            error=None,
        )


@dataclass
class FakeRetriever:
    chunks: list
    calls: list = field(default_factory=list)

    def search(self, query, k):
        self.calls.append((query, k))
        return [SearchResult(chunk=c, score=1) for c in self.chunks[:k]]


def checker(extractor):
    built = []

    def builder(chunks):
        index = FakeRetriever(chunks)
        built.append(index)
        return index

    return ContractChecker(
        extractor, document_loader=document, retriever_builder=builder
    ), built


def test_clause_once_report_roundtrip_and_cost():
    extractor = FakeExtractor()
    service, built = checker(extractor)
    report = service.check("fake.pdf", rules(), "policy.json")
    assert len(extractor.calls) == 1
    assert extractor.calls[0][0] == "Renewal Term"
    assert built[0].calls == [(CLAUSE_QUERIES["Renewal Term"], 5)]
    assert report.overall_status == OverallStatus.FAIL
    assert report.counts[Decision.PASS] == report.counts[Decision.FAIL] == 1
    assert report.clauses[0].retrieved_chunk_ids == ["doc:0"]
    assert report.cost.llm_calls == 1
    assert report.cost.cache_hits == 0
    assert report.cost.input_tokens == 10
    assert report.cost.output_tokens == 2
    assert report.cost.elapsed_ms >= 0
    assert ContractReport.model_validate_json(report.model_dump_json()) == report
    assert report.prompt_version == "v2"
    assert report.model == "fake"


@pytest.mark.parametrize(
    "decisions,expected",
    [
        ([Decision.FAIL, Decision.NEEDS_REVIEW], OverallStatus.FAIL),
        ([Decision.NEEDS_REVIEW, Decision.PASS], OverallStatus.NEEDS_REVIEW),
        ([Decision.PASS, Decision.NOT_APPLICABLE], OverallStatus.PASS),
        ([Decision.NOT_APPLICABLE], OverallStatus.NO_APPLICABLE_RULES),
        ([], OverallStatus.NO_APPLICABLE_RULES),
    ],
)
def test_status_precedence(decisions, expected):
    service, _ = checker(FakeExtractor())
    template = service.check("fake.pdf", rules(), "policy.json").rule_results[0]
    assert (
        overall_status([template.model_copy(update={"decision": d}) for d in decisions])
        == expected
    )
    assert expected.reason


def test_empty_policy_rejected_before_loading(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="empty"):
        load_rules(path)
    service, built = checker(FakeExtractor())
    with pytest.raises(ValueError, match="empty"):
        service.check("fake.pdf", [], path)
    assert not built


def test_quota_propagates_through_real_extractor():
    class QuotaClient:
        def generate_json(self, prompt, schema):
            raise LLMQuotaError("quota exhausted")

    service, _ = checker(Extractor(QuotaClient(), prompt_version="v2"))
    with pytest.raises(LLMQuotaError, match="quota exhausted"):
        service.check("fake.pdf", rules(), "policy.json")


def test_cached_result_cost():
    service, _ = checker(FakeExtractor(cached=True))
    cost = service.check("fake.pdf", rules(), "policy.json").cost
    assert cost.llm_calls == 0
    assert cost.cache_hits == 1


def test_each_used_clause_once_and_quota_after_first_clause():
    from contractguard.extraction.schemas import LiabilityCapExtraction

    class BothExtractor(FakeExtractor):
        quota = False

        def extract(self, clause_type, chunks):
            result = super().extract(clause_type, chunks)
            if clause_type == "Cap On Liability":
                if self.quota:
                    raise LLMQuotaError("second clause quota")
                result.extraction = LiabilityCapExtraction(
                    found=True,
                    has_money_cap=True,
                    cap_amount=None,
                    currency=None,
                    cap_formula=None,
                    has_damages_exclusion=False,
                    evidence_quote="liability shall not exceed",
                    evidence_chunk_id="doc:0",
                )
            return result

    cap = Rule(
        id="cap",
        description="cap",
        clause_type="Cap On Liability",
        field="has_money_cap",
        operator="is_true",
    )
    extractor = BothExtractor()
    service, _ = checker(extractor)
    report = service.check("fake.pdf", [*rules(), cap], "policy.json")
    assert [clause for clause, _ in extractor.calls] == [
        "Renewal Term",
        "Cap On Liability",
    ]
    assert len(report.clauses) == 2
    assert report.cost.llm_calls == 2
    assert report.cost.input_tokens == 20
    extractor.quota = True
    with pytest.raises(LLMQuotaError, match="second clause"):
        service.check("fake.pdf", [*rules(), cap], "policy.json")


def test_cli_json_and_v2_without_network(tmp_path, monkeypatch, capsys):
    from contractguard import check

    policy = tmp_path / "policy.json"
    policy.write_text("[" + rules()[0].model_dump_json() + "]")
    out = tmp_path / "report.json"
    monkeypatch.setattr(
        check, "GeminiClient", lambda: type("Client", (), {"model": "fake"})()
    )
    service, _ = checker(FakeExtractor())

    def factory(extractor):
        assert extractor.prompt_version == "v2"
        return service

    monkeypatch.setattr(check, "ContractChecker", factory)
    assert (
        check.main(["fake.pdf", "--policy", str(policy), "--json-out", str(out)]) == 0
    )
    assert ContractReport.model_validate_json(out.read_text()).overall_status == "PASS"
    assert "Cost:" in capsys.readouterr().out
