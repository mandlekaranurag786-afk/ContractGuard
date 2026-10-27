import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_checker_batch import report

from contractguard.rules.models import Decision, Rule
from evals.eval_followup import accepted, build_report, changes, metrics, score_reports


def fixtures(decision):
    rules = [
        Rule(
            id="cap",
            clause_type="Cap On Liability",
            field="cap_amount",
            description="test",
            operator=">=",
            threshold=100,
        )
    ]
    reports = {"fake.pdf": report(decision)}
    labels = {("fake.pdf", "Cap On Liability"): {"found": "True", "cap_amount": "200"}}
    return reports, rules, labels


def test_trace_groups_full_error_messages():
    from contractguard.followup.models import FollowupStep
    from evals.eval_followup import trace_section

    reports, _, _ = fixtures(Decision.NEEDS_REVIEW)
    message = "ValueError: rejected schema\nfull details"
    step = FollowupStep(
        step=1,
        clause_type="Cap On Liability",
        field="has_money_cap",
        rule_ids=["cap"],
        note="extraction_error",
        error=message,
    )
    reports["fake.pdf"].followup.steps = [step, step.model_copy()]
    output = trace_section("fake", reports)
    assert f"count=2\n{message}" in output
    assert output.count(message) == 1


def test_trace_reports_no_value_separately():
    from contractguard.followup.models import FollowupStep
    from evals.eval_followup import trace_section

    reports, _, _ = fixtures(Decision.NEEDS_REVIEW)
    reports["fake.pdf"].followup.steps = [FollowupStep(
        step=1, clause_type="Cap On Liability", field="has_money_cap",
        rule_ids=["cap"], merge_outcome="NO_VALUE", note="merged",
    )]
    output = trace_section("fake", reports)
    assert "ACCEPTED_CONFIRMATION | NO_VALUE | DISCARDED_UNVERIFIED" in output
    assert "has_money_cap | 0 | 0 | 1 | 0 | 0 | 0" in output


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_batch_passes_followup_version(tmp_path, version):
    from evals.run_checker_batch import run_batch

    class Checker:
        def check(self, *args, followup, followup_version):
            assert followup is True
            assert followup_version == version
            result = report(Decision.NEEDS_REVIEW)
            result.followup.version = followup_version
            return result

    _, rules, _ = fixtures(Decision.PASS)
    run_batch(
        Checker(), [Path("fake.pdf")], rules, Path("policy.json"), tmp_path,
        followup=True, followup_version=version,
    )
    from contractguard.report import ContractReport

    saved = ContractReport.model_validate_json((tmp_path / "fake.json").read_text())
    assert saved.followup.version == version


@pytest.mark.parametrize(
    "decision,label,unsafe",
    [
        (Decision.NOT_APPLICABLE, "200", 1),
        (Decision.PASS, "NONE", 1),
        (Decision.FAIL, "200", 1),
        (Decision.NEEDS_REVIEW, "200", 0),
        (Decision.PASS, "", 0),
    ],
)
def test_unsafe(decision, label, unsafe):
    reports, rules, labels = fixtures(decision)
    labels["fake.pdf", "Cap On Liability"]["cap_amount"] = label
    assert metrics(score_reports(reports, rules, labels))["unsafe"] == unsafe


def test_changed_listing_full_quote_and_blank_verdict():
    before, rules, labels = fixtures(Decision.NEEDS_REVIEW)
    after, _, _ = fixtures(Decision.PASS)
    after["fake.pdf"].rule_results[0].evidence_quote = "Long quote " * 100
    text, rows = build_report(before, after, rules, labels)
    assert len(rows) == 1 and rows[0]["human_verdict"] == ""
    assert rows[0]["followup_quote"] == "Long quote " * 100
    assert rows[0]["chunk_id"] == "doc:0"
    assert rows[0]["unsafe"] == "no" and rows[0]["truth"] == "PASS"
    assert "CHANGED DECISIONS: 1" in text
    assert "Every changed decision must be checked by hand" in text
    old = score_reports(before, rules, labels)
    assert changes(old, old) == []


@pytest.mark.parametrize(
    "unsafe,reduction,expected", [(1, 10, False), (0, 9, False), (0, 10, True)]
)
def test_acceptance(unsafe, reduction, expected):
    assert (
        accepted({"review": 30}, {"review": 30 - reduction, "unsafe": unsafe})
        is expected
    )


def test_missing_after_contract_rejected():
    before, rules, labels = fixtures(Decision.PASS)
    with pytest.raises(ValueError, match="contract sets"):
        build_report(before, {}, rules, labels)


def test_live_followup_main_miss_stops(tmp_path, monkeypatch):
    from contractguard.ingestion import ParsedDocument, ParsedPage
    from contractguard.llm import gemini
    from evals.run_checker_batch import build_checker, run_batch

    class LiveFake:
        model = "fake"
        calls = 0

        def generate_json(self, prompt, schema):
            self.calls += 1
            raise AssertionError("Main extraction must never reach this client")

    live = LiveFake()
    monkeypatch.setattr(gemini, "GeminiClient", lambda: live)
    checker = build_checker(
        live_followup=True,
        main_cache_dir=tmp_path / "main",
        followup_cache_dir=tmp_path / "followup",
    )
    checker.document_loader = lambda path: ParsedDocument(
        doc_id="fake",
        original_filename=path.name,
        page_count=1,
        pages=[
            ParsedPage(
                page_number=1, raw_text="text", clean_text="text", needs_ocr=False
            )
        ],
    )

    class Retriever:
        def search(self, query, k):
            return []

    checker.retriever_builder = lambda _: Retriever()
    _, rules, _ = fixtures(Decision.PASS)
    with pytest.raises(RuntimeError, match="CACHE MISS"):
        run_batch(
            checker,
            [Path("fake.pdf")],
            rules,
            Path("policy.json"),
            tmp_path / "reports",
            followup=True,
            cache_only=False,
        )
    assert live.calls == 0
    assert not list((tmp_path / "reports").glob("*.json"))


def test_live_followup_requires_followup():
    from evals.run_checker_batch import main

    with pytest.raises(SystemExit) as error:
        main(["--live-followup"])
    assert error.value.code == 2


def test_default_followup_is_cache_only(tmp_path, monkeypatch):
    from contractguard.followup.extract import FIELD_SCHEMAS
    from contractguard.llm import gemini
    from evals.run_checker_batch import build_checker

    monkeypatch.setattr(
        gemini, "GeminiClient", lambda: pytest.fail("Must not construct Gemini")
    )
    checker = build_checker(
        main_cache_dir=tmp_path / "main", followup_cache_dir=tmp_path / "followup"
    )
    with pytest.raises(RuntimeError, match="CACHE MISS"):
        checker.followup_client.generate_json(
            "prompt", FIELD_SCHEMAS["Renewal Term", "notice_days"]
        )


def test_quota_keeps_finished_reports(tmp_path, capsys):
    from contractguard.llm.errors import LLMQuotaError
    from evals.run_checker_batch import run_batch

    class Checker:
        calls = 0

        def check(self, *args, **kwargs):
            self.calls += 1
            result = report(Decision.NEEDS_REVIEW)
            if self.calls == 2:
                result.followup.status = "followup_skipped"
            return result

    _, rules, _ = fixtures(Decision.PASS)
    checker = Checker()
    with pytest.raises(LLMQuotaError):
        run_batch(
            checker,
            [Path("a.pdf"), Path("b.pdf"), Path("c.pdf")],
            rules,
            Path("policy.json"),
            tmp_path,
            followup=True,
        )
    assert checker.calls == 2
    assert [p.name for p in tmp_path.glob("*.json")] == ["a.json"]
    assert "completed contracts: 1/3" in capsys.readouterr().out
