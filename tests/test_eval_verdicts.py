"""Verdict diagnostics use saved fakes only; no models or network."""

from collections import Counter

import pytest
from test_checker_batch import report

from contractguard.rules.models import Decision, Rule, TrustPolicy
from evals.eval_rules import outcome_flags
from evals.eval_verdicts import (
    build_report,
    confusion,
    contract_verdicts,
    diagnostic_value,
    main,
    root_cause,
    rule_section,
)


def result(**updates):
    return report(Decision.NEEDS_REVIEW).rule_results[0].model_copy(update=updates)


def record(truth, engine):
    return dict(
        result=result(decision=engine), truth=truth, **outcome_flags(truth, engine)
    )


def test_confusion_and_automatic_catches():
    rows = [record(t, e) for t in Decision for e in Decision]
    rows += [record(None, e) for e in Decision]
    assert confusion(rows) == Counter({(t, e): 1 for t in Decision for e in Decision})
    text = rule_section("total", rows)
    assert "Truth FAIL caught automatically: 1/4" in text
    assert "Truth PASS decided automatically: 1/4" in text
    assert "Truth NEEDS_REVIEW kept for review: 1/4" in text
    assert "Unsafe: 9" in text


@pytest.mark.parametrize(
    "reason,value,label,status,hit,expected",
    [
        ("clause_not_found", 100, "100", "supported", "True", "retrieval_miss"),
        ("low_confidence", 100, "200", "unsupported", "False", "retrieval_miss"),
        ("low_confidence", 100, "200", "unsupported", "True", "low_confidence"),
        ("cap_is_formula", 100, "200", "unsupported", "True", "extraction_wrong"),
        ("cap_is_formula", 100, "100", "unsupported", "False", "correct_but_unproven"),
        ("cap_is_formula", 100, "100", "supported", "True", "cap_formula"),
        ("currency_unverified", 100, "100", "supported", "True", "other"),
        (
            "value_unsupported",
            100,
            "100",
            "unsupported",
            "False",
            "correct_but_unproven",
        ),
        ("value_absence_claim", None, "200", "absence_claim", "", "extraction_wrong"),
        (
            "value_absence_claim",
            None,
            "NONE",
            "absence_claim",
            "True",
            "correct_but_unproven",
        ),
        ("cap_is_formula", None, "", "absence_claim", "False", "cap_formula"),
    ],
)
def test_root_cause_first_match(reason, value, label, status, hit, expected):
    answer = result(reason_code=reason, value=value, field_status=status)
    assert root_cause(answer, {"cap_amount": label}, {"retrieval_hit": hit}) == expected


@pytest.mark.parametrize(
    "status", ["absence_claim", "unsupported", "explicit_negative", "conflicting"]
)
def test_unproven_statuses(status):
    answer = result(field_status=status)
    assert root_cause(answer, {"cap_amount": "100"}, {}) == "correct_but_unproven"


def test_condition_value_and_status_drive_attribution():
    answer = result(
        clause_type="Renewal Term",
        field="notice_days",
        value=999,
        reason_code="condition_unverified",
        condition_field="auto_renews",
        condition_value=False,
        condition_status="absence_claim",
    )
    labels = {"notice_days": "30", "auto_renews": "False"}
    assert diagnostic_value(answer, labels)["field"] == "auto_renews"
    assert (
        root_cause(answer, labels, {"retrieval_hit": "False"}) == "correct_but_unproven"
    )
    labels["auto_renews"] = "True"
    assert root_cause(answer, labels, {"retrieval_hit": "False"}) == "retrieval_miss"
    assert root_cause(answer, labels, {"retrieval_hit": "True"}) == "extraction_wrong"


def test_followup_values_override_csv_predictions_and_policy_trust():
    answer = result(
        value=200, field_status="followup_pattern", field_evidence_source="followup"
    )
    extraction = {"predicted_cap_amount": "100", "retrieval_hit": "False"}
    assert root_cause(answer, {"cap_amount": "200"}, extraction) == "other"
    answer.policy = TrustPolicy.STRICT
    assert (
        root_cause(answer, {"cap_amount": "200"}, extraction) == "correct_but_unproven"
    )


@pytest.mark.parametrize(
    "truth,engine,unsafe",
    [
        (Decision.FAIL, Decision.PASS, True),
        (Decision.PASS, Decision.FAIL, True),
        (Decision.FAIL, Decision.NOT_APPLICABLE, True),
        (Decision.PASS, Decision.NEEDS_REVIEW, False),
        (Decision.FAIL, Decision.FAIL, False),
    ],
)
def test_scored_only_contract_verdict_and_unsafe(truth, engine, unsafe):
    records = {
        ("fake.pdf", "scored"): record(truth, engine),
        ("fake.pdf", "unscored"): record(None, Decision.FAIL),
    }
    row = contract_verdicts({"fake.pdf": report(Decision.FAIL)}, records, 2)[0]
    assert row["truth"] == truth
    assert row["engine"] == (
        "NO_APPLICABLE_RULES" if engine == "NOT_APPLICABLE" else engine
    )
    assert row["unsafe"] is unsafe
    assert row["scored"] == 1 and not row["all_scored"]


def test_all_unscored_and_all_scored_flags():
    records = {
        ("none.pdf", "a"): record(None, Decision.FAIL),
        ("all.pdf", "a"): record(Decision.PASS, Decision.PASS),
    }
    rows = contract_verdicts({"none.pdf": None, "all.pdf": None}, records, 1)
    assert rows[0]["all_scored"]
    assert rows[1]["truth"] == rows[1]["engine"] == "NO_APPLICABLE_RULES"
    assert not rows[1]["unsafe"] and not rows[1]["all_scored"]


def test_full_report_reuses_truth_and_lists_missed_fail():
    rules = [
        Rule(
            id="cap",
            description="cap",
            clause_type="Cap On Liability",
            field="cap_amount",
            operator=">=",
            threshold=200,
        )
    ]
    reports = {"fake.pdf": report(Decision.NEEDS_REVIEW)}
    reports["fake.pdf"].rule_results[0].field_status = "unsupported"
    labels = {("fake.pdf", "Cap On Liability"): {"found": "True", "cap_amount": "100"}}
    extractions = {("fake.pdf", "Cap On Liability"): {"retrieval_hit": "True"}}
    text = build_report(reports, rules, labels, extractions)
    assert "Truth FAIL caught automatically: 0/1" in text
    assert "fake.pdf | cap | truth=FAIL | engine=NEEDS_REVIEW" in text
    assert "root_cause=correct_but_unproven" in text
    assert "correct_but_unproven | 1 | 1" in text
    assert "unsupported | 1 | 1" in text
    assert "Unsafe overall: 0" in text
    with pytest.raises(ValueError, match="Missing extractions"):
        build_report(reports, rules, labels, {})


def test_refuses_existing_output_before_loading(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "evals").mkdir()
    output = tmp_path / "evals/verdict_eval_keep.txt"
    output.write_text("keep")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        main(["--reports", "missing", "--tag", "keep"])
    assert output.read_text() == "keep"
