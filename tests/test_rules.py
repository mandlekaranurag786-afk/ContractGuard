"""Offline rule decisions and validation, using structured extraction fakes."""

import json

import pytest

from contractguard.extraction.confidence import compute_confidence
from contractguard.extraction.schemas import ExtractionResult, RenewalTermExtraction
from contractguard.rules.engine import evaluate_rule, evaluate_rules
from contractguard.rules.models import Decision, Rule, TrustPolicy, load_rules


def inputs(**updates):
    data = dict(
        found=True,
        initial_term_months=None,
        auto_renews=True,
        renewal_term_months=None,
        notice_days=60,
        evidence_quote="Automatically renews with 60 days prior notice.",
        evidence_chunk_id="doc:0",
    )
    data.update(updates)
    ex = RenewalTermExtraction(**data)
    result = ExtractionResult(
        clause_type="Renewal Term",
        extraction=ex,
        checks={"evidence_level": "strict"},
        attempts=1,
        model="fake",
        cached=False,
        latency_ms=None,
        input_tokens=None,
        output_tokens=None,
        error=None,
    )
    return result, compute_confidence(ex, result.checks, result.clause_type)


def rule(**updates):
    data = dict(
        id="notice",
        description="Enough notice",
        clause_type="Renewal Term",
        field="notice_days",
        operator=">=",
        threshold=30,
    )
    data.update(updates)
    return Rule(**data)


def test_mismatched_clause_first():
    result, confidence = inputs()
    result.clause_type = "Cap On Liability"
    result.error = "failed"
    with pytest.raises(ValueError, match="clause types"):
        evaluate_rule(rule(), result, confidence, TrustPolicy.STRICT)


@pytest.mark.parametrize("missing", [True, False])
def test_extraction_failed(missing):
    result, confidence = inputs()
    if missing:
        result.extraction = None
    else:
        result.error = "failed"
    answer = evaluate_rule(rule(), result, confidence, TrustPolicy.STRICT)
    assert answer.reason_code == "extraction_failed"
    assert answer.decision == Decision.NEEDS_REVIEW
    assert answer.evidence_strength is None


@pytest.mark.parametrize(
    "level,code",
    [
        ("NOT_FOUND", "clause_not_found"),
        ("LOW", "low_confidence"),
    ],
)
def test_confidence_gates(level, code):
    from dataclasses import replace

    result, confidence = inputs()
    confidence = replace(confidence, level=level)
    answer = evaluate_rule(
        rule(threshold=999), result, confidence, TrustPolicy.BALANCED
    )
    assert answer.decision == Decision.NEEDS_REVIEW
    assert answer.reason_code == code


def test_condition_gates_before_value():
    result, confidence = inputs(notice_days=None)
    conditional = rule(when={"field": "auto_renews", "equals": False})
    answer = evaluate_rule(conditional, result, confidence, TrustPolicy.STRICT)
    assert answer.reason_code == "condition_unverified"
    answer = evaluate_rule(conditional, result, confidence, TrustPolicy.KEYWORD)
    assert answer.reason_code == "condition_not_met"
    assert answer.decision == Decision.NOT_APPLICABLE
    assert answer.evidence_strength == "weak_signal"
    assert answer.condition_field == "auto_renews"
    assert answer.condition_value is True


def test_value_untrusted():
    result, confidence = inputs(notice_days=None)
    answer = evaluate_rule(rule(), result, confidence, TrustPolicy.BALANCED)
    assert answer.reason_code == "value_absence_claim"
    assert answer.decision == Decision.NEEDS_REVIEW


@pytest.mark.parametrize(
    "operator,threshold,decision",
    [
        (">=", 30, "PASS"),
        (">=", 90, "FAIL"),
        ("<=", 60, "PASS"),
        ("<=", 30, "FAIL"),
        ("==", 60, "PASS"),
        ("==", 30, "FAIL"),
    ],
)
def test_comparisons_and_medium(operator, threshold, decision):
    from dataclasses import replace

    result, confidence = inputs()
    confidence = replace(confidence, level="MEDIUM")
    answer = evaluate_rule(
        rule(operator=operator, threshold=threshold),
        result,
        confidence,
        TrustPolicy.STRICT,
    )
    assert answer.decision == decision
    assert answer.reason_code == "compared"
    assert answer.evidence_strength == "proof"
    assert answer.evidence_chunk_id == "doc:0"


@pytest.mark.parametrize(
    "value,quote,expected",
    [
        (True, "Automatically renews", ["NEEDS_REVIEW", "PASS", "PASS"]),
        (
            False,
            "shall not automatically renew",
            ["NEEDS_REVIEW", "NEEDS_REVIEW", "PASS"],
        ),
    ],
)
def test_policy_differences(value, quote, expected):
    result, confidence = inputs(
        auto_renews=value, notice_days=None, evidence_quote=quote
    )
    boolean = rule(
        field="auto_renews", operator="is_true" if value else "is_false", threshold=None
    )
    answers = [evaluate_rule(boolean, result, confidence, p) for p in TrustPolicy]
    assert [a.decision for a in answers] == expected
    assert answers[-1].evidence_strength == (
        "weak_signal" if value else "explicit_negative"
    )


def test_weakest_condition_strength_and_batch():
    result, confidence = inputs()
    rules = [rule(when={"field": "auto_renews", "equals": True})]
    answers = evaluate_rules(
        rules,
        {result.clause_type: result},
        {result.clause_type: confidence},
        TrustPolicy.KEYWORD,
    )
    assert answers[0].decision == Decision.PASS
    assert answers[0].evidence_strength == "weak_signal"


@pytest.mark.parametrize(
    "currency,quote,code",
    [
        ("EUR", "Liability shall not exceed 200 EUR", "currency_mismatch"),
        ("USD", "Liability shall not exceed 200", "currency_unverified"),
        ("USD", "Liability shall not exceed $200", "compared"),
    ],
)
def test_currency_gate(currency, quote, code):
    from contractguard.extraction.schemas import LiabilityCapExtraction

    result, _ = inputs()
    result.clause_type = "Cap On Liability"
    result.extraction = LiabilityCapExtraction(
        found=True,
        has_money_cap=True,
        cap_amount=200,
        currency=currency,
        cap_formula=None,
        has_damages_exclusion=False,
        evidence_quote=quote,
        evidence_chunk_id="doc:0",
    )
    confidence = compute_confidence(
        result.extraction, result.checks, result.clause_type
    )
    # Isolate the currency gate even when the amount lacks a unit in the quote.
    confidence.field_status["cap_amount"] = "supported"
    cap = rule(clause_type=result.clause_type, field="cap_amount", currency="USD")
    answer = evaluate_rule(cap, result, confidence, TrustPolicy.STRICT)
    assert answer.reason_code == code
    assert answer.decision == (
        Decision.PASS if code == "compared" else Decision.NEEDS_REVIEW
    )


@pytest.mark.parametrize(
    "update,message",
    [
        ({"field": "missing"}, "Unknown field"),
        ({"threshold": None}, "numeric threshold"),
        ({"operator": "<=", "threshold": None}, "numeric threshold"),
        ({"operator": "is_true"}, "boolean field"),
        ({"operator": "is_false"}, "boolean field"),
        ({"when": {"field": "missing", "equals": True}}, "Unknown field"),
        ({"threshold": True}, "validation error"),
    ],
)
def test_load_invalid(tmp_path, update, message):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps([{**rule().model_dump(), **update}]))
    with pytest.raises(ValueError, match=message):
        load_rules(path)


def test_load_list_and_duplicates(tmp_path):
    path = tmp_path / "rules.json"
    data = rule().model_dump()
    path.write_text(json.dumps([data]))
    assert load_rules(path) == [rule()]
    path.write_text(json.dumps([data, data]))
    with pytest.raises(ValueError, match="Duplicate rule ids"):
        load_rules(path)
    path.write_text("{}")
    with pytest.raises(ValueError, match="JSON list"):
        load_rules(path)
