import pytest

from contractguard.extraction.confidence import ConfidenceResult
from contractguard.extraction.schemas import ExtractionResult, LiabilityCapExtraction
from contractguard.followup.merge import merge
from contractguard.followup.models import FieldEvidence
from contractguard.rules.engine import evaluate_rule
from contractguard.rules.models import Rule, TrustPolicy


def inputs():
    ex = LiabilityCapExtraction(
        found=True,
        has_money_cap=True,
        cap_amount=2000000,
        currency="USD",
        cap_formula=None,
        has_damages_exclusion=False,
        evidence_quote="original cap quote",
        evidence_chunk_id="old:0",
    )
    result = ExtractionResult(
        clause_type="Cap On Liability",
        extraction=ex,
        checks={},
        attempts=1,
        model="fake",
        cached=True,
        latency_ms=1,
        input_tokens=1,
        output_tokens=1,
        error=None,
    )
    confidence = ConfidenceResult(
        "HIGH",
        [],
        {},
        {
            "has_damages_exclusion": "absence_claim",
            "has_money_cap": "supported",
            "cap_amount": "supported",
            "currency": "supported",
            "auto_renews": "absence_claim",
        },
    )
    rule = Rule(
        id="exclusion",
        description="test",
        clause_type="Cap On Liability",
        field="has_damages_exclusion",
        operator="is_true",
    )
    return rule, result, confidence


def overlay(
    field="has_damages_exclusion",
    old=False,
    value=True,
    verified=True,
    status="followup_pattern",
):
    new = FieldEvidence(
        clause_type="Cap On Liability",
        field=field,
        value=value,
        evidence_quote="follow-up quote",
        evidence_chunk_id="new:0",
        verified=verified,
        status=status,
    )
    return {field: merge(old, "absence_claim", new)}


@pytest.mark.parametrize(
    "policy,decision",
    [
        (TrustPolicy.STRICT, "NEEDS_REVIEW"),
        (TrustPolicy.KEYWORD, "PASS"),
        (TrustPolicy.BALANCED, "PASS"),
    ],
)
@pytest.mark.parametrize("status", ["supported", "followup_pattern"])
def test_accepted_and_strict(policy, decision, status):
    rule, result, confidence = inputs()
    assert evaluate_rule(rule, result, confidence, policy).decision == "NEEDS_REVIEW"
    answer = evaluate_rule(rule, result, confidence, policy, overlay(status=status))
    assert answer.decision == decision
    assert answer.evidence_strength == ("followup" if decision == "PASS" else None)
    assert answer.evidence_quote == "follow-up quote"
    assert answer.evidence_chunk_id == "new:0"
    assert answer.field_evidence_source == "followup"


def test_conflict_and_discarded():
    rule, result, confidence = inputs()
    answer = evaluate_rule(
        rule, result, confidence, TrustPolicy.KEYWORD, overlay(old=12)
    )
    assert answer.reason_code == "followup_conflict"
    assert "12" in answer.reason and "True" in answer.reason
    assert evaluate_rule(
        rule, result, confidence, TrustPolicy.KEYWORD, overlay(verified=False)
    ) == evaluate_rule(rule, result, confidence, TrustPolicy.KEYWORD)


def test_condition_overlay_own_quote_preserved():
    from contractguard.extraction.schemas import RenewalTermExtraction

    _, result, confidence = inputs()
    result.clause_type = "Renewal Term"
    result.extraction = RenewalTermExtraction(
        found=True,
        initial_term_months=None,
        renewal_term_months=None,
        auto_renews=None,
        notice_days=60,
        evidence_quote="60 days prior notice",
        evidence_chunk_id="old:0",
    )
    confidence.field_status["notice_days"] = "supported"
    rule = Rule(
        id="notice",
        description="test",
        clause_type="Renewal Term",
        field="notice_days",
        operator=">=",
        threshold=30,
        when={"field": "auto_renews", "equals": True},
    )
    answer = evaluate_rule(
        rule, result, confidence, TrustPolicy.KEYWORD, overlay("auto_renews", old=None)
    )
    assert answer.decision == "PASS" and answer.evidence_strength == "followup"
    assert answer.evidence_quote == "60 days prior notice"
    assert answer.field_evidence_source == "extraction"
    assert answer.condition_evidence_source == "followup"
    assert answer.condition_evidence_quote == "follow-up quote"


def test_currency_overlay():
    rule, result, confidence = inputs()
    rule = Rule(
        id="cap",
        description="test",
        clause_type=rule.clause_type,
        field="cap_amount",
        operator=">=",
        threshold=2000000,
        currency="USD",
    )
    confidence.field_status["currency"] = "unsupported"
    answer = evaluate_rule(
        rule,
        result,
        confidence,
        TrustPolicy.KEYWORD,
        overlay("currency", old=None, value="USD", status="supported"),
    )
    assert answer.decision == "PASS" and answer.evidence_strength == "followup"
    assert (
        evaluate_rule(
            rule,
            result,
            confidence,
            TrustPolicy.STRICT,
            overlay("currency", old=None, value="USD", status="supported"),
        ).reason_code
        == "currency_unverified"
    )
