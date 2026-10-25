"""Deterministic company-rule decisions using per-field evidence trust."""

from contractguard.extraction.confidence import ConfidenceResult
from contractguard.extraction.schemas import ExtractionResult
from contractguard.rules.models import Decision, Rule, RuleResult, TrustPolicy


def evaluate_rule(
    rule: Rule,
    result: ExtractionResult,
    confidence: ConfidenceResult,
    policy: TrustPolicy,
) -> RuleResult:
    if rule.clause_type != result.clause_type:
        raise ValueError("Rule and extraction clause types do not match")
    extraction = result.extraction
    value = getattr(extraction, rule.field, None)
    status = confidence.field_status.get(rule.field, "unsupported")
    condition = rule.when
    condition_value = getattr(extraction, condition.field, None) if condition else None
    condition_status = (
        confidence.field_status.get(condition.field, "unsupported")
        if condition
        else None
    )
    currency_status = (
        confidence.field_status.get("currency", "unsupported")
        if rule.currency
        else None
    )
    used: list[str] = []

    def finish(decision: Decision, code: str, reason: str) -> RuleResult:
        strength = None
        if decision != Decision.NEEDS_REVIEW and used:
            weakest = max(
                used,
                key=lambda s: (
                    "supported",
                    "keyword_signal",
                    "explicit_negative",
                ).index(s),
            )
            strength = {
                "supported": "proof",
                "keyword_signal": "weak_signal",
                "explicit_negative": "explicit_negative",
            }[weakest]
        return RuleResult(
            rule_id=rule.id,
            clause_type=rule.clause_type,
            policy=policy,
            decision=decision,
            reason_code=code,
            reason=reason,
            field=rule.field,
            value=value,
            field_status=status,
            condition_field=condition.field if condition else None,
            condition_value=condition_value,
            condition_status=condition_status,
            currency_status=currency_status,
            evidence_strength=strength,
            confidence_level=confidence.level,
            evidence_quote=getattr(extraction, "evidence_quote", None),
            evidence_chunk_id=getattr(extraction, "evidence_chunk_id", None),
        )

    review = Decision.NEEDS_REVIEW
    if extraction is None or result.error:
        return finish(review, "extraction_failed", "Extraction failed or is missing.")
    if confidence.level == "NOT_FOUND":
        return finish(review, "clause_not_found", "The clause was not found.")
    if confidence.level == "LOW":
        return finish(review, "low_confidence", "Extraction confidence is low.")
    if condition:
        if condition_status not in policy.statuses:
            return finish(
                review, "condition_unverified", "The condition is not trusted."
            )
        used.append(condition_status)
        if condition_value != condition.equals:
            return finish(
                Decision.NOT_APPLICABLE,
                "condition_not_met",
                "The trusted condition does not match the required value.",
            )
    if status not in policy.statuses:
        return finish(
            review, f"value_{status}", f"The field status {status} is not trusted."
        )
    used.append(status)
    if rule.currency is not None:
        if currency_status not in policy.statuses:
            return finish(review, "currency_unverified", "The currency is not trusted.")
        currency = getattr(extraction, "currency", None)
        if currency is None or currency.casefold() != rule.currency.casefold():
            return finish(
                review, "currency_mismatch", "The currency does not match the rule."
            )
    if rule.operator in (">=", "<=") and value is None:
        return finish(
            review, "value_absence_claim", "There is no numeric value to compare."
        )
    if rule.operator == ">=":
        passed = value >= rule.threshold
    elif rule.operator == "<=":
        passed = value <= rule.threshold
    elif rule.operator == "==":
        passed = value == rule.threshold
    elif rule.operator == "is_true":
        passed = value is True
    else:
        passed = value is False
    comparison = rule.operator
    if rule.operator in (">=", "<=", "=="):
        comparison += f" {rule.threshold!r}"
    return finish(
        Decision.PASS if passed else Decision.FAIL,
        "compared",
        f"{rule.field}={value!r} {'satisfies' if passed else 'does not satisfy'} "
        f"{comparison}.",
    )


def evaluate_rules(
    rules: list[Rule],
    results_by_clause_type: dict[str, ExtractionResult],
    confidences: dict[str, ConfidenceResult],
    policy: TrustPolicy,
) -> list[RuleResult]:
    """Evaluate in rule order; missing clause mappings raise KeyError."""
    return [
        evaluate_rule(
            rule,
            results_by_clause_type[rule.clause_type],
            confidences[rule.clause_type],
            policy,
        )
        for rule in rules
    ]
