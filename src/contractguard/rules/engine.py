"""Deterministic company-rule decisions using per-field evidence trust."""

from contractguard.extraction.confidence import ConfidenceResult
from contractguard.extraction.schemas import ExtractionResult
from contractguard.followup.models import MergeOutcome, MergeResult
from contractguard.rules.models import Decision, Rule, RuleResult, TrustPolicy


def evaluate_rule(
    rule: Rule,
    result: ExtractionResult,
    confidence: ConfidenceResult,
    policy: TrustPolicy,
    overlay: dict[str, MergeResult] | None = None,
) -> RuleResult:
    if rule.clause_type != result.clause_type:
        raise ValueError("Rule and extraction clause types do not match")
    extraction = result.extraction
    accepted = {
        key: item
        for key, item in (overlay or {}).items()
        if item.outcome
        in {MergeOutcome.ACCEPTED_NEW, MergeOutcome.ACCEPTED_CONFIRMATION}
    }
    for key, item in (overlay or {}).items():
        if key != item.field:
            raise ValueError("Overlay key must match its field")

    def trusted(field, status):
        if field in accepted:
            return policy != TrustPolicy.STRICT and status in {
                "supported",
                "followup_pattern",
            }
        return status in policy.statuses

    def source(field):
        return "followup" if field in accepted else "extraction"

    def quote(field):
        return (
            accepted[field].evidence_quote
            if field in accepted
            else getattr(extraction, "evidence_quote", None)
        )

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
    if rule.field in accepted:
        value = accepted[rule.field].final_value
        status = accepted[rule.field].final_status
    if condition and condition.field in accepted:
        condition_value = accepted[condition.field].final_value
        condition_status = accepted[condition.field].final_status
    currency = getattr(extraction, "currency", None)
    if rule.currency and "currency" in accepted:
        currency = accepted["currency"].final_value
        currency_status = accepted["currency"].final_status

    def use(field, status):
        used.append("followup" if field in accepted else status)

    def finish(decision: Decision, code: str, reason: str) -> RuleResult:
        strength = None
        if decision != Decision.NEEDS_REVIEW and used:
            weakest = max(
                used,
                key=lambda s: (
                    "supported",
                    "keyword_signal",
                    "explicit_negative",
                    "followup",
                ).index(s),
            )
            strength = {
                "supported": "proof",
                "keyword_signal": "weak_signal",
                "explicit_negative": "explicit_negative",
                "followup": "followup",
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
            evidence_quote=quote(rule.field),
            evidence_chunk_id=(
                accepted[rule.field].evidence_chunk_id
                if rule.field in accepted
                else getattr(extraction, "evidence_chunk_id", None)
            ),
            field_evidence_source=source(rule.field),
            condition_evidence_source=source(condition.field) if condition else None,
            condition_evidence_quote=quote(condition.field) if condition else None,
            condition_evidence_chunk_id=(
                accepted[condition.field].evidence_chunk_id
                if condition and condition.field in accepted
                else getattr(extraction, "evidence_chunk_id", None)
                if condition
                else None
            ),
        )

    review = Decision.NEEDS_REVIEW
    if extraction is None or result.error:
        return finish(review, "extraction_failed", "Extraction failed or is missing.")
    if confidence.level == "NOT_FOUND":
        return finish(review, "clause_not_found", "The clause was not found.")
    if confidence.level == "LOW":
        return finish(review, "low_confidence", "Extraction confidence is low.")
    relevant = {rule.field}
    if condition:
        relevant.add(condition.field)
    if rule.currency:
        relevant.add("currency")
    for field in sorted(relevant):
        item = (overlay or {}).get(field)
        if item and item.outcome == MergeOutcome.CONFLICT:
            return finish(
                review,
                "followup_conflict",
                f"Follow-up conflict for {field}: old value {item.old_value!r}, "
                f"new value {item.new_value!r}.",
            )
    if condition:
        if not trusted(condition.field, condition_status):
            return finish(
                review,
                "condition_unverified",
                f"The condition field {condition.field} has status {condition_status} "
                "and is not trusted.",
            )
        use(condition.field, condition_status)
        if condition_value != condition.equals:
            return finish(
                Decision.NOT_APPLICABLE,
                "condition_not_met",
                "The trusted condition does not match the required value.",
            )
    if not trusted(rule.field, status):
        formula = getattr(extraction, "cap_formula", None)
        if rule.field == "cap_amount" and value is None and formula is not None:
            target = f"{rule.threshold} {rule.currency or ''}".strip()
            return finish(
                review,
                "cap_is_formula",
                f"The cap is a formula ({formula}), not a fixed amount, "
                f"so it cannot be compared with {target}.",
            )
        reasons = {
            "absence_claim": "No value was found and the text gives no proof either "
            "way; a person must check.",
            "unsupported": "A value was found but the quote does not prove it.",
            "conflicting": "The value disagrees with the quoted wording.",
        }
        reason = reasons.get(status, f"The field status {status} is not trusted.")
        if rule.field == "cap_amount" and value is not None and formula is not None:
            reason += f" The extraction also found a formula: {formula}."
        return finish(review, f"value_{status}", reason)
    use(rule.field, status)
    if rule.currency is not None:
        if not trusted("currency", currency_status):
            return finish(review, "currency_unverified", "The currency is not trusted.")
        if currency is None or currency.casefold() != rule.currency.casefold():
            return finish(
                review, "currency_mismatch", "The currency does not match the rule."
            )
        if "currency" in accepted:
            use("currency", currency_status)
    if rule.operator in (">=", "<=") and value is None:
        return finish(
            review,
            "value_absence_claim",
            "No value was found and the text gives no proof either way; "
            "a person must check.",
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
