"""Serializable contract decisions with evidence and execution metadata."""

from enum import StrEnum

from pydantic import BaseModel

from contractguard.rules.models import Decision, RuleResult, TrustPolicy


class OverallStatus(StrEnum):
    FAIL = "FAIL"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    PASS = "PASS"
    NO_APPLICABLE_RULES = "NO_APPLICABLE_RULES"

    @property
    def reason(self) -> str:
        return {
            self.FAIL: "At least one evaluated rule failed.",
            self.NEEDS_REVIEW: (
                "At least one rule requires human review; this is not an approval."
            ),
            self.PASS: (
                "At least one rule passed and no rule failed or requires review."
            ),
            self.NO_APPLICABLE_RULES: (
                "No applicable rules were evaluated; this is not an approval."
            ),
        }[self]


def overall_status(results: list[RuleResult]) -> OverallStatus:
    decisions = {result.decision for result in results}
    for decision in (Decision.FAIL, Decision.NEEDS_REVIEW, Decision.PASS):
        if decision in decisions:
            return OverallStatus(decision.value)
    return OverallStatus.NO_APPLICABLE_RULES


class ClauseSummary(BaseModel):
    clause_type: str
    confidence_level: str
    confidence_reasons: list[str]
    retrieved_chunk_ids: list[str]
    extraction_error: str | None


class Cost(BaseModel):
    """Logical generation attempts; response tokens include cached metadata.

    These are not billable-token estimates. Provider-internal retries and mixed
    cache outcomes within one extraction are not exposed by ExtractionResult.
    """

    llm_calls: int
    cache_hits: int
    input_tokens: int
    output_tokens: int
    elapsed_ms: float


class ContractReport(BaseModel):
    contract: str
    policy_path: str
    trust_policy: TrustPolicy
    model: str | None
    prompt_version: str
    overall_status: OverallStatus
    overall_reason: str
    counts: dict[Decision, int]
    rule_results: list[RuleResult]
    clauses: list[ClauseSummary]
    cost: Cost
