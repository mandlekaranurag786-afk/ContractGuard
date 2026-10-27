"""Bounded, auditable follow-up orchestration using an injected LLM client."""

from dataclasses import dataclass
from typing import Literal, Protocol

from contractguard.chunking import Chunk
from contractguard.extraction.confidence import ConfidenceResult
from contractguard.extraction.schemas import ExtractionResult
from contractguard.followup.extract import FieldExtractor
from contractguard.followup.merge import TRUSTED_FOLLOWUP, merge
from contractguard.followup.models import (
    ChangedRule,
    FieldEvidence,
    FollowupReport,
    FollowupStep,
    MergeOutcome,
    MergeResult,
)
from contractguard.llm.client import LLMClient, LLMResponse
from contractguard.llm.errors import LLMQuotaError
from contractguard.retrieval.models import SearchResult
from contractguard.rules.engine import evaluate_rule
from contractguard.rules.models import Decision, Rule, RuleResult, TrustPolicy

FOLLOWUP_QUERIES = {
    "has_damages_exclusion": "neither party shall be liable for indirect incidental "
    "special consequential damages or lost profits",
    "has_money_cap": "total aggregate liability shall not exceed the amount",
    "cap_amount": "total aggregate liability shall not exceed the amount",
    "auto_renews": "this agreement shall automatically renew for successive terms "
    "unless notice of non-renewal",
    "notice_days": "written notice of non-renewal days prior to the expiration "
    "of the term",
    "initial_term_months": "the initial term of this agreement shall be years from "
    "the effective date",
}
TARGET_REASONS = {"value_absence_claim", "value_unsupported", "condition_unverified"}


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[SearchResult]: ...


class BudgetExhausted(RuntimeError):
    """Raised before an attempt that would exceed the per-contract budget."""


class BudgetClient:
    def __init__(self, client: LLMClient, max_calls: int):
        self.client = client
        self.max_calls = max_calls
        self.attempts: list[LLMResponse | None] = []

    def generate_json(self, prompt, schema):
        if len(self.attempts) >= self.max_calls:
            raise BudgetExhausted()
        self.attempts.append(None)
        response = self.client.generate_json(prompt, schema)
        self.attempts[-1] = response
        return response

    def cost(self) -> dict[str, int]:
        cached = [r for r in self.attempts if r is not None and r.cached]
        uncached = [r for r in self.attempts if r is not None and not r.cached]
        return {
            "llm_calls": len(self.attempts) - len(cached),
            "cache_hits": len(cached),
            "input_tokens": sum(r.input_tokens or 0 for r in uncached),
            "output_tokens": sum(r.output_tokens or 0 for r in uncached),
            "cached_input_tokens": sum(r.input_tokens or 0 for r in cached),
            "cached_output_tokens": sum(r.output_tokens or 0 for r in cached),
        }


@dataclass
class LoopResult:
    results: list[RuleResult]
    report: FollowupReport
    cost: dict[str, int]


def _currency_merge(old, status, evidence: FieldEvidence) -> MergeResult:
    if status not in {"supported", "keyword_signal"}:
        # Currency codes are case-insensitive; preserve the original spelling.
        if isinstance(old, str) and isinstance(evidence.value, str):
            if old.casefold() == evidence.value.casefold():
                evidence = evidence.model_copy(update={"value": old})
        return merge(old, status, evidence)
    # Do not invoke merge on a trusted field. Keep its value; only a verified,
    # trusted disagreement can flag a conflict, never replace trusted currency.
    conflict = (
        evidence.verified
        and evidence.status in TRUSTED_FOLLOWUP
        and evidence.value is not None
        and str(old).casefold() != str(evidence.value).casefold()
    )
    return MergeResult(
        outcome=MergeOutcome.CONFLICT if conflict else MergeOutcome.DISCARDED_UNTRUSTED,
        field="currency",
        old_value=old,
        old_status=status,
        new_value=evidence.value,
        new_status=evidence.status,
        final_value=old,
        final_status=status,
        reason="Currencies disagree."
        if conflict
        else "Keep the already-trusted currency.",
        evidence_quote=evidence.evidence_quote,
        evidence_chunk_id=evidence.evidence_chunk_id,
    )


def run_followup(
    rules: list[Rule],
    results: dict[str, ExtractionResult],
    confidences: dict[str, ConfidenceResult],
    original: list[RuleResult],
    retriever: Retriever,
    seen_chunks: dict[str, set[str]],
    client: LLMClient,
    policy: TrustPolicy = TrustPolicy.KEYWORD,
    *,
    version: Literal["v1", "v2"] = "v1",
    original_chunks: dict[str, list[Chunk]] | None = None,
    max_calls: int = 4,
    new_chunks: int = 3,
) -> LoopResult:
    if version not in {"v1", "v2"}:
        raise ValueError("Invalid follow-up version")
    if max_calls < 0 or new_chunks < 1:
        raise ValueError("Invalid follow-up budget or chunk count")
    budget = BudgetClient(client, max_calls)
    extractor = FieldExtractor(budget)
    report = FollowupReport(
        enabled=True, version=version, status="completed", max_calls=max_calls
    )
    seen = {clause: set(ids) for clause, ids in seen_chunks.items()}
    overlays: dict[str, dict[str, MergeResult]] = {}
    targets: dict[tuple[str, str], list[str]] = {}
    target_reasons: dict[tuple[str, str], set[str]] = {}
    for item in original:
        if (
            item.decision == Decision.NEEDS_REVIEW
            and item.reason_code in TARGET_REASONS
        ):
            field = (
                item.condition_field
                if item.reason_code == "condition_unverified"
                else item.field
            )
            if field:
                targets.setdefault((item.clause_type, field), []).append(item.rule_id)
                target_reasons.setdefault((item.clause_type, field), set()).add(
                    item.reason_code
                )

    for (clause, field), ids in targets.items():
        include_original = version == "v2" and bool(
            target_reasons[clause, field]
            & {"value_unsupported", "condition_unverified"}
        )
        if include_original and (
            original_chunks is None or clause not in original_chunks
        ):
            raise ValueError(f"Original chunks required for v2: {clause}")
        step = FollowupStep(
            step=len(report.steps) + 1,
            clause_type=clause,
            field=field,
            rule_ids=ids,
            query=FOLLOWUP_QUERIES.get(field),
            chunk_policy="original_plus_unseen" if include_original else "unseen_only",
        )
        report.steps.append(step)
        if len(budget.attempts) >= max_calls:
            step.note = "budget_exhausted"
            break
        if not step.query:
            step.note = "no_query"
            continue
        status = confidences[clause].field_status.get(field, "unsupported")
        if status in {"supported", "keyword_signal"}:
            step.note = "already_trusted"
            continue
        seen_ids = seen.setdefault(clause, set())
        retrieved = retriever.search(step.query, k=len(seen_ids) + new_chunks)
        chunks = []
        for hit in retrieved:
            if hit.chunk.chunk_id not in seen_ids:
                chunks.append(hit.chunk)
                seen_ids.add(hit.chunk.chunk_id)
                if len(chunks) == new_chunks:
                    break
        step.new_chunk_ids = [c.chunk_id for c in chunks]
        if include_original:
            chunks = [*original_chunks[clause], *chunks]
        if not chunks:
            step.note = "no_new_chunks"
            continue
        step.chunk_ids = [c.chunk_id for c in chunks]
        start = len(budget.attempts)
        try:
            response = extractor.extract(clause, field, chunks)
        except BudgetExhausted:
            step.note = "budget_exhausted"
            break
        except LLMQuotaError:
            step.note = "quota_exhausted"
            report.status = "followup_skipped"
            break
        finally:
            attempts = budget.attempts[start:]
            replies = [r for r in attempts if r is not None]
            step.input_tokens = sum(r.input_tokens or 0 for r in replies)
            step.output_tokens = sum(r.output_tokens or 0 for r in replies)
            step.latency_ms = sum(r.latency_ms or 0 for r in replies)
            step.cached = bool(attempts) and all(
                r is not None and r.cached for r in attempts
            )
        if response.error:
            step.note = "extraction_error"
            step.error = response.error
            continue
        evidence = response.evidence
        step.value, step.quote_excerpt = (
            evidence.value,
            (evidence.evidence_quote or "")[:200],
        )
        step.verified, step.status = evidence.verified, evidence.status
        extraction = results[clause].extraction
        merged = merge(
            getattr(extraction, field, None),
            status,
            evidence,
            old_currency=getattr(extraction, "currency", None),
        )
        overlays.setdefault(clause, {})[field] = merged
        step.merge_outcome, step.note = merged.outcome, "merged"
        if field == "cap_amount":
            currency = evidence.model_copy(
                update={
                    "field": "currency",
                    "value": evidence.currency,
                    "currency": None,
                    "status": evidence.currency_status or "followup_untrusted",
                }
            )
            overlays[clause]["currency"] = _currency_merge(
                getattr(extraction, "currency", None),
                confidences[clause].field_status.get("currency", "unsupported"),
                currency,
            )

    report.calls_used = len(budget.attempts)
    usage = budget.cost()
    report.live_calls = usage["llm_calls"]
    report.cached_calls = usage["cache_hits"]
    report.uncached_input_tokens = usage["input_tokens"]
    report.uncached_output_tokens = usage["output_tokens"]
    if report.status == "followup_skipped":
        return LoopResult(original, report, budget.cost())
    final = []
    for rule, before in zip(rules, original, strict=True):
        after = evaluate_rule(
            rule,
            results[rule.clause_type],
            confidences[rule.clause_type],
            policy,
            overlays.get(rule.clause_type),
        )
        if (
            before.decision != Decision.NEEDS_REVIEW
            and before.decision != after.decision
        ):
            report.steps.append(
                FollowupStep(
                    step=len(report.steps) + 1,
                    clause_type=rule.clause_type,
                    field=rule.field,
                    rule_ids=[rule.id],
                    note="guard_blocked",
                )
            )
            after = before
        if (before.decision, before.reason_code, before.evidence_strength) != (
            after.decision,
            after.reason_code,
            after.evidence_strength,
        ):
            report.changed_rules.append(
                ChangedRule(
                    rule_id=rule.id,
                    before_decision=before.decision,
                    before_reason_code=before.reason_code,
                    after_decision=after.decision,
                    after_reason_code=after.reason_code,
                    evidence_strength=after.evidence_strength,
                )
            )
        final.append(after)
    return LoopResult(final, report, budget.cost())
