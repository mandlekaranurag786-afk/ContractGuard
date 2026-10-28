"""Experimental controller; dependencies live in closures, state holds data only."""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from contractguard.chunking import Chunk
from contractguard.extraction.confidence import ConfidenceResult
from contractguard.extraction.schemas import ExtractionResult
from contractguard.followup import loop as product
from contractguard.followup.extract import FieldExtractor
from contractguard.followup.merge import merge
from contractguard.followup.models import (
    ChangedRule,
    FieldEvidence,
    FollowupReport,
    FollowupStep,
    MergeResult,
)
from contractguard.llm.client import LLMClient
from contractguard.llm.errors import LLMQuotaError
from contractguard.rules.models import Decision, Rule, RuleResult, TrustPolicy


class GraphState(TypedDict):
    targets: list[dict]
    index: int
    seen: dict[str, set[str]]
    chunks: list[Chunk]
    evidence: FieldEvidence | None
    overlays: dict[str, dict[str, MergeResult]]
    report: FollowupReport
    final: list[RuleResult]
    route: str


def run_followup_graph(
    rules: list[Rule],
    results: dict[str, ExtractionResult],
    confidences: dict[str, ConfidenceResult],
    original: list[RuleResult],
    retriever: product.Retriever,
    seen_chunks: dict[str, set[str]],
    client: LLMClient,
    policy: TrustPolicy = TrustPolicy.KEYWORD,
    *,
    version: Literal["v1", "v2"] = "v1",
    original_chunks: dict[str, list[Chunk]] | None = None,
    max_calls: int = 4,
    new_chunks: int = 3,
) -> product.LoopResult:
    if version not in {"v1", "v2"}:
        raise ValueError("Invalid follow-up version")
    if max_calls < 0 or new_chunks < 1:
        raise ValueError("Invalid follow-up budget or chunk count")
    budget = product.BudgetClient(client, max_calls)
    extractor = FieldExtractor(budget)

    def select_targets(state):
        targets = {}
        for item in original:
            if item.decision != Decision.NEEDS_REVIEW:
                continue
            if item.reason_code not in product.TARGET_REASONS:
                continue
            field = (
                item.condition_field
                if item.reason_code == "condition_unverified"
                else item.field
            )
            if field:
                target = targets.setdefault(
                    (item.clause_type, field),
                    dict(
                        clause=item.clause_type,
                        field=field,
                        ids=[],
                        reasons=set(),
                    ),
                )
                target["ids"].append(item.rule_id)
                target["reasons"].add(item.reason_code)
        return {"targets": list(targets.values())}

    def next_target(state):
        index = state["index"]
        if index == len(state["targets"]):
            return {"route": "reevaluate"}
        target = state["targets"][index]
        clause, field = target["clause"], target["field"]
        include_original = version == "v2" and bool(
            target["reasons"] & {"value_unsupported", "condition_unverified"}
        )
        if include_original and (
            original_chunks is None or clause not in original_chunks
        ):
            raise ValueError(f"Original chunks required for v2: {clause}")
        report = state["report"].model_copy(deep=True)
        step = FollowupStep(
            step=len(report.steps) + 1,
            clause_type=clause,
            field=field,
            rule_ids=target["ids"],
            query=product.FOLLOWUP_QUERIES.get(field),
            chunk_policy="original_plus_unseen" if include_original else "unseen_only",
        )
        report.steps.append(step)
        route = "search"
        if report.calls_used >= max_calls:
            step.note, route = "budget_exhausted", "reevaluate"
        elif not step.query:
            step.note, route = "no_query", "next_target"
        elif confidences[clause].field_status.get(field, "unsupported") in {
            "supported",
            "keyword_signal",
        }:
            step.note, route = "already_trusted", "next_target"
        return {"index": index + 1, "report": report, "route": route}

    def search(state):
        report = state["report"].model_copy(deep=True)
        step = report.steps[-1]
        seen = {clause: set(ids) for clause, ids in state["seen"].items()}
        ids = seen.setdefault(step.clause_type, set())
        hits = retriever.search(step.query, k=len(ids) + new_chunks)
        chunks = []
        for hit in hits:
            if hit.chunk.chunk_id not in ids:
                chunks.append(hit.chunk)
                ids.add(hit.chunk.chunk_id)
                if len(chunks) == new_chunks:
                    break
        step.new_chunk_ids = [c.chunk_id for c in chunks]
        if step.chunk_policy == "original_plus_unseen":
            chunks = [*original_chunks[step.clause_type], *chunks]
        if not chunks:
            step.note = "no_new_chunks"
        step.chunk_ids = [c.chunk_id for c in chunks]
        return dict(
            report=report,
            seen=seen,
            chunks=chunks,
            route="extract" if chunks else "next_target",
        )

    def extract(state):
        report = state["report"].model_copy(deep=True)
        step = report.steps[-1]
        start = len(budget.attempts)
        route, evidence = "merge", None
        try:
            response = extractor.extract(step.clause_type, step.field, state["chunks"])
        except product.BudgetExhausted:
            step.note, route = "budget_exhausted", "reevaluate"
        except LLMQuotaError:
            step.note, route = "quota_exhausted", "skipped"
            report.status = "followup_skipped"
        else:
            if response.error:
                step.note, step.error = "extraction_error", response.error
                route = "next_target"
            else:
                evidence = response.evidence
        finally:
            attempts = budget.attempts[start:]
            replies = [r for r in attempts if r is not None]
            step.input_tokens = sum(r.input_tokens or 0 for r in replies)
            step.output_tokens = sum(r.output_tokens or 0 for r in replies)
            step.latency_ms = sum(r.latency_ms or 0 for r in replies)
            step.cached = bool(attempts) and all(
                r is not None and r.cached for r in attempts
            )
            report.calls_used = len(budget.attempts)
        return dict(report=report, evidence=evidence, route=route)

    def merge_evidence(state):
        report = state["report"].model_copy(deep=True)
        step = report.steps[-1]
        clause, field = step.clause_type, step.field
        evidence = state["evidence"]
        step.value = evidence.value
        step.quote_excerpt = (evidence.evidence_quote or "")[:200]
        step.verified, step.status = evidence.verified, evidence.status
        extraction = results[clause].extraction
        merged = merge(
            getattr(extraction, field, None),
            confidences[clause].field_status.get(field, "unsupported"),
            evidence,
            old_currency=getattr(extraction, "currency", None),
        )
        overlays = {
            clause: dict(values) for clause, values in state["overlays"].items()
        }
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
            overlays[clause]["currency"] = product._currency_merge(
                getattr(extraction, "currency", None),
                confidences[clause].field_status.get("currency", "unsupported"),
                currency,
            )
        return {"report": report, "overlays": overlays}

    def reevaluate(state):
        report = state["report"].model_copy(deep=True)
        final = []
        for rule, before in zip(rules, original, strict=True):
            after = product.evaluate_rule(
                rule,
                results[rule.clause_type],
                confidences[rule.clause_type],
                policy,
                state["overlays"].get(rule.clause_type),
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
        return {"report": report, "final": final}

    def skipped(state):
        return {"final": original}

    graph = StateGraph(GraphState)
    for name, node in (
        ("select_targets", select_targets),
        ("next_target", next_target),
        ("search", search),
        ("extract", extract),
        ("merge", merge_evidence),
        ("reevaluate", reevaluate),
        ("skipped", skipped),
    ):
        graph.add_node(name, node)
    graph.add_edge(START, "select_targets")
    graph.add_edge("select_targets", "next_target")
    for node, destinations in (
        ("next_target", ["search", "next_target", "reevaluate"]),
        ("search", ["extract", "next_target"]),
        ("extract", ["merge", "next_target", "reevaluate", "skipped"]),
    ):
        graph.add_conditional_edges(node, lambda s: s["route"], destinations)
    graph.add_edge("merge", "next_target")
    graph.add_edge("reevaluate", END)
    graph.add_edge("skipped", END)
    # At most len(original) unique targets, each with next/search/extract/merge.
    # Add select, final next, reevaluate (or skipped), plus termination headroom.
    # This also handles arbitrarily many skipped targets without spending budget.
    recursion_limit = 4 * len(original) + 8
    state = graph.compile().invoke(
        dict(
            targets=[],
            index=0,
            seen={k: set(v) for k, v in seen_chunks.items()},
            chunks=[],
            evidence=None,
            overlays={},
            final=[],
            route="",
            report=FollowupReport(
                enabled=True, version=version, status="completed", max_calls=max_calls
            ),
        ),
        config={"recursion_limit": recursion_limit},
    )
    report, cost = state["report"], budget.cost()
    report.calls_used = len(budget.attempts)
    report.live_calls, report.cached_calls = cost["llm_calls"], cost["cache_hits"]
    report.uncached_input_tokens = cost["input_tokens"]
    report.uncached_output_tokens = cost["output_tokens"]
    return product.LoopResult(state["final"], report, cost)
