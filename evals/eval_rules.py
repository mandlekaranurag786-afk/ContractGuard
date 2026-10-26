"""Offline rule-policy evaluation against independent hand labels."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contractguard.evaluation.extraction_metrics import FIELD_KINDS, parse_label
from contractguard.extraction.confidence import compute_confidence
from contractguard.extraction.schemas import ExtractionResult
from contractguard.rules.engine import evaluate_rule
from contractguard.rules.models import Decision, Rule, TrustPolicy, load_rules
from scripts.eval_confidence import percent, rebuild_extraction

ROOT = Path(__file__).resolve().parents[1]
RUNS = ("v2", "v2_repeat")
AUTO = {Decision.PASS, Decision.FAIL, Decision.NOT_APPLICABLE}


def true_decision(rule: Rule, labels: dict) -> Decision | None:
    """Check found, condition, rule field, then currency; None means unscored."""
    if parse_label(labels.get("found", ""), bool) is not True:
        return None

    def label_value(field):
        return parse_label(labels.get(field, ""), FIELD_KINDS[rule.clause_type][field])

    if rule.when:
        condition = label_value(rule.when.field)
        if condition is None:
            return None
        if condition == "NONE":
            return Decision.NEEDS_REVIEW
        if condition != rule.when.equals:
            return Decision.NOT_APPLICABLE
    value = label_value(rule.field)
    if value is None:
        return None
    if value == "NONE":
        return Decision.NEEDS_REVIEW
    if rule.currency is not None:
        currency = label_value("currency")
        if currency is None:
            return None
        if currency == "NONE" or currency.casefold() != rule.currency.casefold():
            return Decision.NEEDS_REVIEW
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
    return Decision.PASS if passed else Decision.FAIL


def outcome_flags(truth: Decision | None, engine: Decision) -> dict[str, bool]:
    scored = truth is not None
    return {
        "scored": scored,
        "correct": scored and engine == truth,
        "unsafe": scored and engine in AUTO and engine != truth,
        "review": scored and engine == Decision.NEEDS_REVIEW,
        "unnecessary_review": scored
        and engine == Decision.NEEDS_REVIEW
        and truth in AUTO,
    }


def select_policy(metrics: dict) -> tuple[TrustPolicy, list[str]]:
    chosen = TrustPolicy.STRICT
    lines = ["Start at STRICT."]
    for candidate in (TrustPolicy.KEYWORD, TrustPolicy.BALANCED):
        safe = all(metrics[run][candidate]["unsafe"] == 0 for run in RUNS)
        savings = [
            metrics[run][chosen]["review"] - metrics[run][candidate]["review"]
            for run in RUNS
        ]
        accepted = safe and all(saving >= 3 for saving in savings)
        lines.append(
            f"{candidate}: unsafe-free in both runs={safe}; fewer reviews "
            f"versus {chosen}={savings}; {'accepted' if accepted else 'blocked'}."
        )
        if accepted:
            chosen = candidate
    return chosen, lines


def load_rows(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keyed = {(row["contract"], row["clause_type"]): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"Duplicate contract/clause rows in {path}")
    return keyed


def evaluate(rows: dict, labels: dict, rules: list[Rule], policy: TrustPolicy):
    records = []
    for key, row in rows.items():
        label = labels[key]
        extraction = rebuild_extraction(row)
        checks = json.loads(row["checks"])
        confidence = compute_confidence(extraction, checks, row["clause_type"])
        result = ExtractionResult(
            clause_type=row["clause_type"],
            extraction=extraction,
            checks=checks,
            attempts=int(row.get("attempts") or 0),
            model=row.get("model") or None,
            cached=row.get("cached") == "True",
            latency_ms=None,
            input_tokens=None,
            output_tokens=None,
            error=row.get("error") or None,
        )
        for rule in rules:
            if rule.clause_type != row["clause_type"]:
                continue
            truth = true_decision(rule, label)
            answer = evaluate_rule(rule, result, confidence, policy)
            records.append(
                {
                    "contract": row["contract"],
                    "truth": truth or "UNSCORED",
                    **answer.model_dump(mode="json"),
                    **outcome_flags(truth, answer.decision),
                }
            )
    return records


def summarize(run: str, policy: TrustPolicy, records: list[dict], rules: list[Rule]):
    scored = [r for r in records if r["scored"]]
    totals = {
        name: sum(r[name] for r in scored)
        for name in ("correct", "unsafe", "review", "unnecessary_review")
    }
    engine = Counter(r["decision"] for r in scored)
    truth = Counter(r["truth"] for r in scored)
    strengths = Counter(r["evidence_strength"] for r in scored if r["decision"] in AUTO)
    lines = [
        f"RUN: {run} | POLICY: {policy}",
        f"Rule-rows: {len(records)}; scored: {len(scored)}; "
        f"UNSCORED: {len(records) - len(scored)}",
        "decision | engine | truth",
    ]
    lines.extend(f"{d} | {engine[d]} | {truth[d]}" for d in Decision)
    lines.extend(
        [
            f"Correct: {totals['correct']}/{len(scored)}",
            f"Review: {totals['review']}/{len(scored)} "
            f"({percent(totals['review'], len(scored))})",
            f"UNNECESSARY_REVIEW: {totals['unnecessary_review']}",
            f"UNSAFE: {totals['unsafe']}",
        ]
    )
    for row in scored:
        if row["unsafe"]:
            lines.append(
                " | ".join(
                    f"{field}={row[field]!r}"
                    for field in (
                        "contract",
                        "rule_id",
                        "truth",
                        "decision",
                        "field",
                        "value",
                        "field_status",
                        "condition_status",
                        "currency_status",
                        "evidence_strength",
                    )
                )
                + f" | quote excerpt={(row['evidence_quote'] or '')[:300]!r}"
            )
    lines.append("rule | scored | correct | unsafe | review")
    for rule in rules:
        subset = [r for r in scored if r["rule_id"] == rule.id]
        lines.append(
            f"{rule.id} | {len(subset)} | "
            + " | ".join(
                str(sum(r[k] for r in subset)) for k in ("correct", "unsafe", "review")
            )
        )
    lines.append(
        "Auto-decisions by evidence_strength: "
        + ", ".join(
            f"{s}={strengths[s]}" for s in ("proof", "weak_signal", "explicit_negative")
        )
    )
    return "\n".join(lines), totals


def main() -> None:
    output = ROOT / "evals"
    paths = [output / "rules_eval_v3.txt"] + [
        output / f"rule_results_{run}_{policy.value.lower()}_v3.csv"
        for run in RUNS
        for policy in TrustPolicy
    ]
    if any(path.exists() for path in paths):
        raise FileExistsError("Refusing to overwrite existing rule evaluation outputs")
    rules = load_rules(ROOT / "policies" / "eval_policy.json")
    labels = load_rows(output / "extraction_labels_v2.csv")
    sections = [
        "Offline rule evaluation; labels: evals/extraction_labels_v2.csv",
        "All metrics and decision/strength counts exclude UNSCORED rule-rows.",
        "Truth order: found=False -> UNSCORED; condition empty -> UNSCORED, "
        "NONE -> NEEDS_REVIEW, unequal -> NOT_APPLICABLE (ignore rule field); "
        "rule field empty -> UNSCORED, NONE -> NEEDS_REVIEW; currency empty -> "
        "UNSCORED, NONE/different -> NEEDS_REVIEW; then compare.",
        "Label found=False rows: "
        f"{sum(r['found'] == 'False' for r in labels.values())}",
    ]
    metrics, outputs = {}, {}
    for run in RUNS:
        rows = load_rows(output / f"extraction_results_{run}.csv")
        if rows.keys() != labels.keys():
            raise ValueError(f"Instances differ between {run} and labels")
        metrics[run] = {}
        for policy in TrustPolicy:
            records = evaluate(rows, labels, rules, policy)
            report, metrics[run][policy] = summarize(run, policy, records, rules)
            sections.append(report)
            outputs[output / f"rule_results_{run}_{policy.value.lower()}_v3.csv"] = (
                records
            )
    chosen, selection = select_policy(metrics)
    sections.append(
        "PRE-REGISTERED SELECTION\n"
        + "\n".join(selection)
        + f"\nSelected policy: {chosen}"
    )
    report = "\n\n".join(sections) + "\n"
    for path, records in outputs.items():
        with path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0])
            writer.writeheader()
            writer.writerows(records)
    with paths[0].open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(report, end="")


if __name__ == "__main__":
    main()
