"""Diagnose saved verdicts against hand labels, entirely offline."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contractguard.evaluation.extraction_metrics import FIELD_KINDS, parse_label
from contractguard.report import OverallStatus, overall_status
from contractguard.rules.models import Decision, RuleResult, TrustPolicy, load_rules
from evals.eval_followup import load_reports, score_reports
from evals.eval_rules import load_rows

LABELS = Path("evals/extraction_labels_v2.csv")
POLICY = Path("policies/eval_policy.json")
EXTRACTIONS = Path("evals/extraction_results_v2.csv")
CATEGORIES = (
    "retrieval_miss",
    "low_confidence",
    "extraction_wrong",
    "correct_but_unproven",
    "cap_formula",
    "other",
)


def diagnostic_value(result: RuleResult, labels: dict) -> dict:
    condition = result.reason_code == "condition_unverified"
    field = result.condition_field if condition else result.field
    if field is None:
        raise ValueError("condition_unverified requires a condition field")
    raw = labels.get(field, "")
    label = parse_label(raw, FIELD_KINDS[result.clause_type][field])
    value = result.condition_value if condition else result.value
    expected = None if label == "NONE" else label
    # Blank labels remain unknown, including when truth is NOT_APPLICABLE
    # because a labelled condition is unequal to the rule's condition.
    equal = (
        None
        if label is None
        else (
            value == expected and isinstance(value, bool) == isinstance(expected, bool)
        )
    )
    return dict(
        field=field,
        value=value,
        label=raw,
        status=result.condition_status if condition else result.field_status,
        source=(
            result.condition_evidence_source
            if condition
            else result.field_evidence_source
        ),
        equal=equal,
    )


def root_cause(result: RuleResult, labels: dict, extraction: dict) -> str:
    """First matching category wins; comparison uses the saved final verdict."""
    fact = diagnostic_value(result, labels)
    hit = parse_label(extraction.get("retrieval_hit", ""), bool)
    if result.reason_code == "clause_not_found" or (
        fact["equal"] is False and hit is False
    ):
        return "retrieval_miss"
    if result.reason_code == "low_confidence":
        return "low_confidence"
    if fact["equal"] is False:
        return "extraction_wrong"
    trusted = fact["status"] in result.policy.statuses
    if fact["source"] == "followup":
        trusted = result.policy != TrustPolicy.STRICT and fact["status"] in {
            "supported",
            "followup_pattern",
        }
    if fact["equal"] is True and not trusted:
        return "correct_but_unproven"
    if result.reason_code == "cap_is_formula":
        return "cap_formula"
    return "other"


def confusion(records: list[dict]) -> Counter:
    return Counter((r["truth"], r["result"].decision) for r in records if r["scored"])


def matrix(counts: Counter, statuses) -> list[str]:
    statuses = list(statuses)
    return ["truth / engine | " + " | ".join(statuses)] + [
        str(truth) + " | " + " | ".join(str(counts[truth, d]) for d in statuses)
        for truth in statuses
    ]


def rule_section(name: str, records: list[dict]) -> str:
    counts = confusion(records)
    lines = [name, *matrix(counts, Decision)]
    for decision, description in (
        (Decision.FAIL, "Truth FAIL caught automatically"),
        (Decision.PASS, "Truth PASS decided automatically"),
        (Decision.NEEDS_REVIEW, "Truth NEEDS_REVIEW kept for review"),
    ):
        total = sum(counts[decision, d] for d in Decision)
        lines.append(f"{description}: {counts[decision, decision]}/{total}")
    lines.append(f"Unsafe: {sum(r['unsafe'] for r in records)}")
    return "\n".join(lines)


def contract_verdicts(reports: dict, records: dict, rule_count: int) -> list[dict]:
    rows = []
    for contract in sorted(reports):
        scored = [
            r for (name, _), r in records.items() if name == contract and r["scored"]
        ]
        engine = overall_status([r["result"] for r in scored])
        truth = overall_status(
            [r["result"].model_copy(update={"decision": r["truth"]}) for r in scored]
        )
        rows.append(
            dict(
                contract=contract,
                scored=len(scored),
                all_scored=len(scored) == rule_count,
                truth=truth,
                engine=engine,
                unsafe=engine != truth and engine != OverallStatus.NEEDS_REVIEW,
            )
        )
    return rows


def detail(contract: str, rule_id: str, row: dict, labels: dict, extractions: dict):
    result = row["result"]
    key = (contract, result.clause_type)
    fact = diagnostic_value(result, labels[key])
    cause = root_cause(result, labels[key], extractions[key])
    return (
        f"{contract} | {rule_id} | truth={row['truth']} | "
        f"engine={result.decision} ({result.reason_code}) | "
        f"field={result.field} | predicted={result.value!r} | "
        f"label={labels[key].get(result.field, '')!r} | "
        f"field_status={result.field_status} | root_cause={cause} | "
        f"diagnostic_field={fact['field']} | diagnostic_value={fact['value']!r} | "
        f"diagnostic_label={fact['label']!r} | diagnostic_status={fact['status']}"
    )


def build_report(reports: dict, rules: list, labels: dict, extractions: dict) -> str:
    expected = {(name, rule.clause_type) for name in reports for rule in rules}
    for name, rows in (("labels", labels), ("extractions", extractions)):
        if missing := expected - rows.keys():
            raise ValueError(f"Missing {name} rows: {sorted(missing)}")
    records = score_reports(reports, rules, labels)
    scored = {k: r for k, r in records.items() if r["scored"]}
    sections = [
        f"Contracts: {len(reports)}\nScored rule-rows: {len(scored)}/{len(records)}",
        "A) RULE LEVEL (scored rows only; truth rows, engine columns)",
        rule_section("TOTAL", list(scored.values())),
    ]
    sections += [
        rule_section(rule.id, [r for (_, rid), r in scored.items() if rid == rule.id])
        for rule in rules
    ]
    missed = [
        detail(*key, row, labels, extractions)
        for key, row in scored.items()
        if row["truth"] == Decision.FAIL and row["result"].decision != Decision.FAIL
    ]
    sections.append("B) MISSED FAILS\n" + "\n".join(missed or ["(none)"]))
    counts, statuses, others = Counter(), Counter(), []
    for (contract, rid), row in scored.items():
        if not row["unnecessary_review"]:
            continue
        result = row["result"]
        key = (contract, result.clause_type)
        cause = root_cause(result, labels[key], extractions[key])
        counts[cause, rid] += 1
        if cause == "correct_but_unproven":
            statuses[diagnostic_value(result, labels[key])["status"], rid] += 1
        if cause == "other":
            others.append(detail(contract, rid, row, labels, extractions))

    def category_table(counter, categories):
        ids = [rule.id for rule in rules]
        lines = ["category / rule | " + " | ".join(ids) + " | TOTAL"]
        for category in categories:
            values = [counter[category, rid] for rid in ids]
            lines.append(
                str(category)
                + " | "
                + " | ".join(map(str, values))
                + f" | {sum(values)}"
            )
        totals = [
            sum(v for (_, key), v in counter.items() if key == rid) for rid in ids
        ]
        lines.append("TOTAL | " + " | ".join(map(str, totals)) + f" | {sum(totals)}")
        return "\n".join(lines)

    sections.extend(
        [
            "C) UNNECESSARY REVIEW ROOT CAUSES (first match wins)\n"
            + category_table(counts, CATEGORIES),
            "correct_but_unproven by status\n"
            + category_table(
                statuses, sorted({status for status, _ in statuses}, key=str)
            ),
            "Other rows\n" + "\n".join(others or ["(none)"]),
        ]
    )
    contracts = contract_verdicts(reports, records, len(rules))
    lines = [
        "D) CONTRACT LEVEL (both sides use the SAME scored rules only)",
        *matrix(Counter((r["truth"], r["engine"]) for r in contracts), OverallStatus),
        f"Unsafe overall: {sum(r['unsafe'] for r in contracts)}",
        f"All {len(rules)} rules scored: {sum(r['all_scored'] for r in contracts)}",
        "contract | scored rules | all rules scored | truth | engine | unsafe overall",
    ]
    contract_lines = [
        f"{r['contract']} | {r['scored']}/{len(rules)} | {r['all_scored']} | "
        f"{r['truth']} | {r['engine']} | {r['unsafe']}"
        for r in contracts
    ]
    lines += contract_lines
    lines.append("Contracts with differing overall verdicts:")
    lines += [
        line
        for line, row in zip(contract_lines, contracts, strict=True)
        if row["truth"] != row["engine"]
    ] or ["(none)"]
    sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args(argv)
    if not args.tag or not args.tag.replace("_", "").replace("-", "").isalnum():
        parser.error("Invalid output tag")
    output = Path(f"evals/verdict_eval_{args.tag}.txt")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    reports = load_reports(args.reports)
    labels, extractions = load_rows(LABELS), load_rows(EXTRACTIONS)
    rules = load_rules(POLICY)
    text = (
        f"Reports: {args.reports}\nLabels: {LABELS}\nPolicy: {POLICY}\n"
        f"Retrieval diagnostics: {EXTRACTIONS}\n"
        "Predictions: saved report values (including follow-up overlays).\n"
        "Blank labels are unknown; NONE means expected null.\n\n"
        + build_report(reports, rules, labels, extractions)
    )
    with output.open("x", encoding="utf-8") as handle:
        handle.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
