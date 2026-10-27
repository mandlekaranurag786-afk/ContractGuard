"""Compare saved checker reports offline using one shared label truth function."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contractguard.followup.models import MergeOutcome
from contractguard.report import ContractReport
from contractguard.rules.models import Decision, Rule, load_rules
from evals.eval_rules import load_rows, outcome_flags, true_decision

NOTES = (
    "merged",
    "no_new_chunks",
    "no_query",
    "budget_exhausted",
    "extraction_error",
    "guard_blocked",
    "quota_exhausted",
)
CHANGE_FIELDS = (
    "contract",
    "rule",
    "before_decision",
    "before_reason_code",
    "after_decision",
    "after_reason_code",
    "field",
    "value",
    "evidence_strength",
    "truth",
    "unsafe",
    "followup_quote",
    "chunk_id",
    "human_verdict",
)


def load_reports(directory: Path) -> dict[str, ContractReport]:
    reports = {}
    for path in sorted(directory.glob("*.json")):
        report = ContractReport.model_validate_json(path.read_text(encoding="utf-8"))
        name = Path(report.contract).name
        if name in reports:
            raise ValueError(f"Duplicate contract: {name}")
        reports[name] = report
    if not reports:
        raise ValueError(f"No reports in {directory}")
    return reports


def score_reports(
    reports: dict[str, ContractReport], rules: list[Rule], labels: dict
) -> dict:
    records = {}
    by_id = {r.id: r for r in rules}
    for contract, report in reports.items():
        if len(report.rule_results) != len(rules) or {
            r.rule_id for r in report.rule_results
        } != set(by_id):
            raise ValueError(f"Rule set differs for {contract}")
        for result in report.rule_results:
            rule = by_id[result.rule_id]
            if rule.field != result.field or rule.clause_type != result.clause_type:
                raise ValueError(f"Rule definition mismatch: {contract}/{rule.id}")
            truth = true_decision(rule, labels[contract, rule.clause_type])
            records[contract, rule.id] = dict(
                result=result, truth=truth, **outcome_flags(truth, result.decision)
            )
    return records


def metrics(records: dict) -> dict:
    scored = [r for r in records.values() if r["scored"]]
    return dict(
        scored=len(scored),
        correct=sum(r["correct"] for r in scored),
        unsafe=sum(r["unsafe"] for r in scored),
        review=sum(r["review"] for r in scored),
        all_review=sum(
            r["result"].decision == Decision.NEEDS_REVIEW for r in records.values()
        ),
    )


def accepted(before: dict, after: dict) -> bool:
    return after["unsafe"] == 0 and before["review"] - after["review"] >= 10


def changes(before: dict, after: dict) -> list[dict]:
    if before.keys() != after.keys():
        raise ValueError("Before/after rule rows differ")
    changed = []
    for (contract, rule), new in after.items():
        old = before[contract, rule]["result"]
        result = new["result"]
        if old.decision == result.decision:
            continue
        quote, chunk_id = result.evidence_quote, result.evidence_chunk_id
        if (
            result.field_evidence_source != "followup"
            and result.condition_evidence_source == "followup"
        ):
            quote, chunk_id = (
                result.condition_evidence_quote,
                result.condition_evidence_chunk_id,
            )
        changed.append(
            dict(
                contract=contract,
                rule=rule,
                before_decision=old.decision.value,
                before_reason_code=old.reason_code,
                after_decision=result.decision.value,
                after_reason_code=result.reason_code,
                field=result.field,
                value=result.value,
                evidence_strength=result.evidence_strength,
                truth=new["truth"] or "UNSCORED",
                unsafe="yes" if new["unsafe"] else "no",
                followup_quote=quote or "",
                chunk_id=chunk_id or "",
                human_verdict="",
            )
        )
    return changed


def score_section(name: str, records: dict, rules: list[Rule]) -> str:
    counts = metrics(records)
    scored = [r for r in records.values() if r["scored"]]
    decisions = Counter(r["result"].decision for r in scored)
    all_decisions = Counter(r["result"].decision for r in records.values())
    lines = [
        name,
        f"Scored rule-rows: {counts['scored']}/{len(records)}",
        "Decision counts (scored): "
        + ", ".join(f"{d}={decisions[d]}" for d in Decision),
        "Decision counts (all): "
        + ", ".join(f"{d}={all_decisions[d]}" for d in Decision),
        f"UNSAFE: {counts['unsafe']}",
        f"Correct (scored): {counts['correct']}",
        f"Review (scored): {counts['review']}",
        f"Review (all {len(records)} rows): {counts['all_review']}",
        "rule | scored | correct | unsafe | review",
    ]
    for rule in rules:
        subset = {k: v for k, v in records.items() if k[1] == rule.id}
        m = metrics(subset)
        lines.append(
            rule.id
            + " | "
            + " | ".join(str(m[k]) for k in ("scored", "correct", "unsafe", "review"))
        )
    return "\n".join(lines)


def trace_section(name: str, reports: dict[str, ContractReport]) -> str:
    steps = [s for p in reports.values() for s in p.followup.steps]
    notes = Counter(s.note for s in steps)
    outcomes = Counter((s.field, s.merge_outcome) for s in steps if s.merge_outcome)
    calls = [p.followup.calls_used for p in reports.values()]

    def total(field):
        values = [
            getattr(p.followup, field) if p.followup.calls_used else 0
            for p in reports.values()
        ]
        return (
            str(sum(values))
            if all(v is not None for v in values)
            else "not available (legacy report)"
        )

    lines = [
        f"TRACE: {name}",
        "Steps by note: "
        + ", ".join(
            f"{note}={notes[note]}"
            for note in (*NOTES, *sorted(set(notes) - set(NOTES)))
        ),
        "Merge outcomes by field: field | " + " | ".join(o.value for o in MergeOutcome),
    ]
    for field in sorted({field for field, _ in outcomes}):
        lines.append(
            field + " | " + " | ".join(str(outcomes[field, o]) for o in MergeOutcome)
        )
    if not outcomes:
        lines.append("(no merges)")
    errors = Counter(s.error for s in steps if s.error)
    lines.append("Errors by message:")
    if not errors:
        lines.append("(none recorded)")
    for message, count in sorted(errors.items()):
        lines.append(f"count={count}\n{message}")
    lines.extend(
        [
            f"Follow-up calls per contract: total={sum(calls)}, "
            f"median={statistics.median(calls):g}, max={max(calls)}",
            f"Follow-up live calls: {total('live_calls')}",
            f"Follow-up cached calls: {total('cached_calls')}",
            "Follow-up uncached input/output tokens: "
            f"{total('uncached_input_tokens')}/{total('uncached_output_tokens')}",
            "All-pipeline uncached input/output tokens: "
            f"{sum(p.cost.input_tokens for p in reports.values())}/"
            f"{sum(p.cost.output_tokens for p in reports.values())}",
        ]
    )
    return "\n".join(lines)


def build_report(before, after, rules, labels) -> tuple[str, list[dict]]:
    if before.keys() != after.keys():
        raise ValueError("Before/after contract sets differ")
    for name in before:
        if (
            before[name].trust_policy,
            before[name].prompt_version,
            before[name].model,
        ) != (after[name].trust_policy, after[name].prompt_version, after[name].model):
            raise ValueError(f"Run configuration mismatch: {name}")
    old, new = score_reports(before, rules, labels), score_reports(after, rules, labels)
    changed = changes(old, new)
    sections = [
        score_section("BEFORE", old, rules),
        score_section("AFTER", new, rules),
        f"CHANGED DECISIONS: {len(changed)}",
    ]
    for row in changed:
        sections.append(
            f"{row['contract']} | {row['rule']} | {row['before_decision']} "
            f"({row['before_reason_code']}) -> {row['after_decision']} "
            f"({row['after_reason_code']}) | field={row['field']} | "
            f"value={row['value']!r} | strength={row['evidence_strength']} | "
            f"truth={row['truth']} | unsafe={row['unsafe']}\n"
            f"Quote: {row['followup_quote']}\nChunk: {row['chunk_id']}"
        )
    sections += [trace_section("BEFORE", before), trace_section("AFTER", after)]
    timing = [
        "TIMING: elapsed_ms per contract (p95 uses nearest rank)",
        "run | median | p95",
    ]
    for name, reports in (("before", before), ("after", after)):
        values = sorted(p.cost.elapsed_ms for p in reports.values())
        timing.append(
            f"{name} | {statistics.median(values):.2f} | "
            f"{values[math.ceil(0.95 * len(values)) - 1]:.2f}"
        )
    sections.append("\n".join(timing))
    old_m, new_m = metrics(old), metrics(new)
    sections.append(
        "PRE-REGISTERED ACCEPTANCE\n"
        f"UNSAFE on scored rows: {new_m['unsafe']} (required: 0)\n"
        f"Scored review reduction: {old_m['review'] - new_m['review']} "
        "(required: at least 10)\n"
        f"Result: {'ACCEPT' if accepted(old_m, new_m) else 'REJECT'}\n"
        "Every changed decision must be checked by hand before the decision is final."
    )
    return "\n\n".join(sections) + "\n", changed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args(argv)
    if not args.tag or not args.tag.replace("_", "").replace("-", "").isalnum():
        parser.error("Invalid output tag")
    output = Path(f"evals/followup_eval_{args.tag}.txt")
    manual = Path(f"evals/followup_manual_check_{args.tag}.csv")
    if output.exists() or manual.exists():
        raise FileExistsError("Refusing to overwrite existing follow-up outputs")
    rules = load_rules("policies/eval_policy.json")
    labels = load_rows(Path("evals/extraction_labels_v2.csv"))
    before, after = load_reports(args.before), load_reports(args.after)
    if set(before) != {name for name, _ in labels}:
        raise ValueError("Reports must cover every labelled contract")
    report, changed = build_report(before, after, rules, labels)
    report = (
        f"Before: {args.before}\nAfter: {args.after}\n"
        "Labels: evals/extraction_labels_v2.csv\n"
        "Policy: policies/eval_policy.json\n\n" + report
    )
    with manual.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CHANGE_FIELDS)
        writer.writeheader()
        writer.writerows(changed)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(report, end="")


if __name__ == "__main__":
    main()
