"""Evaluate quote-based confidence using saved extraction CSVs only."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from contractguard.evaluation.extraction_metrics import (
    FIELD_KINDS,
    parse_label,
    score_row,
)
from contractguard.extraction.confidence import NUMERIC_FIELDS, compute_confidence
from contractguard.extraction.schemas import (
    LiabilityCapExtraction,
    RenewalTermExtraction,
)

ROOT = Path(__file__).resolve().parents[1]


def rebuild_extraction(row: dict):
    if row.get("error") or not row.get("predicted_found"):
        return None
    clause = row["clause_type"]
    values = {
        field: parse_label(row.get("predicted_" + field, ""), kind)
        for field, kind in {"found": bool, **FIELD_KINDS[clause]}.items()
    }
    values.update(
        evidence_quote=row.get("evidence_quote") or None,
        evidence_chunk_id=row.get("evidence_chunk_id") or None,
    )
    schema = (
        RenewalTermExtraction if clause == "Renewal Term" else LiabilityCapExtraction
    )
    return schema.model_validate(values)


def percent(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "N/A"


def evaluate(path: Path) -> str:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter()
    numeric = Counter()
    statuses = Counter()
    residual = []
    high_wrong, downgraded = [], []
    wrong_total = caught = 0
    for row in rows:
        clause = row["clause_type"]
        extraction = rebuild_extraction(row)
        confidence = compute_confidence(extraction, json.loads(row["checks"]), clause)
        labels = {
            k.removeprefix("label_"): v
            for k, v in row.items()
            if k.startswith("label_")
        }
        scores = score_row(clause, labels, extraction)
        wrong = [
            k.removesuffix("_correct")
            for k, v in scores.items()
            if k.endswith("_correct") and not v
        ]
        correct = not wrong
        counts[confidence.level, correct] += 1
        wrong_total += not correct
        caught += not correct and confidence.level != "HIGH"
        identity = f"{row['contract']} | {clause}"
        if confidence.level == "HIGH" and not correct:
            details = "; ".join(
                f"{field}: label={labels.get(field, '')!r}, "
                f"prediction={getattr(extraction, field, None)!r}"
                for field in wrong
            )
            high_wrong.append(
                f"{identity} | wrong fields: {details} | reasons={confidence.reasons!r}"
                f" | quote excerpt={(row.get('evidence_quote') or '')[:300]!r}"
            )
            if all(
                confidence.field_status.get(field) in ("supported", "keyword_signal")
                for field in wrong
            ):
                residual.append(high_wrong[-1])
        for key, correct_field in scores.items():
            if key.endswith("_correct"):
                field = key.removesuffix("_correct")
                status = confidence.field_status.get(field, "unsupported")
                statuses[clause, field, status, correct_field] += 1
        if correct and confidence.level != "HIGH":
            downgraded.append(
                f"{identity} | {confidence.level} | reasons={confidence.reasons!r}"
            )
        for field in NUMERIC_FIELDS[clause]:
            if field + "_correct" in scores:
                support = confidence.field_support.get(field, "not_applicable")
                numeric[field, support, scores[field + "_correct"]] += 1
    lines = [f"RUN: {path.name}", "level | rows | fully correct | wrong"]
    for level in ("HIGH", "MEDIUM", "LOW", "NOT_FOUND"):
        good, bad = counts[level, True], counts[level, False]
        lines.append(f"{level} | {good + bad} | {good} | {bad}")
    high = sum(counts["HIGH", correct] for correct in (True, False))
    lines.extend(
        [
            f"HIGH-but-wrong: {len(high_wrong)}/{high} "
            f"({percent(len(high_wrong), high)})",
            *high_wrong,
            f"Errors caught: {caught}/{wrong_total} ({percent(caught, wrong_total)})",
            f"Review rate: {len(rows) - high}/{len(rows)} "
            f"({percent(len(rows) - high, len(rows))})",
            f"Correct-but-downgraded: {len(downgraded)}",
            *downgraded,
            "Numeric support x correctness (scored fields only):",
            "field | support | correct | wrong",
        ]
    )
    for field in dict.fromkeys(f for fields in NUMERIC_FIELDS.values() for f in fields):
        for support in ("supported", "unsupported", "not_applicable"):
            lines.append(
                f"{field} | {support} | {numeric[field, support, True]} | "
                f"{numeric[field, support, False]}"
            )
    lines.append(
        "not_applicable: null numeric prediction (or failed extraction); "
        "excluded from the 2x2 supported/unsupported table."
    )
    lines.extend(
        [
            "Field status x correctness (scored fields only; nonzero rows):",
            "clause_type | field | status | correct | wrong",
        ]
    )
    for clause, field, status in sorted({key[:3] for key in statuses}):
        lines.append(
            f"{clause} | {field} | {status} | "
            f"{statuses[clause, field, status, True]} | "
            f"{statuses[clause, field, status, False]}"
        )
    lines.extend(
        [
            f"HIGH-but-wrong after field status: {len(residual)}",
            *(residual or ["None"]),
            "This count requires EVERY wrong field to be supported or keyword_signal; "
            "it does not change confidence levels.",
            "No support rule was specified for found, currency or cap_formula: "
            "present values are unsupported; null/False values are absence_claim.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    report = (
        "\n\n".join(
            evaluate(ROOT / "evals" / f"extraction_results_{tag}.csv")
            for tag in ("v2", "v2_repeat")
        )
        + "\n"
    )
    (ROOT / "evals" / "confidence_eval_v3.txt").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
