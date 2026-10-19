"""Build the hand-labeling sheet and guide for extraction evaluation.

Usage:
    uv run python -m scripts.build_extraction_labels
"""

from __future__ import annotations

import csv
import difflib
import sys
from pathlib import Path

from contractguard.extraction.prompt import FIELD_DEFINITIONS
from evals.eval_retrieval import (
    KNOWN_BAD_LABELS,
    MASTER_CLAUSES_CSV,
    PARSED_DIR,
    build_filename_to_docid_map,
    load_cuad_answers,
    normalize_filename,
)

LABELS_OUT = Path("evals/extraction_labels.csv")
GUIDE_OUT = Path("docs/labeling_guide.md")

CLAUSE_TYPES = ("Renewal Term", "Cap On Liability")
REQUIRED_CUAD_COLUMNS = (
    "Filename",
    "Renewal Term",
    "Renewal Term-Answer",
    "Notice Period To Terminate Renewal",
    "Notice Period To Terminate Renewal- Answer",
    "Expiration Date-Answer",
    "Cap On Liability",
    "Cap On Liability-Answer",
    "Uncapped Liability",
    "Uncapped Liability-Answer",
)
FIELDNAMES = (
    "contract",
    "clause_type",
    "is_abstention_case",
    "cuad_span",
    "cuad_renewal_answer",
    "cuad_notice_span",
    "cuad_notice_answer",
    "cuad_expiration_answer",
    "cuad_cap_answer",
    "cuad_uncapped_span",
    "cuad_uncapped_answer",
    "found",
    "initial_term_months",
    "auto_renews",
    "renewal_term_months",
    "notice_days",
    "has_money_cap",
    "cap_amount",
    "currency",
    "cap_formula",
    "has_damages_exclusion",
    "label_notes",
)


def _require_cuad_columns(fieldnames: list[str] | None) -> None:
    """Fail safely if the CUAD schema has changed."""
    available = fieldnames or []
    missing = [name for name in REQUIRED_CUAD_COLUMNS if name not in available]
    if not missing:
        return

    print("Required CUAD columns were not found exactly:", file=sys.stderr)
    for name in missing:
        similar = difflib.get_close_matches(name, available, n=5, cutoff=0.35)
        print(f"  {name!r}", file=sys.stderr)
        print(f"    Similar real column names: {similar or '(none)'}", file=sys.stderr)
    raise SystemExit(1)


def _load_cuad_rows() -> dict[str, dict[str, str]]:
    with MASTER_CLAUSES_CSV.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        _require_cuad_columns(reader.fieldnames)
        return {
            normalize_filename(row["Filename"]): row
            for row in reader
        }


def _build_rows() -> list[dict[str, str | bool]]:
    # Reuse the retrieval evaluator's parsed CUAD answers and its contract set.
    cuad_answers = load_cuad_answers(MASTER_CLAUSES_CSV)
    cuad_rows = _load_cuad_rows()
    filename_to_docid = build_filename_to_docid_map(PARSED_DIR)
    rows: list[dict[str, str | bool]] = []

    for contract in filename_to_docid:
        normalized_contract = normalize_filename(contract)
        if normalized_contract not in cuad_answers:
            continue
        raw_cuad_row = cuad_rows[normalized_contract]

        for clause_type in CLAUSE_TYPES:
            is_abstention_case = (
                clause_type == "Renewal Term"
                and (
                    not cuad_answers[normalized_contract][clause_type]
                    or (normalized_contract, clause_type) in KNOWN_BAD_LABELS
                )
            )
            row: dict[str, str | bool] = {
                field: "" for field in FIELDNAMES
            }
            row.update(
                contract=contract,
                clause_type=clause_type,
                is_abstention_case=is_abstention_case,
                cuad_span="\n\n".join(cuad_answers[normalized_contract][clause_type]),
            )

            if clause_type == "Renewal Term":
                row.update(
                    cuad_renewal_answer=raw_cuad_row["Renewal Term-Answer"],
                    cuad_notice_span=raw_cuad_row[
                        "Notice Period To Terminate Renewal"
                    ],
                    cuad_notice_answer=raw_cuad_row[
                        "Notice Period To Terminate Renewal- Answer"
                    ],
                    cuad_expiration_answer=raw_cuad_row["Expiration Date-Answer"],
                )
            else:
                row.update(
                    cuad_cap_answer=raw_cuad_row["Cap On Liability-Answer"],
                    cuad_uncapped_span=raw_cuad_row["Uncapped Liability"],
                    cuad_uncapped_answer=raw_cuad_row["Uncapped Liability-Answer"],
                )

            if is_abstention_case:
                row["found"] = False
            rows.append(row)

    return sorted(rows, key=lambda row: (str(row["clause_type"]), str(row["contract"])))


def _write_guide() -> None:
    definitions = "\n\n".join(
        f"## {clause_type}\n\n{FIELD_DEFINITIONS[clause_type]}"
        for clause_type in CLAUSE_TYPES
    )
    GUIDE_OUT.write_text(
        "# Extraction labeling guide\n\n"
        "These labels are the answer key for the extraction eval.\n\n"
        f"{definitions}\n\n"
        "## Labeling rules\n\n"
        "- Label from the contract text, not from what the model might say.\n"
        "- If CUAD's answer looks wrong, label what the contract says and explain "
        "in `label_notes`.\n"
        "- Leave a field empty if the contract does not state it.\n"
        "- Booleans: True / False. Numbers: digits only (12, not \"twelve\").\n",
        encoding="utf-8",
    )


def main() -> None:
    if LABELS_OUT.exists():
        print(
            f"Refusing to overwrite existing hand-label file: {LABELS_OUT}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    rows = _build_rows()
    if len(rows) != 60:
        raise RuntimeError(f"Expected 60 extraction-label rows, found {len(rows)}")

    LABELS_OUT.parent.mkdir(exist_ok=True)
    with LABELS_OUT.open("x", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    _write_guide()
    print(f"Wrote {len(rows)} rows to {LABELS_OUT}")
    print(f"Wrote labeling guide to {GUIDE_OUT}")


if __name__ == "__main__":
    main()
