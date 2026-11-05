"""Build a blind CUAD-assisted sheet without importing any evaluation pipeline."""

import ast
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "evals/heldout_selection.csv"
DESTINATION = ROOT / "evals/heldout_labels.csv"
BASELINE = ROOT / "evals/extraction_labels_v2.csv"
MASTER = ROOT / "data/contracts/master_clauses.csv"
PDFS = ROOT / "data/heldout"
CLAUSES = ("Renewal Term", "Cap On Liability")

# Copied from build_extraction_labels.py:124-137 to avoid importing its evaluator.
CUAD_COLUMNS = {
    "Renewal Term": {
        "cuad_renewal_answer": "Renewal Term-Answer",
        "cuad_notice_span": "Notice Period To Terminate Renewal",
        "cuad_notice_answer": "Notice Period To Terminate Renewal- Answer",
        "cuad_expiration_answer": "Expiration Date-Answer",
    },
    "Cap On Liability": {
        "cuad_cap_answer": "Cap On Liability-Answer",
        "cuad_uncapped_span": "Uncapped Liability",
        "cuad_uncapped_answer": "Uncapped Liability-Answer",
    },
}
# eval_retrieval.py:59-63; its key includes the normalized PDF extension.
KNOWN_BAD_LABELS = {
    ("operaltd04302020ex414serviceagreementpdf", "Renewal Term"),
}


def name_key(name: str) -> str:
    # CUAD Filename can omit the extension; preserve dots within such names.
    stem = name[:-4] if name.lower().endswith(".pdf") else name
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def parse_spans(cell: str) -> list[str]:
    """Copy of eval_retrieval.py:200-210; no evaluator imports for blind labels."""
    cell = cell.strip()
    if not cell:
        return []
    try:
        parsed = ast.literal_eval(cell)
        if isinstance(parsed, list):
            return [str(s) for s in parsed]
        return [str(parsed)]
    except (ValueError, SyntaxError):
        return [cell] if cell != "[]" else []


def main() -> None:
    if DESTINATION.exists():
        raise FileExistsError(f"Refusing to overwrite hand labels: {DESTINATION}")
    with SELECTION.open(newline="", encoding="utf-8") as handle:
        selected = [row for row in csv.DictReader(handle) if row["status"] == "ok"]
    assert len(selected) == 12, f"Expected 12 selected contracts, got {len(selected)}"
    assert sum(row["set"] == "main" for row in selected) == 10
    assert sum(row["set"] == "no_answer" for row in selected) == 2
    keys = [name_key(row["contract"]) for row in selected]
    assert len(set(keys)) == 12, "Duplicate selected contracts"

    with BASELINE.open(newline="", encoding="utf-8") as handle:
        fields = next(csv.reader(handle))
    with MASTER.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"Filename", *CLAUSES}
        required.update(column for columns in CUAD_COLUMNS.values()
                        for column in columns.values())
        assert required <= set(reader.fieldnames or []), "Missing CUAD columns"
        master_rows = list(reader)
    master = {name_key(row["Filename"]): row for row in master_rows}
    assert len(master) == len(master_rows), "Ambiguous normalized CUAD filenames"
    assert set(keys) <= master.keys(), "Selected contract missing from CUAD master"

    pdf_names = {}
    for path in PDFS.iterdir():
        if path.is_file() and path.suffix.lower() == ".pdf":
            key = name_key(path.name)
            assert key not in pdf_names, "Ambiguous held-out PDF filenames"
            pdf_names[key] = path.name
    assert set(keys) <= pdf_names.keys(), "Selected PDF missing from data/heldout"

    rows = []
    for selected_row, key in zip(selected, keys, strict=True):
        contract = pdf_names[key]
        assert contract == selected_row["contract"], "Selection must use PDF basename"
        raw = master[key]
        for clause in CLAUSES:
            # build_extraction_labels.py:104-142: retain its exact defaults.
            spans = parse_spans(raw[clause])
            normalized_contract = re.sub(r"[^a-z0-9]", "", contract.lower())
            abstention = clause == "Renewal Term" and (
                not spans or (normalized_contract, clause) in KNOWN_BAD_LABELS
            )
            row = dict.fromkeys(fields, "")
            row.update(contract=contract, clause_type=clause,
                       is_abstention_case=abstention, cuad_span="\n\n".join(spans))
            row.update({field: raw[column]
                        for field, column in CUAD_COLUMNS[clause].items()})
            if abstention:
                row["found"] = False
            assert set(row) == set(fields), "Unexpected v2 header"
            rows.append(row)
    assert len(rows) == 24
    with DESTINATION.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with DESTINATION.open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == fields
    print(f"Wrote {len(rows)} rows to {DESTINATION.relative_to(ROOT)}")
    print("Header equals v2 header: True")
    for row in rows[:2]:
        print({field: str(value)[:80] if field.startswith("cuad_") else value
               for field, value in row.items()})


if __name__ == "__main__":
    main()
