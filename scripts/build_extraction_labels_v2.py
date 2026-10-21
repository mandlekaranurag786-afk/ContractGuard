"""Create reviewed labels v2 without modifying baseline labels."""

import csv
from pathlib import Path

SOURCE = Path("evals/extraction_labels.csv")
DESTINATION = Path("evals/extraction_labels_v2.csv")


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    changes = []

    def update(prefix, clause, values, reason, overwrite=False):
        matches = [
            (i, r)
            for i, r in enumerate(rows, 1)
            if r["contract"].startswith(prefix) and r["clause_type"] == clause
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one target row for {prefix}/{clause}, got {len(matches)}"
            )
        number, row = matches[0]
        changed = False
        for field, value in values.items():
            old = row[field]
            if old == value or (old and not overwrite):
                continue
            changes.append((number, row["contract"], field, old, value))
            row[field] = value
            changed = True
        if changed:
            old = row["label_notes"]
            row["label_notes"] = (old + " " if old else "") + reason
            changes.append(
                (number, row["contract"], "label_notes", old, row["label_notes"])
            )

    renewal = "Renewal Term"
    update(
        "AlliedEsports",
        renewal,
        {
            "is_abstention_case": "False",
            "found": "True",
            "initial_term_months": "60",
            "renewal_term_months": "NONE",
            "notice_days": "NONE",
        },
        "V2: A 5-year term is stated; no renewal provision; "
        "under our definition found=True.",
        True,
    )
    update(
        "OPERALTD",
        renewal,
        {
            "is_abstention_case": "False",
            "found": "True",
            "auto_renews": "False",
            "notice_days": "NONE",
        },
        'V2: "may be extended or renewed by the Parties in a writing executed by ..." '
        'exists; CUAD span is still wrong; the Part 1 note '
        '"no renewal clause exists" was incorrect.',
        True,
    )
    for prefix in (
        "2ThemartComInc",
        "BELLICUMPHARMACEUTICALS",
        "MRSFIELDSORIGINALCOOKIESINC",
        "UpjohnInc",
        "EuromediaHoldingsCorp",
        "PfHospitalityGroupInc",
    ):
        update(
            prefix,
            renewal,
            {"notice_days": "NONE"},
            "V2: Option / notice-to-renew, not notice to stop renewal; expected empty.",
        )
    update(
        "BravatekSolutionsInc",
        renewal,
        {"notice_days": "NONE"},
        "V2: Termination for convenience is excluded by definition; expected empty.",
    )
    for prefix in (
        "KitovPharmaLtd",
        "Magenta Therapeutics",
        "ParatekPharmaceuticalsInc",
        "PareteumCorp",
    ):
        values = {"notice_days": "NONE"}
        if prefix != "Magenta Therapeutics":
            values["renewal_term_months"] = "NONE"
        update(
            prefix,
            renewal,
            values,
            "V2: Duration(s) redacted, not stated; expected empty.",
        )
    for prefix in (
        "Magenta Therapeutics",
        "PACIRA PHARMACEUTICALS",
        "ParatekPharmaceuticalsInc",
        "PareteumCorp",
    ):
        update(
            prefix,
            "Cap On Liability",
            {"cap_amount": "NONE"},
            "V2: Cap amount redacted; expected empty.",
        )
    # All target validations finish before any output is written.
    with DESTINATION.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for number, contract, field, old, new in changes:
        print(f"row {number} | {contract} | {field}: {old!r} -> {new!r}")


if __name__ == "__main__":
    main()
