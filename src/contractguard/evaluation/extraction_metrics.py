"""Pure scoring for partially labelled clause extractions."""

from decimal import Decimal

from contractguard.extraction.checks import Extraction

FIELD_KINDS = {
    "Renewal Term": {
        "initial_term_months": int,
        "auto_renews": bool,
        "renewal_term_months": int,
        "notice_days": int,
    },
    "Cap On Liability": {
        "has_money_cap": bool,
        "cap_amount": float,
        "currency": str,
        "cap_formula": str,
        "has_damages_exclusion": bool,
    },
}


def parse_label(value: str, kind: type) -> None | bool | int | float | str:
    """Parse a CSV label using bool/int/float/str; blank means unlabelled."""
    value = value.strip()
    if not value:
        return None
    if kind is bool:
        if value not in ("True", "False"):
            raise ValueError(f"Invalid boolean label: {value!r}")
        return value == "True"
    if kind not in (int, float, str):
        raise ValueError(f"Unsupported label kind: {kind!r}")
    return kind(value)


def score_row(clause_type: str, label_row: dict, extraction: Extraction | None) -> dict:
    """Score labelled facts; formula presence is always scored on Cap rows.

    Field results use '<field>_correct' keys. Extra values do not introduce
    scores, except for cap_formula's explicit presence/absence requirement.
    A failed extraction is wrong for every scored field, including absence.
    """
    scores = {
        "found_correct": extraction is not None
        and extraction.found == parse_label(label_row["found"], bool),
        "extra_values": [],
    }
    for field, kind in FIELD_KINDS[clause_type].items():
        label = parse_label(label_row.get(field, ""), kind)
        prediction = getattr(extraction, field, None)
        if label is None and prediction is not None:
            scores["extra_values"].append(field)
        if label is None and field != "cap_formula":
            continue
        if extraction is None:
            correct = False
        elif field == "cap_formula":
            correct = (label is not None) == (prediction is not None)
        elif field == "cap_amount":
            correct = prediction is not None and abs(
                Decimal(str(prediction)) - Decimal(str(label))
            ) <= Decimal("0.01")
        elif field == "currency":
            correct = (
                prediction is not None and prediction.casefold() == label.casefold()
            )
        else:
            correct = prediction == label
        scores[f"{field}_correct"] = correct
    return scores
