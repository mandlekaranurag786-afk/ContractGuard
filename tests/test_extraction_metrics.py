from __future__ import annotations

import pytest

from contractguard.evaluation.extraction_metrics import parse_label, score_row
from contractguard.extraction.schemas import (
    LiabilityCapExtraction,
    RenewalTermExtraction,
)


def renewal(**values) -> RenewalTermExtraction:
    return RenewalTermExtraction.model_validate(
        {
            **dict.fromkeys(RenewalTermExtraction.model_fields),
            "found": True,
            **values,
        }
    )


def cap(**values) -> LiabilityCapExtraction:
    return LiabilityCapExtraction.model_validate(
        {
            **dict.fromkeys(LiabilityCapExtraction.model_fields),
            "found": True,
            "has_money_cap": False,
            "has_damages_exclusion": False,
            **values,
        }
    )


@pytest.mark.parametrize(
    "value,kind,expected",
    [
        ("", int, None),
        ("  ", str, None),
        ("True", bool, True),
        ("False", bool, False),
        ("12", int, 12),
        ("2.5", float, 2.5),
        ("USD", str, "USD"),
    ],
)
def test_parse_label(value, kind, expected) -> None:
    result = parse_label(value, kind)
    assert result == expected
    assert type(result) is type(expected)


def test_invalid_boolean_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="boolean"):
        parse_label("no", bool)


def test_unlabelled_fields_are_not_scored_and_extras_include_false_and_zero() -> None:
    scores = score_row(
        "Renewal Term",
        {"found": "True", "notice_days": "30"},
        renewal(notice_days=30, auto_renews=False, initial_term_months=0),
    )
    assert scores == {
        "found_correct": True,
        "notice_days_correct": True,
        "extra_values": ["initial_term_months", "auto_renews"],
    }


@pytest.mark.parametrize(
    "label,prediction,correct",
    [
        ("", None, True),
        ("", "fees", False),
        ("fees", None, False),
        ("fees", "different wording", True),
        ("fees", "", True),
    ],
)
def test_cap_formula_scores_presence_even_for_empty_label(label, prediction, correct):
    scores = score_row(
        "Cap On Liability",
        {"found": "True", "cap_formula": label},
        cap(cap_formula=prediction),
    )
    assert scores["cap_formula_correct"] is correct
    assert ("cap_formula" in scores["extra_values"]) == (
        label == "" and prediction is not None
    )


@pytest.mark.parametrize(
    "amount,correct",
    [
        (100.0, True),
        (100.01, True),
        (99.99, True),
        (100.011, False),
        (None, False),
    ],
)
def test_amount_uses_absolute_cent_tolerance(amount, correct):
    scores = score_row(
        "Cap On Liability",
        {"found": "True", "cap_amount": "100"},
        cap(cap_amount=amount),
    )
    assert scores["cap_amount_correct"] is correct


def test_currency_is_case_insensitive():
    scores = score_row(
        "Cap On Liability", {"found": "True", "currency": "USD"}, cap(currency="usd")
    )
    assert scores["currency_correct"] is True


@pytest.mark.parametrize(
    "clause,label,keys",
    [
        (
            "Renewal Term",
            {"found": "False", "auto_renews": "False", "notice_days": "30"},
            {"found_correct", "auto_renews_correct", "notice_days_correct"},
        ),
        (
            "Cap On Liability",
            {"found": "True", "has_money_cap": "False"},
            {"found_correct", "has_money_cap_correct", "cap_formula_correct"},
        ),
    ],
)
def test_missing_extraction_is_wrong_for_every_scored_field(clause, label, keys):
    scores = score_row(clause, label, None)
    assert scores.pop("extra_values") == []
    assert scores == dict.fromkeys(keys, False)


@pytest.mark.parametrize("found", [True, False])
def test_abstention_requires_found_false(found):
    scores = score_row("Renewal Term", {"found": "False"}, renewal(found=found))
    assert scores["found_correct"] is (not found)
