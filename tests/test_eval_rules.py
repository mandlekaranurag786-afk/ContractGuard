import pytest

from contractguard.rules.models import Decision, Rule, TrustPolicy, load_rules
from evals.eval_rules import ROOT, RUNS, outcome_flags, select_policy, true_decision


def notice_rule():
    return Rule(
        id="notice",
        description="notice",
        clause_type="Renewal Term",
        field="notice_days",
        operator=">=",
        threshold=30,
        when={"field": "auto_renews", "equals": True},
    )


@pytest.mark.parametrize(
    "updates,expected",
    [
        ({"notice_days": ""}, None),
        ({"auto_renews": ""}, None),
        ({"notice_days": "NONE"}, Decision.NEEDS_REVIEW),
        ({"auto_renews": "NONE"}, Decision.NEEDS_REVIEW),
        ({"auto_renews": "False"}, Decision.NOT_APPLICABLE),
        ({"found": "False"}, None),
        ({"notice_days": "20"}, Decision.FAIL),
        ({}, Decision.PASS),
        ({"notice_days": "NONE", "auto_renews": "False"}, Decision.NOT_APPLICABLE),
        ({"notice_days": "", "auto_renews": "False"}, Decision.NOT_APPLICABLE),
        ({"notice_days": "", "auto_renews": "NONE"}, Decision.NEEDS_REVIEW),
        ({"notice_days": "NONE", "auto_renews": ""}, None),
    ],
)
def test_truth(updates, expected):
    labels = {"found": "True", "notice_days": "60", "auto_renews": "True", **updates}
    assert true_decision(notice_rule(), labels) == expected


@pytest.mark.parametrize(
    "currency,expected",
    [
        ("", None),
        ("NONE", Decision.NEEDS_REVIEW),
        ("EUR", Decision.NEEDS_REVIEW),
        ("usd", Decision.PASS),
    ],
)
def test_truth_currency(currency, expected):
    rule = load_rules(ROOT / "policies/eval_policy.json")[0]
    assert (
        true_decision(
            rule, {"found": "True", "cap_amount": "2000000", "currency": currency}
        )
        == expected
    )


def test_null_rule_field_precedes_missing_currency():
    rule = load_rules(ROOT / "policies/eval_policy.json")[0]
    assert (
        true_decision(
            rule,
            {
                "found": "True",
                "cap_amount": "NONE",
                "currency": "",
            },
        )
        == Decision.NEEDS_REVIEW
    )


@pytest.mark.parametrize(
    "truth,engine,unsafe",
    [
        (Decision.PASS, Decision.NOT_APPLICABLE, True),
        (Decision.NEEDS_REVIEW, Decision.PASS, True),
        (Decision.NEEDS_REVIEW, Decision.FAIL, True),
        (Decision.NEEDS_REVIEW, Decision.NOT_APPLICABLE, True),
        (None, Decision.FAIL, False),
        (Decision.PASS, Decision.NEEDS_REVIEW, False),
    ],
)
def test_unsafe(truth, engine, unsafe):
    flags = outcome_flags(truth, engine)
    assert flags["unsafe"] is unsafe
    assert flags["unnecessary_review"] == (
        truth == Decision.PASS and engine == Decision.NEEDS_REVIEW
    )


@pytest.mark.parametrize(
    "unsafe,savings,expected",
    [
        (1, 3, TrustPolicy.STRICT),
        (0, 2, TrustPolicy.STRICT),
        (0, 3, TrustPolicy.KEYWORD),
    ],
)
def test_selection(unsafe, savings, expected):
    metrics = {
        run: {
            TrustPolicy.STRICT: {"unsafe": 0, "review": 20},
            TrustPolicy.KEYWORD: {"unsafe": 0, "review": 17},
            TrustPolicy.BALANCED: {"unsafe": 1, "review": 0},
        }
        for run in RUNS
    }
    metrics["v2_repeat"][TrustPolicy.KEYWORD] = {
        "unsafe": unsafe,
        "review": 20 - savings,
    }
    assert select_policy(metrics)[0] == expected


def test_selection_compares_balanced_to_current_choice():
    metrics = {
        run: {
            TrustPolicy.STRICT: {"unsafe": 0, "review": 20},
            TrustPolicy.KEYWORD: {"unsafe": 0, "review": 17},
            TrustPolicy.BALANCED: {"unsafe": 0, "review": 15},
        }
        for run in RUNS
    }
    assert select_policy(metrics)[0] == TrustPolicy.KEYWORD
    for run in RUNS:
        metrics[run][TrustPolicy.BALANCED]["review"] = 14
    assert select_policy(metrics)[0] == TrustPolicy.BALANCED
