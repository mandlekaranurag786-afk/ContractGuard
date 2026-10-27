"""Pure follow-up safety tests, without search or generation."""

import pytest
from pydantic import ValidationError

from contractguard.chunking import Chunk
from contractguard.followup.merge import merge
from contractguard.followup.models import FieldEvidence, MergeOutcome
from contractguard.followup.patterns import followup_status


@pytest.mark.parametrize(
    "field,value,quote,expected",
    [
        (
            "has_damages_exclusion",
            True,
            "UNDER NO CIRCUMSTANCE ... SHALL EITHER PARTY "
            "BE LIABLE TO THE OTHER PARTY FOR ANY LOST PROFITS, LOSS OF OPPORTUNITY OR "
            "OTHER SPECIAL, PUNITIVE, INDIRECT, OR CONSEQUENTIAL DAMAGES",
            "followup_pattern",
        ),
        (
            "has_damages_exclusion",
            True,
            "Supplier shall be liable for consequential "
            "damages arising from its breach.",
            "followup_untrusted",
        ),
        (
            "has_damages_exclusion",
            True,
            "The limitations above shall not apply to consequential damages.",
            "followup_untrusted",
        ),
        (
            "has_damages_exclusion",
            True,
            "Supplier disclaims warranties. Consequential damages are discussed below.",
            "followup_untrusted",
        ),
        (
            "has_damages_exclusion",
            True,
            "Supplier disclaims warranties; consequential damages are discussed below.",
            "followup_untrusted",
        ),
        (
            "has_damages_exclusion",
            True,
            "Neither party shall be liable for consequential damages.",
            "followup_pattern",
        ),
        (
            "has_damages_exclusion",
            True,
            "Supplier disclaims indirect damages but shall "
            "be liable for consequential damages.",
            "followup_untrusted",
        ),
        (
            "has_damages_exclusion",
            True,
            "Supplier excludes indirect damages but nothing "
            "in this agreement shall limit liability.",
            "followup_untrusted",
        ),
        (
            "has_money_cap",
            True,
            "THE MAXIMUM LIABILITY OF EACH PARTY ... SHALL NOT "
            "EXCEED $2,000,000.00 (EXCLUDING ...).",
            "followup_pattern",
        ),
        (
            "has_money_cap",
            True,
            "Liability for death shall be unlimited.",
            "followup_untrusted",
        ),
        (
            "has_money_cap",
            True,
            "Liability is limited to $2,000,000.00 but this limit shall not apply.",
            "followup_untrusted",
        ),
        (
            "auto_renews",
            True,
            "This Agreement shall automatically renew for successive "
            "one (1) year terms",
            "followup_pattern",
        ),
        ("auto_renews", True, "shall not automatically renew", "followup_untrusted"),
        (
            "auto_renews",
            True,
            "automatic billing but shall not automatically renew",
            "followup_untrusted",
        ),
        ("auto_renews", True, "will renew automatically", "followup_pattern"),
        ("notice_days", 60, "sixty (60) days prior written notice", "supported"),
        ("renewal_term_months", 12, "successive one (1) year terms", "supported"),
        (
            "renewal_term_months",
            60,
            "successive one (1) year terms",
            "followup_untrusted",
        ),
        ("initial_term_months", 36, "initial term of three years", "supported"),
        ("cap_amount", 2000000, "SHALL NOT EXCEED $2,000,000.00", "supported"),
        ("notice_days", None, "60 days prior notice", "followup_untrusted"),
        ("notice_days", True, "1 day prior notice", "followup_untrusted"),
    ],
)
def test_patterns(field, value, quote, expected):
    assert followup_status(field, value, quote) == expected


@pytest.mark.parametrize(
    "field", ["has_money_cap", "has_damages_exclusion", "auto_renews"]
)
@pytest.mark.parametrize("value", [False, None])
def test_negative_booleans_never_trusted(field, value):
    assert (
        followup_status(
            field,
            value,
            "maximum liability; automatically renew; "
            "neither party shall be liable for indirect damages",
        )
        == "followup_untrusted"
    )


def evidence(**updates):
    return FieldEvidence(
        **{
            "clause_type": "Renewal Term",
            "field": "notice_days",
            "value": 60,
            "evidence_quote": "60 days prior notice",
            "evidence_chunk_id": "doc:0",
            "verified": True,
            "status": "supported",
            **updates,
        }
    )


@pytest.mark.parametrize(
    "old,status,updates,outcome",
    [
        (30, "unsupported", {"verified": False}, "DISCARDED_UNVERIFIED"),
        (30, "unsupported", {"value": None}, "NO_VALUE"),
        (30, "unsupported", {"value": False}, "NO_VALUE"),
        (30, "unsupported", {"value": None, "verified": False}, "NO_VALUE"),
        (30, "unsupported", {"value": False, "verified": False}, "NO_VALUE"),
        (30, "unsupported", {"status": "followup_untrusted"}, "DISCARDED_UNTRUSTED"),
        (None, "absence_claim", {}, "ACCEPTED_NEW"),
        (
            False,
            "absence_claim",
            {"field": "auto_renews", "value": True, "status": "followup_pattern"},
            "ACCEPTED_NEW",
        ),
        (
            False,
            "explicit_negative",
            {"field": "auto_renews", "value": True, "status": "followup_pattern"},
            "CONFLICT",
        ),
        (60, "unsupported", {}, "ACCEPTED_CONFIRMATION"),
        (30, "unsupported", {}, "CONFLICT"),
        (60, "conflicting", {}, "CONFLICT"),
        (None, "conflicting", {}, "CONFLICT"),
        (0, "unsupported", {"value": 0}, "ACCEPTED_CONFIRMATION"),
    ],
)
def test_merge_order_and_preservation(old, status, updates, outcome):
    result = merge(old, status, evidence(**updates))
    assert result.outcome == outcome
    if result.outcome in {
        MergeOutcome.NO_VALUE,
        MergeOutcome.CONFLICT,
        MergeOutcome.DISCARDED_UNTRUSTED,
        MergeOutcome.DISCARDED_UNVERIFIED,
    }:
        assert result.final_value == old
        assert result.final_status == status
    else:
        assert result.final_value == result.new_value
        assert result.final_status == result.new_status


@pytest.mark.parametrize("status", ["supported", "keyword_signal"])
@pytest.mark.parametrize("updates", [{}, {"verified": False}, {"value": None}])
def test_trusted_old_raises(status, updates):
    with pytest.raises(ValueError, match="trusted field"):
        merge(60, status, evidence(**updates))


@pytest.mark.parametrize(
    "field,quote,expected",
    [
        (
            "has_money_cap",
            "liability for any claims, including but not limited to negligence",
            "followup_untrusted",
        ),
        (
            "has_money_cap",
            "Liability is not limited to fees paid.",
            "followup_untrusted",
        ),
        (
            "has_money_cap",
            "To the maximum extent permitted by law, neither party "
            "shall have any liability for consequential damages",
            "followup_untrusted",
        ),
        (
            "has_money_cap",
            "Each party's maximum aggregate liability shall be limited "
            "to the fees paid",
            "followup_pattern",
        ),
        (
            "has_damages_exclusion",
            "Neither party shall be liable for any indirect or consequential damages.",
            "followup_pattern",
        ),
        (
            "has_damages_exclusion",
            "In no event shall either party be liable for lost profits.",
            "followup_pattern",
        ),
        (
            "has_damages_exclusion",
            "Supplier shall be liable for consequential damages.",
            "followup_untrusted",
        ),
        (
            "auto_renews",
            "This Agreement shall renew automatically for one year.",
            "followup_pattern",
        ),
        ("auto_renews", "shall not renew automatically", "followup_untrusted"),
    ],
)
def test_refined_patterns(field, quote, expected):
    assert followup_status(field, True, quote) == expected


@pytest.mark.parametrize(
    "phrase",
    [
        "maximum liability",
        "maximum aggregate liability",
        "maximum total liability",
        "maximum amount",
    ],
)
def test_specific_maximum_phrases(phrase):
    assert (
        followup_status("has_money_cap", True, f"The {phrase} for liability is $200.")
        == "followup_pattern"
    )


def test_missing_renewal_keyword_does_not_crash(monkeypatch):
    from contractguard.followup import patterns

    monkeypatch.setitem(patterns.BOOLEAN_KEYWORDS, "auto_renews", r"\babsent\b")
    assert (
        followup_status("auto_renews", True, "renew automatically")
        == "followup_pattern"
    )
    assert followup_status("auto_renews", True, "shall not renew automatically") == (
        "followup_untrusted"
    )


@pytest.mark.parametrize(
    "currency,expected",
    [
        ("EUR", "CONFLICT"),
        ("usd", "ACCEPTED_CONFIRMATION"),
        (None, "ACCEPTED_CONFIRMATION"),
    ],
)
def test_currency_before_confirmation(currency, expected):
    new = evidence(
        clause_type="Cap On Liability",
        field="cap_amount",
        currency=currency,
        value=500000,
    )
    result = merge(500000, "unsupported", new, old_currency="USD")
    assert result.outcome == expected
    if expected == "CONFLICT":
        assert result.final_status == "unsupported"
        assert result.final_value == 500000


def test_currency_only_for_amount():
    with pytest.raises(ValidationError, match="only valid for cap_amount"):
        evidence(currency="USD")


def test_verify_new_chunks_only_and_roundtrip():
    def chunk(index, start, text):
        return Chunk(
            chunk_id=f"doc:{index}",
            doc_id="doc",
            text=text,
            char_start=start,
            char_end=start + len(text),
            page_start=1,
            page_end=1,
        )

    new = evidence()
    assert not new.verify([]).verified
    assert new.verify([chunk(0, 0, new.evidence_quote)]).verified
    chunks = [chunk(0, 0, "60 days "), chunk(1, 8, "prior notice")]
    assert new.verify(chunks).verified
    assert not evidence(evidence_chunk_id="old:0").verify(chunks).verified
    assert not new.verify(
        [chunk(0, 0, "60 days "), chunk(1, 20, "prior notice")]
    ).verified
    assert (
        not evidence(evidence_quote="60 days ... prior notice").verify(chunks).verified
    )
    assert FieldEvidence.model_validate_json(new.model_dump_json()) == new
