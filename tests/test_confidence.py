import pytest

from contractguard.extraction.confidence import (
    compute_confidence,
    field_status,
    field_support,
    number_forms,
)
from contractguard.extraction.schemas import (
    LiabilityCapExtraction,
    RenewalTermExtraction,
)
from scripts.eval_confidence import evaluate, rebuild_extraction


def extraction(
    field="notice_days", value=60, quote="sixty (60) days prior written notice"
):
    schema = LiabilityCapExtraction if field == "cap_amount" else RenewalTermExtraction
    data = dict.fromkeys(schema.model_fields)
    data.update(found=True, evidence_quote=quote, evidence_chunk_id="doc:0")
    if schema is LiabilityCapExtraction:
        data.update(has_money_cap=True, has_damages_exclusion=False)
    data[field] = value
    return schema.model_validate(data)


@pytest.mark.parametrize(
    "field,value,quote,status",
    [
        ("notice_days", 60, "sixty (60) days prior written notice", "supported"),
        (
            "renewal_term_months",
            12,
            "successive one (1) year terms unless sixty (60) days notice",
            "supported",
        ),
        (
            "renewal_term_months",
            60,
            "successive one (1) year terms unless sixty (60) days notice",
            "unsupported",
        ),
        (
            "notice_days",
            60,
            "successive one (1) year terms unless sixty (60) days notice",
            "supported",
        ),
        (
            "notice_days",
            365,
            "one (1) year written notice prior to the expiration",
            "supported",
        ),
        ("notice_days", 90, "three months prior written notice", "supported"),
        ("cap_amount", 2000000, "SHALL NOT EXCEED $2,000,000.00", "supported"),
        (
            "cap_amount",
            500000,
            "LIABILITY: FIVE HUNDRED THOUSAND DOLLARS ($5OO,000)",
            "supported",
        ),
        ("notice_days", None, "notice", "not_applicable"),
        ("notice_days", 60, "160 days notice", "unsupported"),
        ("notice_days", 60, "60 days", "unsupported"),
        ("notice_days", 60, "60 " + "x" * 21 + " days notice", "unsupported"),
        ("notice_days", 60, "60 days " + "x" * 121 + " notice", "unsupported"),
        ("cap_amount", 2, "liability $2,000", "unsupported"),
        ("initial_term_months", 24, "initial term of two years", "supported"),
    ],
)
def test_support(field, value, quote, status):
    assert field_support(extraction(field, value, quote), field, quote) == status


def test_number_forms():
    assert ("three hundred sixty-five", "days") in number_forms(365, "notice_days")
    assert ("one", "weeks") in number_forms(7, "notice_days")
    assert ("two million", "dollars") in number_forms(2000000, "cap_amount")
    assert ("2,000,000.00", "dollars") in number_forms(2000000, "cap_amount")
    assert ("zero", "months") in number_forms(0, "initial_term_months")


def test_money_words_still_require_keyword():
    quote = "FIVE HUNDRED THOUSAND DOLLARS ($5OO,000)"
    assert field_support(extraction("cap_amount", 500000), "cap_amount", quote) == (
        "unsupported"
    )


@pytest.mark.parametrize(
    "value,quote",
    [
        (
            36,
            "and shall continue for three years thereafter (the “Initial Term”). "
            "Unless either party gives written notice of termination of this Agreement "
            "at least 60 days prior to the end of the Initial Term, or any successive "
            "three-year term, this Agreement shall automatically renew for successive "
            "additional three-year terms",
        ),
        (
            12,
            "This Agreement shall commence as of the Effective Date and shall "
            "continue in full force and effect for an initial term of\nthree (3) years "
            "from the Promotion Commencement Date, divided into three "
            "one-year periods. "
            "Unless terminated in\naccordance with the provisions of Section 18, this "
            "Agreement shall automatically renew for each subsequent one-year\nterm.",
        ),
    ],
)
def test_exact_hyphenated_quotes(value, quote):
    ex = extraction("renewal_term_months", value, quote)
    assert field_support(ex, "renewal_term_months", quote) == "supported"


@pytest.mark.parametrize(
    "field,value,quote,status",
    [
        ("auto_renews", True, "shall not automatically renew", "conflicting"),
        ("auto_renews", False, "shall not automatically renew", "explicit_negative"),
        (
            "auto_renews",
            False,
            "may be renewed by mutual written agreement",
            "explicit_negative",
        ),
        ("auto_renews", True, "automatically renews", "keyword_signal"),
        ("auto_renews", False, "automatically renews", "conflicting"),
        ("auto_renews", True, "agreement", "unsupported"),
        ("auto_renews", False, "agreement", "absence_claim"),
        ("has_damages_exclusion", False, "consequential damages", "conflicting"),
        ("has_damages_exclusion", False, "agreement", "absence_claim"),
        ("has_damages_exclusion", True, "no consequential damages", "keyword_signal"),
        ("has_money_cap", True, "maximum liability", "keyword_signal"),
        ("has_money_cap", False, "maximum liability", "conflicting"),
        ("has_money_cap", False, "no maximum liability", "conflicting"),
        ("has_money_cap", True, "SHALL NOT EXCEED $2,000,000.00", "keyword_signal"),
        ("has_money_cap", False, "SHALL NOT EXCEED $2,000,000.00", "conflicting"),
        (
            "has_damages_exclusion",
            True,
            "Neither party shall be liable for special, incidental "
            "or consequential damages",
            "keyword_signal",
        ),
        (
            "has_damages_exclusion",
            False,
            "Neither party shall be liable for special, incidental "
            "or consequential damages",
            "conflicting",
        ),
        ("has_money_cap", True, "agreement", "unsupported"),
        ("has_money_cap", False, "agreement", "absence_claim"),
        ("has_damages_exclusion", True, "agreement", "unsupported"),
        ("notice_days", 60, "60 days prior notice", "supported"),
        ("notice_days", 60, "agreement", "unsupported"),
        ("notice_days", None, "agreement", "absence_claim"),
    ],
)
def test_field_status(field, value, quote, status):
    ex = extraction("cap_amount" if field.startswith("has_") else "notice_days")
    ex = ex.model_copy(update={field: value})
    assert field_status(ex, field, quote) == status


def test_notice_explicit_negative_and_status_metadata():
    ex = extraction(value=None, quote="shall not automatically renew").model_copy(
        update={"auto_renews": False}
    )
    result = compute_confidence(ex, {"evidence_level": "strict"}, "Renewal Term")
    assert result.level == "HIGH"
    assert result.field_status["notice_days"] == "explicit_negative"
    assert set(result.field_status) == set(type(ex).model_fields)


@pytest.mark.parametrize(
    "evidence,expected",
    [
        ("strict", "HIGH"),
        ("in_context", "HIGH"),
        ("composite", "MEDIUM"),
        ("not_found", "LOW"),
    ],
)
def test_levels(evidence, expected):
    checks = dict(
        evidence_level=evidence, required_fields_present=True, not_found_is_clean=True
    )
    result = compute_confidence(extraction(), checks, "Renewal Term")
    assert result.level == expected
    if evidence == "composite":
        assert "composite_evidence" in result.reasons


def test_all_reasons_and_precedence():
    result = compute_confidence(
        extraction(value=99),
        {
            "evidence_level": "composite",
            "required_fields_present": False,
            "not_found_is_clean": False,
        },
        "Renewal Term",
    )
    assert result.level == "LOW"
    assert len(result.reasons) == 4
    assert "notice_days_not_supported_by_quote" in result.reasons
    assert compute_confidence(None, {}, "Renewal Term").reasons == ["extraction_failed"]
    absent = extraction(value=None).model_copy(update={"found": False})
    result = compute_confidence(absent, {}, "Renewal Term")
    assert result.level == "NOT_FOUND"
    assert "absence_not_trusted" in result.reasons


def test_report(tmp_path):
    import csv
    import json

    row = {
        "contract": "test",
        "clause_type": "Renewal Term",
        "label_found": "True",
        "label_notice_days": "90",
        "predicted_found": "True",
        "predicted_notice_days": "60",
        "evidence_quote": "60 days prior notice",
        "evidence_chunk_id": "doc:0",
        "checks": json.dumps({"evidence_level": "strict"}),
    }
    path = tmp_path / "results.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    report = evaluate(path)
    assert "HIGH-but-wrong: 1/1 (100.0%)" in report
    assert "notice_days | supported | 0 | 1" in report
    assert "label='90', prediction=60" in report
    assert rebuild_extraction({**row, "error": "failed"}) is None


@pytest.mark.parametrize(
    "currency,quote,status",
    [
        ("USD", "...SHALL NOT EXCEED $2,000,000.00 ...", "supported"),
        ("USD", "No currency is stated", "unsupported"),
        (None, "...SHALL NOT EXCEED $2,000,000.00 ...", "absence_claim"),
        ("usd", "US Dollars", "supported"),
        ("USD", "USD 200", "supported"),
        ("USD", "US$200", "supported"),
        ("USD", "two hundred dollars", "supported"),
        ("EUR", "EUR 200", "supported"),
        ("EUR", "$200", "unsupported"),
    ],
)
def test_currency_status(currency, quote, status):
    ex = extraction("cap_amount", 2000000).model_copy(update={"currency": currency})
    assert field_status(ex, "currency", quote) == status
