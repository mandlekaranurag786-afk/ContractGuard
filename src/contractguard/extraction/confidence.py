"""Conservative, quote-based confidence heuristics (not probabilities)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from contractguard.extraction.checks import Extraction, normalize

NUMERIC_FIELDS = {
    "Renewal Term": ("notice_days", "renewal_term_months", "initial_term_months"),
    "Cap On Liability": ("cap_amount",),
}
KEYWORDS = {
    "notice_days": r"\b(?:notice|prior|written)\b",
    "renewal_term_months": (
        r"\b(?:renew\w*|successive|extension\w*|extend\w*|additional)\b"
    ),
    "initial_term_months": r"\b(?:initial|commence\w*|effective date|term of)\b",
    "cap_amount": r"\b(?:exceed\w*|limit\w*|liability|liable)\b",
}
_SMALL = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()


def _words(n: int) -> str:
    if n < 20:
        return _SMALL[n]
    if n < 100:
        return _TENS[n // 10] + ("-" + _words(n % 10) if n % 10 else "")
    for scale, name in ((1_000_000, "million"), (1000, "thousand"), (100, "hundred")):
        if n >= scale:
            return (
                _words(n // scale)
                + " "
                + name
                + (" " + _words(n % scale) if n % scale else "")
            )
    raise ValueError(n)


def number_forms(value: int | float, field: str) -> list[tuple[str, str]]:
    """Return number spellings paired with canonical plural unit words.

    Money uses 'dollars'; matching also accepts USD or a preceding dollar sign.
    English spellings cover nonnegative integers below one billion.
    """
    n = Decimal(str(value))
    if not n.is_finite() or n < 0:
        return []
    quantities = []
    if field == "notice_days":
        quantities.append((n, "days"))
        quantities.extend(
            (n / divisor, unit)
            for divisor, unit in ((30, "months"), (365, "years"), (7, "weeks"))
            if n % divisor == 0
        )
    elif field in ("initial_term_months", "renewal_term_months"):
        quantities.append((n, "months"))
        if n % 12 == 0:
            quantities.append((n / 12, "years"))
    elif field == "cap_amount":
        quantities.append((n, "dollars"))
    else:
        raise ValueError(f"Not a numeric extraction field: {field}")
    forms = []
    for amount, unit in quantities:
        digits = format(amount, "f")
        if "." in digits:
            digits = digits.rstrip("0").rstrip(".")
        variants = [digits]
        if field == "cap_amount":
            variants.append(format(Decimal(digits), ",f"))
            if amount == int(amount):
                variants.extend([digits + ".00", f"{int(amount):,}.00"])
        if amount == int(amount) and amount < 1_000_000_000:
            variants.append(_words(int(amount)))
        forms.extend((variant, unit) for variant in variants)
    return list(dict.fromkeys(forms))


def field_support(extraction: Extraction, field: str, quote: str) -> str:
    value = getattr(extraction, field, None)
    if value is None:
        return "not_applicable"
    text = normalize(quote).casefold()
    keywords = list(re.finditer(KEYWORDS[field], text))
    for number, unit in number_forms(value, field):
        pattern = re.escape(number).replace(r"\-", r"[-\s]")
        # Do not match a substring of a larger number, decimal, or word.
        boundary = r"[\w.,]" if number[0].isdigit() else r"[\w-]"
        end = rf"(?!{boundary})|(?=-{unit[:-1]}s?\b)"
        for match in re.finditer(rf"(?<!{boundary}){pattern}(?:{end})", text):
            after = text[match.end() : match.end() + 20]
            unit_pattern = (
                r"\b(?:dollars?|usd)\b"
                if unit == "dollars"
                else (rf"\b{unit[:-1]}s?\b")
            )
            has_unit = re.search(unit_pattern, after) is not None
            if field == "cap_amount":
                has_unit |= text[max(0, match.start() - 1) : match.start()] == "$"
            near_keyword = any(
                keyword.start() <= match.end() + 120
                and keyword.end() >= match.start() - 120
                for keyword in keywords
            )
            if has_unit and near_keyword:
                return "supported"
    return "unsupported"


BOOLEAN_KEYWORDS = {
    "auto_renews": r"\b(?:automatic|automatically|tacit|tacitly)\b",
    "has_money_cap": r"\b(?:exceed|limited to|maximum|cap)\b",
    "has_damages_exclusion": (
        r"\b(?:consequential|indirect|special|punitive|lost profits|loss of profit)\b"
    ),
}
NEGATION = r"\b(?:not|no|neither|nor|never|without)\b"
EXPLICIT_RENEWAL_NEGATIVE = (
    r"\b(?:not automatically|shall not renew|mutual written agreement|"
    r"mutual agreement|by amendment|option to extend|option to renew|"
    r"may be extended|may be renewed)\b"
)


def field_status(extraction: Extraction, field: str, quote: str) -> str:
    """Describe local support, without changing confidence or interpreting law.

    For fields without specified rules (found, cap_formula and evidence fields),
    null/False is an absence claim; other values are unsupported. This avoids
    inventing support criteria for those fields.
    """
    value = getattr(extraction, field, None)
    if field in KEYWORDS:
        if value is not None:
            return field_support(extraction, field, quote)
        if (
            field == "notice_days"
            and field_status(extraction, "auto_renews", quote) == "explicit_negative"
        ):
            return "explicit_negative"
        return "absence_claim"
    if value is None:
        return "absence_claim"
    if field == "currency":
        text = normalize(quote).casefold()
        code = value.strip().casefold()
        pattern = (
            r"\$|\b(?:usd|us dollars|dollars)\b"
            if code == "usd"
            else rf"(?<!\w){re.escape(code)}(?!\w)"
        )
        return "supported" if code and re.search(pattern, text) else "unsupported"
    if field not in BOOLEAN_KEYWORDS:
        return "absence_claim" if value is False else "unsupported"
    text = normalize(quote).casefold()
    matches = list(re.finditer(BOOLEAN_KEYWORDS[field], text))
    # Legal negation can express a cap or exclusion, not contradict it.
    if field != "auto_renews":
        if value is True:
            return "keyword_signal" if matches else "unsupported"
        return "conflicting" if matches else "absence_claim"
    positive = any(
        not re.search(NEGATION, text[max(0, match.start() - 40) : match.start()])
        for match in matches
    )
    if value is True:
        return (
            "keyword_signal"
            if positive
            else ("conflicting" if matches else "unsupported")
        )
    if field == "auto_renews" and re.search(EXPLICIT_RENEWAL_NEGATIVE, text):
        return "explicit_negative"
    return "conflicting" if positive else "absence_claim"


@dataclass(frozen=True)
class ConfidenceResult:
    level: str
    reasons: list[str]
    field_support: dict[str, str]
    field_status: dict[str, str]


def compute_confidence(
    extraction_or_none: Extraction | None, checks: dict, clause_type: str
) -> ConfidenceResult:
    extraction = extraction_or_none
    if extraction is None:
        return ConfidenceResult("LOW", ["extraction_failed"], {}, {})
    support = {
        field: field_support(extraction, field, extraction.evidence_quote or "")
        for field in NUMERIC_FIELDS[clause_type]
    }
    reasons = []
    if not extraction.found:
        reasons.append("absence_not_trusted")
    evidence = checks.get("evidence_level")
    low = False
    if evidence not in ("strict", "in_context", "composite"):
        reasons.append("evidence_not_found")
        low = True
    for check in ("required_fields_present", "not_found_is_clean"):
        if checks.get(check) is False:
            reasons.append(check + "_failed")
            low = True
    if evidence == "composite":
        reasons.append("composite_evidence")
    reasons.extend(
        field + "_not_supported_by_quote"
        for field, status in support.items()
        if status == "unsupported"
    )
    if not extraction.found:
        level = "NOT_FOUND"
    elif low:
        level = "LOW"
    elif reasons:
        level = "MEDIUM"
    else:
        level = "HIGH"
    statuses = {
        field: field_status(extraction, field, extraction.evidence_quote or "")
        for field in type(extraction).model_fields
    }
    return ConfidenceResult(level, reasons, support, statuses)
