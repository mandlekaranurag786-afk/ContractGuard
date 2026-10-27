"""Conservative sentence-local signals, not legal interpretation."""

import re
from types import SimpleNamespace

from contractguard.extraction.checks import normalize
from contractguard.extraction.confidence import (
    BOOLEAN_KEYWORDS,
    KEYWORDS,
    NEGATION,
    field_support,
)
from contractguard.rules.models import Scalar

EXCLUSION = (
    r"\b(?:not be liable|not liable|in no event|neither party shall be liable|"
    r"no liability|waives|excludes|disclaims|under no circumstances?)\b"
)
DAMAGES = (
    r"\b(?:consequential|indirect|special|punitive|incidental|"
    r"lost profits|loss of profit)\b"
)
REVERSAL = (
    r"\b(?:shall not apply|nothing in this agreement shall limit|"
    r"nothing in this agreement shall exclude)\b"
)
LIMIT = (
    r"\b(?:shall not exceed|not to exceed|(?<!not )(?<!not be )limited to|"
    r"maximum (?:liability|aggregate liability|total liability|amount)|"
    r"in no event exceed)\b"
)
RENEWAL = (
    r"\b(?:automatically renew|automatically extend|renew automatically|"
    r"automatic renewal)\b"
)


def followup_status(field: str, value: Scalar, quote: str) -> str:
    if field in KEYWORDS:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "followup_untrusted"
        extraction = SimpleNamespace(**{field: value})
        return (
            "supported"
            if field_support(extraction, field, quote) == "supported"
            else "followup_untrusted"
        )
    if value is not True:
        return "followup_untrusted"
    text = normalize(quote).casefold()
    # Preserve decimal points and ellipses; semicolons separate assertions.
    for sentence in re.split(r";|(?<!\.)\.(?![\d.])", text):
        if field == "has_damages_exclusion":
            reversed_liability = any(
                not re.search(NEGATION, sentence[max(0, m.start() - 40) : m.start()])
                for m in re.finditer(r"\bshall be liable for\b", sentence)
            )
            if (
                re.search(EXCLUSION, sentence)
                and re.search(DAMAGES, sentence)
                and not re.search(REVERSAL, sentence)
                and not reversed_liability
            ):
                return "followup_pattern"
        elif field == "has_money_cap":
            if (
                re.search(r"\b(?:liability|liable)\b", sentence)
                and re.search(LIMIT, sentence)
                and not re.search(
                    r"\b(?:shall not apply|unlimited|no limit)\b", sentence
                )
            ):
                return "followup_pattern"
        elif field == "auto_renews":
            for match in re.finditer(RENEWAL, sentence):
                # Use the existing 40-character negation rule on this occurrence,
                # not an unrelated automatic/tacit keyword elsewhere in the quote.
                keyword = re.search(BOOLEAN_KEYWORDS[field], match.group())
                start = match.start() + (keyword.start() if keyword else 0)
                if not re.search(NEGATION, sentence[max(0, start - 40) : start]):
                    return "followup_pattern"
    return "followup_untrusted"
