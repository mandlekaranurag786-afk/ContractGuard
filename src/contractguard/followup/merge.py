"""Fail-closed merging: discarded or conflicting evidence never replaces a value."""

from contractguard.followup.models import FieldEvidence, MergeOutcome, MergeResult
from contractguard.rules.models import Scalar

TRUSTED_FOLLOWUP = frozenset({"supported", "followup_pattern"})


def merge(
    old_value: Scalar,
    old_status: str,
    new: FieldEvidence,
    old_currency: str | None = None,
) -> MergeResult:
    if old_status in {"supported", "keyword_signal"}:
        raise ValueError("A trusted field must never receive a follow-up")

    def finish(outcome: MergeOutcome, reason: str) -> MergeResult:
        accepted = outcome in {
            MergeOutcome.ACCEPTED_NEW,
            MergeOutcome.ACCEPTED_CONFIRMATION,
        }
        return MergeResult(
            outcome=outcome,
            field=new.field,
            old_value=old_value,
            old_status=old_status,
            new_value=new.value,
            new_status=new.status,
            final_value=new.value if accepted else old_value,
            final_status=new.status if accepted else old_status,
            reason=reason,
            evidence_quote=new.evidence_quote,
            evidence_chunk_id=new.evidence_chunk_id,
        )

    if new.value is None or new.value is False:
        return finish(
            MergeOutcome.NO_VALUE, "The follow-up supplies no positive value."
        )
    if not new.verified:
        return finish(
            MergeOutcome.DISCARDED_UNVERIFIED, "The new quote was not verified."
        )
    if new.status not in TRUSTED_FOLLOWUP:
        return finish(MergeOutcome.DISCARDED_UNTRUSTED, "The follow-up is not trusted.")
    if old_status == "conflicting":
        return finish(
            MergeOutcome.CONFLICT, "Existing conflicting evidence needs review."
        )
    if (
        new.field == "cap_amount"
        and old_currency is not None
        and new.currency is not None
        and old_currency.casefold() != new.currency.casefold()
    ):
        return finish(MergeOutcome.CONFLICT, "The currencies disagree.")
    if old_value is None:
        return finish(
            MergeOutcome.ACCEPTED_NEW, "Verified evidence supplies a new value."
        )
    if old_value is False and old_status == "absence_claim":
        return finish(
            MergeOutcome.ACCEPTED_NEW, "Positive evidence replaces an absence claim."
        )
    if old_value is False and old_status == "explicit_negative":
        return finish(
            MergeOutcome.CONFLICT, "Positive evidence contradicts an explicit negative."
        )
    # Python equates True and 1; do not treat those as the same extracted fact.
    if old_value == new.value and isinstance(old_value, bool) == isinstance(
        new.value, bool
    ):
        return finish(
            MergeOutcome.ACCEPTED_CONFIRMATION, "Verified evidence confirms the value."
        )
    return finish(MergeOutcome.CONFLICT, "The old and new values disagree.")
