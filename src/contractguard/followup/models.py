"""Evidence and merge records; no retrieval or generation is performed here."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contractguard.chunking import Chunk
from contractguard.extraction.checks import evidence_grounded, evidence_in_context
from contractguard.rules.models import Decision, Scalar


class FieldEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    clause_type: str
    field: str
    value: Scalar
    currency: str | None = None
    currency_status: str | None = None
    evidence_quote: str | None
    evidence_chunk_id: str | None
    verified: bool
    status: str

    @model_validator(mode="after")
    def currency_only_for_amount(self) -> Self:
        if self.currency is not None and self.field != "cap_amount":
            raise ValueError("Currency is only valid for cap_amount")
        return self

    def verify(self, chunks: list[Chunk]) -> Self:
        """Return a copy verified against exactly the chunks sent for extraction.

        Composite evidence is insufficient. Citation membership is required even
        when the quote crosses adjacent chunks. Never trust an incoming flag.
        """
        verified = any(c.chunk_id == self.evidence_chunk_id for c in chunks) and (
            evidence_grounded(self, chunks) or evidence_in_context(self, chunks)
        )
        return self.model_copy(update={"verified": verified})


class MergeOutcome(StrEnum):
    ACCEPTED_NEW = "ACCEPTED_NEW"
    ACCEPTED_CONFIRMATION = "ACCEPTED_CONFIRMATION"
    NO_VALUE = "NO_VALUE"
    DISCARDED_UNVERIFIED = "DISCARDED_UNVERIFIED"
    DISCARDED_UNTRUSTED = "DISCARDED_UNTRUSTED"
    CONFLICT = "CONFLICT"


class MergeResult(BaseModel):
    outcome: MergeOutcome
    field: str
    old_value: Scalar
    old_status: str
    new_value: Scalar
    new_status: str
    final_value: Scalar
    final_status: str
    reason: str
    evidence_quote: str | None = None
    evidence_chunk_id: str | None = None


class FollowupStep(BaseModel):
    step: int
    clause_type: str
    field: str
    rule_ids: list[str]
    query: str | None = None
    new_chunk_ids: list[str] = Field(default_factory=list)
    chunk_policy: Literal["unseen_only", "original_plus_unseen"] = "unseen_only"
    chunk_ids: list[str] = Field(default_factory=list)
    value: Scalar = None
    quote_excerpt: str = ""
    verified: bool = False
    status: str | None = None
    merge_outcome: MergeOutcome | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0
    cached: bool = False
    note: str = ""
    error: str | None = None


class ChangedRule(BaseModel):
    rule_id: str
    before_decision: Decision
    before_reason_code: str
    after_decision: Decision
    after_reason_code: str
    evidence_strength: str | None


class FollowupReport(BaseModel):
    enabled: bool = False
    version: Literal["v1", "v2"] = "v1"
    status: Literal["disabled", "completed", "followup_skipped"] = "disabled"
    calls_used: int = 0
    max_calls: int = 4
    steps: list[FollowupStep] = Field(default_factory=list)
    changed_rules: list[ChangedRule] = Field(default_factory=list)
    live_calls: int | None = None
    cached_calls: int | None = None
    uncached_input_tokens: int | None = None
    uncached_output_tokens: int | None = None
