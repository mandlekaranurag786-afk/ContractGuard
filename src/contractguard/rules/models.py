"""Validated rule definitions, trust policies and decision records."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt, model_validator

from contractguard.extraction.extractor import SCHEMAS

Scalar = bool | int | float | str | None


class Decision(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    equals: Scalar


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str
    description: str
    clause_type: str
    field: str
    operator: Literal[">=", "<=", "==", "is_true", "is_false"]
    threshold: StrictInt | StrictFloat | None = None
    currency: str | None = None
    when: Condition | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> Rule:
        schema = SCHEMAS.get(self.clause_type)
        if schema is None:
            raise ValueError(f"Unknown clause type: {self.clause_type}")
        for field in [self.field] + ([self.when.field] if self.when else []):
            if field not in schema.model_fields:
                raise ValueError(f"Unknown field {field!r} for {self.clause_type}")
        annotation = schema.model_fields[self.field].annotation
        types = set(get_args(annotation) or (annotation,)) - {type(None)}
        if self.operator in (">=", "<="):
            if self.threshold is None:
                raise ValueError(f"{self.operator} requires a numeric threshold")
        if self.operator in (">=", "<=", "==") and not types <= {int, float}:
            raise ValueError(f"{self.operator} requires a numeric field")
        if self.operator in ("is_true", "is_false") and types != {bool}:
            raise ValueError(f"{self.operator} requires a boolean field")
        if self.currency is not None and "currency" not in schema.model_fields:
            raise ValueError(f"{self.clause_type} has no currency field")
        return self


class TrustPolicy(StrEnum):
    STRICT = "STRICT"
    KEYWORD = "KEYWORD"
    BALANCED = "BALANCED"

    @property
    def statuses(self) -> frozenset[str]:
        statuses = {"supported"}
        if self != TrustPolicy.STRICT:
            statuses.add("keyword_signal")
        if self == TrustPolicy.BALANCED:
            statuses.add("explicit_negative")
        return frozenset(statuses)


class RuleResult(BaseModel):
    rule_id: str
    clause_type: str
    policy: TrustPolicy
    decision: Decision
    reason_code: str
    reason: str
    field: str
    value: Scalar
    field_status: str
    condition_field: str | None
    condition_value: Scalar
    condition_status: str | None
    currency_status: str | None
    evidence_strength: Literal["proof", "weak_signal", "explicit_negative"] | None
    confidence_level: str
    evidence_quote: str | None
    evidence_chunk_id: str | None


def load_rules(path: str | Path) -> list[Rule]:
    """Load a JSON list, rejecting invalid definitions and duplicate ids."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Expected a JSON list of rules")
        if not data:
            raise ValueError("Rule list must not be empty")
        rules = [Rule.model_validate(item) for item in data]
        ids = [rule.id for rule in rules]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate rule ids")
        return rules
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot load rules from {path}: {exc}") from exc
