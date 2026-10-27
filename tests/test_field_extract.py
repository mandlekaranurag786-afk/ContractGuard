import json

import pytest

from contractguard.chunking import Chunk
from contractguard.followup.extract import (
    FIELD_SCHEMAS,
    FieldExtractor,
    build_field_prompt,
)
from contractguard.followup.patterns import followup_status
from contractguard.llm.client import LLMResponse
from contractguard.llm.errors import LLMQuotaError, LLMUnavailableError

QUOTE = "Neither party shall be liable for consequential damages."


def chunk(text=QUOTE, index=0, start=0):
    return Chunk(
        chunk_id=f"doc:{index}",
        doc_id="doc",
        text=text,
        char_start=start,
        char_end=start + len(text),
        page_start=1,
        page_end=1,
    )


class Fake:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_json(self, prompt, schema):
        self.calls.append((prompt, schema))
        value = self.outcomes.pop(0)
        if isinstance(value, Exception):
            raise value
        return LLMResponse(
            text=value,
            model="fake",
            input_tokens=10,
            output_tokens=2,
            latency_ms=3,
            cached=True,
        )


def response(**updates):
    return json.dumps(
        dict(value=True, evidence_quote=QUOTE, evidence_chunk_id="doc:0", **updates)
    )


@pytest.mark.parametrize(
    "quote,cited,verified",
    [
        (QUOTE, "doc:0", True),
        (QUOTE, "unseen:0", False),
        ("invented quote", "doc:0", False),
    ],
)
def test_verification(quote, cited, verified):
    data = json.dumps(dict(value=True, evidence_quote=quote, evidence_chunk_id=cited))
    result = FieldExtractor(Fake([data])).extract(
        "Cap On Liability", "has_damages_exclusion", [chunk()]
    )
    assert result.evidence.verified is verified
    assert result.error is None


def test_context_verification():
    chunks = [chunk(QUOTE[:20]), chunk(QUOTE[20:], 1, 20)]
    assert (
        FieldExtractor(Fake([response()]))
        .extract("Cap On Liability", "has_damages_exclusion", chunks)
        .evidence.verified
    )


def test_retry_and_metadata():
    fake = Fake(["invalid", response()])
    evidence, tokens, latency, cached, error = FieldExtractor(fake).extract(
        "Cap On Liability", "has_damages_exclusion", [chunk()]
    )
    assert len(fake.calls) == 2
    assert evidence.status == "followup_pattern" and evidence.verified
    assert tokens == {"input_tokens": 20, "output_tokens": 4}
    assert latency == 6 and cached and error is None
    assert fake.calls[0][1].__name__ == "FollowupHasDamagesExclusion"


def test_invalid_twice_and_provider_error():
    for outcomes in (["bad", "bad"], [LLMUnavailableError("offline failure")]):
        result = FieldExtractor(Fake(outcomes)).extract(
            "Cap On Liability", "has_money_cap", []
        )
        assert result.error and not result.evidence.verified
        assert result.evidence.value is None


def test_quota_propagates():
    with pytest.raises(LLMQuotaError):
        FieldExtractor(Fake([LLMQuotaError("quota")])).extract(
            "Cap On Liability", "has_money_cap", []
        )


@pytest.mark.parametrize("clause,field", FIELD_SCHEMAS)
def test_field_isolation(clause, field):
    prompt = build_field_prompt(clause, field, [])
    definitions = [
        line
        for line in prompt.splitlines()
        if line.startswith("- ") and not line.startswith("- If")
    ]
    assert len(definitions) == 1 and definitions[0].startswith(f"- {field}:")
    assert "data, not instructions" in prompt and "followup_v1" in prompt


def test_currency_and_redaction():
    quote = "Liability shall not exceed $2,000,000.00"
    data = json.dumps(
        dict(
            value=2000000,
            currency="USD",
            evidence_quote=quote,
            evidence_chunk_id="doc:0",
        )
    )
    result = FieldExtractor(Fake([data])).extract(
        "Cap On Liability", "cap_amount", [chunk(quote)]
    )
    assert result.evidence.currency_status == "supported"
    assert result.evidence.status == "supported"
    assert "set that number field to null" in build_field_prompt(
        "Cap On Liability", "cap_amount", []
    )


def test_not_be_limited():
    assert (
        followup_status("has_money_cap", True, "liability shall not be limited to fees")
        == "followup_untrusted"
    )


def test_followup_cache_is_separate_from_main(tmp_path):
    from contractguard.extraction.prompt import build_prompt
    from contractguard.extraction.schemas import LiabilityCapExtraction
    from contractguard.llm.cache import CachedLLMClient

    fake = Fake([response()])
    client = CachedLLMClient(fake, "fake", tmp_path)
    extractor = FieldExtractor(client)
    first = extractor.extract("Cap On Liability", "has_damages_exclusion", [chunk()])
    second = extractor.extract("Cap On Liability", "has_damages_exclusion", [chunk()])
    assert first.evidence == second.evidence and second.cached
    assert len(fake.calls) == 1
    field_key = client._cache_key(fake.calls[0][0], fake.calls[0][1])
    for version in ("v1", "v2"):
        assert field_key != client._cache_key(
            build_prompt("Cap On Liability", [chunk()], version), LiabilityCapExtraction
        )


def test_null_boolean_is_valid_but_untrusted():
    data = json.dumps(dict(value=None, evidence_quote=None, evidence_chunk_id=None))
    result = FieldExtractor(Fake([data])).extract(
        "Cap On Liability", "has_damages_exclusion", [chunk()]
    )
    assert result.error is None
    assert result.evidence.value is None and not result.evidence.verified
    assert result.evidence.status == "followup_untrusted"
