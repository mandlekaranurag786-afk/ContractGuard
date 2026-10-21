import hashlib

import pytest

from contractguard.chunking import Chunk
from contractguard.extraction.prompt import build_prompt
from contractguard.extraction.schemas import RenewalTermExtraction
from contractguard.llm.cache import CachedLLMClient


def dummy_chunks():
    return [
        Chunk(
            chunk_id=f"dummy:{i}",
            doc_id="dummy",
            text=text,
            page_start=1,
            page_end=1,
            char_start=i * 100,
            char_end=i * 100 + len(text),
        )
        for i, text in enumerate(
            ["First dummy contract chunk.", "Second dummy contract chunk."]
        )
    ]


@pytest.mark.parametrize(
    "clause,digest",
    [
        (
            "Renewal Term",
            "376e2e6bbb84d0de776fcea019ca88e1d422ed6a1c53fcfcfba6d42f93518c7c",
        ),
        (
            "Cap On Liability",
            "0d4ffff68f1e1018da8b75718dbc9f67318bb1d90cda648c11977fe0067f2c64",
        ),
    ],
)
def test_v1_frozen_sha256(clause, digest):
    prompt = build_prompt(clause, dummy_chunks())
    assert hashlib.sha256(prompt.encode()).hexdigest() == digest
    assert build_prompt(clause, dummy_chunks(), version="v1") == prompt


def test_v2_rules_and_clause_isolation():
    renewal = build_prompt("Renewal Term", dummy_chunks(), version="v2")
    cap = build_prompt("Cap On Liability", dummy_chunks(), version="v2")
    for text in (renewal, cap):
        assert "Never copy a placeholder into any field." in text
        assert "set that number field to null" in text
    assert "notice to exercise a renewal option" in renewal
    assert (
        "If the contract does not renew automatically, notice_days is null." in renewal
    )
    assert "true if ANY of the given chunks excludes" in cap
    assert "punitive damages or lost profits" in cap
    assert "A fixed amount is not a formula" in cap
    assert "Keep cap_formula only if a real calculation" in cap
    assert "cap_formula" not in renewal
    assert "has_damages_exclusion" not in renewal
    assert "notice_days" not in cap
    assert "auto_renews" not in cap


def test_prompt_versions_have_distinct_cache_keys(tmp_path):
    client = CachedLLMClient(None, "test", tmp_path)
    keys = [
        client._cache_key(
            build_prompt("Renewal Term", dummy_chunks(), version=v),
            RenewalTermExtraction,
        )
        for v in ("v1", "v2")
    ]
    assert keys[0] != keys[1]
