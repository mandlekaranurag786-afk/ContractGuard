from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_grounding import (  # noqa: E402
    diagnose,
    normalized_offsets,
    prefix_mismatch,
)


@pytest.mark.parametrize(
    "quote,cited,other,full,expected",
    [
        ("alpha beta", "alpha", "alpha beta", "alpha beta", "other_chunk"),
        ("alpha beta", "alpha", "beta", "alpha beta", "crosses_chunk_boundary"),
        ("alpha beta", "alphabeta", "", "alphabeta", "whitespace"),
        ("alpha ... beta", "alpha", "beta", "alpha omitted beta", "joined_passages"),
        ("alpha … beta", "alpha", "beta", "alpha omitted beta", "joined_passages"),
        ("alpha changed", "alpha original", "", "alpha original", "mismatch"),
    ],
)
def test_first_passing_test_determines_class(quote, cited, other, full, expected):
    chunks = [
        SimpleNamespace(chunk_id="doc:0", text=cited, char_start=0),
        SimpleNamespace(chunk_id="doc:1", text=other, char_start=len(cited)),
    ]
    category, _, _ = diagnose(quote, "doc:0", chunks, full)
    assert category == expected


def test_raw_offsets_preserve_hidden_characters_at_mismatch():
    raw = "  alpha\u200b\n\n 4\nbeta"
    assert normalized_offsets(raw)[:5] == list(range(2, 7))
    lines = prefix_mismatch("alpha beta", raw, 0)
    assert lines[-1] == f"T6 DOCUMENT next 80: {raw[7:]!r}"
    assert lines[-2] == "T6 QUOTE next 80: ' beta'"
