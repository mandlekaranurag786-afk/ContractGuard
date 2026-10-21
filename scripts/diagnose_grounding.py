"""Diagnose saved grounding failures without changing extraction or results.

Usage: uv run python scripts/diagnose_grounding.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contractguard.chunking.fixed import _concatenate_pages
from contractguard.extraction.checks import normalize
from evals.eval_retrieval import (  # noqa: E402
    PARSED_DIR,
    build_filename_to_docid_map,
    chunk_document,
    load_parsed_document,
    normalize_filename,
)

RESULTS = Path("evals/extraction_results.csv")
OUTPUT = Path("evals/grounding_diagnosis.txt")


def without_whitespace(text: str) -> str:
    return re.sub(r"\s+", "", text)


def normalized_offsets(raw: str) -> list[int]:
    """Map normalized characters back to raw offsets for repr diagnostics.

    Use the existing normalizer for character translations/deletions, and
    retain the first whitespace offset for each collapsed whitespace run.
    """
    characters = []
    offsets = []
    pending_space = None
    for index, character in enumerate(raw):
        if character.isspace():
            if characters and pending_space is None:
                pending_space = index
            continue
        translated = normalize(character)
        if not translated:
            continue
        if pending_space is not None:
            characters.append(" ")
            offsets.append(pending_space)
            pending_space = None
        characters.extend(translated)
        offsets.extend([index] * len(translated))
    assert "".join(characters) == normalize(raw)
    return offsets


def prefix_mismatch(quote: str, full_text: str, fallback_offset: int) -> list[str]:
    normalized_quote, normalized_full = normalize(quote), normalize(full_text)
    ends = [match.end() for match in re.finditer(r"\S+", normalized_quote)]
    longest = 0
    position = -1
    for end in ends:
        found_at = normalized_full.find(normalized_quote[:end])
        if found_at < 0:
            break
        longest, position = end, found_at
    if longest:
        quote_offset = normalized_offsets(quote)[longest - 1] + 1
        document_offset = normalized_offsets(full_text)[position + longest - 1] + 1
        note = "first occurrence of longest prefix"
    else:
        quote_offset, document_offset = 0, fallback_offset
        note = "no prefix found; document context starts at cited chunk (or 0)"
    return [
        f"T6 prefix: {normalized_quote[:longest]!r}",
        f"T6 document raw offset: {document_offset} ({note})",
        f"T6 QUOTE next 80: {quote[quote_offset : quote_offset + 80]!r}",
        f"T6 DOCUMENT next 80: {full_text[document_offset : document_offset + 80]!r}",
    ]


def diagnose(
    quote: str, cited_id: str, chunks, full_text: str
) -> tuple[str, str, list[str]]:
    """Run T1-T5 in order; classify by first match, or explain T6 mismatch."""
    cited = next((chunk for chunk in chunks if chunk.chunk_id == cited_id), None)
    q, full = normalize(quote), normalize(full_text)
    cited_text = normalize(cited.text) if cited else ""
    strict = bool(q) and q in cited_text
    lines = [f"Cited chunk: {cited_id}", f"Quote: {quote!r}", f"T1 strict: {strict}"]
    if strict:
        raise ValueError(f"Saved failure now passes strict grounding: {cited_id}")
    other_ids = [
        chunk.chunk_id
        for chunk in chunks
        if chunk.chunk_id != cited_id and q and q in normalize(chunk.text)
    ]
    lines.append(f"T2 other chunk: {other_ids or False}")
    full_match = bool(q) and q in full
    lines.append(f"T3 full text: {full_match}")
    compact_quote, compact_full = without_whitespace(q), without_whitespace(full)
    compact_cited_match = bool(compact_quote) and compact_quote in without_whitespace(
        cited_text
    )
    compact_full_match = bool(compact_quote) and compact_quote in compact_full
    lines.append(
        f"T4 no whitespace: cited={compact_cited_match}, full={compact_full_match}"
    )
    parts = [normalize(part) for part in re.split(r"\.{3,}|\u2026", quote)]
    parts = [part for part in parts if part]
    has_ellipsis = bool(re.search(r"\.{3,}|\u2026", quote))
    part_matches = (
        [(part in full, without_whitespace(part) in compact_full) for part in parts]
        if has_ellipsis
        else []
    )
    joined = bool(part_matches) and all(a or b for a, b in part_matches)
    lines.append(
        f"T5 ellipsis: {joined}; parts (normalized, no-whitespace)={part_matches}"
        if has_ellipsis
        else "T5 ellipsis: N/A (no ellipsis)"
    )
    if other_ids:
        category, details = "other_chunk", "T2: " + ", ".join(other_ids)
    elif full_match:
        category, details = "crosses_chunk_boundary", "T3: full text only"
    elif compact_cited_match or compact_full_match:
        category = "whitespace"
        details = "T4: " + ("cited chunk" if compact_cited_match else "full text only")
    elif joined:
        category, details = "joined_passages", f"T5: all {len(parts)} parts found"
    else:
        category, details = "mismatch", "T6: see prefix and repr context below"
        lines.extend(
            prefix_mismatch(quote, full_text, cited.char_start if cited else 0)
        )
    if not cited:
        details += "; cited chunk ID missing"
    return category, details, lines


def main() -> None:
    with RESULTS.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("predicted_found", row.get("found", "")).lower() == "true"
            and json.loads(row["checks"]).get("evidence_grounded") is False
        ]
    docids = {
        normalize_filename(filename): docid
        for filename, docid in build_filename_to_docid_map(PARSED_DIR).items()
    }
    table = ["contract | clause_type | class | details", "--- | --- | --- | ---"]
    reports = []
    counts = Counter()
    for number, row in enumerate(rows, 1):
        docid = docids[normalize_filename(row["contract"])]
        document = load_parsed_document(PARSED_DIR / f"{docid}.json")
        chunks = chunk_document(document)
        full_text, _ = _concatenate_pages(document)
        assert all(full_text[c.char_start : c.char_end] == c.text for c in chunks)
        category, details, lines = diagnose(
            row["evidence_quote"], row["evidence_chunk_id"], chunks, full_text
        )
        counts[category] += 1
        table.append(
            f"{row['contract']} | {row['clause_type']} | {category} | {details}"
        )
        reports.append(
            "\n".join(
                [
                    f"[{number}] {row['contract']} | {row['clause_type']} | {category}",
                    *lines,
                ]
            )
        )
    output = "\n".join(
        [
            f"Grounding diagnosis: {len(rows)} saved found=True grounding failures",
            "Full text: chunk_document's _concatenate_pages "
            "(clean_text joined by newline).",
            "All chunk raw offset slices verified. T1-T5 use existing normalize().",
            "T6 contexts are raw characters mapped from the normalized prefix match.",
            "",
            *table,
            "",
            f"Class counts: {dict(counts)}",
            "",
            "\n\n".join(reports),
            "",
        ]
    )
    OUTPUT.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
