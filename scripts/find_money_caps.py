"""Find likely money-cap language in the selected parsed contracts.

Usage:
    uv run python -m scripts.find_money_caps
"""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

from evals.eval_retrieval import (
    PARSED_DIR,
    build_filename_to_docid_map,
    load_parsed_document,
    normalize_filename,
)

CONTRACT_PREFIXES = (
    "2ThemartComInc",
    "AlliedEsportsEntertainmentInc",
    "Apollo Endosurgery",
    "EuromediaHoldingsCorp",
    "MIDDLEBROOKPHARMACEUTICALS",
    "PenntexMidstreamPartnersLp",
    "PfHospitalityGroupInc",
    "RangeResourcesLouisianaInc",
    "SigaTechnologiesInc",
)
PHRASES = (
    "shall not exceed",
    "will not exceed",
    "not to exceed",
    "aggregate liability",
    "total liability",
    "maximum liability",
    "liability shall be limited",
    "liability is limited",
    "limited to an amount",
)
OUTPUT_PATH = Path("evals/money_cap_search.txt")
CONTEXT_CHARS = 300
SEARCH_PATTERN = re.compile(
    "|".join(re.escape(phrase) for phrase in PHRASES), re.IGNORECASE
)


def _selected_contracts() -> list[tuple[str, str]]:
    """Resolve the requested filename prefixes using the retrieval eval's loader."""
    filename_to_docid = build_filename_to_docid_map(PARSED_DIR)
    selected: list[tuple[str, str]] = []

    for prefix in CONTRACT_PREFIXES:
        normalized_prefix = normalize_filename(prefix)
        matches = [
            (filename, doc_id)
            for filename, doc_id in filename_to_docid.items()
            if normalize_filename(filename).startswith(normalized_prefix)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one contract for prefix {prefix!r}; found {len(matches)}"
            )
        selected.extend(matches)
    return selected


def _full_text(doc_id: str) -> str:
    document = load_parsed_document(PARSED_DIR / f"{doc_id}.json")
    return "\n\n".join(page.clean_text for page in document.pages)


def _write_search_output(destination: StringIO) -> None:
    for filename, doc_id in _selected_contracts():
        text = _full_text(doc_id)
        matches = list(SEARCH_PATTERN.finditer(text))

        print(filename, file=destination)
        if not matches:
            print("NO MATCHES", file=destination)
        else:
            for match in matches:
                start = max(0, match.start() - CONTEXT_CHARS)
                end = min(len(text), match.end() + CONTEXT_CHARS)
                print(f'\nMATCH: "{match.group()}"', file=destination)
                print(text[start:end], file=destination)
        print("\n" + "=" * 80 + "\n", file=destination)


def main() -> None:
    output = StringIO()
    _write_search_output(output)
    result = output.getvalue()
    OUTPUT_PATH.write_text(result, encoding="utf-8")
    print(result, end="")


if __name__ == "__main__":
    main()
