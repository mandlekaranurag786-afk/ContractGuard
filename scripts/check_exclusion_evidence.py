"""Inspect exclusion wording in retrieved and full contract text, offline."""

import csv
import os
import re
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.local_retrieval import LocalRetrieval  # noqa: E402

PATTERN = re.compile(
    r"\b(?:consequential|indirect|incidental|special|punitive|lost\s+profits|loss\s+of\s+profit)\b",
    re.IGNORECASE,
)


def read_cap_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            r["contract"]: r
            for r in csv.DictReader(handle)
            if r["clause_type"] == "Cap On Liability"
        }


def main():
    retrieval = LocalRetrieval()
    labels = read_cap_rows(Path("evals/extraction_labels_v2.csv"))
    old = read_cap_rows(Path("evals/extraction_results.csv"))
    new = read_cap_rows(Path("evals/extraction_results_v2.csv"))
    lines = [
        "Exclusion evidence check (hybrid top 5, Cap On Liability query)",
        "Contexts show 100 raw characters before/after each match (repr).",
        "Full-text hits include all candidates, not just exclusion sentences.",
    ]
    for prefix in ("PACIRA PHARMACEUTICALS", "PareteumCorp", "XpresspaGroupInc"):
        matches = [name for name in labels if name.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one contract for {prefix}")
        filename = matches[0]
        chunks, text = retrieval.document(filename)
        ids = retrieval.chunk_ids(filename, "Cap On Liability")
        lines += [
            "",
            filename,
            "has_damages_exclusion: "
            f"label={labels[filename]['has_damages_exclusion']}; "
            f"v1={old[filename]['predicted_has_damages_exclusion']}; "
            f"v2={new[filename]['predicted_has_damages_exclusion']}",
            "Retrieved chunk_ids (rank order):",
            *ids,
            "Retrieved matches:",
        ]
        by_id = {c.chunk_id: c for c in chunks}
        for chunk_id in ids:
            chunk = by_id[chunk_id]
            found = list(PATTERN.finditer(chunk.text))
            if not found:
                lines.append(f"{chunk_id}: no matches")
            for match in found:
                context = chunk.text[max(0, match.start() - 100) : match.end() + 100]
                lines.append(
                    f"{chunk_id} offset={chunk.char_start + match.start()} "
                    f"match={match.group()!r}: {context!r}"
                )
        lines.append(
            "Main exclusion sentence candidates (consequential + liability wording):"
        )
        for sentence in re.finditer(r".+?(?:[.!?](?=\s|$)|$)", text, re.DOTALL):
            value = sentence.group()
            if not (
                re.search(r"\bconsequential\b", value, re.IGNORECASE)
                and re.search(r"\bliab(?:le|ility)\b", value, re.IGNORECASE)
            ):
                continue
            start = sentence.start() + len(value) - len(value.lstrip())
            end = sentence.end()
            containing = [
                c.chunk_id
                for c in chunks
                if c.char_start <= start and c.char_end >= end
            ]
            overlapping = [
                c.chunk_id for c in chunks if c.char_start < end and c.char_end > start
            ]
            lines.append(
                f"sentence_range=[{start}, {end}) contains={containing} "
                f"overlaps={overlapping} retrieved_overlap="
                f"{[i for i in overlapping if i in ids]}: {value.strip()!r}"
            )
        lines.append("Full-document matches (raw offsets in chunk_document source):")
        for match in PATTERN.finditer(text):
            containing = [
                c.chunk_id
                for c in chunks
                if c.char_start <= match.start() and c.char_end >= match.end()
            ]
            context = text[max(0, match.start() - 100) : match.end() + 100]
            lines.append(
                f"offset={match.start()} match={match.group()!r} "
                f"chunks={containing} retrieved={any(i in ids for i in containing)}: "
                f"{context!r}"
            )
    output = "\n".join(lines) + "\n"
    Path("evals/exclusion_evidence_check.txt").write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
