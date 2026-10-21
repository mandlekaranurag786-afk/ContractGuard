"""Compare extraction CSVs, validating their label provenance before scoring."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_rows(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keyed = {(r["contract"], r["clause_type"]): r for r in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"Duplicate instances in {path}")
    return keyed


def compare(old_path: Path, new_path: Path, *, retrieval_lookup=None) -> str:
    summaries = [
        json.loads(
            p.with_name(p.name.replace("extraction_results", "extraction_summary", 1))
            .with_suffix(".json")
            .read_text()
        )
        for p in (old_path, new_path)
    ]
    declared = [Path(s["labels"]).resolve() for s in summaries if s.get("labels")]
    old, new = load_rows(old_path), load_rows(new_path)
    if not declared:
        # Legacy runs have no labels path: require a unique exact label-cell match.
        candidates = []
        for candidate in old_path.parent.glob("extraction_labels*.csv"):
            source = load_rows(candidate)
            if all(
                key in source
                and any(k.startswith("label_") for k in row)
                and all(
                    value == source[key].get(k.removeprefix("label_"), "")
                    for k, value in row.items()
                    if k.startswith("label_")
                )
                for rows in (old, new)
                for key, row in rows.items()
            ):
                candidates.append(candidate.resolve())
        if len(candidates) != 1:
            raise ValueError("Legacy runs require one uniquely matching labels file")
        declared = candidates
    if len(set(declared)) != 1:
        raise ValueError("Summaries declare different labels files")
    labels = load_rows(declared[0])
    if old.keys() != new.keys():
        raise ValueError(
            "Runs contain different instances; refusing partial comparison"
        )
    # Legacy summaries lack provenance: verify every embedded label cell instead.
    for rows in (old, new):
        for key, row in rows.items():
            if key not in labels:
                raise ValueError(f"Instance missing from labels file: {key}")
            label_keys = [k for k in row if k.startswith("label_")]
            if not label_keys:
                raise ValueError(f"No embedded labels for {key}")
            for field in label_keys:
                if row[field] != labels[key].get(field.removeprefix("label_"), ""):
                    raise ValueError(f"Label mismatch: {key}, {field}")
    totals = {}
    regressions, improvements = [], []
    row_regressions, row_improvements, retrieval_lines = [], [], []
    prediction_changes = retrieval_differences = retrieval_unavailable = 0
    levels = {None: 0, "not_found": 0, "composite": 1, "in_context": 2, "strict": 3}
    level_counts = [{k: 0 for k in levels}, {k: 0 for k in levels}]
    for key in sorted(old):
        a, b = old[key], new[key]
        prediction_changes += sum(
            a[k] != b[k] for k in set(a) & set(b) if k.startswith("predicted_")
        )
        status = [
            all(
                v == "True"
                for k, v in row.items()
                if k.endswith("_correct") and v != ""
            )
            for row in (a, b)
        ]
        if status[0] != status[1]:
            (row_improvements if status[1] else row_regressions).append(
                f"{key[0]} | {key[1]} | {status[0]} -> {status[1]}"
            )
        ids = []
        origins = []
        for row in (a, b):
            if row.get("retrieved_chunk_ids"):
                ids.append(json.loads(row["retrieved_chunk_ids"]))
                origins.append("recorded")
            elif retrieval_lookup:
                ids.append(retrieval_lookup(*key))
                origins.append("recomputed (historical ids unavailable)")
            else:
                ids.append(None)
                origins.append("not available")
        if None in ids:
            retrieval_unavailable += 1
            identity = "not available"
        else:
            identity = "IDENTICAL" if ids[0] == ids[1] else "DIFFERENT"
            retrieval_differences += ids[0] != ids[1]
        retrieval_lines.append(
            f"{key[0]} | {key[1]} | {identity} | old={origins[0]}; new={origins[1]}"
        )
        for column in sorted(set(a) & set(b)):
            if not column.endswith("_correct") or a[column] == "" or b[column] == "":
                continue
            if a[column] not in ("True", "False") or b[column] not in ("True", "False"):
                raise ValueError(f"Invalid correctness value: {key}, {column}")
            field = column.removesuffix("_correct")
            counts = totals.setdefault((key[1], field), [0, 0, 0])
            before, after = a[column] == "True", b[column] == "True"
            counts[0] += before
            counts[1] += after
            counts[2] += 1
            if before != after:
                entry = (
                    f"{key[0]} | {key[1]} | {field} | label={a.get('label_' + field)!r}"
                    f" | old={a.get('predicted_' + field)!r}"
                    f" | new={b.get('predicted_' + field)!r}"
                )
                (improvements if after else regressions).append(entry)
        evidence = [json.loads(r["checks"]).get("evidence_level") for r in (a, b)]
        for i, level in enumerate(evidence):
            if level not in levels:
                raise ValueError(f"Unknown evidence level: {level}")
            level_counts[i][level] += 1
        if evidence[0] != evidence[1]:
            entry = (
                f"{key[0]} | {key[1]} | evidence_level | label=grounded evidence"
                f" | old={evidence[0]!r} | new={evidence[1]!r}"
            )
            (
                improvements
                if levels[evidence[1]] > levels[evidence[0]]
                else regressions
            ).append(entry)
    lines = [
        f"Old: {old_path}",
        f"New: {new_path}",
        f"Labels: {declared[0]}",
        "Label cells verified against the declared file in both runs.",
        f"Compared instances: {len(old)}",
        "Evidence ordering: strict > in_context > composite > not_found/None",
        "",
        "clause_type | field | old correct/scored | new correct/scored",
    ]
    for (clause, field), (a, b, n) in sorted(totals.items()):
        lines.append(f"{clause} | {field} | {a}/{n} | {b}/{n}")
    lines += ["", "evidence_level | old count | new count"]
    for level in levels:
        lines.append(f"{level} | {level_counts[0][level]} | {level_counts[1][level]}")
    lines += [
        "",
        f"REGRESSIONS: {len(regressions)}",
        *regressions,
        "",
        f"IMPROVEMENTS: {len(improvements)}",
        *improvements,
        "",
        "ROW-LEVEL",
        f"Fully-correct regressions: {len(row_regressions)}",
        *row_regressions,
        f"Fully-correct improvements: {len(row_improvements)}",
        *row_improvements,
        "Field-level prediction changes (all predicted_* columns): "
        f"{prediction_changes}",
        "",
        "RETRIEVAL IDENTITY (rank order matters)",
        *retrieval_lines,
        f"Rows with different retrieval: {retrieval_differences}",
        f"Rows with retrieval not available: {retrieval_unavailable}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--recompute-retrieval", action="store_true")
    args = parser.parse_args()
    lookup = None
    if args.recompute_retrieval:
        import os
        import sys

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from evals.local_retrieval import LocalRetrieval

        lookup = LocalRetrieval().chunk_ids
    output = compare(args.old, args.new, retrieval_lookup=lookup)
    args.out.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
