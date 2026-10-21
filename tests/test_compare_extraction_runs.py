import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.compare_extraction_runs import compare  # noqa: E402


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_compare_changes_and_label_provenance(tmp_path):
    labels = tmp_path / "labels.csv"
    write_csv(
        labels,
        [
            {"contract": name, "clause_type": "Renewal Term", "found": "True"}
            for name in ("a", "b")
        ],
    )
    paths = [tmp_path / f"extraction_results_{tag}.csv" for tag in ("old", "new")]
    for index, path in enumerate(paths):
        write_csv(
            path,
            [
                {
                    "contract": name,
                    "clause_type": "Renewal Term",
                    "label_found": "True",
                    "predicted_found": str(bool(i == index)),
                    "found_correct": str(bool(i == index)),
                    "checks": json.dumps({"evidence_level": "strict"}),
                    "retrieved_chunk_ids": json.dumps([f"doc:{index}"]),
                }
                for i, name in enumerate(("a", "b"))
            ],
        )
        path.with_name(path.name.replace("results", "summary")).with_suffix(
            ".json"
        ).write_text(json.dumps({"labels": str(labels)}))
    output = compare(*paths)
    assert "REGRESSIONS: 1" in output
    assert "IMPROVEMENTS: 1" in output
    assert "Rows with different retrieval: 2" in output
    assert "Field-level prediction changes (all predicted_* columns): 2" in output
    assert "Fully-correct regressions: 1" in output
    assert "Fully-correct improvements: 1" in output
    assert "Rows with different retrieval: 0" in compare(paths[0], paths[0])
    assert (
        "a | Renewal Term | found | label='True' | old='True' | new='False'" in output
    )
    assert (
        "b | Renewal Term | found | label='True' | old='False' | new='True'" in output
    )
    summary = tmp_path / "extraction_summary_old.json"
    summary.write_text("{}")
    assert compare(*paths) == output  # Legacy provenance verified from label cells.
    summary.write_text(json.dumps({"labels": str(tmp_path / "different.csv")}))
    with pytest.raises(ValueError, match="different labels"):
        compare(*paths)
