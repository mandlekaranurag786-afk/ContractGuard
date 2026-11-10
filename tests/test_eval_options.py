"""Evaluation input options use tiny local fixtures, never models or live clients."""

import csv
import sys
from pathlib import Path

import pytest
from test_checker_batch import report

from contractguard.rules.models import Decision, Rule, TrustPolicy
from evals import eval_extraction, eval_rules, eval_verdicts, run_checker_batch


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize(
    "module,required,option,attribute,default,value,expected",
    [
        (eval_extraction, ["--tag", "test"], "--parsed-dir", "parsed_dir",
         eval_extraction.PARSED_DIR, ["custom/parsed"], Path("custom/parsed")),
        (run_checker_batch, [], "--labels", "labels",
         eval_extraction.LABELS, ["custom.csv"], Path("custom.csv")),
        (run_checker_batch, [], "--contracts-dir", "contracts_dir",
         Path("data/contracts"), ["custom/pdfs"], Path("custom/pdfs")),
        (eval_rules, [], "--labels", "labels",
         Path("evals/extraction_labels_v2.csv"), ["custom.csv"], Path("custom.csv")),
        (eval_rules, [], "--runs", "runs",
         eval_rules.RUNS, ["custom", "repeat"], ["custom", "repeat"]),
        (eval_verdicts, ["--reports", "reports", "--tag", "test"], "--labels",
         "labels", eval_verdicts.LABELS, ["custom.csv"], Path("custom.csv")),
        (eval_verdicts, ["--reports", "reports", "--tag", "test"], "--extractions",
         "extractions", eval_verdicts.EXTRACTIONS,
         ["results.csv"], Path("results.csv")),
    ],
)
def test_option_defaults_and_overrides(
    module, required, option, attribute, default, value, expected
):
    assert getattr(module.parse_args(required), attribute) == default
    parsed = module.parse_args([*required, option, *value])
    assert getattr(parsed, attribute) == expected


def test_selected_contracts_preserves_label_order(tmp_path):
    labels = write_rows(tmp_path / "labels.csv", [
        {"contract": "z.PDF", "clause_type": "Renewal Term"},
        {"contract": "z.PDF", "clause_type": "Cap On Liability"},
        {"contract": "a.pdf", "clause_type": "Renewal Term"},
    ])
    assert run_checker_batch.selected_contracts(labels, tmp_path) == [
        tmp_path / "z.PDF", tmp_path / "a.pdf",
    ]


def cap_rule():
    return Rule(
        id="cap", description="cap", clause_type="Cap On Liability",
        field="cap_amount", operator=">=", threshold=100,
    )


@pytest.mark.parametrize("decision", list(Decision))
def test_absent_clause_automatic_decisions(decision):
    fake = report(decision)
    fake.rule_results[0].evidence_quote = "q" * 150 + "TRUNCATED"
    key = ("fake.pdf", "Cap On Liability")
    labels = {key: {"found": "False"}}
    text = eval_verdicts.build_report(
        {"fake.pdf": fake}, [cap_rule()], labels, {key: {}},
    )
    section = text.split("Automatic decisions on absent clauses (amendment 3)\n")[1]
    automatic = decision in eval_rules.AUTO
    assert f"Count: {int(automatic)}" in section
    assert (f"fake.pdf | cap | {decision} | compared" in section) is automatic
    assert ("q" * 150 in section) is automatic
    assert "TRUNCATED" not in section


@pytest.mark.parametrize("found", ["True", "", "NONE"])
def test_no_absent_clause_section_without_false_labels(found):
    key = ("fake.pdf", "Cap On Liability")
    text = eval_verdicts.build_report(
        {"fake.pdf": report(Decision.PASS)}, [cap_rule()],
        {key: {"found": found, "cap_amount": "100"}}, {key: {}},
    )
    assert "Automatic decisions on absent clauses" not in text


def test_absent_clause_section_matches_contract_and_clause():
    reports = {
        "absent.pdf": report(Decision.PASS, "absent.pdf"),
        "present.pdf": report(Decision.PASS, "present.pdf"),
    }
    labels = {
        ("absent.pdf", "Cap On Liability"): {"found": "False"},
        ("present.pdf", "Cap On Liability"): {"found": "True", "cap_amount": "100"},
        ("present.pdf", "Renewal Term"): {"found": "False"},
    }
    text = eval_verdicts.build_report(
        reports, [cap_rule()], labels, dict.fromkeys(labels, {}),
    )
    section = text.split("Automatic decisions on absent clauses (amendment 3)\n")[1]
    assert "absent.pdf | cap | PASS" in section
    assert "present.pdf" not in section
    assert "Count: 1" in section


def test_extraction_main_forwards_parsed_dir(tmp_path, monkeypatch):
    for name in (
        "LABELS", "PARSED_DIR", "PROMPT_VERSION", "RESULTS_OUT", "SUMMARY_OUT",
    ):
        monkeypatch.setattr(eval_extraction, name, getattr(eval_extraction, name))
    monkeypatch.setattr(sys, "argv", [
        "eval", "--tag", "test", "--parsed-dir", str(tmp_path), "--cache-only",
    ])

    def fake_run(*args, **kwargs):
        assert eval_extraction.PARSED_DIR == tmp_path
        return 0

    monkeypatch.setattr(eval_extraction, "run_eval", fake_run)
    assert eval_extraction.main() == 0


def test_verdict_main_uses_custom_files_and_prints_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "evals").mkdir()
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "fake.json").write_text(report(Decision.PASS).model_dump_json())
    labels = write_rows(tmp_path / "labels.csv", [{
        "contract": "fake.pdf", "clause_type": "Cap On Liability", "found": "False",
    }])
    extractions = write_rows(tmp_path / "extractions.csv", [{
        "contract": "fake.pdf", "clause_type": "Cap On Liability",
    }])
    monkeypatch.setattr(eval_verdicts, "load_rules", lambda _: [cap_rule()])
    eval_verdicts.main([
        "--reports", str(reports), "--labels", str(labels),
        "--extractions", str(extractions), "--tag", "test",
    ])
    text = (tmp_path / "evals/verdict_eval_test.txt").read_text()
    assert f"Labels: {labels}" in text
    assert f"Retrieval diagnostics: {extractions}" in text
    assert "Count: 1" in text


def test_rule_main_uses_custom_runs_and_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_rules, "ROOT", tmp_path)
    output = tmp_path / "evals"
    output.mkdir()
    rows = [{
        "contract": "fake.pdf", "clause_type": "Cap On Liability", "found": "False",
    }]
    labels = write_rows(tmp_path / "labels.csv", rows)
    write_rows(output / "extraction_results_custom.csv", rows)
    monkeypatch.setattr(eval_rules, "load_rules", lambda _: [cap_rule()])

    def evaluate(extractions, actual_labels, rules, policy):
        assert actual_labels == eval_rules.load_rows(labels)
        return [{"contract": "fake.pdf", "policy": policy}]

    monkeypatch.setattr(eval_rules, "evaluate", evaluate)
    monkeypatch.setattr(eval_rules, "summarize", lambda *args: (
        "fake summary", {"unsafe": 0, "review": 0},
    ))
    eval_rules.main(["--labels", str(labels), "--runs", "custom", "--tag", "test"])
    assert {p.name for p in output.glob("rule_results_*.csv")} == {
        f"rule_results_custom_{policy.value.lower()}_test.csv" for policy in TrustPolicy
    }
    text = (output / "rules_eval_test.txt").read_text()
    assert f"labels: {labels}" in text
    assert "all selected runs" in text


def test_batch_custom_inputs_skip_design_comparison(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "evals").mkdir()
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    (pdfs / "fake.pdf").touch()
    labels = write_rows(tmp_path / "labels.csv", [{
        "contract": "fake.pdf", "clause_type": "Cap On Liability",
    }])
    monkeypatch.setattr(run_checker_batch, "load_rules", lambda _: [cap_rule()])
    monkeypatch.setattr(run_checker_batch, "build_checker", lambda **kwargs: None)
    monkeypatch.setattr(run_checker_batch, "_load_model", lambda _: None)

    def batch(checker, contracts, *args, **kwargs):
        assert contracts == [pdfs / "fake.pdf"]
        return [report(Decision.PASS)]

    def no_comparison(*args):
        pytest.fail("Custom inputs must not read the design-set results")

    monkeypatch.setattr(run_checker_batch, "run_batch", batch)
    monkeypatch.setattr(run_checker_batch, "compare_offline", no_comparison)
    assert run_checker_batch.main([
        "--labels", str(labels), "--contracts-dir", str(pdfs), "--tag", "test",
    ]) == 0
    text = (tmp_path / "evals/checker_batch_summary_test.txt").read_text()
    assert "comparison skipped" in text
