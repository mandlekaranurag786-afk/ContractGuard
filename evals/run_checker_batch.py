"""Run the product checker on the extraction-eval contracts; cache-only by default."""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Literal

# Set before importing the embedding library: this script never downloads models.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contractguard.checker import ContractChecker
from contractguard.extraction.extractor import Extractor
from contractguard.extraction.prompt import PRODUCT_PROMPT_VERSION
from contractguard.llm.cache import DEFAULT_CACHE_DIR, CachedLLMClient
from contractguard.llm.errors import LLMQuotaError
from contractguard.report import ContractReport, OverallStatus
from contractguard.retrieval.vector import MODEL_NAME, _load_model
from contractguard.rules.models import Decision, Rule, TrustPolicy, load_rules
from evals.eval_extraction import LABELS, TOP_K, CacheOnlyClient
from evals.eval_rules import load_rows

POLICY = Path("policies/eval_policy.json")
OFFLINE_RESULTS = Path("evals/rule_results_v2_keyword_v2.csv")
SUMMARY = Path("evals/checker_batch_summary.txt")
REPORTS = Path("evals/reports")


def selected_contracts() -> list[Path]:
    """Reuse the extraction eval's labels source, preserving its contract order."""
    return [
        Path("data/contracts") / name
        for name in dict.fromkeys(contract for contract, _ in load_rows(LABELS))
    ]


def build_checker(
    *,
    live_followup: bool = False,
    followup_cache_dir: Path = DEFAULT_CACHE_DIR,
    main_cache_dir: Path = DEFAULT_CACHE_DIR,
) -> ContractChecker:
    """Main extraction can never reach a live client, even in live-followup mode."""
    main = CacheOnlyClient()
    followup = CacheOnlyClient()
    if live_followup:
        from contractguard.llm.gemini import GeminiClient

        followup = GeminiClient()
    return ContractChecker(
        Extractor(
            CachedLLMClient(main, model=main.model, cache_dir=main_cache_dir),
            prompt_version=PRODUCT_PROMPT_VERSION,
        ),
        policy=TrustPolicy.KEYWORD,
        top_k=TOP_K,
        followup_client=CachedLLMClient(
            followup, model=followup.model, cache_dir=followup_cache_dir
        ),
    )


def run_batch(
    checker: ContractChecker,
    contracts: list[Path],
    rules: list[Rule],
    policy_path: Path,
    report_dir: Path,
    *,
    followup: bool = False,
    followup_version: Literal["v1", "v2"] = "v1",
    cache_only: bool = True,
) -> list[ContractReport]:
    """Stop immediately on any error; never overwrite an existing report."""
    outputs = [report_dir / f"{p.stem}.json" for p in contracts]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Contract stems must be unique")
    for output in outputs:
        if output.exists():
            raise FileExistsError(output)
    report_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for i, (contract, output) in enumerate(zip(contracts, outputs, strict=True), 1):
        print(f"[{i}/{len(contracts)}] {contract.name}", flush=True)
        try:
            report = (
                checker.check(
                    contract, rules, policy_path, followup=True,
                    followup_version=followup_version,
                )
                if followup
                else checker.check(contract, rules, policy_path)
            )
            if report.followup.status == "followup_skipped":
                raise LLMQuotaError("Follow-up quota exhausted")
        except LLMQuotaError:
            print(
                f"Quota exhausted at {contract.name}; "
                f"completed contracts: {len(reports)}/{len(contracts)}. "
                "Finished reports retained.",
                flush=True,
            )
            raise
        if cache_only and report.cost.llm_calls:
            raise RuntimeError("Cache-only batch unexpectedly recorded LLM calls")
        with output.open("x", encoding="utf-8") as handle:
            handle.write(report.model_dump_json(indent=2) + "\n")
        reports.append(report)
        print(
            f"{contract.name} | follow-up calls={report.followup.calls_used} | "
            f"cached={report.followup.cached_calls or 0} | "
            f"live={report.followup.live_calls or 0}",
            flush=True,
        )
    return reports


def summarize(
    reports: list[ContractReport], model_load_ms: float, cache_only: bool = True
) -> str:
    decisions = Counter(r.decision for p in reports for r in p.rule_results)
    statuses = Counter(p.overall_status for p in reports)
    reasons = Counter(r.reason_code for p in reports for r in p.rule_results)
    timings = sorted(p.cost.elapsed_ms for p in reports)
    total = sum(decisions.values())
    reviews = decisions[Decision.NEEDS_REVIEW]
    rate = 100 * reviews / total if total else 0
    p95 = timings[math.ceil(0.95 * len(timings)) - 1] if timings else 0
    lines = [
        f"Policy: {POLICY} | KEYWORD | prompt={PRODUCT_PROMPT_VERSION}",
        f"Cache-only: ON (main extraction) | cache={DEFAULT_CACHE_DIR}",
        f"Follow-up cache-only: {'ON' if cache_only else 'OFF'}",
        "Embedding model: loaded once; shared by pipeline via _load_model lru_cache",
        f"Model load time (separate): {model_load_ms:.2f} ms",
        "contract | overall_status | PASS | FAIL | NEEDS_REVIEW | "
        "NOT_APPLICABLE | llm_calls | cache_hits | elapsed_ms",
    ]
    for p in reports:
        counts = Counter(r.decision for r in p.rule_results)
        lines.append(
            f"{Path(p.contract).name} | {p.overall_status} | "
            + " | ".join(str(counts[d]) for d in Decision)
            + f" | {p.cost.llm_calls} | {p.cost.cache_hits} | {p.cost.elapsed_ms:.2f}"
        )
    lines.extend(
        [
            "Overall status counts: "
            + ", ".join(f"{s}={statuses[s]}" for s in OverallStatus),
            "Decision counts: " + ", ".join(f"{d}={decisions[d]}" for d in Decision),
            f"Review rate: {reviews}/{total} ({rate:.2f}%)",
            "Reason code counts: "
            + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())),
            f"Median elapsed_ms: {statistics.median(timings) if timings else 0:.2f}",
            f"p95 elapsed_ms (nearest rank): {p95:.2f}",
            f"Total llm_calls: {sum(p.cost.llm_calls for p in reports)}",
            f"Total cache_hits: {sum(p.cost.cache_hits for p in reports)}",
        ]
    )
    return "\n".join(lines)


def compare_offline(reports: list[ContractReport], path: Path) -> str:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    old = {(r["contract"], r["rule_id"]): r for r in rows}
    new = {
        (Path(p.contract).name, r.rule_id): r for p in reports for r in p.rule_results
    }
    if len(old) != len(rows) or len(new) != sum(len(p.rule_results) for p in reports):
        raise ValueError("Duplicate contract/rule keys")
    if old.keys() != new.keys():
        raise ValueError(f"Rule-row keys differ: {old.keys() ^ new.keys()}")
    identical = decisions = allowed = 0
    differences = []
    for key, result in new.items():
        before = tuple(
            old[key][f] or None
            for f in ("decision", "reason_code", "evidence_strength")
        )
        after = (result.decision.value, result.reason_code, result.evidence_strength)
        decisions += before[0] != after[0]
        if before == after:
            identical += 1
            continue
        exception = (
            before[0] == after[0]
            and before[2] == after[2]
            and after[1] == "cap_is_formula"
            and (before[1] or "").startswith("value_")
        )
        allowed += exception
        differences.append(
            f"{key[0]} | {key[1]} | batch={after!r} | offline={before!r}"
            + (" | allowed cap_is_formula reason change" if exception else "")
        )
    return "\n".join(
        [
            f"CONSISTENCY: {path}",
            "Compared fields: (decision, reason_code, evidence_strength); "
            "reason text ignored",
            f"Rows compared: {len(new)}",
            f"Rows identical: {identical}",
            f"Allowed cap_is_formula changes: {allowed}",
            f"Rows identical after allowed exception: {identical + allowed}",
            f"Decision differences: {decisions}",
            f"Unexpected differences: {len(differences) - allowed}",
            *differences,
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-only",
        action="store_true",
        default=True,
        help="Main extraction is always cache-only",
    )
    parser.add_argument("--followup", action="store_true")
    parser.add_argument("--followup-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--live-followup", action="store_true")
    parser.add_argument("--followup-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--tag")
    args = parser.parse_args(argv)
    if args.live_followup and not args.followup:
        parser.error("--live-followup requires --followup")
    if args.tag and (not args.tag.replace("_", "").replace("-", "").isalnum()):
        parser.error("tag must contain letters, numbers, underscores or hyphens")
    summary = (
        Path(f"evals/checker_batch_summary_{args.tag}.txt") if args.tag else SUMMARY
    )
    reports_dir = Path(f"evals/reports_{args.tag}") if args.tag else REPORTS
    if summary.exists():
        raise FileExistsError(summary)
    contracts = selected_contracts()
    for contract in contracts:
        if not contract.is_file():
            raise FileNotFoundError(contract)
    rules = load_rules(POLICY)
    checker = build_checker(
        live_followup=args.live_followup, followup_cache_dir=args.followup_cache_dir
    )
    start = perf_counter()
    _load_model(MODEL_NAME)
    load_ms = (perf_counter() - start) * 1000
    try:
        reports = run_batch(
            checker,
            contracts,
            rules,
            POLICY,
            reports_dir,
            followup=args.followup,
            followup_version=args.followup_version,
            cache_only=not args.live_followup,
        )
    except LLMQuotaError:
        return 1
    text = (
        summarize(reports, load_ms, not args.live_followup)
        + "\n\n"
        + compare_offline(reports, OFFLINE_RESULTS)
    )
    with summary.open("x", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
