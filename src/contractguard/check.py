"""Check a PDF: python -m contractguard.check PDF --policy PATH."""

import argparse
from pathlib import Path

from contractguard.checker import ContractChecker
from contractguard.extraction.extractor import Extractor
from contractguard.extraction.prompt import PRODUCT_PROMPT_VERSION
from contractguard.llm.cache import DEFAULT_CACHE_DIR, CachedLLMClient
from contractguard.llm.errors import LLMQuotaError
from contractguard.llm.gemini import GeminiClient
from contractguard.rules.models import load_rules


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--followup", action="store_true")
    parser.add_argument("--followup-version", choices=("v1", "v2"), default="v1")
    args = parser.parse_args(argv)
    try:
        rules = load_rules(args.policy)
        client = GeminiClient()
        extractor = Extractor(
            CachedLLMClient(client, model=client.model, cache_dir=DEFAULT_CACHE_DIR),
            prompt_version=PRODUCT_PROMPT_VERSION,
        )
        checker = ContractChecker(extractor)
        report = (
            checker.check(
                args.pdf, rules, args.policy, followup=True,
                followup_version=args.followup_version,
            )
            if args.followup
            else checker.check(args.pdf, rules, args.policy)
        )
    except (LLMQuotaError, ValueError, OSError) as exc:
        parser.exit(1, f"Check failed: {exc}\n")
    print(f"{report.overall_status}: {report.overall_reason}")
    for result in report.rule_results:
        strength = (
            "weak signal (keyword only)"
            if result.evidence_strength == "weak_signal"
            else result.evidence_strength
        )
        print(
            f"{result.rule_id} | {result.decision} | {result.value!r} | "
            f"{result.field_status} | {strength} | {result.reason}"
        )
        print(f"  Quote: {(result.evidence_quote or '')[:160]}")
    print(f"Cost: {report.cost.model_dump_json()}")
    if report.followup.enabled:
        print(
            f"Follow-up: {report.followup.status} | calls "
            f"{report.followup.calls_used}/{report.followup.max_calls}"
        )
        for change in report.followup.changed_rules:
            print(
                f"  {change.rule_id}: {change.before_decision} "
                f"({change.before_reason_code}) -> {change.after_decision} "
                f"({change.after_reason_code}) | {change.evidence_strength}"
            )
    if args.json_out:
        args.json_out.write_text(
            report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
