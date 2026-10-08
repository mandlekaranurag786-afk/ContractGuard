"""Command-line entry point for batch PDF ingestion."""

from __future__ import annotations

import argparse
from pathlib import Path

from contractguard.ingestion import IngestionSummary, ingest_directory


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, print a batch summary, and return an exit status."""
    parser = argparse.ArgumentParser(description="Ingest PDFs in a directory.")
    parser.add_argument(
        "input_dir", nargs="?", default=Path("data/contracts"), type=Path
    )
    parser.add_argument("--output-dir", default=Path("data/parsed"), type=Path)
    args = parser.parse_args(argv)

    summary = ingest_directory(args.input_dir, args.output_dir)
    if summary.files_found == 0:
        print(f"0 PDFs found in {args.input_dir}")
        return 1

    _print_summary(summary)
    return 1 if summary.failed else 0


def _print_summary(summary: IngestionSummary) -> None:
    print("filename\tdoc_id\tpage_count\tneeds_ocr_pages")
    for parsed_document in summary.ingested:
        flagged_pages = sum(page.needs_ocr for page in parsed_document.pages)
        print(
            f"{parsed_document.original_filename}\t{parsed_document.doc_id[:12]}"
            f"\t{parsed_document.page_count}\t{flagged_pages}"
        )
    for failure in summary.failed:
        print(f"{failure.filename}\tERROR\t-\t{failure.error}")

    print("Totals")
    print(f"files ingested\t{len(summary.ingested)}")
    print(f"files failed\t{len(summary.failed)}")
    print(f"total pages\t{summary.total_pages}")
    print(f"total flagged pages\t{summary.flagged_pages}")


if __name__ == "__main__":
    raise SystemExit(main())
