"""Batch PDF ingestion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from contractguard.ingestion.pdf import ParsedDocument, PDFIngestionError, ingest_pdf


@dataclass(frozen=True)
class IngestionFailure:
    """A PDF that could not be ingested during a batch run."""

    filename: str
    error: str


@dataclass
class IngestionSummary:
    """Aggregate results for a non-recursive directory ingestion."""

    ingested: list[ParsedDocument]
    failed: list[IngestionFailure]
    total_pages: int
    flagged_pages: int

    @property
    def files_found(self) -> int:
        """Return the number of PDFs discovered in the input directory."""
        return len(self.ingested) + len(self.failed)


def ingest_directory(input_dir: str | Path, output_dir: str | Path) -> IngestionSummary:
    """Ingest immediate PDF children of *input_dir* into *output_dir*."""
    source_directory = Path(input_dir)
    destination_directory = Path(output_dir)
    pdf_paths = sorted(
        (
            path
            for path in source_directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.name.lower(),
    )

    ingested: list[ParsedDocument] = []
    failed: list[IngestionFailure] = []
    total_pages = 0
    flagged_pages = 0

    for pdf_path in pdf_paths:
        try:
            parsed_document = ingest_pdf(pdf_path, output_dir=destination_directory)
        except PDFIngestionError as exc:
            failed.append(IngestionFailure(filename=pdf_path.name, error=str(exc)))
            continue

        ingested.append(parsed_document)
        total_pages += parsed_document.page_count
        flagged_pages += sum(page.needs_ocr for page in parsed_document.pages)

    return IngestionSummary(
        ingested=ingested,
        failed=failed,
        total_pages=total_pages,
        flagged_pages=flagged_pages,
    )
