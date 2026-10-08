"""PDF ingestion for ContractGuard."""

from contractguard.ingestion.batch import (
    IngestionFailure,
    IngestionSummary,
    ingest_directory,
)
from contractguard.ingestion.pdf import (
    MIN_TEXT_CHARACTERS,
    ParsedDocument,
    ParsedPage,
    PDFIngestionError,
    ingest_pdf,
)

__all__ = [
    "MIN_TEXT_CHARACTERS",
    "PDFIngestionError",
    "ParsedDocument",
    "ParsedPage",
    "IngestionFailure",
    "IngestionSummary",
    "ingest_directory",
    "ingest_pdf",
]
