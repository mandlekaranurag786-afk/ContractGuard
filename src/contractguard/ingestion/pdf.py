"""Read PDFs into the normalized document representation used by ContractGuard."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pymupdf
from pydantic import BaseModel

MIN_TEXT_CHARACTERS = 20


class PDFIngestionError(ValueError):
    """Raised when a source file cannot be ingested as a readable PDF."""


class ParsedPage(BaseModel):
    """Text extracted from one PDF page."""

    page_number: int
    raw_text: str
    clean_text: str
    needs_ocr: bool


class ParsedDocument(BaseModel):
    """A PDF and its page-level extraction results."""

    doc_id: str
    original_filename: str
    page_count: int
    pages: list[ParsedPage]


def ingest_pdf(
    pdf_path: str | Path,
    output_dir: str | Path = Path("data/parsed"),
) -> ParsedDocument:
    """Extract every page from *pdf_path* and persist a ParsedDocument JSON file.

    Pages with fewer than ``MIN_TEXT_CHARACTERS`` non-whitespace characters are
    retained and marked for a later OCR stage.
    """
    source_path = Path(pdf_path)
    if not source_path.is_file():
        raise PDFIngestionError(f"PDF file does not exist: {source_path}")

    try:
        source_bytes = source_path.read_bytes()
    except OSError as exc:
        raise PDFIngestionError(f"Unable to read PDF file: {source_path}") from exc

    try:
        document = pymupdf.open(source_path)
    except Exception as exc:
        raise PDFIngestionError(
            f"Unable to open PDF '{source_path}': the file is corrupt or unsupported."
        ) from exc

    try:
        if not document.is_pdf:
            raise PDFIngestionError(f"File is not a PDF: {source_path}")

        pages = [_parse_page(page, index + 1) for index, page in enumerate(document)]
    except PDFIngestionError:
        raise
    except Exception as exc:
        raise PDFIngestionError(
            f"Unable to extract PDF '{source_path}': the file may be corrupt."
        ) from exc
    finally:
        document.close()

    parsed_document = ParsedDocument(
        doc_id=hashlib.sha256(source_bytes).hexdigest(),
        original_filename=source_path.name,
        page_count=len(pages),
        pages=pages,
    )

    destination = Path(output_dir) / f"{parsed_document.doc_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        parsed_document.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return parsed_document


def _parse_page(page: pymupdf.Page, page_number: int) -> ParsedPage:
    raw_text = page.get_text("text", sort=True)
    clean_text = _clean_text(raw_text)
    return ParsedPage(
        page_number=page_number,
        raw_text=raw_text,
        clean_text=clean_text,
        needs_ocr=len(re.sub(r"\s+", "", clean_text)) < MIN_TEXT_CHARACTERS,
    )


def _clean_text(raw_text: str) -> str:
    """Normalize horizontal whitespace without flattening document structure."""
    normalized_lines = [
        re.sub(r"[ \t]+", " ", line).rstrip() for line in raw_text.splitlines()
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(normalized_lines))
