from __future__ import annotations

import hashlib

import pymupdf
import pytest

from contractguard.ingestion import (
    PDFIngestionError,
    ingest_pdf,
    load_parsed_document,
)
from contractguard.ingestion.pdf import _clean_text


def _create_pdf(path, page_texts: list[str]) -> None:
    document = pymupdf.open()
    try:
        for text in page_texts:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        document.save(path)
    finally:
        document.close()


def test_ingest_pdf_extracts_pages_and_writes_json(tmp_path) -> None:
    source = tmp_path / "agreement.pdf"
    _create_pdf(source, ["This agreement contains enough text to avoid an OCR flag."])

    parsed = ingest_pdf(source, output_dir=tmp_path / "parsed")

    assert parsed.doc_id == hashlib.sha256(source.read_bytes()).hexdigest()
    assert parsed.original_filename == "agreement.pdf"
    assert parsed.page_count == 1
    assert parsed.pages[0].page_number == 1
    assert parsed.pages[0].clean_text.startswith("This agreement")
    assert parsed.pages[0].needs_ocr is False
    assert (tmp_path / "parsed" / f"{parsed.doc_id}.json").is_file()


def test_load_parsed_document_round_trips_ingested_pdf(tmp_path) -> None:
    source = tmp_path / "agreement.pdf"
    _create_pdf(source, ["This agreement can be persisted and loaded again."])

    original = ingest_pdf(source, output_dir=tmp_path / "parsed")
    json_path = tmp_path / "parsed" / f"{original.doc_id}.json"

    loaded = load_parsed_document(json_path)

    assert loaded == original


def test_ingest_pdf_retains_empty_page_and_flags_it_for_ocr(tmp_path) -> None:
    source = tmp_path / "scanned.pdf"
    _create_pdf(source, ["", "Text is present on this second page of the agreement."])

    parsed = ingest_pdf(source, output_dir=tmp_path / "parsed")

    assert parsed.page_count == 2
    assert parsed.pages[0].page_number == 1
    assert parsed.pages[0].raw_text == ""
    assert parsed.pages[0].clean_text == ""
    assert parsed.pages[0].needs_ocr is True
    assert parsed.pages[1].needs_ocr is False


def test_clean_text_preserves_lines_paragraphs_and_hyphens() -> None:
    raw_text = "First   line with a hyphen-\nsecond  line  \n\n\nThird   paragraph"

    assert (
        _clean_text(raw_text)
        == "First line with a hyphen-\nsecond line\n\nThird paragraph"
    )


def test_ingest_pdf_flags_whitespace_only_page_for_ocr(tmp_path) -> None:
    source = tmp_path / "whitespace.pdf"
    _create_pdf(source, ["   \t  "])

    parsed = ingest_pdf(source, output_dir=tmp_path / "parsed")

    assert parsed.pages[0].needs_ocr is True


def test_ingest_same_file_twice_keeps_document_id(tmp_path) -> None:
    source = tmp_path / "agreement.pdf"
    _create_pdf(source, ["This agreement contains enough text to avoid an OCR flag."])

    first = ingest_pdf(source, output_dir=tmp_path / "parsed")
    second = ingest_pdf(source, output_dir=tmp_path / "parsed")

    assert first.doc_id == second.doc_id


def test_ingest_pdf_reports_corrupt_files_clearly(tmp_path) -> None:
    source = tmp_path / "broken.pdf"
    source.write_bytes(b"not a PDF")

    with pytest.raises(PDFIngestionError, match="corrupt or unsupported"):
        ingest_pdf(source, output_dir=tmp_path / "parsed")


def test_ingest_pdf_reports_missing_files_clearly(tmp_path) -> None:
    missing_source = tmp_path / "missing.pdf"

    with pytest.raises(PDFIngestionError, match="does not exist"):
        ingest_pdf(missing_source, output_dir=tmp_path / "parsed")
