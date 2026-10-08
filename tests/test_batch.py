from __future__ import annotations

from pathlib import Path

import pymupdf

from contractguard.ingestion import ingest_directory


def _create_pdf(path: Path, text: str) -> None:
    document = pymupdf.open()
    try:
        page = document.new_page()
        page.insert_text((72, 72), text)
        document.save(path)
    finally:
        document.close()


def test_ingest_directory_writes_json_for_real_pdfs(tmp_path: Path) -> None:
    input_dir = tmp_path / "contracts"
    output_dir = tmp_path / "parsed"
    input_dir.mkdir()
    _create_pdf(
        input_dir / "first.pdf", "A first agreement with enough extracted text."
    )
    _create_pdf(
        input_dir / "second.pdf", "A second agreement with enough extracted text."
    )

    summary = ingest_directory(input_dir, output_dir)

    assert len(summary.ingested) == 2
    assert summary.failed == []
    assert summary.total_pages == 2
    assert all(
        (output_dir / f"{document.doc_id}.json").is_file()
        for document in summary.ingested
    )


def test_ingest_directory_returns_empty_summary_for_empty_folder(
    tmp_path: Path,
) -> None:
    summary = ingest_directory(tmp_path, tmp_path / "parsed")

    assert summary.files_found == 0
    assert summary.ingested == []
    assert summary.failed == []
    assert summary.total_pages == 0
    assert summary.flagged_pages == 0
