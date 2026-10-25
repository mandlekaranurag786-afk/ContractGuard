from pathlib import Path

import pymupdf

from contractguard import pipeline
from contractguard.chunking import chunk_document
from evals import eval_extraction, eval_retrieval


def test_evals_share_pipeline_queries_and_helpers():
    for module in (eval_extraction, eval_retrieval):
        assert module.CLAUSE_QUERIES is pipeline.CLAUSE_QUERIES
        assert module.normalize_filename is pipeline.normalize_filename
        assert (
            module.build_filename_to_docid_map is pipeline.build_filename_to_docid_map
        )
        assert module.chunk_document is chunk_document


def test_pdf_loader_uses_existing_parser_without_persistent_artifacts(
    tmp_path, monkeypatch
):
    source = tmp_path / "contract.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "The contract requires sixty days prior notice.")
        pdf.save(source)
    original = pipeline.ingest_pdf
    destinations = []

    def record(path, output_dir):
        destinations.append(Path(output_dir))
        return original(path, output_dir=output_dir)

    monkeypatch.setattr(pipeline, "ingest_pdf", record)
    document = pipeline.load_document(source)
    assert document.original_filename == "contract.pdf"
    assert "sixty days prior notice" in pipeline.chunk_document(document)[0].text
    assert not destinations[0].exists()
