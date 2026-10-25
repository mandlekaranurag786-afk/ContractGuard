"""Shared document loading and retrieval configuration for checks and evals."""

import re
from pathlib import Path
from tempfile import TemporaryDirectory

from contractguard.chunking import Chunk, chunk_document
from contractguard.ingestion import ParsedDocument, ingest_pdf, load_parsed_document
from contractguard.retrieval.hybrid import HybridSearchIndex

CLAUSE_QUERIES = {
    "Renewal Term": "what is the renewal term",
    "Cap On Liability": "what is the liability cap",
}


def load_document(pdf_path: str | Path) -> ParsedDocument:
    """Use the existing PDF parser without leaving a parsed-data artifact."""
    with TemporaryDirectory(prefix="contractguard-parse-") as directory:
        return ingest_pdf(pdf_path, output_dir=directory)


def build_retriever(chunks: list[Chunk]) -> HybridSearchIndex:
    return HybridSearchIndex.from_chunks(chunks)


def normalize_filename(name: str) -> str:
    """Match filenames despite punctuation, whitespace and case differences."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def build_filename_to_docid_map(parsed_dir: Path) -> dict[str, str]:
    mapping = {}
    for json_path in parsed_dir.glob("*.json"):
        doc = load_parsed_document(json_path)
        mapping[doc.original_filename] = doc.doc_id
    return mapping


__all__ = [
    "CLAUSE_QUERIES",
    "build_filename_to_docid_map",
    "build_retriever",
    "chunk_document",
    "load_document",
    "load_parsed_document",
    "normalize_filename",
]
