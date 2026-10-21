"""Read-only reconstruction of the eval's deterministic hybrid top-5 context."""

from functools import lru_cache

from contractguard.chunking.fixed import _concatenate_pages
from contractguard.retrieval.hybrid import HybridSearchIndex
from evals.eval_retrieval import (
    CLAUSE_QUERIES,
    PARSED_DIR,
    build_filename_to_docid_map,
    chunk_document,
    load_parsed_document,
    normalize_filename,
)


class LocalRetrieval:
    def __init__(self):
        self.docids = build_filename_to_docid_map(PARSED_DIR)

    @lru_cache
    def document(self, filename):
        matches = [
            docid
            for name, docid in self.docids.items()
            if normalize_filename(name) == normalize_filename(filename)
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one parsed document for {filename}")
        doc = load_parsed_document(PARSED_DIR / f"{matches[0]}.json")
        chunks = chunk_document(doc)
        text, _ = _concatenate_pages(doc)
        return chunks, text

    @lru_cache
    def index(self, filename):
        return HybridSearchIndex.from_chunks(self.document(filename)[0])

    @lru_cache
    def chunk_ids(self, filename, clause):
        return [
            r.chunk.chunk_id
            for r in self.index(filename).search(CLAUSE_QUERIES[clause], k=5)
        ]
