"""
Run the chunker on one contract and print the chunks so you can read them
yourself, instead of trusting a summary.

Usage:
    uv run python scripts/inspect_chunks.py "data/contracts/BELLICUM....PDF"
    uv run python scripts/inspect_chunks.py "data/contracts/BELLICUM....PDF" --around 146200
"""

import argparse
from pathlib import Path

from contractguard.ingestion import ingest_pdf
from contractguard.chunking import chunk_document  # adjust import if named differently


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--around", type=int, default=None,
                        help="print whichever chunk(s) contain this character offset")
    parser.add_argument("--chunk-index", type=int, default=None,
                        help="print one specific chunk by its window index, e.g. 171")
    args = parser.parse_args()

    parsed = ingest_pdf(args.pdf_path)
    chunks = chunk_document(parsed, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"Total chunks: {len(chunks)}\n")

    if args.chunk_index is not None:
        target_id = f"{parsed.doc_id}:{args.chunk_index}"
        matches = [c for c in chunks if c.chunk_id == target_id]
        if not matches:
            print(f"No chunk with id {target_id}")
        for c in matches:
            _print_chunk(c)
        return

    if args.around is not None:
        matches = [c for c in chunks if c.char_start <= args.around < c.char_end]
        print(f"Chunk(s) covering offset {args.around}:\n")
        for c in matches:
            _print_chunk(c)
        return

    # default: just show the first couple of chunks so you can see the shape
    for c in chunks[:2]:
        _print_chunk(c)


def _print_chunk(c):
    print(f"--- {c.chunk_id}  (chars {c.char_start}-{c.char_end}, pages {c.page_start}-{c.page_end}) ---")
    print(c.text)
    print()




if __name__ == "__main__":
    main()