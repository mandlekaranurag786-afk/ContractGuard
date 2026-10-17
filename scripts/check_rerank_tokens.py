"""Inspect cross-encoder token lengths for the retrieval eval candidates."""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

from transformers import AutoTokenizer

# Let this standalone script reuse the eval's exact data-loading helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contractguard.chunking.fixed import chunk_document
from contractguard.ingestion import load_parsed_document
from contractguard.retrieval.hybrid import HybridSearchIndex
from contractguard.retrieval.vector import InMemoryVectorIndex
from evals.eval_retrieval import (
    CLAUSE_QUERIES,
    KNOWN_BAD_LABELS,
    MASTER_CLAUSES_CSV,
    PARSED_DIR,
    TOP_K,
    best_overlap_against_spans,
    build_filename_to_docid_map,
    load_cuad_answers,
    normalize_filename,
)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOKEN_LIMIT = 512


def _pair_encoding(tokenizer, query: str, text: str):
    return tokenizer(
        query,
        text,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=True,
    )


def _answer_start_token(
    tokenizer, query: str, text: str, spans: list[str]
) -> int | None:
    encoding = _pair_encoding(tokenizer, query, text)
    sequence_ids = encoding.sequence_ids()
    for span in spans:
        start = text.lower().find(span.lower())
        if start < 0:
            continue
        for position, (sequence_id, offsets) in enumerate(
            zip(sequence_ids, encoding["offset_mapping"], strict=True)
        ):
            if sequence_id == 1 and offsets[0] <= start < offsets[1]:
                return position
    return None


def _first_hit(results, answer_spans: list[str]):
    for result in results:
        if best_overlap_against_spans(result.chunk.text, answer_spans) >= 0.5:
            return result
    return None


def _print_longest_pairs(pairs: dict[tuple[str, str], tuple]) -> None:
    print("\n10 longest pairs:")
    for _, (chunk, token_count) in sorted(
        pairs.items(), key=lambda item: item[1][1], reverse=True
    )[:10]:
        print(
            f"  {chunk.chunk_id} | chars={len(chunk.text)} | "
            f"tokens={token_count} | {chunk.text[:150]!r}"
        )


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    cuad_answers = load_cuad_answers(MASTER_CLAUSES_CSV)
    filename_to_docid = build_filename_to_docid_map(PARSED_DIR)
    pairs: dict[tuple[str, str], tuple] = {}
    chunk_lengths: list[int] = []
    long_first_hits: list[tuple[str, str, str, int, int | None]] = []
    scored_instances = 0

    for filename, doc_id in filename_to_docid.items():
        key = normalize_filename(filename)
        if key not in cuad_answers:
            continue

        doc = load_parsed_document(PARSED_DIR / f"{doc_id}.json")
        chunks = chunk_document(doc)
        chunk_lengths.extend(len(chunk.text) for chunk in chunks)
        dense = InMemoryVectorIndex.from_chunks(chunks)
        hybrid = HybridSearchIndex.from_chunks(chunks)

        for clause_type, query in CLAUSE_QUERIES.items():
            if (key, clause_type) in KNOWN_BAD_LABELS:
                continue
            answer_spans = cuad_answers[key][clause_type]
            if not answer_spans:
                continue

            scored_instances += 1
            for retriever_name, index in (("dense", dense), ("hybrid", hybrid)):
                results = index.search(query, k=TOP_K)
                for result in results:
                    pair_key = (query, result.chunk.chunk_id)
                    if pair_key not in pairs:
                        encoding = _pair_encoding(tokenizer, query, result.chunk.text)
                        pairs[pair_key] = (
                            result.chunk,
                            len(encoding["input_ids"]),
                        )

                first_hit = _first_hit(results, answer_spans)
                if first_hit is None:
                    continue
                token_count = pairs[(query, first_hit.chunk.chunk_id)][1]
                if token_count > TOKEN_LIMIT:
                    start_token = _answer_start_token(
                        tokenizer, query, first_hit.chunk.text, answer_spans
                    )
                    long_first_hits.append(
                        (
                            filename,
                            clause_type,
                            retriever_name,
                            token_count,
                            start_token,
                        )
                    )

    over_limit = sum(token_count > TOKEN_LIMIT for _, token_count in pairs.values())
    print(f"Scored instances: {scored_instances}")
    print(f"Total unique pairs: {len(pairs)}")
    print(f"Pairs over 512 tokens: {over_limit}")
    print(
        "Chunk chars: "
        f"min={min(chunk_lengths)} median={statistics.median(chunk_lengths):.1f} "
        f"max={max(chunk_lengths)} "
        f"over_1000={sum(length > 1000 for length in chunk_lengths)}"
    )
    _print_longest_pairs(pairs)

    print("\nOver-512 first-hit chunks at overlap 0.5:")
    if not long_first_hits:
        print("  none")
    for (
        filename,
        clause_type,
        retriever_name,
        token_count,
        start_token,
    ) in long_first_hits:
        if start_token is None:
            position = "answer_start_token=not found"
        else:
            cut_off = "cut off" if start_token >= TOKEN_LIMIT else "not cut off"
            position = f"answer_start_token={start_token} ({cut_off})"
        print(
            f"  {retriever_name} | {filename} | {clause_type} | "
            f"tokens={token_count} | {position}"
        )


if __name__ == "__main__":
    main()
