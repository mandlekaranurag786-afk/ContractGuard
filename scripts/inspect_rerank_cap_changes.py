"""Inspect liability-cap rank-one changes caused by MiniLM reranking."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contractguard.chunking.fixed import chunk_document
from contractguard.ingestion import load_parsed_document
from contractguard.retrieval.rerank import CrossEncoderScorer, RerankingRetriever
from contractguard.retrieval.vector import InMemoryVectorIndex
from evals.eval_retrieval import (
    CLAUSE_QUERIES,
    KNOWN_BAD_LABELS,
    MASTER_CLAUSES_CSV,
    PARSED_DIR,
    RERANK_POOL_SIZE,
    TOP_K,
    best_overlap_against_spans,
    build_filename_to_docid_map,
    load_cuad_answers,
    normalize_filename,
)

CLAUSE_TYPE = "Cap On Liability"
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
REPORT_PATH = Path("evals/rerank_cap_inspection.txt")
TARGETS = {
    "lost rank 1": (
        "MRSFIELDSORIGINALCOOKIES",
        "ParatekPharmaceuticals",
        "UpjohnInc",
        "RangeResourcesLouisiana",
        "PACIRA PHARMACEUTICALS",
        "IMAGEWARESYSTEMS",
    ),
    "gained rank 1": (
        "MIDDLEBROOKPHARMACEUTICALS",
        "AlliedEsportsEntertainment",
    ),
}


def _matching_result(results, answer_spans: list[str]):
    for rank, result in enumerate(results, start=1):
        overlap = best_overlap_against_spans(result.chunk.text, answer_spans)
        if overlap >= 0.5:
            return rank, result, overlap
    return None


def _find_filename(prefix: str, filename_to_docid: dict[str, str]) -> str:
    matches = [
        filename
        for filename in filename_to_docid
        if filename.upper().startswith(prefix.upper())
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one contract for {prefix!r}, found {matches}")
    return matches[0]


def _print_chunk(output, label: str, result, overlap: float) -> None:
    print(
        f"{label}: {result.chunk.chunk_id} | overlap={overlap:.3f}", file=output
    )
    if result.rerank_score is not None:
        print(f"cross_encoder_score={result.rerank_score:.6f}", file=output)
    print(result.chunk.text[:600], file=output)


def main() -> None:
    cuad_answers = load_cuad_answers(MASTER_CLAUSES_CSV)
    filename_to_docid = build_filename_to_docid_map(PARSED_DIR)
    query = CLAUSE_QUERIES[CLAUSE_TYPE]
    scorer = CrossEncoderScorer(MODEL_NAME)

    with REPORT_PATH.open("w", encoding="utf-8") as output:
        for change, prefixes in TARGETS.items():
            print(f"{'=' * 80}\n{change.upper()}\n{'=' * 80}", file=output)
            for prefix in prefixes:
                filename = _find_filename(prefix, filename_to_docid)
                key = normalize_filename(filename)
                if (key, CLAUSE_TYPE) in KNOWN_BAD_LABELS:
                    raise RuntimeError(f"{filename} is excluded by the eval")

                answer_spans = cuad_answers[key][CLAUSE_TYPE]
                if not answer_spans:
                    raise RuntimeError(f"{filename} has no CUAD answer span")

                doc_id = filename_to_docid[filename]
                doc = load_parsed_document(PARSED_DIR / f"{doc_id}.json")
                chunks = chunk_document(doc)
                dense = InMemoryVectorIndex.from_chunks(chunks)
                reranked = RerankingRetriever(dense, scorer, pool_size=RERANK_POOL_SIZE)
                dense_results = dense.search(query, k=TOP_K)
                reranked_results = reranked.search(query, k=TOP_K)

                dense_first = dense_results[0]
                dense_overlap = best_overlap_against_spans(
                    dense_first.chunk.text, answer_spans
                )
                reranked_first = reranked_results[0]
                reranked_overlap = best_overlap_against_spans(
                    reranked_first.chunk.text, answer_spans
                )
                match = _matching_result(reranked_results, answer_spans)

                print(f"\n{filename}", file=output)
                print(f"CUAD answer: {answer_spans[0][:400]}", file=output)
                _print_chunk(output, "dense rank 1", dense_first, dense_overlap)
                _print_chunk(
                    output, "reranked rank 1", reranked_first, reranked_overlap
                )
                print(
                    "reranked rank-1 pair tokens: "
                    f"{scorer.max_pair_tokens(query, [reranked_first.chunk.text])}",
                    file=output,
                )

                if match is None:
                    print("CUAD matching chunk after reranking: not found", file=output)
                    continue

                rank, matching_result, overlap = match
                print(
                    "CUAD matching chunk after reranking: "
                    f"rank={rank} chunk_id={matching_result.chunk.chunk_id} "
                    f"overlap={overlap:.3f} "
                    f"cross_encoder_score={matching_result.rerank_score:.6f}",
                    file=output,
                )
                print(
                    "CUAD matching pair tokens: "
                    f"{scorer.max_pair_tokens(query, [matching_result.chunk.text])}",
                    file=output,
                )


if __name__ == "__main__":
    main()
