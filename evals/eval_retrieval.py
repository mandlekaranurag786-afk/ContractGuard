"""
Retrieval eval harness.

What it does:
  For each of our 30 selected contracts, for each clause type we care about,
  look up CUAD's expert-labeled answer, run our retrieval pipeline, and record
  where (if anywhere) in the ranked results the correct chunk shows up.

Why:
  Replaces manually reading printed search results with an automatic,
  repeatable score across all 30 contracts. This becomes our "before" number;
  every future retrieval change (hybrid search, reranking) gets measured
  against it with a re-run of this same script.

Usage:
    uv run python evals/eval_retrieval.py
"""

import ast
import csv
import re
from collections.abc import Callable
from pathlib import Path

from contractguard.chunking import Chunk
from contractguard.chunking.fixed import chunk_document
from contractguard.evaluation.metrics import first_hit_rank, mrr_at_k, recall_at_k
from contractguard.ingestion import load_parsed_document
from contractguard.retrieval.hybrid import HybridSearchIndex
from contractguard.retrieval.vector import InMemoryVectorIndex

# ---- Config -----------------------------------------------------------

PARSED_DIR = Path("data/parsed")
MASTER_CLAUSES_CSV = Path("data/contracts/master_clauses.csv")
SELECTED_CONTRACTS_CSV = Path("data/contracts/selected_30_contracts.csv")
RESULTS_OUT = Path("evals/retrieval_results_dense.csv")
HYBRID_RESULTS_OUT = Path("evals/retrieval_results_hybrid.csv")
TOP_K = 20

# clause_type -> (CUAD raw-span column name, query we run against our index)
CLAUSE_QUERIES = {
    "Renewal Term": "what is the renewal term",
    "Cap On Liability": "what is the liability cap",
}

# Contracts manually verified to have incorrect CUAD labels.
# (normalized_filename, clause_type) -> reason
KNOWN_BAD_LABELS = {
    ("operaltd04302020ex414serviceagreementpdf", "Renewal Term"):
        "CUAD answer span points to a data-protection clause, not a renewal "
        "clause. Verified: no renewal clause exists anywhere in this PDF.",
}

# ---- Filename matching -----------------------------------------------------

def normalize_filename(name: str) -> str:
    """
    Loosen filename matching so small formatting differences (A&R vs A_R,
    extra spaces, case) don't silently drop a contract from the CUAD join.
    Keep only alphanumeric characters, lowercased.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ---- Scoring ------------------------------------------------------------

def normalize(text: str) -> set[str]:
    """Lowercase, strip punctuation, split into a set of words."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def word_overlap_score(chunk_text: str, answer_text: str) -> float:
    """
    Fraction of the ANSWER's words found in the chunk.
    Asymmetric on purpose: a long chunk containing the whole answer should
    score 1.0 even though the chunk has lots of extra words the answer
    doesn't have. We care whether the chunk *contains* the answer's content.
    """
    answer_words = normalize(answer_text)
    if not answer_words:
        return 0.0
    chunk_words = normalize(chunk_text)
    return len(answer_words & chunk_words) / len(answer_words)


def best_overlap_against_spans(chunk_text: str, answer_spans: list[str]) -> float:
    """CUAD sometimes gives more than one correct span for a clause.
    A chunk matching ANY one of them counts as the real answer."""
    if not answer_spans:
        return 0.0
    return max(word_overlap_score(chunk_text, span) for span in answer_spans)


# ---- Loading CUAD answers -------------------------------------------------

def parse_cuad_list_cell(cell: str) -> list[str]:
    """
    CUAD stores multi-span answers as a Python-list-looking string, e.g.
        "['span one.', 'span two.']"
    or an empty list "[]" when the clause doesn't exist in that contract.
    We parse it defensively with ast.literal_eval rather than assuming
    it's always well-formed.
    """
    cell = cell.strip()
    if not cell:
        return []
    try:
        parsed = ast.literal_eval(cell)
        if isinstance(parsed, list):
            return [str(s) for s in parsed]
        return [str(parsed)]
    except (ValueError, SyntaxError):
        # Fallback: treat the raw cell as a single span.
        return [cell] if cell != "[]" else []


def load_cuad_answers(csv_path: Path) -> dict[str, dict[str, list[str]]]:
    """Returns {normalized_filename: {clause_type: [answer spans]}}"""
    answers: dict[str, dict[str, list[str]]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = normalize_filename(row["Filename"])
            answers[key] = {}
            for clause_type in CLAUSE_QUERIES:
                answers[key][clause_type] = parse_cuad_list_cell(row[clause_type])
    return answers


# ---- Loading our 30 selected contracts + mapping filename -> doc_id -------

def build_filename_to_docid_map(parsed_dir: Path) -> dict[str, str]:
    """
    Our parsed JSON files are named <doc_id>.json and each one stores
    original_filename internally. We need the reverse lookup to join
    against master_clauses.csv, which only has filenames.
    """
    mapping = {}
    for json_path in parsed_dir.glob("*.json"):
        doc = load_parsed_document(json_path)
        mapping[doc.original_filename] = doc.doc_id
    return mapping


# ---- Main eval loop -------------------------------------------------------

def run_eval(index_factory: Callable[[list[Chunk]], object]) -> list[dict]:
    cuad_answers = load_cuad_answers(MASTER_CLAUSES_CSV)
    filename_to_docid = build_filename_to_docid_map(PARSED_DIR)

    results = []

    for filename, doc_id in filename_to_docid.items():
        key = normalize_filename(filename)
        if key not in cuad_answers:
            print(f"  [skip] {filename}: not found in master_clauses.csv")
            continue
        answer_lookup = cuad_answers[key]

        parsed_path = PARSED_DIR / f"{doc_id}.json"
        doc = load_parsed_document(parsed_path)
        chunks = chunk_document(doc)
        if not chunks:
            print(f"  [skip] {filename}: produced zero chunks")
            continue

        index = index_factory(chunks)

        for clause_type, query in CLAUSE_QUERIES.items():
            bad_label_reason = KNOWN_BAD_LABELS.get((key, clause_type))
            if bad_label_reason:
                results.append({
                    "contract": filename,
                    "clause_type": clause_type,
                    "cuad_answer_exists": False,
                    "best_overlap_rank": None,
                    "best_overlap": None,
                    "first_hit_rank_05": None,
                    "first_hit_rank_07": None,
                    "first_hit_rank_09": None,
                    "note": f"Excluded -- known bad CUAD label: {bad_label_reason}",
                })
                continue

            answer_spans = answer_lookup[clause_type]

            if not answer_spans:
                # CUAD itself found no such clause in this contract.
                # Not a retrieval failure -- exclude from scoring.
                results.append({
                    "contract": filename,
                    "clause_type": clause_type,
                    "cuad_answer_exists": False,
                    "best_overlap_rank": None,
                    "best_overlap": None,
                    "first_hit_rank_05": None,
                    "first_hit_rank_07": None,
                    "first_hit_rank_09": None,
                    "note": "CUAD answer empty -- excluded from recall scoring",
                })
                continue

            search_results = index.search(query, k=TOP_K)

            per_rank_scores = []
            for rank, r in enumerate(search_results, start=1):
                score = best_overlap_against_spans(r.chunk.text, answer_spans)
                per_rank_scores.append((rank, r.chunk.chunk_id, score))

            if per_rank_scores:
                # Best (rank, score) pair -- highest overlap, earliest rank on ties
                best_overlap_rank, best_chunk_id, best_score = max(
                    per_rank_scores, key=lambda x: (x[2], -x[0])
                )
            else:
                # Retrieval returned nothing at all for this query.
                best_overlap_rank, best_chunk_id, best_score = None, None, 0.0

            rank_overlaps = [score for _, _, score in per_rank_scores]
            first_hit_rank_05 = first_hit_rank(rank_overlaps, 0.5)
            first_hit_rank_07 = first_hit_rank(rank_overlaps, 0.7)
            first_hit_rank_09 = first_hit_rank(rank_overlaps, 0.9)

            results.append({
                "contract": filename,
                "clause_type": clause_type,
                "cuad_answer_exists": True,
                "best_overlap_rank": best_overlap_rank,
                "best_chunk_id": best_chunk_id,
                "best_overlap": round(best_score, 3),
                "rank_overlaps": rank_overlaps,
                "first_hit_rank_05": first_hit_rank_05,
                "first_hit_rank_07": first_hit_rank_07,
                "first_hit_rank_09": first_hit_rank_09,
                "note": "",
            })

            first_hit_str = (
                f"{first_hit_rank_05:2d}" if first_hit_rank_05 is not None else " -"
            )
            best_rank_str = (
                f"{best_overlap_rank:2d}" if best_overlap_rank is not None else " -"
            )
            print(f"  {filename[:50]:50s} | {clause_type:20s} | "
                  f"first_hit@0.5={first_hit_str} "
                  f"best_overlap_rank={best_rank_str} overlap={best_score:.2f}")

    _assert_reporting_consistency(results)
    return results

def save_results(results: list[dict], results_out: Path = RESULTS_OUT):
    results_out.parent.mkdir(exist_ok=True)
    with open(results_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["contract", "clause_type", "cuad_answer_exists",
                        "best_overlap_rank", "best_chunk_id", "best_overlap",
                        "first_hit_rank_05", "first_hit_rank_07",
                        "first_hit_rank_09", "note"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nSaved {len(results)} rows to {results_out}")


def _assert_reporting_consistency(results: list[dict]) -> None:
    scored = [row for row in results if row["cuad_answer_exists"]]
    first_hit_ranks = [row["first_hit_rank_05"] for row in scored]

    for k in (1, 10):
        reported_hits = sum(
            rank is not None and rank <= k for rank in first_hit_ranks
        )
        recall_hits = round(recall_at_k(first_hit_ranks, k) * len(scored))
        if reported_hits != recall_hits:
            raise RuntimeError(
                f"first_hit_rank_05 count ({reported_hits}) does not match "
                f"recall@{k} count ({recall_hits})"
            )


def _print_recall_table(scored: list[dict]):
    for threshold in (0.5, 0.7, 0.9):
        first_hit_ranks = [
            first_hit_rank(r["rank_overlaps"], threshold) for r in scored
        ]
        for k in (1, 3, 5, 10, 20):
            recall = recall_at_k(first_hit_ranks, k)
            hits = sum(rank is not None and rank <= k for rank in first_hit_ranks)
            print(f"  recall@{k:<2d} (overlap>={threshold}): {recall:.2%} "
                  f"({hits}/{len(scored)})")
        mrr = mrr_at_k(first_hit_ranks, k=10)
        print(f"  MRR@10 (overlap>={threshold}): {mrr:.3f}")
        print()


def print_summary(results: list[dict]):
    scored = [r for r in results if r["cuad_answer_exists"]]
    skipped = [r for r in results if not r["cuad_answer_exists"]]

    print(f"\n{'='*60}")
    print(f"Total clause instances evaluated: {len(scored)}")
    print(f"Skipped (CUAD answer empty):      {len(skipped)}")
    _print_recall_table(scored)

    for clause_type in CLAUSE_QUERIES:
        clause_scored = [r for r in scored if r["clause_type"] == clause_type]
        print(f"{clause_type}: {len(clause_scored)} instances")
        _print_recall_table(clause_scored)


if __name__ == "__main__":
    print("Running retrieval eval across selected contracts...\n")
    print("DENSE-ONLY BASELINE")
    dense_results = run_eval(InMemoryVectorIndex.from_chunks)
    save_results(dense_results)
    print_summary(dense_results)

    print("HYBRID SEARCH")
    hybrid_results = run_eval(HybridSearchIndex.from_chunks)
    save_results(hybrid_results, HYBRID_RESULTS_OUT)
    print_summary(hybrid_results)
