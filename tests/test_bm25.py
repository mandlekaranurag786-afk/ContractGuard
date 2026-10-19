from __future__ import annotations

from contractguard.chunking import Chunk
from contractguard.retrieval.bm25 import BM25Index, tokenize


def _chunk(index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"doc-1:{index}",
        doc_id="doc-1",
        text=text,
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(text),
    )


def _corpus() -> list[Chunk]:
    return [
        _chunk(0, "The liability cap limits damages to fees paid."),
        _chunk(1, "The renewal term extends for twelve months."),
        _chunk(2, "Invoices are due thirty days after delivery."),
    ]


def test_exact_keyword_ranks_matching_chunk_first() -> None:
    results = BM25Index.from_chunks(_corpus()).search("liability")

    assert results[0].chunk.chunk_id == "doc-1:0"
    assert results[0].score == results[0].bm25_score


def test_query_with_no_shared_vocabulary_returns_no_results() -> None:
    assert BM25Index.from_chunks(_corpus()).search("volcano") == []


def test_tokenization_is_case_insensitive_and_strips_punctuation() -> None:
    assert tokenize("Liability-CAP!") == ["liability", "cap"]

    results = BM25Index.from_chunks(_corpus()).search("LIABILITY!")

    assert results[0].chunk.chunk_id == "doc-1:0"


def test_tokenizer_preserves_duplicate_words_for_bm25_term_frequency() -> None:
    assert tokenize("cap cap CAP") == ["cap", "cap", "cap"]






# import os
# from dotenv import load_dotenv
# from google import genai

# # Load the variables from the .env file
# load_dotenv() 

# # Now the client will automatically find the GEMINI_API_KEY in the environment
# client = genai.Client()

# from google import genai

# client = genai.Client()
# response = client.models.generate_content(
#     model="gemini-3.5-flash-lite",
#     contents="Who is the PM of India?",
# )
# print(response.text)