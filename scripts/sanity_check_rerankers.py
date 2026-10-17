"""Compare offline MiniLM and BGE reranker scores on four simple passages."""

from __future__ import annotations

import gc
import os

os.environ["HF_HUB_OFFLINE"] = "1"

from contractguard.retrieval.rerank import CrossEncoderScorer

QUERY = "what is the liability cap"
TEXTS = [
    "In no event shall either party's total liability under this Agreement "
    "exceed the fees paid in the twelve months before the claim.",
    "In no event shall either party be liable for any indirect, incidental "
    "or consequential damages.",
    "This Agreement shall be governed by the laws of the State of Delaware.",
    "15. Limitations of Liability ........ 8   16. Arbitration ........ 9",
]
MODELS = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "BAAI/bge-reranker-base",
)


def _print_scores(model_name: str) -> None:
    scorer = CrossEncoderScorer(model_name)
    scores = scorer.score(QUERY, TEXTS)
    ranks = {
        index: rank
        for rank, (index, _) in enumerate(
            sorted(enumerate(scores), key=lambda item: item[1], reverse=True),
            start=1,
        )
    }

    print(f"\n{model_name}")
    if model_name == "BAAI/bge-reranker-base":
        print(f"model config num_labels: {scorer._model.config.num_labels}")
        print(f"model class: {type(scorer._model.model).__name__}")
    for index, (text, score) in enumerate(zip(TEXTS, scores, strict=True), start=1):
        print(f"{index}. score={score:.6f} rank={ranks[index - 1]} text={text!r}")

    del scorer
    gc.collect()


def main() -> None:
    for model_name in MODELS:
        _print_scores(model_name)


if __name__ == "__main__":
    main()
