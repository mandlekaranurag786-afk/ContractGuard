"""Injectable orchestration from a PDF to an evidence-bearing rule report."""

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Protocol

from contractguard.chunking import Chunk
from contractguard.extraction.confidence import compute_confidence
from contractguard.extraction.schemas import ExtractionResult
from contractguard.ingestion import ParsedDocument
from contractguard.pipeline import (
    CLAUSE_QUERIES,
    build_retriever,
    chunk_document,
    load_document,
)
from contractguard.report import ClauseSummary, ContractReport, Cost, overall_status
from contractguard.retrieval.models import SearchResult
from contractguard.rules.engine import evaluate_rules
from contractguard.rules.models import Decision, Rule, TrustPolicy


class ClauseExtractor(Protocol):
    @property
    def prompt_version(self) -> str: ...

    def extract(self, clause_type: str, chunks: list[Chunk]) -> ExtractionResult: ...


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[SearchResult]: ...


class ContractChecker:
    def __init__(
        self,
        extractor: ClauseExtractor,
        policy: TrustPolicy = TrustPolicy.KEYWORD,
        top_k: int = 5,
        *,
        document_loader: Callable[[Path], ParsedDocument] = load_document,
        retriever_builder: Callable[[list[Chunk]], Retriever] = build_retriever,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.extractor = extractor
        self.policy = policy
        self.top_k = top_k
        self.document_loader = document_loader
        self.retriever_builder = retriever_builder

    def check(
        self,
        pdf_path: str | Path,
        rules: list[Rule],
        policy_path: str | Path,
    ) -> ContractReport:
        if not rules:
            raise ValueError("Rule list must not be empty")
        clauses = list(dict.fromkeys(rule.clause_type for rule in rules))
        for clause in clauses:
            if clause not in CLAUSE_QUERIES:
                raise ValueError(f"No retrieval query for {clause}")
        started = perf_counter()
        chunks = chunk_document(self.document_loader(Path(pdf_path)))
        retriever = self.retriever_builder(chunks)
        results, confidences, summaries = {}, {}, []
        for clause in clauses:
            retrieved = [
                r.chunk for r in retriever.search(CLAUSE_QUERIES[clause], k=self.top_k)
            ]
            result = self.extractor.extract(clause, retrieved)
            results[clause] = result
            confidence = compute_confidence(
                None if result.error else result.extraction,
                result.checks,
                clause,
            )
            confidences[clause] = confidence
            summaries.append(
                ClauseSummary(
                    clause_type=clause,
                    confidence_level=confidence.level,
                    confidence_reasons=confidence.reasons,
                    retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieved],
                    extraction_error=result.error,
                )
            )
        decisions = evaluate_rules(rules, results, confidences, self.policy)
        status = overall_status(decisions)
        models = list(dict.fromkeys(r.model for r in results.values() if r.model))
        return ContractReport(
            contract=str(pdf_path),
            policy_path=str(policy_path),
            trust_policy=self.policy,
            model=", ".join(models) or None,
            prompt_version=self.extractor.prompt_version,
            overall_status=status,
            overall_reason=status.reason,
            counts={d: sum(r.decision == d for r in decisions) for d in Decision},
            rule_results=decisions,
            clauses=summaries,
            cost=Cost(
                llm_calls=sum(r.attempts for r in results.values() if not r.cached),
                cache_hits=sum(r.attempts for r in results.values() if r.cached),
                input_tokens=sum(r.input_tokens or 0 for r in results.values()),
                output_tokens=sum(r.output_tokens or 0 for r in results.values()),
                elapsed_ms=(perf_counter() - started) * 1000,
            ),
        )
