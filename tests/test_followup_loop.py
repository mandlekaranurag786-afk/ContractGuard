"""Bounded follow-ups with fake retrieval and generation only."""

import json

import pytest

from contractguard.checker import ContractChecker
from contractguard.chunking import Chunk
from contractguard.extraction.confidence import ConfidenceResult
from contractguard.extraction.extractor import SCHEMAS, Extractor
from contractguard.extraction.schemas import ExtractionResult
from contractguard.followup.loop import FOLLOWUP_QUERIES, run_followup
from contractguard.ingestion import ParsedDocument, ParsedPage
from contractguard.llm.client import LLMResponse
from contractguard.llm.errors import LLMQuotaError, LLMUnavailableError
from contractguard.retrieval.models import SearchResult
from contractguard.rules.engine import evaluate_rules
from contractguard.rules.models import Decision, Rule, TrustPolicy

QUOTE = (
    "UNDER NO CIRCUMSTANCE SHALL EITHER PARTY BE LIABLE TO THE OTHER PARTY "
    "FOR ANY LOST PROFITS OR CONSEQUENTIAL DAMAGES"
)


def chunk(index, text=QUOTE):
    return Chunk(
        chunk_id=f"doc:{index}",
        doc_id="doc",
        text=text,
        char_start=index * 1000,
        char_end=index * 1000 + len(text),
        page_start=1,
        page_end=1,
    )


class Retriever:
    def __init__(self, chunks=None):
        self.chunks = chunks if chunks is not None else [chunk(i) for i in range(20)]
        self.calls = []

    def search(self, query, k):
        self.calls.append((query, k))
        return [SearchResult(chunk=c, score=1) for c in self.chunks[:k]]


class Client:
    def __init__(self, outcomes=None, cached=False):
        self.outcomes = list(outcomes or [])
        self.calls = []
        self.cached = cached

    def generate_json(self, prompt, schema):
        self.calls.append(prompt)
        outcome = (
            self.outcomes.pop(0)
            if self.outcomes
            else dict(value=True, evidence_quote=QUOTE, evidence_chunk_id="doc:1")
        )
        if isinstance(outcome, Exception):
            raise outcome
        return LLMResponse(
            text=outcome if isinstance(outcome, str) else json.dumps(outcome),
            model="fake",
            input_tokens=10,
            output_tokens=2,
            latency_ms=3,
            cached=self.cached,
        )


def setup(fields=("has_damages_exclusion",)):
    rules, results, confidences = [], {}, {}
    for field in fields:
        clause = (
            "Cap On Liability"
            if field in {"cap_amount", "has_money_cap", "has_damages_exclusion"}
            else "Renewal Term"
        )
        schema = SCHEMAS[clause]
        data = dict.fromkeys(schema.model_fields)
        data.update(found=True, evidence_quote="original", evidence_chunk_id="doc:0")
        if clause == "Cap On Liability":
            data.update(
                has_money_cap=False, has_damages_exclusion=False, currency="USD"
            )
        results[clause] = ExtractionResult(
            clause_type=clause,
            extraction=schema(**data),
            checks={},
            attempts=1,
            model="fake",
            cached=True,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            error=None,
        )
        confidences[clause] = ConfidenceResult(
            "HIGH", [], {}, {f: "absence_claim" for f in schema.model_fields}
        )
        numeric = not field.startswith("has_") and field != "auto_renews"
        rules.append(
            Rule(
                id=str(len(rules)),
                description="test",
                clause_type=clause,
                field=field,
                operator=">=" if numeric else "is_true",
                threshold=1 if numeric else None,
                currency="USD" if field == "cap_amount" else None,
            )
        )
    return rules, results, confidences


def run(client, fields=("has_damages_exclusion",), retriever=None, **kwargs):
    rules, results, confidences = setup(fields)
    original = evaluate_rules(rules, results, confidences, TrustPolicy.KEYWORD)
    outcome = run_followup(
        rules,
        results,
        confidences,
        original,
        retriever or Retriever(),
        {c: {"doc:0"} for c in results},
        client,
        **kwargs,
    )
    return outcome, original


def test_success_and_deduplication():
    client = Client()
    result, _ = run(client, ("has_damages_exclusion", "has_damages_exclusion"))
    assert len(client.calls) == result.report.calls_used == 1
    assert all(
        r.decision == "PASS" and r.evidence_strength == "followup"
        for r in result.results
    )
    assert result.report.steps[0].rule_ids == ["0", "1"]
    assert result.results[0].evidence_quote == QUOTE


@pytest.mark.parametrize("cached", [True, False])
def test_budget_and_seen(cached):
    client = Client(
        [dict(value=None, evidence_quote=None, evidence_chunk_id=None, currency=None)]
        + [dict(value=None, evidence_quote=None, evidence_chunk_id=None)] * 3,
        cached=cached,
    )
    result, _ = run(
        client,
        (
            "cap_amount",
            "has_money_cap",
            "has_damages_exclusion",
            "initial_term_months",
            "notice_days",
        ),
    )
    assert len(client.calls) == result.report.calls_used == 4
    assert result.report.steps[-1].note == "budget_exhausted"
    cap_ids = [
        i
        for s in result.report.steps
        if s.clause_type == "Cap On Liability"
        for i in s.new_chunk_ids
    ]
    assert len(cap_ids) == len(set(cap_ids)) and "doc:0" not in cap_ids
    assert result.cost["cache_hits" if cached else "llm_calls"] == 4


def test_retries_cannot_exceed_budget():
    client = Client(["bad"] * 10)
    result, _ = run(client, ("has_money_cap", "has_damages_exclusion"), max_calls=3)
    assert len(client.calls) == result.report.calls_used == 3
    assert result.report.steps[0].note == "extraction_error"
    assert result.report.steps[-1].note == "budget_exhausted"


def test_no_new_chunks_or_query():
    client = Client()
    result, _ = run(client, retriever=Retriever([chunk(0)]))
    assert result.report.steps[0].note == "no_new_chunks" and not client.calls
    result, _ = run(client, ("renewal_term_months",))
    assert result.report.steps[0].note == "no_query" and not client.calls


@pytest.mark.parametrize(
    "update,outcome",
    [
        ({"evidence_chunk_id": "unseen:0"}, "DISCARDED_UNVERIFIED"),
        ({"value": False}, "NO_VALUE"),
        (
            {"value": None, "evidence_quote": None, "evidence_chunk_id": None},
            "NO_VALUE",
        ),
    ],
)
def test_discarded(update, outcome):
    data = dict(value=True, evidence_quote=QUOTE, evidence_chunk_id="doc:1")
    data.update(update)
    result, original = run(Client([data]))
    assert result.results == original
    assert result.report.steps[0].merge_outcome == outcome


@pytest.mark.parametrize(
    "currency,decision", [("USD", "PASS"), ("EUR", "NEEDS_REVIEW")]
)
@pytest.mark.parametrize("old_currency_status", ["absence_claim", "supported"])
def test_cap_and_currency(currency, decision, old_currency_status):
    quote = f"Liability shall not exceed $2,000,000.00 {currency}"
    client = Client(
        [
            dict(
                value=2000000,
                currency=currency,
                evidence_quote=quote,
                evidence_chunk_id="doc:1",
            )
        ]
    )
    rules, results, confidences = setup(("cap_amount",))
    confidences["Cap On Liability"].field_status["currency"] = old_currency_status
    original = evaluate_rules(rules, results, confidences, TrustPolicy.KEYWORD)
    outcome = run_followup(
        rules,
        results,
        confidences,
        original,
        Retriever([chunk(0), chunk(1, quote)]),
        {"Cap On Liability": {"doc:0"}},
        client,
    )
    assert outcome.results[0].decision == decision
    if currency != "USD":
        assert outcome.results[0].reason_code == "followup_conflict"


def test_quota_rolls_back_earlier_success():
    client = Client(
        [
            dict(value=True, evidence_quote=QUOTE, evidence_chunk_id="doc:1"),
            LLMQuotaError("quota"),
        ]
    )
    result, original = run(client, ("has_damages_exclusion", "has_money_cap"))
    assert result.results == original
    assert result.report.status == "followup_skipped"
    assert result.report.calls_used == 2 and not result.report.changed_rules


def test_error_counts_call():
    result, original = run(Client([LLMUnavailableError("failed")]))
    assert result.report.calls_used == 1 and result.results == original
    assert result.report.steps[0].note == "extraction_error"
    assert "LLMUnavailableError: failed" in result.report.steps[0].error


def test_guard_blocks_fake_overlay(monkeypatch):
    from contractguard.followup import loop

    rules, results, confidences = setup()
    original = evaluate_rules(rules, results, confidences, TrustPolicy.KEYWORD)
    original[0] = original[0].model_copy(update={"decision": Decision.PASS})
    monkeypatch.setattr(
        loop,
        "evaluate_rule",
        lambda *args: original[0].model_copy(update={"decision": Decision.FAIL}),
    )
    result = run_followup(
        rules, results, confidences, original, Retriever(), {}, Client()
    )
    assert (
        result.results == original and result.report.steps[-1].note == "guard_blocked"
    )


def fake_checker():
    cap = dict(
        found=True,
        has_money_cap=True,
        cap_amount=2000000,
        currency="USD",
        cap_formula=None,
        has_damages_exclusion=False,
        evidence_quote="Liability shall not exceed $2,000,000.00",
        evidence_chunk_id="doc:0",
    )
    client = Client(
        [cap, dict(value=True, evidence_quote=QUOTE, evidence_chunk_id="doc:1")]
    )

    def loader(path):
        return ParsedDocument(
            doc_id="doc",
            original_filename=path.name,
            page_count=1,
            pages=[
                ParsedPage(
                    page_number=1,
                    raw_text="original",
                    clean_text="original",
                    needs_ocr=False,
                )
            ],
        )

    class InitialRetriever(Retriever):
        def search(self, query, k):
            if query not in FOLLOWUP_QUERIES.values():
                return [SearchResult(chunk=chunk(0, cap["evidence_quote"]), score=1)]
            return super().search(query, k)

    service = ContractChecker(
        Extractor(client, "v2"),
        document_loader=loader,
        retriever_builder=lambda _: InitialRetriever(),
    )
    return service, client, setup()[0]


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_checker_and_cli_fake(tmp_path, monkeypatch, capsys, version):
    from contractguard import check

    service, client, rules = fake_checker()
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps([r.model_dump() for r in rules]))
    monkeypatch.setattr(
        check, "GeminiClient", lambda: type("Fake", (), {"model": "fake"})()
    )
    monkeypatch.setattr(check, "ContractChecker", lambda _: service)
    out = tmp_path / "report.json"
    assert (
        check.main(
            [
                "fake-xpresspa.pdf",
                "--policy",
                str(policy),
                "--followup",
                "--followup-version",
                version,
                "--json-out",
                str(out),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "PASS: At least one rule passed" in printed
    assert "Follow-up: completed | calls 1/4" in printed
    assert "followup" in printed and QUOTE in printed
    report = json.loads(out.read_text())
    assert report["followup"]["version"] == version
    assert report["cost"]["input_tokens"] == 20
    assert report["cost"]["llm_calls"] == 2
    assert report["followup"]["changed_rules"][0]["after_decision"] == "PASS"
    assert len(client.calls) == 2


def test_followup_off_does_not_call():
    service, client, rules = fake_checker()
    report = service.check("fake.pdf", rules, "policy.json")
    assert report.overall_status == "NEEDS_REVIEW"
    assert not report.followup.enabled and report.followup.status == "disabled"
    assert len(client.calls) == 1


def test_condition_target_uses_condition_query():
    rules, results, confidences = setup(("notice_days",))
    clause = "Renewal Term"
    results[clause].extraction.notice_days = 60
    confidences[clause].field_status["notice_days"] = "supported"
    rules[0] = rules[0].model_copy(update={"when": None})
    from contractguard.rules.models import Condition

    rules[0].when = Condition(field="auto_renews", equals=True)
    original = evaluate_rules(rules, results, confidences, TrustPolicy.KEYWORD)
    quote = "This Agreement shall automatically renew for successive terms"
    retriever = Retriever([chunk(0), chunk(1, quote)])
    result = run_followup(
        rules,
        results,
        confidences,
        original,
        retriever,
        {clause: {"doc:0"}},
        Client([dict(value=True, evidence_quote=quote, evidence_chunk_id="doc:1")]),
    )
    assert result.report.steps[0].field == "auto_renews"
    assert retriever.calls[0][0] == FOLLOWUP_QUERIES["auto_renews"]
    assert result.results[0].decision == "PASS"
    assert result.results[0].condition_evidence_quote == quote


def test_mixed_cache_retry_cost():
    class Mixed(Client):
        def generate_json(self, prompt, schema):
            response = super().generate_json(prompt, schema)
            return response.model_copy(update={"cached": len(self.calls) == 1})

    client = Mixed(
        ["invalid", dict(value=True, evidence_quote=QUOTE, evidence_chunk_id="doc:1")]
    )
    result, _ = run(client)
    assert result.report.calls_used == 2
    assert result.cost == dict(
        llm_calls=1,
        cache_hits=1,
        input_tokens=10,
        output_tokens=2,
        cached_input_tokens=10,
        cached_output_tokens=2,
    )
    assert result.report.steps[0].input_tokens == 20
    assert not result.report.steps[0].cached


def test_checker_quota_rolls_back_and_retains_cost():
    service, client, rules = fake_checker()
    client.outcomes[-1] = LLMQuotaError("quota")
    report = service.check("fake.pdf", rules, "policy.json", followup=True)
    assert report.overall_status == "NEEDS_REVIEW"
    assert report.followup.status == "followup_skipped"
    assert report.cost.llm_calls == 2 and report.followup.calls_used == 1
    assert not report.followup.changed_rules


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize(
    "reason", ["value_absence_claim", "value_unsupported", "condition_unverified"]
)
@pytest.mark.parametrize("has_unseen", [True, False])
def test_versioned_chunk_policy_and_verification(version, reason, has_unseen):
    rules, results, confidences = setup()
    clause = "Cap On Liability"
    original = evaluate_rules(rules, results, confidences, TrustPolicy.KEYWORD)
    original[0] = original[0].model_copy(update={
        "reason_code": reason, "condition_field": "has_damages_exclusion",
    })
    originals = [chunk(8), chunk(0)]
    retrieved = [chunk(0), chunk(8)]
    if has_unseen:
        retrieved += [chunk(i) for i in range(1, 6)]
    client = Client([dict(value=True, evidence_quote=QUOTE, evidence_chunk_id="doc:8")])
    result = run_followup(
        rules, results, confidences, original, Retriever(retrieved),
        {clause: {"doc:8", "doc:0"}}, client,
        version=version, original_chunks={clause: originals},
    )
    include_original = version == "v2" and reason != "value_absence_claim"
    sent = (["doc:8", "doc:0"] if include_original else []) + (
        ["doc:1", "doc:2", "doc:3"] if has_unseen else []
    )
    step = result.report.steps[0]
    assert result.report.version == version
    assert step.chunk_ids == sent
    assert step.new_chunk_ids == (["doc:1", "doc:2", "doc:3"] if has_unseen else [])
    assert step.chunk_policy == (
        "original_plus_unseen" if include_original else "unseen_only"
    )
    assert len(client.calls) == result.report.calls_used == bool(sent)
    if sent:
        from contractguard.followup.extract import build_field_prompt

        by_id = {c.chunk_id: c for c in [*originals, *retrieved]}
        assert client.calls[0] == build_field_prompt(
            clause, "has_damages_exclusion", [by_id[i] for i in sent]
        )
        assert step.verified is include_original
        assert step.merge_outcome == (
            "ACCEPTED_NEW" if include_original else "DISCARDED_UNVERIFIED"
        )
    else:
        assert step.note == "no_new_chunks"


@pytest.mark.parametrize("reverse", [False, True])
def test_v2_combines_reasons_for_same_target(reverse):
    rules, results, confidences = setup(("has_damages_exclusion",) * 2)
    clause = "Cap On Liability"
    original = evaluate_rules(rules, results, confidences, TrustPolicy.KEYWORD)
    reasons = ["value_absence_claim", "condition_unverified"]
    if reverse:
        reasons.reverse()
    for item, reason in zip(original, reasons, strict=True):
        item.reason_code = reason
        item.condition_field = "has_damages_exclusion"
    client = Client()
    result = run_followup(
        rules, results, confidences, original, Retriever(), {clause: {"doc:0"}},
        client, version="v2", original_chunks={clause: [chunk(0)]},
    )
    assert len(client.calls) == 1
    assert result.report.steps[0].rule_ids == ["0", "1"]
    assert result.report.steps[0].chunk_ids == ["doc:0", "doc:1", "doc:2", "doc:3"]


def test_v2_original_only_retries_still_obey_four_attempt_budget():
    fields = ("has_damages_exclusion", "has_money_cap", "cap_amount")
    rules, results, confidences = setup(fields)
    clause = "Cap On Liability"
    original = evaluate_rules(rules, results, confidences, TrustPolicy.KEYWORD)
    for item in original:
        item.reason_code = "value_unsupported"
    client = Client(["invalid"] * 10)
    result = run_followup(
        rules, results, confidences, original, Retriever([chunk(0)]),
        {clause: {"doc:0"}}, client, version="v2",
        original_chunks={clause: [chunk(0)]},
    )
    assert result.report.calls_used == len(client.calls) == 4
    assert result.report.steps[-1].note == "budget_exhausted"


def test_checker_v2_passes_original_chunks_in_main_order():
    service, client, rules = fake_checker()
    client.outcomes[0]["has_damages_exclusion"] = True
    client.outcomes[1]["evidence_chunk_id"] = "doc:8"
    cap_quote = client.outcomes[0]["evidence_quote"]

    class OrderedRetriever(Retriever):
        def search(self, query, k):
            if query not in FOLLOWUP_QUERIES.values():
                return [SearchResult(chunk=c, score=1) for c in (
                    chunk(8), chunk(0, cap_quote),
                )]
            return []

    service.retriever_builder = lambda _: OrderedRetriever()
    report = service.check("fake.pdf", rules, "policy.json", True, "v2")
    assert report.followup.version == "v2"
    assert report.followup.steps[0].chunk_ids == ["doc:8", "doc:0"]
    assert report.followup.steps[0].verified
    assert report.overall_status == "PASS"
