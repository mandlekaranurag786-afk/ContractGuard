"""Request-history integration uses local fakes, temporary DBs and a movable clock."""

import hashlib
import json
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from contractguard.api import app as api
from contractguard.api.history import HistoryError, HistoryStore
from contractguard.llm.errors import LLMQuotaError
from contractguard.report import ClauseSummary, ContractReport, Cost, OverallStatus
from contractguard.rules.models import Decision, RuleResult, TrustPolicy

NOW = datetime(2026, 11, 11, 9, 30, tzinfo=timezone.utc)
PDF = b"%PDF-fake upload handled by the injected document loader"
POLICY = [
    {"id": f"cap-{i}", "description": "cap", "clause_type": "Cap On Liability",
     "field": "cap_amount", "operator": ">=", "threshold": 100 + i}
    for i in range(2)
]


class Clock:
    def __init__(self):
        self.now = NOW

    def __call__(self):
        return self.now


class FakeChecker:
    def __init__(self):
        self.calls = []
        self.error = None

    def check(self, path, rules, policy_path, **kwargs):
        self.calls.append((path, path.read_bytes(), kwargs))
        if self.error is not None:
            raise self.error
        results = [RuleResult(
            rule_id=rule.id, clause_type=rule.clause_type,
            policy=TrustPolicy.KEYWORD, decision=Decision.PASS,
            reason_code="value_compared", reason="SECRET_REASON",
            field=rule.field, value="SECRET_VALUE", field_status="supported",
            condition_field=None, condition_value="SECRET_CONDITION",
            condition_status=None, currency_status=None, evidence_strength="proof",
            confidence_level="HIGH", evidence_quote="SECRET_QUOTE",
            evidence_chunk_id="SECRET_CHUNK", condition_evidence_quote="SECRET_QUOTE",
        ) for rule in rules]
        return ContractReport(
            contract=str(path), policy_path=str(policy_path),
            trust_policy=TrustPolicy.KEYWORD, model="fake",
            prompt_version=api.PRODUCT_PROMPT_VERSION,
            overall_status=OverallStatus.PASS, overall_reason=OverallStatus.PASS.reason,
            counts={Decision.PASS: len(results)}, rule_results=results,
            clauses=[ClauseSummary(
                clause_type="Cap On Liability", confidence_level="HIGH",
                confidence_reasons=["SECRET_REASON"],
                retrieved_chunk_ids=["SECRET_CHUNK"],
                extraction_error=None,
            )],
            cost=Cost(
                llm_calls=2, cache_hits=3, input_tokens=100, output_tokens=20,
                cached_input_tokens=50, cached_output_tokens=10, elapsed_ms=7,
            ),
        )


def fake_document(path):
    assert path.is_file()
    return SimpleNamespace(page_count=1, pages=[SimpleNamespace(needs_ocr=False)])


@pytest.fixture
def build(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Real service initialization is forbidden")

    monkeypatch.setattr(api, "GeminiClient", forbidden)
    monkeypatch.setattr(api, "warm_up_model", forbidden)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(POLICY))
    with ExitStack() as stack:
        instances = []

        def factory(
            *, off=False, automatic=False, policy_path=None,
            document_loader=fake_document, **settings_changes,
        ):
            store = HistoryStore(tmp_path / f"history-{len(instances)}.db")
            settings = replace(
                api.ApiSettings(None, 1024, 200, tmp_path / "cache"),
                **settings_changes,
            )
            if automatic:
                settings = replace(settings, db_path=store.db_path)
            checker, clock = FakeChecker(), Clock()
            app = api.create_app(
                checker=checker, settings=settings, policy_path=policy_path or policy,
                document_loader=document_loader, clock=clock,
                history=None if off or automatic else store,
            )
            client = stack.enter_context(TestClient(app))
            instance = SimpleNamespace(
                store=store, settings=settings, checker=checker, clock=clock,
                app=app, client=client, policy=policy,
            )
            instances.append(instance)
            return instance

        yield factory


def post(instance, upload_bytes=PDF, **kwargs):
    return instance.client.post(
        "/check", files={"file": (r"C:\uploads\vendor.pdf", upload_bytes)}, **kwargs,
    )


def latest(instance):
    rows = instance.store.list_checks()
    return instance.store.get_check(rows[0]["id"])


def assert_error_row(instance, response, status, code, counted):
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
    row = latest(instance)
    assert row["http_status"] == status and row["error_code"] == code
    assert row["counted"] is counted
    assert row["file_name"] == "vendor.pdf"
    assert row["created_at"] == "2026-11-11T09:30:00.000000Z"
    assert row["elapsed_ms"] >= 0
    for field in ("overall_status", "report_json", "llm_calls", "cache_hits",
                  "input_tokens", "output_tokens"):
        assert row[field] is None
    assert row["rules"] == []
    return row


@pytest.mark.parametrize("followup", [False, True])
def test_success_record(build, monkeypatch, followup):
    instance = build()
    ticks = iter([10.0, 11.25])
    monkeypatch.setattr(api, "perf_counter", lambda: next(ticks))
    response = post(instance, data={"followup": str(followup).lower()})
    assert response.status_code == 200
    assert len(instance.store.list_checks()) == 1
    row = latest(instance)
    assert row["http_status"] == 200 and row["error_code"] is None
    assert row["counted"] is True and row["overall_status"] == "PASS"
    assert row["file_name"] == "vendor.pdf"
    assert row["file_sha256"] == hashlib.sha256(PDF).hexdigest()
    assert len(row["policy_hash"]) == 64
    assert int(row["policy_hash"], 16) >= 0
    assert row["prompt_version"] == api.PRODUCT_PROMPT_VERSION
    assert row["followup"] is followup
    assert row["created_at"] == "2026-11-11T09:30:00.000000Z"
    assert row["elapsed_ms"] == 1250
    for field in ("llm_calls", "cache_hits", "input_tokens", "output_tokens"):
        assert row[field] == response.json()["cost"][field]
    assert row["rules"] == [
        {"rule_id": r["rule_id"], "decision": r["decision"], "reason": r["reason_code"]}
        for r in response.json()["rule_results"]
    ]
    path, _, options = instance.checker.calls[0]
    assert not path.exists()
    assert options == ({"followup": True, "followup_version": "v2"} if followup else {})
    assert not instance.app.state.check_lock.locked()


def test_metadata_allowlist(build):
    instance = build()
    assert post(instance).status_code == 200
    row = latest(instance)
    for secret in ("SECRET_QUOTE", "SECRET_REASON", "SECRET_VALUE",
                   "SECRET_CONDITION", "SECRET_CHUNK"):
        assert secret not in json.dumps(row)
    stored = json.loads(row["report_json"])
    assert set(stored) == {
        "contract", "policy_path", "trust_policy", "model", "prompt_version",
        "overall_status", "overall_reason", "counts", "cost", "rule_results", "clauses",
    }
    assert set(stored["rule_results"][0]) == {
        "rule_id", "clause_type", "policy", "decision", "reason_code", "field",
        "field_status", "condition_field", "condition_status", "currency_status",
        "evidence_strength", "confidence_level", "field_evidence_source",
        "condition_evidence_source",
    }
    assert set(stored["clauses"][0]) == {"clause_type", "confidence_level"}


def test_full_report_matches_response(build):
    instance = build(history_mode="full")
    response = post(instance)
    assert response.status_code == 200
    assert json.loads(latest(instance)["report_json"]) == response.json()


def test_daily_cap_retry_and_expiration(build):
    instance = build(daily_cap=2)
    assert post(instance).status_code == 200
    instance.clock.now += timedelta(hours=1)
    assert post(instance).status_code == 200
    instance.clock.now = NOW + timedelta(hours=2, microseconds=100000)
    response = post(instance)
    assert response.status_code == 429
    assert response.json()["detail"]["code"] == "daily_limit_reached"
    assert response.headers["Retry-After"] == "79200"
    row = latest(instance)
    assert row["counted"] is False and row["file_sha256"] is None
    assert len(instance.checker.calls) == 2
    instance.clock.now = NOW + timedelta(hours=24)
    response = post(instance)
    assert response.status_code == 429 and response.headers["Retry-After"] == "1"
    instance.clock.now += timedelta(microseconds=1)
    assert post(instance).status_code == 200
    assert len(instance.checker.calls) == 3


@pytest.mark.parametrize("case,status,code", [
    ("large", 413, "file_too_large"), ("text", 415, "not_a_pdf"),
    ("empty", 422, "empty_pdf"),
])
def test_validation_errors_do_not_consume_cap(build, case, status, code):
    empty = case == "empty"
    loader_calls = []

    def loader(path):
        loader_calls.append(path)
        if empty:
            return SimpleNamespace(page_count=0, pages=[])
        return fake_document(path)

    instance = build(daily_cap=1, document_loader=loader)
    content = b"x" * 1025 if case == "large" else b"plain text"
    if case == "empty":
        content = PDF
    response = post(instance, content)
    row = assert_error_row(instance, response, status, code, False)
    assert row["file_sha256"] == (
        None if case == "large" else hashlib.sha256(content).hexdigest()
    )
    assert not instance.checker.calls
    assert instance.store.counted_since(NOW) == 0
    assert all(not path.exists() for path in loader_calls)
    empty = False
    assert post(instance).status_code == 200


def test_busy_row_does_not_release_owners_lock(build):
    instance = build()
    instance.app.state.check_lock.acquire()
    try:
        response = post(instance)
        assert instance.app.state.check_lock.locked()
    finally:
        instance.app.state.check_lock.release()
    row = assert_error_row(instance, response, 429, "busy", False)
    assert response.headers["Retry-After"] == "10"
    assert row["file_sha256"] is None
    assert not instance.checker.calls


@pytest.mark.parametrize("error,status,code", [
    (LLMQuotaError("quota"), 503, "llm_quota_exhausted"),
    (RuntimeError("failure"), 500, "internal_error"),
])
def test_checker_errors_are_counted(build, error, status, code):
    instance = build(daily_cap=1)
    instance.checker.error = error
    response = post(instance)
    row = assert_error_row(instance, response, status, code, True)
    assert row["file_sha256"] == hashlib.sha256(PDF).hexdigest()
    assert not instance.checker.calls[0][0].exists()
    assert not instance.app.state.check_lock.locked()
    if code == "llm_quota_exhausted":
        assert response.headers["Retry-After"] == "60"
    assert post(instance).json()["detail"]["code"] == "daily_limit_reached"
    assert len(instance.checker.calls) == 1


def test_auth_failure_has_no_history(build):
    instance = build(api_key="correct")
    assert post(instance, headers={"X-API-Key": "wrong"}).status_code == 401
    assert instance.store.list_checks() == []
    assert not instance.checker.calls


@pytest.mark.parametrize("method", ["counted_since", "oldest_counted_since"])
def test_history_read_failure_does_not_write(build, monkeypatch, method):
    instance = build(daily_cap=1)
    if method == "oldest_counted_since":
        assert post(instance).status_code == 200
    before = instance.store.list_checks()
    calls = len(instance.checker.calls)

    def fail(*args):
        raise HistoryError("unavailable")

    def forbidden(*args):
        pytest.fail("Must not record after a history read failure")

    monkeypatch.setattr(instance.store, method, fail)
    monkeypatch.setattr(instance.store, "record", forbidden)
    response = post(instance)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "history_unavailable"
    assert len(instance.checker.calls) == calls
    assert instance.store.list_checks() == before
    assert not instance.app.state.check_lock.locked()


def test_counted_write_failure_latches_closed(build, monkeypatch, caplog):
    instance = build()
    original = instance.store.record

    def fail(*args):
        raise HistoryError("unavailable")

    monkeypatch.setattr(instance.store, "record", fail)
    assert post(instance).status_code == 200
    assert instance.store.list_checks() == []
    assert instance.app.state.history_failed
    assert any(record.exc_info for record in caplog.records)
    monkeypatch.setattr(instance.store, "record", original)
    response = post(instance)
    assert_error_row(instance, response, 503, "history_unavailable", False)
    assert len(instance.checker.calls) == 1


def test_runtime_write_failure_preserves_response_and_latches(build, monkeypatch):
    instance = build()

    def fail(*args):
        raise RuntimeError("unexpected history failure")

    monkeypatch.setattr(instance.store, "record", fail)
    assert post(instance).status_code == 200
    assert instance.app.state.history_failed
    assert not instance.app.state.check_lock.locked()
    assert not instance.checker.calls[0][0].exists()
    response = post(instance)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "history_unavailable"
    assert len(instance.checker.calls) == 1
    assert not instance.app.state.check_lock.locked()
    assert instance.store.list_checks() == []


def test_default_app_uses_isolated_history_database(tmp_path, monkeypatch):
    assert api.ApiSettings.from_env().db_path == tmp_path / "history.db"
    for name in ("API_KEY", "MAX_UPLOAD_MB", "MAX_PAGES"):
        monkeypatch.delenv("CONTRACTGUARD_" + name, raising=False)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(POLICY))
    checker = FakeChecker()
    app = api.create_app(
        checker=checker, policy_path=policy_path,
        document_loader=fake_document, clock=Clock(),
    )
    with TestClient(app) as client:
        response = client.post("/check", files={"file": ("vendor.pdf", PDF)})
    assert response.status_code == 200
    assert len(checker.calls) == 1
    assert (tmp_path / "history.db").is_file()
    assert len(HistoryStore(tmp_path / "history.db").list_checks()) == 1


def test_uncounted_write_failure_does_not_latch(build, monkeypatch):
    instance = build()
    original = instance.store.record

    def fail(*args):
        raise HistoryError("unavailable")

    monkeypatch.setattr(instance.store, "record", fail)
    assert post(instance, b"text").status_code == 415
    assert not instance.app.state.history_failed
    monkeypatch.setattr(instance.store, "record", original)
    assert post(instance).status_code == 200


def test_history_off(build, caplog):
    instance = build(off=True, daily_cap=1)
    assert post(instance).status_code == 200
    assert post(instance).status_code == 200
    assert instance.client.get("/health").json()["history"] is False
    assert instance.store.list_checks() == []
    assert "History disabled; no daily cap" in caplog.text


def test_history_health_and_automatic_construction(build):
    instance = build(automatic=True, daily_cap=7, history_mode="full")
    health = instance.client.get("/health").json()
    assert health["history"] is True
    assert health["daily_cap"] == 7 and health["history_mode"] == "full"
    assert post(instance).status_code == 200
    assert len(instance.store.list_checks()) == 1


def test_history_init_error_propagates(tmp_path):
    parent = tmp_path / "not-a-directory"
    parent.write_text("file")
    settings = api.ApiSettings(None, 1024, 200, tmp_path, db_path=parent / "history.db")
    with pytest.raises(HistoryError):
        api.create_app(checker=FakeChecker(), settings=settings, clock=Clock())


def test_policy_hash_is_canonical(build, tmp_path):
    compact, pretty, changed = [tmp_path / f"{name}.json" for name in (
        "compact", "pretty", "changed",
    )]
    compact.write_text(json.dumps(POLICY, separators=(",", ":")))
    pretty.write_text(json.dumps(POLICY, indent=4, sort_keys=True))
    changed.write_text(json.dumps([{**POLICY[0], "threshold": 999}, POLICY[1]]))
    hashes = []
    for path in (compact, pretty, changed):
        instance = build(policy_path=path)
        assert post(instance).status_code == 200
        hashes.append(latest(instance)["policy_hash"])
    assert hashes[0] == hashes[1]
    assert hashes[0] != hashes[2]


def test_record_written_before_lock_release(build, monkeypatch):
    instance = build()
    original = instance.store.record

    def record(check, rules):
        assert instance.app.state.check_lock.locked()
        return original(check, rules)

    monkeypatch.setattr(instance.store, "record", record)
    assert post(instance).status_code == 200
    assert not instance.app.state.check_lock.locked()
