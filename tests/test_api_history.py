"""SQLite history tests use temporary databases and fixed timestamps only."""

import sqlite3
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from contractguard.api.app import ApiSettings
from contractguard.api.history import (
    CheckRecord,
    HistoryError,
    HistoryStore,
    RuleRecord,
)

NOW = datetime(2026, 11, 11, 9, 30, tzinfo=timezone.utc)


def check(**changes):
    return replace(CheckRecord(
        created_at=NOW, file_name="Vendor.pdf", file_sha256="abc",
        policy_hash="policy", prompt_version="v2", followup=False,
        http_status=200, error_code=None, overall_status="PASS", counted=True,
        llm_calls=2, cache_hits=1, input_tokens=100, output_tokens=20,
        elapsed_ms=12.5, report_json='{"result": "PASS"}',
    ), **changes)


@pytest.fixture
def store(tmp_path):
    return HistoryStore(tmp_path / "nested" / "history.db")


def test_initialization_and_reopening_preserve_rows(store):
    assert store.db_path.is_file()
    connection = sqlite3.connect(store.db_path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        connection.close()
    check_id = store.record(check())
    reopened = HistoryStore(store.db_path)
    assert reopened.get_check(check_id) == store.get_check(check_id)


def test_unknown_schema_version(tmp_path):
    path = tmp_path / "future.db"
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA user_version=99")
    finally:
        connection.close()
    with pytest.raises(HistoryError, match="99"):
        HistoryStore(path)


def test_parent_is_regular_file(tmp_path):
    parent = tmp_path / "file"
    parent.write_text("not a directory")
    with pytest.raises(HistoryError) as error:
        HistoryStore(parent / "history.db")
    assert isinstance(error.value.__cause__, OSError)


def test_round_trip_with_rules(store):
    record = check(followup=True)
    rules = [RuleRecord(f"rule-{i}", "PASS", f"reason-{i}") for i in range(5)]
    check_id = store.record(record, rules)
    assert isinstance(check_id, int)
    assert store.get_check(check_id) == {
        **asdict(record), "created_at": "2026-11-11T09:30:00.000000Z",
        "id": check_id, "rules": [asdict(rule) for rule in rules],
    }
    result = store.get_check(check_id)
    assert result["followup"] is True
    assert result["counted"] is True


def test_failed_rule_insert_rolls_back_entire_transaction(store):
    with pytest.raises(HistoryError) as error:
        store.record(check(), [
            RuleRecord("valid", "PASS", None), RuleRecord("invalid", None, None),
        ])
    assert isinstance(error.value.__cause__, sqlite3.IntegrityError)
    assert store.list_checks() == []
    connection = sqlite3.connect(store.db_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM check_rules").fetchone()[0] == 0
    finally:
        connection.close()


def test_naive_created_at_rejected(store):
    with pytest.raises(ValueError, match="timezone-aware"):
        store.record(check(created_at=NOW.replace(tzinfo=None)))
    assert store.list_checks() == []


def test_counted_window_and_oldest(store):
    store.record(check(created_at=NOW - timedelta(microseconds=1)))
    store.record(check(created_at=NOW, counted=False))
    store.record(check(created_at=NOW + timedelta(hours=1)))
    store.record(check(created_at=NOW))
    assert store.counted_since(NOW) == 2
    oldest = store.oldest_counted_since(NOW)
    assert oldest == NOW
    assert oldest.tzinfo is timezone.utc
    later = NOW + timedelta(hours=2)
    assert store.counted_since(later) == 0
    assert store.oldest_counted_since(later) is None


def test_non_utc_storage_and_comparison(store):
    local = NOW.astimezone(timezone(timedelta(hours=5, minutes=30)))
    check_id = store.record(check(created_at=local))
    assert store.get_check(check_id)["created_at"] == "2026-11-11T09:30:00.000000Z"
    assert store.counted_since(local) == 1
    assert store.oldest_counted_since(local) == NOW
    assert store.counted_since(local + timedelta(microseconds=1)) == 0
    assert len(store.list_checks(date_from=local)) == 1
    assert store.list_checks(date_to=local) == []
    assert len(store.list_checks(date_to=local + timedelta(microseconds=1))) == 1


@pytest.mark.parametrize("method", [
    lambda store, value: store.counted_since(value),
    lambda store, value: store.oldest_counted_since(value),
    lambda store, value: store.list_checks(date_from=value),
    lambda store, value: store.list_checks(date_to=value),
])
def test_naive_query_times_rejected(store, method):
    with pytest.raises(ValueError, match="timezone-aware"):
        method(store, NOW.replace(tzinfo=None))


def test_list_order_and_literal_name_filters(store):
    names = ["Vendor.PDF", "100%.pdf", "under_score.pdf", "bang!.pdf", "plain.pdf"]
    ids = [store.record(check(file_name=name)) for name in names]
    assert [row["id"] for row in store.list_checks()] == ids[::-1]
    for query, expected in [
        ("vEnDoR", "Vendor.PDF"), ("%", "100%.pdf"),
        ("_", "under_score.pdf"), ("!", "bang!.pdf"),
    ]:
        assert [row["file_name"] for row in store.list_checks(name=query)] == [expected]
    assert store.list_checks(name="%' OR 1=1 --") == []


def test_exact_filters_and_date_boundaries(store):
    first = store.record(check(file_sha256="abc", overall_status="PASS"))
    second = store.record(check(
        created_at=NOW + timedelta(hours=1), file_sha256="abcd",
        overall_status=None, error_code="bad_pdf", http_status=422, counted=False,
    ))
    store.record(check(created_at=NOW + timedelta(hours=2), overall_status="FAIL"))
    assert [row["id"] for row in store.list_checks(sha256="abcd")] == [second]
    assert [row["id"] for row in store.list_checks(status="PASS")] == [first]
    assert [row["id"] for row in store.list_checks(status="bad_pdf")] == [second]
    assert store.list_checks(status="bad") == []
    assert store.list_checks(sha256="' OR 1=1 --") == []
    assert [row["id"] for row in store.list_checks(
        date_from=NOW, date_to=NOW + timedelta(hours=1),
    )] == [first]
    assert [row["id"] for row in store.list_checks(
        name="vendor", sha256="abcd", status="bad_pdf",
        date_from=NOW, date_to=NOW + timedelta(hours=2),
    )] == [second]


def test_limit_clamps_and_offset(store):
    ids = [store.record(check()) for _ in range(201)]
    assert len(store.list_checks()) == 50
    assert [row["id"] for row in store.list_checks(limit=0)] == [ids[-1]]
    assert len(store.list_checks(limit=999)) == 200
    page = store.list_checks(limit=2, offset=1)
    assert [row["id"] for row in page] == ids[-3:-1][::-1]
    assert store.list_checks(offset=-1) == store.list_checks(offset=0)
    assert store.list_checks(offset=201) == []


def test_list_omits_report_and_returns_booleans(store):
    check_id = store.record(check(counted=False))
    listed = store.list_checks()[0]
    assert "report_json" not in listed
    assert listed["followup"] is False
    assert listed["counted"] is False
    full = store.get_check(check_id)
    assert full["report_json"] == check().report_json
    assert full["rules"] == []
    assert set(listed) == set(full) - {"report_json", "rules"}


def test_missing_check(store):
    assert store.get_check(999) is None


def test_settings_defaults_and_old_constructor(tmp_path, monkeypatch):
    for name in ("CONTRACTGUARD_DB_PATH", "CONTRACTGUARD_DAILY_CAP",
                 "CONTRACTGUARD_HISTORY_MODE"):
        monkeypatch.delenv(name, raising=False)
    old = ApiSettings(None, 1024, 200, tmp_path)
    for settings in (old, ApiSettings.from_env()):
        assert settings.db_path == (
            None if settings is old else Path(".cache/history.db")
        )
        assert settings.daily_cap == 30
        assert settings.history_mode == "metadata"


def test_settings_from_env(tmp_path, monkeypatch):
    path = tmp_path / "custom.db"
    monkeypatch.setenv("CONTRACTGUARD_DB_PATH", str(path))
    monkeypatch.setenv("CONTRACTGUARD_DAILY_CAP", "7")
    monkeypatch.setenv("CONTRACTGUARD_HISTORY_MODE", "full")
    settings = ApiSettings.from_env()
    assert settings.db_path == path
    assert settings.daily_cap == 7
    assert settings.history_mode == "full"


@pytest.mark.parametrize("changes", [{"daily_cap": 0}, {"history_mode": "other"}])
def test_invalid_settings(tmp_path, changes):
    with pytest.raises(ValueError):
        ApiSettings(None, 1024, 200, tmp_path, **changes)


@pytest.mark.parametrize("name,value", [
    ("CONTRACTGUARD_DAILY_CAP", "0"), ("CONTRACTGUARD_HISTORY_MODE", "other"),
])
def test_invalid_settings_from_env(monkeypatch, name, value):
    monkeypatch.setenv("CONTRACTGUARD_DAILY_CAP", "30")
    monkeypatch.setenv("CONTRACTGUARD_HISTORY_MODE", "metadata")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        ApiSettings.from_env()
