"""Read-only history routes use temporary storage and existing offline fakes."""

import hashlib
from datetime import timedelta

import pytest
from test_api_history_check import NOW, PDF
from test_api_history_check import build as history_build

build = history_build


@pytest.fixture
def populated(build):
    instance = build()
    for hour, name, content, status in (
        (0, "FirstVendor.pdf", PDF, 200),
        (1, "SecondVendor.pdf", PDF + b"second", 200),
        (2, "invalid.txt", b"not a pdf", 415),
    ):
        instance.clock.now = NOW + timedelta(hours=hour)
        response = instance.client.post("/check", files={"file": (name, content)})
        assert response.status_code == status
    return instance


def ids(response):
    assert response.status_code == 200
    return [row["id"] for row in response.json()["checks"]]


@pytest.mark.parametrize("path", ["/checks", "/checks/1"])
@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_auth_required(build, path, headers):
    instance = build(api_key="correct")
    response = instance.client.get(path, headers=headers)
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "unauthorized"
    allowed = instance.client.get(path, headers={"X-API-Key": "correct"})
    assert allowed.status_code == (200 if path == "/checks" else 404)


def test_empty_history(build):
    instance = build()
    response = instance.client.get("/checks")
    assert response.status_code == 200
    assert response.json() == {"checks": [], "limit": 50, "offset": 0}


def test_list_order_and_no_reports(populated):
    response = populated.client.get("/checks")
    assert ids(response) == [3, 2, 1]
    for row in response.json()["checks"]:
        assert "report_json" not in row and "report" not in row
    assert len(populated.store.list_checks()) == 3


@pytest.mark.parametrize("query,expected", [
    ({"name": "sTvEnD"}, [1]),
    ({"sha256": hashlib.sha256(PDF).hexdigest()}, [1]),
    ({"sha256": hashlib.sha256(PDF).hexdigest()[:10]}, []),
    ({"status": "PASS"}, [2, 1]),
    ({"status": "not_a_pdf"}, [3]),
    ({"date_from": (NOW + timedelta(hours=1)).isoformat()}, [3, 2]),
    ({"date_to": (NOW + timedelta(hours=1)).isoformat()}, [1]),
    ({"date_from": NOW.isoformat(),
      "date_to": (NOW + timedelta(hours=1)).isoformat()}, [1]),
    ({"name": "vendor", "status": "PASS",
      "sha256": hashlib.sha256(PDF + b"second").hexdigest()}, [2]),
])
def test_filters(populated, query, expected):
    assert ids(populated.client.get("/checks", params=query)) == expected


@pytest.mark.parametrize("field", ["date_from", "date_to"])
def test_naive_date_is_utc(populated, field):
    boundary = NOW + timedelta(hours=1)
    naive = populated.client.get("/checks", params={
        field: boundary.replace(tzinfo=None).isoformat(),
    })
    aware = populated.client.get("/checks", params={field: boundary.isoformat()})
    assert naive.status_code == aware.status_code == 200
    assert naive.json() == aware.json()


def test_pagination(populated):
    first = populated.client.get("/checks", params={"limit": 2, "offset": 0})
    second = populated.client.get("/checks", params={"limit": 2, "offset": 2})
    assert ids(first) == [3, 2]
    assert ids(second) == [1]
    assert second.json()["limit"] == 2 and second.json()["offset"] == 2


@pytest.mark.parametrize("query", [{"limit": 0}, {"limit": 201}, {"offset": -1}])
def test_invalid_pagination(build, query):
    assert build().client.get("/checks", params=query).status_code == 422


def test_success_detail(populated):
    response = populated.client.get("/checks/1")
    assert response.status_code == 200
    detail = response.json()
    assert isinstance(detail["report"], dict)
    assert detail["report"]["contract"] == "FirstVendor.pdf"
    assert "report_json" not in detail
    assert detail["rules"] == [
        {"rule_id": "cap-0", "decision": "PASS", "reason": "value_compared"},
        {"rule_id": "cap-1", "decision": "PASS", "reason": "value_compared"},
    ]


@pytest.mark.parametrize("path", ["/checks", "/checks/1"])
def test_metadata_has_no_secrets(populated, path):
    response = populated.client.get(path)
    assert response.status_code == 200
    for secret in ("SECRET_QUOTE", "SECRET_REASON", "SECRET_VALUE"):
        assert secret not in response.text


def test_error_detail(populated):
    response = populated.client.get("/checks/3")
    assert response.status_code == 200
    detail = response.json()
    assert detail["report"] is None and detail["rules"] == []
    assert detail["error_code"] == "not_a_pdf"
    assert "report_json" not in detail


def test_missing_and_invalid_id(build):
    instance = build()
    response = instance.client.get("/checks/999")
    assert response.status_code == 404
    assert response.json() == {"detail": {
        "code": "not_found", "message": "No check with this id.",
    }}
    assert instance.client.get("/checks/not-an-int").status_code == 422


@pytest.mark.parametrize("path", ["/checks", "/checks/1"])
def test_disabled_history(build, path):
    response = build(off=True).client.get(path)
    assert response.status_code == 503
    assert response.json() == {"detail": {
        "code": "history_unavailable", "message": "Check history is unavailable.",
    }}


@pytest.mark.parametrize("path,method", [
    ("/checks", "list_checks"), ("/checks/1", "get_check"),
])
def test_store_failure(build, monkeypatch, caplog, path, method):
    from contractguard.api.history import HistoryError

    instance = build()

    def fail(*args, **kwargs):
        raise HistoryError("fake storage failure")

    monkeypatch.setattr(instance.store, method, fail)
    response = instance.client.get(path)
    assert response.status_code == 503
    assert response.json() == {"detail": {
        "code": "history_unavailable", "message": "Check history is unavailable.",
    }}
    assert any(r.exc_info and r.name == "contractguard.api" for r in caplog.records)


@pytest.mark.parametrize("path", ["/checks", "/checks/1"])
def test_reads_ignore_check_lock_and_failed_write_flag(populated, path):
    populated.app.state.history_failed = True
    populated.app.state.check_lock.acquire()
    try:
        assert populated.client.get(path).status_code == 200
        assert populated.app.state.check_lock.locked()
    finally:
        populated.app.state.check_lock.release()
