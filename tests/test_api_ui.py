"""Demo page delivery and security checks; no browser or live model calls."""

import logging
from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient
from test_api import FakeChecker

from contractguard.api import app as api

HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; connect-src 'self'; "
        "script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}
ERROR_CODES = (
    "unauthorized", "busy", "daily_limit_reached", "file_too_large", "not_a_pdf",
    "unreadable_pdf", "empty_pdf", "too_many_pages", "scanned_pdf",
    "llm_quota_exhausted", "history_unavailable", "internal_error",
)


class PageParser(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.elements = {}
        self.tags = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append(tag)
        if "id" in attrs:
            self.elements[attrs["id"]] = (tag, attrs)


@pytest.fixture
def client(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Real service initialization is forbidden")

    monkeypatch.setattr(api, "GeminiClient", forbidden)
    monkeypatch.setattr(api, "warm_up_model", forbidden)
    settings = api.ApiSettings("correct", 1024, 200, tmp_path / "cache")
    app = api.create_app(checker=FakeChecker(), settings=settings,
                         policy_path="policies/eval_policy.json")
    with TestClient(app) as test_client:
        yield test_client


def test_root_is_public_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "/" not in client.get("/openapi.json").json()["paths"]


def test_root_without_configured_key(tmp_path):
    settings = api.ApiSettings(None, 1024, 200, tmp_path / "cache")
    with TestClient(api.create_app(checker=FakeChecker(), settings=settings,
                                  policy_path="policies/eval_policy.json")) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_security_headers(client):
    response = client.get("/")
    for name, value in HEADERS.items():
        assert response.headers[name] == value


def test_page_elements_and_password(client):
    parser = PageParser(client.get("/").text)
    assert {
        "warning", "limits", "auth-notice", "file", "followup", "check-button",
        "status", "error", "result", "tab-history", "history-panel",
    } <= parser.elements.keys()
    assert parser.tags.count("style") == parser.tags.count("script") == 1


def test_warning_and_error_codes(client):
    html = client.get("/").text
    assert "Google Gemini" in html
    assert "Do not upload real or confidential contracts" in html
    for removed in ("X-API-Key", "api-key", "Enter your API key"):
        assert removed not in html
    assert "This server has API-key authentication turned on." in html
    for code in ERROR_CODES:
        assert code in html


def test_no_external_resources_or_unsafe_apis(client):
    html = client.get("/").text
    for forbidden in (
        "http://", "https://", "innerHTML", "outerHTML", "insertAdjacentHTML",
        "document.write", "eval(", "console.", "localStorage", "sessionStorage",
        "document.cookie",
    ):
        assert forbidden not in html


def test_wrong_key_not_in_responses_or_logs(client, caplog):
    caplog.set_level(logging.DEBUG)
    key = "WRONG-KEY-12345"
    headers = {"X-API-Key": key}
    responses = [
        client.post("/check", headers=headers,
                    files={"file": ("demo.pdf", b"%PDF-fake")}),
        client.get("/checks", headers=headers),
    ]
    for response in responses:
        assert response.status_code == 401
        assert key not in response.text
    assert key not in caplog.text
    assert all(key not in str(record.__dict__) for record in caplog.records)


def test_existing_health_and_policy(client):
    assert client.get("/health").json()["max_pages"] == 200
    assert client.get("/policy", headers={"X-API-Key": "correct"}).json()["rules"]


def test_model_found_column_text(client):
    html = client.get("/").text
    for expected in (
        "Model found", "formatValue", "nothing found",
        "not proven by the quote", "absence is not proven",
    ):
        assert expected in html


def test_model_found_preserves_page_security(client):
    html = client.get("/").text
    for forbidden in (
        "innerHTML", "outerHTML", "insertAdjacentHTML", "document.write",
        "eval(", "console.", "localStorage", "sessionStorage", "document.cookie",
        "http://", "https://",
    ):
        assert forbidden not in html


def test_model_found_has_no_suggested_decision(client):
    assert "Suggested" not in client.get("/").text
