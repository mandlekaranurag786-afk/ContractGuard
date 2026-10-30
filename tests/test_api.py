"""API tests use local fakes; no provider calls or embedding model loading."""

from dataclasses import replace
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from contractguard.api import app as api
from contractguard.ingestion import ParsedDocument
from contractguard.llm.errors import LLMQuotaError
from contractguard.report import ContractReport, Cost, OverallStatus
from contractguard.rules.models import TrustPolicy, load_rules

POLICY = Path("policies/eval_policy.json")


def pdf_bytes(text_pages=(True,), encrypted=False):
    with pymupdf.open() as document:
        for text in text_pages:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), "This is a tiny contract with a text layer.")
        if encrypted:
            return document.tobytes(
                encryption=pymupdf.PDF_ENCRYPT_AES_256,
                owner_pw="owner", user_pw="password",
            )
        return document.tobytes()


@pytest.fixture
def temp_paths(monkeypatch):
    paths = []
    original = api.NamedTemporaryFile

    def tracked(*args, **kwargs):
        temporary = original(*args, **kwargs)
        paths.append(Path(temporary.name))
        return temporary

    monkeypatch.setattr(api, "NamedTemporaryFile", tracked)
    yield paths
    assert all(not path.exists() for path in paths)


class FakeChecker:
    def __init__(self, error=None):
        self.calls = []
        self.error = error
        self.report = None

    def check(
        self, pdf_path, rules, policy_path, followup=False, followup_version="v1"
    ):
        assert pdf_path.is_file()
        assert pdf_path.suffix == ".pdf"
        self.calls.append(
            (
                pdf_path, rules, policy_path, followup, followup_version,
                pdf_path.read_bytes(),
            )
        )
        if self.error:
            raise self.error
        status = OverallStatus.NO_APPLICABLE_RULES
        self.report = ContractReport(
            contract=str(pdf_path),
            policy_path=str(policy_path),
            trust_policy=TrustPolicy.KEYWORD,
            model="fake",
            prompt_version="v2",
            overall_status=status,
            overall_reason=status.reason,
            counts={},
            rule_results=[],
            clauses=[],
            cost=Cost(
                llm_calls=0, cache_hits=0, input_tokens=0, output_tokens=0,
                cached_input_tokens=0, cached_output_tokens=0, elapsed_ms=0,
            ),
        )
        return self.report


@pytest.fixture(autouse=True)
def no_real_services(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("CONTRACTGUARD_POLICY", raising=False)
    monkeypatch.delenv("CONTRACTGUARD_CACHE_DIR", raising=False)
    for name in ("API_KEY", "MAX_UPLOAD_MB", "MAX_PAGES"):
        monkeypatch.delenv("CONTRACTGUARD_" + name, raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Real service initialization is forbidden in API tests")

    monkeypatch.setattr(api, "GeminiClient", forbidden)
    monkeypatch.setattr(api, "warm_up_model", forbidden)


def test_health_without_api_key():
    with TestClient(api.create_app(checker=FakeChecker())) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok", "prompt_version": "v2", "rules": len(load_rules(POLICY)),
        "auth": False, "max_upload_mb": 10, "max_pages": 200,
    }


def test_policy():
    with TestClient(api.create_app(checker=FakeChecker())) as client:
        response = client.get("/policy")
    assert response.status_code == 200
    assert response.json()["policy"] == POLICY.name
    assert [r["id"] for r in response.json()["rules"]] == [
        r.id for r in load_rules(POLICY)
    ]


@pytest.mark.parametrize("filename", ["../vendor.pdf", r"C:\uploads\vendor.pdf"])
def test_check_default_cleans_temp_and_redacts_path(filename):
    checker = FakeChecker()
    data = pdf_bytes()
    with TestClient(api.create_app(checker=checker)) as client:
        response = client.post("/check", files={"file": (filename, data)})
    assert response.status_code == 200
    report = ContractReport.model_validate(response.json())
    path, rules, policy, followup, version, content = checker.calls[0]
    assert len(checker.calls) == 1
    assert rules == load_rules(POLICY) and policy == POLICY
    assert followup is False and version == "v1"
    assert content == data
    assert not path.exists()
    assert report.contract == "vendor.pdf"
    assert str(path) not in response.text
    assert checker.report.contract == str(path)


def test_check_followup_v2():
    checker = FakeChecker()
    with TestClient(api.create_app(checker=checker)) as client:
        response = client.post(
            "/check", files={"file": ("vendor.pdf", pdf_bytes())},
            data={"followup": "true"},
        )
    assert response.status_code == 200
    assert checker.calls[0][3:5] == (True, "v2")
    assert not checker.calls[0][0].exists()


def test_temp_deleted_on_checker_error():
    checker = FakeChecker(RuntimeError("fake failure"))
    with TestClient(api.create_app(checker=checker)) as client:
        response = client.post("/check", files={"file": ("vendor.pdf", pdf_bytes())})
        assert response.status_code == 500
    assert not checker.calls[0][0].exists()


def test_policy_loaded_once_and_explicit_path_overrides_env(tmp_path, monkeypatch):
    path = tmp_path / "policy.json"
    path.write_text(POLICY.read_text())
    monkeypatch.setenv("CONTRACTGUARD_POLICY", str(tmp_path / "missing.json"))
    app = api.create_app(checker=FakeChecker(), policy_path=path)
    path.write_text("[]")
    with TestClient(app) as client:
        assert client.get("/policy").json()["rules"]


def test_invalid_env_policy_rejected_at_creation(tmp_path, monkeypatch):
    path = tmp_path / "bad.json"
    path.write_text("[]")
    monkeypatch.setenv("CONTRACTGUARD_POLICY", str(path))
    with pytest.raises(ValueError, match="empty"):
        api.create_app(checker=FakeChecker())


@pytest.mark.parametrize("cache_dir", [None, ".cache/custom"])
def test_real_checker_wiring_is_deferred_and_uses_api_cache(monkeypatch, cache_dir):
    events = []
    inner = type("FakeClient", (), {"model": "fake"})()
    monkeypatch.setattr(api, "GeminiClient", lambda: events.append("client") or inner)
    monkeypatch.setattr(api, "warm_up_model", lambda: events.append("warm"))
    if cache_dir:
        monkeypatch.setenv("CONTRACTGUARD_CACHE_DIR", cache_dir)
    app = api.create_app()
    assert events == []
    with TestClient(app):
        assert events == ["client", "warm"]
        extractor = app.state.checker.extractor
        assert extractor.prompt_version == "v2"
        assert extractor.client._inner is inner
        assert extractor.client._cache_dir == Path(cache_dir or ".cache/api_llm_cache")


def assert_error(response, status, code, app, paths):
    assert response.status_code == status
    assert set(response.json()) == {"detail"}
    assert response.json()["detail"]["code"] == code
    assert isinstance(response.json()["detail"]["message"], str)
    assert response.json()["detail"]["message"]
    assert "secret" not in response.text and "/tmp/" not in response.text
    assert all(str(path) not in response.text and not path.exists() for path in paths)
    assert app.state.check_lock.acquire(blocking=False)
    app.state.check_lock.release()


@pytest.mark.parametrize("header", [None, "wrong"])
def test_auth_required(header, temp_paths):
    settings = replace(api.ApiSettings.from_env(), api_key="correct")
    app = api.create_app(checker=FakeChecker(), settings=settings)
    headers = {} if header is None else {"X-API-Key": header}
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["auth"] is True
        assert_error(client.get("/policy", headers=headers),
                     401, "unauthorized", app, temp_paths)
        assert_error(client.post("/check", headers=headers,
                                 files={"file": ("a.pdf", pdf_bytes())}),
                     401, "unauthorized", app, temp_paths)
        assert not temp_paths
        response = client.get("/policy", headers={"X-API-Key": "correct"})
        assert response.status_code == 200
        assert client.post("/check", headers={"X-API-Key": "correct"},
                           files={"file": ("a.pdf", pdf_bytes())}).status_code == 200


def test_auth_disabled_warning_once(caplog):
    app = api.create_app(checker=FakeChecker())
    with TestClient(app) as client:
        assert client.get("/health").json()["auth"] is False
        assert client.get("/policy").status_code == 200
        response = client.post("/check", files={"file": ("a.pdf", pdf_bytes())})
        assert response.status_code == 200
    assert sum(r.name == "contractguard.api" and
               r.message == "API key not set; auth disabled"
               for r in caplog.records) == 1


@pytest.mark.parametrize(
    "case,status,code",
    [
        ("large", 413, "file_too_large"),
        ("text", 415, "not_a_pdf"),
        ("empty_file", 415, "not_a_pdf"),
        ("corrupt", 422, "unreadable_pdf"),
        ("encrypted", 422, "unreadable_pdf"),
        ("zero_pages", 422, "empty_pdf"),
        ("many_pages", 422, "too_many_pages"),
        ("scanned", 422, "scanned_pdf"),
        ("quota", 503, "llm_quota_exhausted"),
        ("internal", 500, "internal_error"),
    ],
)
def test_errors_cleanup_and_release(case, status, code, temp_paths, caplog):
    data = pdf_bytes()
    settings = api.ApiSettings.from_env()
    loader = api.load_document
    checker = FakeChecker()
    if case == "large":
        settings = replace(settings, max_upload_bytes=10)
    elif case == "text":
        data = b"plain text"
    elif case == "empty_file":
        data = b""
    elif case == "corrupt":
        data = b"%PDF-garbage"
    elif case == "encrypted":
        data = pdf_bytes(encrypted=True)
    elif case == "zero_pages":
        def loader(path):
            return ParsedDocument(doc_id="empty", original_filename=path.name,
                                  page_count=0, pages=[])
    elif case == "many_pages":
        settings = replace(settings, max_pages=2)
        data = pdf_bytes((True, True, True))
    elif case == "scanned":
        data = pdf_bytes((False, False))
    elif case == "quota":
        checker.error = LLMQuotaError("secret /tmp/x")
    elif case == "internal":
        checker.error = RuntimeError("secret /tmp/x")
    app = api.create_app(checker=checker, settings=settings, document_loader=loader)
    with TestClient(app) as client:
        response = client.post("/check", files={"file": ("a.pdf", data)})
        assert_error(response, status, code, app, temp_paths)
    assert len(temp_paths) == 1
    if case == "quota":
        assert response.headers["Retry-After"] == "60"
    if case == "internal":
        assert any(r.exc_info and r.name == "contractguard.api" for r in caplog.records)
    if case not in {"quota", "internal"}:
        assert not checker.calls


@pytest.mark.parametrize("case", ["prefix", "partly_scanned"])
def test_accepted_pdf_variants(case, temp_paths):
    data = b"junk\n" + pdf_bytes() if case == "prefix" else pdf_bytes((True, False))
    app = api.create_app(checker=FakeChecker())
    with TestClient(app) as client:
        response = client.post("/check", files={"file": ("a.pdf", data)})
    assert response.status_code == 200
    assert all(str(path) not in response.text for path in temp_paths)
    assert not app.state.check_lock.locked()


def test_busy_then_success(temp_paths):
    app = api.create_app(checker=FakeChecker())
    with TestClient(app) as client:
        app.state.check_lock.acquire()
        try:
            response = client.post("/check", files={"file": ("a.pdf", pdf_bytes())})
            assert app.state.check_lock.locked()
            assert not temp_paths
        finally:
            app.state.check_lock.release()
        assert_error(response, 429, "busy", app, temp_paths)
        assert response.headers["Retry-After"] == "10"
        response = client.post("/check", files={"file": ("a.pdf", pdf_bytes())})
        assert response.status_code == 200


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("CONTRACTGUARD_API_KEY", "")
    monkeypatch.setenv("CONTRACTGUARD_MAX_UPLOAD_MB", "0.5")
    monkeypatch.setenv("CONTRACTGUARD_MAX_PAGES", "3")
    monkeypatch.setenv("CONTRACTGUARD_CACHE_DIR", ".cache/test")
    settings = api.ApiSettings.from_env()
    assert settings == api.ApiSettings(None, 524288, 3, Path(".cache/test"))
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        settings.max_pages = 4
