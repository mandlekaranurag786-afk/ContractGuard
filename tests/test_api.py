"""API tests use local fakes; no provider calls or embedding model loading."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from contractguard.api import app as api
from contractguard.report import ContractReport, Cost, OverallStatus
from contractguard.rules.models import TrustPolicy, load_rules

POLICY = Path("policies/eval_policy.json")


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

    def forbidden(*args, **kwargs):
        pytest.fail("Real service initialization is forbidden in API tests")

    monkeypatch.setattr(api, "GeminiClient", forbidden)
    monkeypatch.setattr(api, "warm_up_model", forbidden)


def test_health_without_api_key():
    with TestClient(api.create_app(checker=FakeChecker())) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok", "prompt_version": "v2", "rules": len(load_rules(POLICY))
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
    with TestClient(api.create_app(checker=checker)) as client:
        response = client.post("/check", files={"file": (filename, b"fake PDF")})
    assert response.status_code == 200
    report = ContractReport.model_validate(response.json())
    path, rules, policy, followup, version, content = checker.calls[0]
    assert len(checker.calls) == 1
    assert rules == load_rules(POLICY) and policy == POLICY
    assert followup is False and version == "v1"
    assert content == b"fake PDF"
    assert not path.exists()
    assert report.contract == "vendor.pdf"
    assert str(path) not in response.text
    assert checker.report.contract == str(path)


def test_check_followup_v2():
    checker = FakeChecker()
    with TestClient(api.create_app(checker=checker)) as client:
        response = client.post(
            "/check", files={"file": ("vendor.pdf", b"PDF")}, data={"followup": "true"}
        )
    assert response.status_code == 200
    assert checker.calls[0][3:5] == (True, "v2")
    assert not checker.calls[0][0].exists()


def test_temp_deleted_on_checker_error():
    checker = FakeChecker(RuntimeError("fake failure"))
    with TestClient(api.create_app(checker=checker)) as client:
        with pytest.raises(RuntimeError, match="fake failure"):
            client.post("/check", files={"file": ("vendor.pdf", b"PDF")})
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
