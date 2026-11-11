import pytest


@pytest.fixture(autouse=True)
def isolated_history_database(monkeypatch, tmp_path):
    """Tests must never write to the real history database."""
    monkeypatch.setenv("CONTRACTGUARD_DB_PATH", str(tmp_path / "history.db"))
    monkeypatch.delenv("CONTRACTGUARD_DAILY_CAP", raising=False)
    monkeypatch.delenv("CONTRACTGUARD_HISTORY_MODE", raising=False)
