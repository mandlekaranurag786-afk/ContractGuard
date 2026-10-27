import pytest
from pydantic import ValidationError
from test_field_extract import Fake, chunk, response

from contractguard.followup.extract import FieldExtractor
from contractguard.llm.errors import LLMUnavailableError
from scripts import smoke_followup


def test_dry_run_never_loads_or_calls(monkeypatch, capsys):
    def forbidden():
        raise AssertionError("dry run must only print schemas")

    monkeypatch.setattr(smoke_followup, "load_chunks", forbidden)
    assert smoke_followup.main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "FollowupHasDamagesExclusion" in output
    assert "LiabilityCapExtraction" in output


@pytest.mark.parametrize("payload", [response(), "invalid JSON"])
def test_smoke_allows_only_one_generation(payload):
    fake = Fake([payload])
    client = smoke_followup.OneCallClient(fake)
    extractor = FieldExtractor(client)
    if payload == "invalid JSON":
        with pytest.raises(ValidationError):
            extractor.extract(smoke_followup.CLAUSE, smoke_followup.FIELD, [chunk()])
    else:
        assert extractor.extract(
            smoke_followup.CLAUSE, smoke_followup.FIELD, [chunk()]
        ).evidence.verified
    assert len(fake.calls) == 1


def test_provider_cause_is_preserved():
    try:
        try:
            raise ValueError("full provider schema error\nsecond line")
        except ValueError as cause:
            raise LLMUnavailableError("Gemini API request failed") from cause
    except LLMUnavailableError as error:
        result = FieldExtractor(Fake([error])).extract(
            smoke_followup.CLAUSE, smoke_followup.FIELD, [chunk()]
        )
    assert "LLMUnavailableError: Gemini API request failed" in result.error
    assert (
        "builtins.ValueError: full provider schema error\nsecond line" in result.error
    )
