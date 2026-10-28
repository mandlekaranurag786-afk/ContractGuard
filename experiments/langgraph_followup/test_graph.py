"""Replay the actual product scenarios with both controllers and compare all output."""

import inspect
import sys
from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("langgraph.graph")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

import test_followup_loop as scenarios

from contractguard import checker
from contractguard.followup.loop import run_followup
from experiments.langgraph_followup.graph import run_followup_graph


@pytest.fixture(autouse=True, params=["loop", "graph"])
def controller(request, monkeypatch):
    primary, reference = (
        (run_followup, run_followup_graph)
        if request.param == "loop"
        else (run_followup_graph, run_followup)
    )

    def compared(*args, **kwargs):
        other_args, other_kwargs = deepcopy((args, kwargs))
        actual = primary(*args, **kwargs)
        expected = reference(*other_args, **other_kwargs)
        assert actual.results == expected.results
        assert actual.report == expected.report
        assert actual.cost == expected.cost
        actual_inputs = inspect.signature(run_followup).bind(*args, **kwargs).arguments
        other_inputs = (
            inspect.signature(run_followup).bind(*other_args, **other_kwargs).arguments
        )
        for dependency in ("client", "retriever"):
            assert actual_inputs[dependency].calls == other_inputs[dependency].calls
        return actual

    monkeypatch.setattr(scenarios, "run_followup", compared)
    monkeypatch.setattr(checker, "run_followup", compared)
    return compared


# Reuse the test functions and their original parametrization, without copying
# scenarios or assertions. Their module globals are patched by the fixture above.
for _name, _test in vars(scenarios).items():
    if _name.startswith("test_") and inspect.isfunction(_test):
        globals()[_name] = _test


def test_six_targets_budget_and_recursion(controller):
    fields = (
        "cap_amount",
        "has_money_cap",
        "has_damages_exclusion",
        "initial_term_months",
        "notice_days",
        "auto_renews",
    )
    client = scenarios.Client(
        [dict(value=None, evidence_quote=None, evidence_chunk_id=None, currency=None)]
        * 6
    )
    result, _ = scenarios.run(client, fields, version="v2")
    assert len(client.calls) == result.report.calls_used == 4
    assert len(result.report.steps) == 5
    assert result.report.steps[-1].note == "budget_exhausted"


def test_six_targets_all_processed_with_larger_test_budget(controller):
    # Six full iterations exceed LangGraph's historical default of 25 supersteps.
    fields = (
        "cap_amount",
        "has_money_cap",
        "has_damages_exclusion",
        "initial_term_months",
        "notice_days",
        "auto_renews",
    )
    client = scenarios.Client(
        [dict(value=None, evidence_quote=None, evidence_chunk_id=None, currency=None)]
        * 6
    )
    result, _ = scenarios.run(client, fields, version="v2", max_calls=6)
    assert len(result.report.steps) == result.report.calls_used == 6


def test_identical_public_signature():
    assert inspect.signature(run_followup_graph) == inspect.signature(run_followup)


def test_state_is_data_only(monkeypatch):
    from functools import wraps

    from langgraph.graph import StateGraph
    from pydantic import TypeAdapter

    from experiments.langgraph_followup.graph import GraphState

    add = StateGraph.add_node
    adapter = TypeAdapter(GraphState)
    visited = []

    def checked_add(graph, name, node, **kwargs):
        @wraps(node)
        def checked(state):
            assert set(state) == set(GraphState.__annotations__)
            adapter.dump_json(state, warnings="error")
            visited.append(name)
            return node(state)

        return add(graph, name, checked, **kwargs)

    monkeypatch.setattr(StateGraph, "add_node", checked_add)
    scenarios.run(scenarios.Client(), version="v2")
    assert set(visited) == {
        "select_targets",
        "next_target",
        "search",
        "extract",
        "merge",
        "reevaluate",
    }
