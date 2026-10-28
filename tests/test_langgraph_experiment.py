"""Optional experiment bridge: pyproject's testpaths contains only tests/."""

import pytest

pytest.importorskip("langgraph.graph")

from experiments.langgraph_followup import test_graph as suite

controller = suite.controller
for _name, _test in vars(suite).items():
    if _name.startswith("test_"):
        globals()[_name] = _test
