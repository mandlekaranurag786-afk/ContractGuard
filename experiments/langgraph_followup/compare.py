"""Offline equivalence and engineering measurements; never constructs a live LLM."""

from __future__ import annotations

import ast
import math
import os
import statistics
import subprocess
import sys
import tomllib
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contractguard.extraction.confidence import compute_confidence
from contractguard.followup.loop import run_followup
from contractguard.rules.models import load_rules
from evals.run_checker_batch import POLICY, build_checker, selected_contracts
from experiments.langgraph_followup.graph import run_followup_graph

ROOT = Path(__file__).resolve().parents[2]


def differences(left, right, path=""):
    """Every differing leaf, including missing entries and sequence order."""
    if isinstance(left, dict) and isinstance(right, dict):
        output = []
        for key in sorted(left.keys() | right.keys()):
            child = f"{path}.{key}"
            if key not in left or key not in right:
                output.append(f"{child}: missing from one implementation")
            else:
                output.extend(differences(left[key], right[key], child))
        return output
    if isinstance(left, list) and isinstance(right, list):
        output = []
        if len(left) != len(right):
            output.append(f"{path}.length: loop={len(left)} graph={len(right)}")
        for i, (a, b) in enumerate(zip(left, right)):
            output.extend(differences(a, b, f"{path}[{i}]"))
        return output
    return [] if left == right else [f"{path}: loop={left!r} graph={right!r}"]


def output_data(result):
    return dict(
        results=[r.model_dump(mode="json") for r in result.results],
        report=result.report.model_dump(mode="json"),
        cost=result.cost,
    )


def line_counts(path, entry):
    source = path.read_text(encoding="utf-8")
    selected = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and (
            node.name == entry or node.name.startswith("_")
        ):
            selected.update(range(node.lineno, node.end_lineno + 1))
    lines = source.splitlines()
    nonblank = {
        i
        for i, line in enumerate(lines, 1)
        if line.strip() and not line.lstrip().startswith("#")
    }
    return len(selected & nonblank), len(nonblank)


def exclusive_packages():
    """All-platform lock reachability; preserve version forks and activated extras."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages = lock["package"]
    root = next(p for p in packages if p["name"] == "contractguard")

    def closure(dependencies):
        pending, visited, identities = list(dependencies), set(), set()
        while pending:
            dep = pending.pop()
            matches = [
                p
                for p in packages
                if p["name"] == dep["name"]
                and ("version" not in dep or p["version"] == dep["version"])
                and ("source" not in dep or p["source"] == dep["source"])
            ]
            if not matches:
                raise ValueError(f"Unresolved lock dependency: {dep}")
            for package in matches:
                identity = (package["name"], package["version"])
                visit = (*identity, tuple(dep.get("extra", [])))
                if visit in visited:
                    continue
                visited.add(visit)
                identities.add(identity)
                pending.extend(package.get("dependencies", []))
                for extra in dep.get("extra", []):
                    pending.extend(package["optional-dependencies"][extra])
        return identities

    baseline = list(root.get("dependencies", []))
    for group, deps in root.get("dev-dependencies", {}).items():
        if group != "langgraph-exp":
            baseline.extend(deps)
    experimental = root["dev-dependencies"]["langgraph-exp"]
    return sorted(closure(experimental) - closure(baseline))


def test_counts():
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "tests/test_followup_loop.py",
        "tests/test_langgraph_experiment.py",
    ]
    result = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=True
    )
    counts = Counter()
    for line in result.stdout.splitlines():
        if line.startswith("tests/test_followup_loop.py::"):
            counts["product"] += 1
        elif line.startswith("tests/test_langgraph_experiment.py::"):
            parameter = line.rsplit("[", 1)[-1]
            for implementation in ("loop", "graph"):
                if parameter.startswith(implementation):
                    counts[implementation] += 1
    if not all(counts[name] for name in ("product", "loop", "graph")):
        raise ValueError("Could not reliably count collected controller tests")
    return counts


def main():
    output = Path(__file__).with_name("comparison.txt")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    rules = load_rules(POLICY)
    paths = selected_contracts()
    checker = build_checker(live_followup=False)
    inner_extractor, inner_builder = checker.extractor, checker.retriever_builder
    captured = {}

    class CaptureExtractor:
        prompt_version = inner_extractor.prompt_version

        def extract(self, clause, chunks):
            result = inner_extractor.extract(clause, chunks)
            captured["results"][clause] = result
            captured["chunks"][clause] = list(chunks)
            return result

    def capture_retriever(chunks):
        index = inner_builder(chunks)
        captured["retriever"] = index
        return index

    checker.extractor = CaptureExtractor()
    checker.retriever_builder = capture_retriever
    timings, diffs, rows, steps = {"loop": [], "graph": []}, [], 0, 0
    cached, live = Counter(), Counter()
    lines = [
        f"LangGraph: {version('langgraph')}",
        "Version: v2; main and follow-up clients: cache-only (miss raises).",
        "contract | rule rows | steps | calls | differences | loop ms | graph ms",
    ]
    for i, path in enumerate(paths):
        print(f"[{i + 1}/{len(paths)}] {path.name}", flush=True)
        captured.update(results={}, chunks={})
        main_report = checker.check(path, rules, POLICY)
        assert main_report.cost.llm_calls == 0
        cached["main"] += main_report.cost.cache_hits
        results = captured["results"]
        confidences = {
            clause: compute_confidence(
                None if r.error else r.extraction,
                r.checks,
                clause,
            )
            for clause, r in results.items()
        }
        args = (
            rules,
            results,
            confidences,
            main_report.rule_results,
            captured["retriever"],
            {
                c: {x.chunk_id for x in chunks}
                for c, chunks in captured["chunks"].items()
            },
            checker.followup_client,
            checker.policy,
        )
        functions = [("loop", run_followup), ("graph", run_followup_graph)]
        # Alternate order to reduce systematic warm-cache timing advantage.
        if i % 2:
            functions.reverse()
        outcomes, elapsed = {}, {}
        for name, controller in functions:
            start = perf_counter()
            outcomes[name] = controller(
                *args, version="v2", original_chunks=captured["chunks"]
            )
            elapsed[name] = (perf_counter() - start) * 1000
            timings[name].append(elapsed[name])
            cached[name] += outcomes[name].cost["cache_hits"]
            live[name] += outcomes[name].cost["llm_calls"]
            assert outcomes[name].cost["llm_calls"] == 0
        changed = differences(
            output_data(outcomes["loop"]), output_data(outcomes["graph"]), path.name
        )
        diffs.extend(changed)
        count = len(outcomes["loop"].results)
        trace_count = len(outcomes["loop"].report.steps)
        rows += count
        steps += trace_count
        lines.append(
            f"{path.name} | {count} | {trace_count} | "
            f"{outcomes['loop'].report.calls_used} | {len(changed)} | "
            f"{elapsed['loop']:.3f} | {elapsed['graph']:.3f}"
        )
    lines += [
        f"Contracts compared: {len(paths)}",
        f"Rule rows compared: {rows}",
        f"Trace steps compared: {steps}",
        "Comparison: complete results, reports (ordered steps, outcomes, changes, "
        "status, calls, metadata), and cost.",
        f"Differences: {len(diffs)}",
        *diffs,
        "Equivalence: " + ("FAIL" if diffs else "PASS"),
        f"Live LLM calls: main=0, loop={live['loop']}, graph={live['graph']}",
        f"Cache hits: main={cached['main']}, loop={cached['loop']}, "
        f"graph={cached['graph']}",
    ]
    for name, path, entry in (
        ("loop", ROOT / "src/contractguard/followup/loop.py", "run_followup"),
        ("graph", Path(__file__).with_name("graph.py"), "run_followup_graph"),
    ):
        controller, whole = line_counts(path, entry)
        times = sorted(timings[name])
        lines += [
            f"{name} nonblank/noncomment controller lines: {controller}; "
            f"whole module: {whole}",
            f"{name} controller time ms: median={statistics.median(times):.3f}, "
            f"p95={times[math.ceil(0.95 * len(times)) - 1]:.3f}",
        ]
    lines += [
        "Line count: entry function and top-level private helpers, including "
        "docstrings; loop includes _currency_merge. Shared tool classes excluded.",
        "Timing: one run/controller/contract, alternating order; includes query "
        "retrieval, cache reads, validation, and graph build/compile/invoke. "
        "Excludes main extraction and document/index preparation; p95 nearest rank.",
    ]
    counts = test_counts()
    lines += [
        f"Original loop scenario test cases: {counts['product']}",
        f"Experiment test cases: loop-primary={counts['loop']}, "
        f"graph-primary={counts['graph']} (each compares both implementations)",
        "recursion_limit = 4 * len(original) + 8: four nodes per unique target "
        "plus selection, final routing, termination and headroom.",
    ]
    try:
        packages = exclusive_packages()
    except (KeyError, ValueError) as exc:
        lines.append(
            f"Exclusive langgraph-exp packages: cannot determine reliably: {exc}"
        )
    else:
        lines += [
            f"Exclusive langgraph-exp locked packages: {len(packages)}",
            "Method: all-platform lock dependency reachability, including version "
            "forks and activated extras, minus production and all other groups; "
            "counts (name, version), not current-platform installed distributions.",
            ", ".join(f"{name}=={v}" for name, v in packages),
        ]
    text = "\n".join(lines) + "\n"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(text)
    print(text, end="")
    return bool(diffs)


if __name__ == "__main__":
    raise SystemExit(main())
