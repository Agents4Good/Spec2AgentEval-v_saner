"""Trace-based evaluation of the Academic Triage Agent (spec 014).

Oracle: the behavioral graph in spec.md.
  - B1 (Exact Sciences/Technology papers) -> exactly one call to search_arxiv
  - B2 (Life Sciences/Health papers)      -> exactly one call to search_pubmed
  - B3 (single concept explanation)       -> exactly one call to get_wikipedia_summary
  - B4 (empty input), B5 (multi-domain), B6 (out of scope) -> no tool call

Only the trace returned by the benchmark wrapper is used as evidence.
"""

import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from trace_capture import traced
from agent import agent

T1, T2, T3 = "search_arxiv", "search_pubmed", "get_wikipedia_summary"
ALL_TOOLS = {T1, T2, T3}

TOOL_SCENARIOS = {
    "B1": ("Find academic papers about transformer architectures in artificial intelligence.", T1),
    "B2": ("Find research papers about immunotherapy treatments for melanoma.", T2),
    "B3": ("Explain the concept of photosynthesis.", T3),
}

NO_TOOL_SCENARIOS = {
    "B4_empty": "",
    "B4_whitespace": "   \n\t ",
    "B5_multi_domain": "Find papers about artificial intelligence and papers about melanoma.",
    "B6_out_of_scope": "Book me a flight to Paris next Friday.",
}

_cache = {}


def run(key, request):
    if key not in _cache:
        _cache[key] = traced(agent)(request)
    return _cache[key]


def assert_trace_valid(trace):
    assert isinstance(trace, dict), "Trace not available"
    for field in ("status", "trajectory", "tool_calls"):
        assert field in trace, f"Trace missing '{field}': insufficient evidence"
    assert isinstance(trace["tool_calls"], list)
    assert isinstance(trace["trajectory"], list)


def called_names(trace):
    return [c.get("name") for c in trace["tool_calls"]]


# ---------------------------------------------------------------- B1/B2/B3


@pytest.mark.parametrize("key", list(TOOL_SCENARIOS))
def test_agent_executes_without_raising(key):
    request, _ = TOOL_SCENARIOS[key]
    _, trace = run(key, request)
    assert_trace_valid(trace)
    assert trace["status"] == "ok", f"Agent raised: {trace.get('error')}"


@pytest.mark.parametrize("key", list(TOOL_SCENARIOS))
def test_exactly_one_tool_call_of_expected_tool(key):
    """Guardrail (2)/(3) + B1-B3 constraints: only the mapped tool, exactly once."""
    request, expected = TOOL_SCENARIOS[key]
    _, trace = run(key, request)
    assert_trace_valid(trace)
    names = called_names(trace)
    assert names == [expected], f"{key}: expected exactly [{expected}], got {names}"


@pytest.mark.parametrize("key", list(TOOL_SCENARIOS))
def test_tool_correctness_deepeval(key):
    """Same requirement expressed with DeepEval's ToolCorrectnessMetric."""
    request, expected = TOOL_SCENARIOS[key]
    output, trace = run(key, request)
    assert_trace_valid(trace)
    test_case = LLMTestCase(
        input=request,
        actual_output=output if isinstance(output, str) else str(output),
        tools_called=[
            ToolCall(name=c.get("name"), input_parameters=c.get("input") or {})
            for c in trace["tool_calls"]
        ],
        expected_tools=[ToolCall(name=expected)],
    )
    metric = ToolCorrectnessMetric(threshold=1.0, should_exact_match=True)
    metric.measure(test_case)
    assert metric.score >= 1.0, f"{key}: {metric.reason}"


@pytest.mark.parametrize("key", list(TOOL_SCENARIOS))
def test_tool_call_has_execution_evidence(key):
    """The tool call must carry input, timestamps and an output or error."""
    request, expected = TOOL_SCENARIOS[key]
    _, trace = run(key, request)
    assert_trace_valid(trace)
    calls = [c for c in trace["tool_calls"] if c.get("name") == expected]
    assert calls, f"No evidence of {expected} call in trace"
    for c in calls:
        assert isinstance(c.get("input"), dict) and c["input"], "Tool input missing/empty"
        assert any(isinstance(v, str) and v.strip() for v in c["input"].values()), \
            "Tool input does not carry a query"
        assert c.get("start") and c.get("end"), "Tool call lacks timestamps"
        assert c["start"] <= c["end"], "Tool call end precedes start"
        assert "output" in c or "error" in c, "Tool call lacks output/error evidence"


@pytest.mark.parametrize("key", list(TOOL_SCENARIOS))
def test_tool_appears_in_trajectory(key):
    """Trajectory must reflect the tool invocation (B1-B3 -> invoke T and synthesize)."""
    request, expected = TOOL_SCENARIOS[key]
    _, trace = run(key, request)
    assert_trace_valid(trace)
    traj = trace["trajectory"]
    assert expected in traj, f"{expected} not observed in trajectory {traj}"
    assert len([t for t in traj if t in ALL_TOOLS]) == 1, \
        f"More than one domain tool in trajectory {traj}"


# ---------------------------------------------------------------- B4/B5/B6


@pytest.mark.parametrize("key", list(NO_TOOL_SCENARIOS))
def test_error_paths_invoke_no_tool(key):
    """B4, B5, B6: the graph terminates before any tool invocation."""
    _, trace = run(key, NO_TOOL_SCENARIOS[key])
    assert_trace_valid(trace)
    assert trace["status"] == "ok", f"Agent raised: {trace.get('error')}"
    assert trace["tool_calls"] == [], f"{key}: unexpected tool calls {called_names(trace)}"
    assert not (set(trace["trajectory"]) & ALL_TOOLS), \
        f"{key}: tool observed in trajectory {trace['trajectory']}"
