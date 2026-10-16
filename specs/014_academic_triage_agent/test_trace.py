"""Trace-based evaluation of the academic triage agent (specification 014).

The agent is invoked only through the benchmark wrapper, which returns the
output and the execution trace. Each requirement is checked against the trace
(tool calls, trajectory and run status). The output is checked only for its
type, for non-blank content and for the exact standard error messages; it is
never judged semantically. When the trace does not show the evidence for a
requirement, its test fails.

Tool correctness uses DeepEval's deterministic ToolCorrectnessMetric; no LLM
judge is involved.

Behavioral graph used as the oracle, derived from the specification:

    empty or whitespace request          -> B4 error, no tool
    request spanning several domains     -> B5 error, no tool
    request outside academic scope       -> B6 error, no tool
    papers, Exact Sciences/Technology    -> search_arxiv -> synthesis
    papers, Life Sciences/Health         -> search_pubmed -> synthesis
    explanation of a single concept      -> get_wikipedia_summary -> synthesis
    tool fails or returns no results     -> B7 error, no further tool

Run with: deepeval test run test_trace.py
"""

import functools
from dataclasses import dataclass
from typing import Optional, Tuple

import pytest
from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from trace_capture import traced
from agent import agent


SEARCH_ARXIV = "search_arxiv"  # T1
SEARCH_PUBMED = "search_pubmed"  # T2
GET_WIKIPEDIA_SUMMARY = "get_wikipedia_summary"  # T3
TOOL_NAMES = {SEARCH_ARXIV, SEARCH_PUBMED, GET_WIKIPEDIA_SUMMARY}

EMPTY_REQUEST_ERROR = "Error: The user request cannot be empty."
MULTIPLE_DOMAINS_ERROR = (
    "Error: The query spans multiple distinct domains. "
    "Please make one specific research request at a time."
)
OUT_OF_SCOPE_ERROR = "Error: Query outside of academic scope."
EXECUTION_ERROR = "Error: Unable to get response from agent."
STANDARD_ERRORS = {
    EMPTY_REQUEST_ERROR,
    MULTIPLE_DOMAINS_ERROR,
    OUT_OF_SCOPE_ERROR,
    EXECUTION_ERROR,
}

# Trace fields the evaluation reads. Without them there is no evidence.
TRACE_EVIDENCE_FIELDS = ("status", "error", "trajectory", "tool_calls")


@dataclass(frozen=True)
class Scenario:
    """One agent invocation, the behavior it triggers, and its graph path."""

    name: str
    request: str
    behavior: str
    expected_tools: Tuple[str, ...] = ()
    expected_error: Optional[str] = None


TOOL_SCENARIOS = [
    Scenario(
        "exact_sciences_papers",
        "Find academic papers about artificial intelligence.",
        "B1",
        expected_tools=(SEARCH_ARXIV,),
    ),
    Scenario(
        "life_sciences_papers",
        "Find research papers about melanoma.",
        "B2",
        expected_tools=(SEARCH_PUBMED,),
    ),
    Scenario(
        "exact_sciences_concept",
        "What is entropy?",
        "B3",
        expected_tools=(GET_WIKIPEDIA_SUMMARY,),
    ),
    Scenario(
        "life_sciences_concept",
        "What is melanoma?",
        "B3",
        expected_tools=(GET_WIKIPEDIA_SUMMARY,),
    ),
    Scenario(
        "social_sciences_concept",
        "Explain the concept of cognitive dissonance.",
        "B3",
        expected_tools=(GET_WIKIPEDIA_SUMMARY,),
    ),
    Scenario(
        "humanities_concept",
        "Explain the concept of existentialism.",
        "B3",
        expected_tools=(GET_WIKIPEDIA_SUMMARY,),
    ),
    Scenario(
        "arts_concept",
        "What is chiaroscuro?",
        "B3",
        expected_tools=(GET_WIKIPEDIA_SUMMARY,),
    ),
]

NO_TOOL_SCENARIOS = [
    Scenario("empty_request", "", "B4", expected_error=EMPTY_REQUEST_ERROR),
    Scenario(
        "whitespace_request",
        " \n\t ",
        "B4",
        expected_error=EMPTY_REQUEST_ERROR,
    ),
    Scenario(
        "multiple_domains",
        "Find papers about artificial intelligence and melanoma.",
        "B5",
        expected_error=MULTIPLE_DOMAINS_ERROR,
    ),
    Scenario(
        "personal_advice",
        "Should I accept a job offer that pays less but is closer to home?",
        "B6",
        expected_error=OUT_OF_SCOPE_ERROR,
    ),
    Scenario(
        "transaction",
        "Book a table for two at an Italian restaurant tonight.",
        "B6",
        expected_error=OUT_OF_SCOPE_ERROR,
    ),
    Scenario(
        "creative_generation",
        "Write a short poem about the ocean.",
        "B6",
        expected_error=OUT_OF_SCOPE_ERROR,
    ),
]

# A valid request whose tool is expected to return no results.
FAILURE_SCENARIOS = [
    Scenario(
        "no_results",
        "Find research papers about Zorbonian melanoma syndrome.",
        "B7",
        expected_tools=(SEARCH_PUBMED,),
        expected_error=EXECUTION_ERROR,
    ),
]

ALL_SCENARIOS = TOOL_SCENARIOS + NO_TOOL_SCENARIOS + FAILURE_SCENARIOS
ERROR_SCENARIOS = NO_TOOL_SCENARIOS + FAILURE_SCENARIOS


@functools.lru_cache(maxsize=None)
def _run(scenario):
    """Invoke the agent once per scenario, through the benchmark wrapper only."""
    output, trace = traced(agent)(scenario.request)
    return scenario, output, trace


def _scenario_id(scenario):
    return f"{scenario.behavior}-{scenario.name}"


@pytest.fixture(params=ALL_SCENARIOS, ids=_scenario_id)
def any_run(request):
    return _run(request.param)


@pytest.fixture(params=TOOL_SCENARIOS, ids=_scenario_id)
def tool_run(request):
    return _run(request.param)


@pytest.fixture(params=ERROR_SCENARIOS, ids=_scenario_id)
def error_run(request):
    return _run(request.param)


@pytest.fixture(params=FAILURE_SCENARIOS, ids=_scenario_id)
def failure_run(request):
    return _run(request.param)


def _is_empty(value):
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return isinstance(value, (list, tuple, dict)) and len(value) == 0


def _single_call(trace):
    calls = trace["tool_calls"]
    assert len(calls) == 1, f"expected exactly one tool call, observed {len(calls)}"
    return calls[0]


def _failed_or_empty(call):
    return "error" in call or _is_empty(call.get("output"))


def test_trace_contains_execution_evidence(any_run):
    """Trace format: the fields used as evidence are present."""
    _, _, trace = any_run
    missing = [field for field in TRACE_EVIDENCE_FIELDS if field not in trace]
    assert not missing, f"trace lacks evidence fields: {missing}"


def test_run_completes_without_unhandled_error(any_run):
    """B7, P1: an execution failure becomes the standard error, not an exception."""
    _, _, trace = any_run
    assert trace["status"] == "ok", f"agent raised: {trace['error']}"
    assert trace["error"] is None, trace["error"]


def test_output_is_string(any_run):
    """P1: the output is always a string."""
    _, output, _ = any_run
    assert isinstance(output, str), f"output is {type(output).__name__}"


def test_tool_calls_match_behavioral_graph(any_run):
    """P2, P3, B1-B7: the executed tools are exactly the graph path for the request."""
    scenario, _, trace = any_run
    test_case = LLMTestCase(
        input=scenario.request,
        tools_called=[
            ToolCall(
                name=call["name"],
                input_parameters=call.get("input"),
                output=call.get("output"),
            )
            for call in trace["tool_calls"]
        ],
        expected_tools=[ToolCall(name=name) for name in scenario.expected_tools],
    )
    metric = ToolCorrectnessMetric(threshold=1.0, should_exact_match=True)
    assert_test(test_case=test_case, metrics=[metric])


def test_trajectory_tools_match_behavioral_graph(any_run):
    """P2, B1-B7: the trajectory records the same tool path as the graph."""
    scenario, _, trace = any_run
    tool_steps = [step for step in trace["trajectory"] if step in TOOL_NAMES]
    expected = list(scenario.expected_tools)
    assert tool_steps == expected, f"trajectory tools {tool_steps} != {expected}"


def test_routing_precedes_and_synthesis_follows_tool(tool_run):
    """Goal, B1-B3: a node selects the tool before it runs, and a node follows it."""
    scenario, _, trace = tool_run
    steps = trace["trajectory"]
    tool = scenario.expected_tools[0]
    assert tool in steps, f"{tool} is missing from the trajectory"
    position = steps.index(tool)
    assert any(step not in TOOL_NAMES for step in steps[:position]), (
        "no node ran before the tool, so tool selection is not evidenced"
    )
    assert any(step not in TOOL_NAMES for step in steps[position + 1 :]), (
        "no node ran after the tool, so the synthesis step is not evidenced"
    )


def test_tool_result_is_evidenced_for_synthesis(tool_run):
    """B1-B3: the tool returned the data the synthesized response is based on."""
    _, _, trace = tool_run
    call = _single_call(trace)
    assert "error" not in call, f"tool failed: {call['error']}"
    assert not _is_empty(call.get("output")), (
        "tool returned no data, so the synthesis is not evidenced"
    )


def test_output_follows_tool_result_branch(tool_run):
    """B1-B3, B7: a failed or empty tool result gives the standard error."""
    _, output, trace = tool_run
    call = _single_call(trace)
    if _failed_or_empty(call):
        assert output == EXECUTION_ERROR
    else:
        assert isinstance(output, str) and output.strip(), "expected a response"
        assert output not in STANDARD_ERRORS, (
            "tool returned data but output is a standard error"
        )


def test_standard_error_returned(error_run):
    """B4-B7: the exact standard error message is returned."""
    scenario, output, _ = error_run
    assert output == scenario.expected_error


def test_failed_or_empty_result_is_evidenced(failure_run):
    """B7: the trace shows the tool failed or returned nothing, so B7 was exercised."""
    _, _, trace = failure_run
    call = _single_call(trace)
    assert _failed_or_empty(call), "tool returned data; the B7 trigger is not evidenced"
