"""Trace-based evaluation tests for spec 012 - Explore Travel Destination.

Oracle: the behavioral graph in spec.md
    User Request -> B5 (empty?) -> yes: B7
                                -> no:  B1 -> {B2 get_weather | B3 search_destination_news | B4 no tool} -> B7

All evidence is taken from the trace returned by `traced(agent)(...)`.
Missing evidence makes the test fail; nothing is inferred from the output.
"""

from datetime import datetime

import pytest
from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from trace_capture import traced
from agent import agent

WEATHER_TOOL = "get_weather"
NEWS_TOOL = "search_destination_news"
SPEC_TOOLS = {WEATHER_TOOL, NEWS_TOOL}

SCENARIOS = {
    "weather": "What is the current weather in Lisbon right now?",
    "news": "Are there any recent safety advisories or news for travelers in Bangkok?",
    "tips": "What are some traditional foods I should try when visiting Rome?",
    "empty": "",
    "blank": "   \n\t ",
}

EXPECTED_TOOL = {
    "weather": [WEATHER_TOOL],
    "news": [NEWS_TOOL],
    "tips": [],
    "empty": [],
    "blank": [],
}


# --------------------------------------------------------------------------- #
# Fixtures: each scenario invokes the agent exactly once.
# --------------------------------------------------------------------------- #
_cache = {}


def _run(name):
    if name not in _cache:
        output, trace = traced(agent)(SCENARIOS[name])
        _cache[name] = (output, trace)
    return _cache[name]


@pytest.fixture(scope="module", params=list(SCENARIOS))
def scenario(request):
    output, trace = _run(request.param)
    return request.param, output, trace


@pytest.fixture(scope="module")
def weather_run():
    return _run("weather")


@pytest.fixture(scope="module")
def news_run():
    return _run("news")


@pytest.fixture(scope="module")
def tips_run():
    return _run("tips")


@pytest.fixture(scope="module", params=["empty", "blank"])
def empty_run(request):
    return _run(request.param)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _require_trace(trace):
    assert isinstance(trace, dict), "No trace returned by instrumentation"
    for key in ("status", "trajectory", "tool_calls", "runs_collected"):
        assert key in trace, f"Trace missing required field '{key}'"
    assert isinstance(trace["trajectory"], list), "trajectory must be a list"
    assert isinstance(trace["tool_calls"], list), "tool_calls must be a list"


def _require_execution_evidence(trace):
    _require_trace(trace)
    assert trace["runs_collected"] > 0, "Trace has no collected runs: no evidence of execution"
    assert len(trace["trajectory"]) > 0, "Empty trajectory: no evidence of execution"


def _tool_names(trace):
    return [c.get("name") for c in trace["tool_calls"]]


def _parse_ts(value):
    assert isinstance(value, str) and value, "Tool call missing timestamp"
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _assert_tool_correctness(query, trace, expected):
    test_case = LLMTestCase(
        input=query,
        actual_output="<not evaluated>",
        tools_called=[
            ToolCall(name=c["name"], input_parameters=c.get("input") or {})
            for c in trace["tool_calls"]
        ],
        expected_tools=[ToolCall(name=n) for n in expected],
    )
    assert_test(test_case, [ToolCorrectnessMetric(threshold=1.0, should_exact_match=True)])


def _assert_tool_call_evidence(call):
    assert call.get("name") in SPEC_TOOLS, f"Unexpected tool '{call.get('name')}'"
    assert isinstance(call.get("input"), dict) and call["input"], (
        f"Tool '{call['name']}' call has no input arguments in trace"
    )
    assert ("output" in call) or ("error" in call), (
        f"Tool '{call['name']}' call has neither output nor error in trace"
    )
    assert _parse_ts(call.get("start")) <= _parse_ts(call.get("end")), (
        f"Tool '{call['name']}' call ends before it starts"
    )


# --------------------------------------------------------------------------- #
# Invariants across all scenarios
# --------------------------------------------------------------------------- #
def test_agent_does_not_propagate_exceptions(scenario):
    """B6: exceptions must not propagate; the trace must record a normal return."""
    _name, _output, trace = scenario
    _require_trace(trace)
    assert trace["status"] == "ok", f"Agent raised: {trace.get('error')}"


def test_at_most_one_tool_call_per_execution(scenario):
    """B2/B3 constraint: invoke at most one tool per execution."""
    _name, _output, trace = scenario
    _require_trace(trace)
    assert len(trace["tool_calls"]) <= 1, f"Multiple tool calls: {_tool_names(trace)}"


def test_only_specified_tools_are_called(scenario):
    _name, _output, trace = scenario
    _require_trace(trace)
    unknown = [n for n in _tool_names(trace) if n not in SPEC_TOOLS]
    assert not unknown, f"Tools not declared in the specification were called: {unknown}"


def test_tool_calls_are_observed_in_trajectory(scenario):
    """Each recorded tool call must also appear in the trajectory."""
    _name, _output, trace = scenario
    _require_trace(trace)
    for name in _tool_names(trace):
        assert name in trace["trajectory"], f"Tool '{name}' not present in trajectory"


def test_tool_usage_matches_behavioral_graph(scenario):
    """Graph branch selection via DeepEval ToolCorrectnessMetric."""
    name, _output, trace = scenario
    _require_trace(trace)
    if EXPECTED_TOOL[name]:
        _require_execution_evidence(trace)
    _assert_tool_correctness(SCENARIOS[name], trace, EXPECTED_TOOL[name])


# --------------------------------------------------------------------------- #
# B1 -> B2: weather intent
# --------------------------------------------------------------------------- #
def test_weather_request_calls_get_weather(weather_run):
    _output, trace = weather_run
    _require_execution_evidence(trace)
    assert _tool_names(trace) == [WEATHER_TOOL], (
        f"Expected exactly one '{WEATHER_TOOL}' call, got {_tool_names(trace)}"
    )
    _assert_tool_call_evidence(trace["tool_calls"][0])


def test_weather_tool_not_first_in_trajectory(weather_run):
    """B1 (interpret) must precede B2: the tool cannot be the first step."""
    _output, trace = weather_run
    _require_execution_evidence(trace)
    assert WEATHER_TOOL in trace["trajectory"], "get_weather not observed in trajectory"
    assert trace["trajectory"].index(WEATHER_TOOL) > 0, (
        "get_weather executed before any interpretation step"
    )


def test_weather_request_does_not_search_news(weather_run):
    _output, trace = weather_run
    _require_trace(trace)
    assert NEWS_TOOL not in _tool_names(trace)
    assert NEWS_TOOL not in trace["trajectory"]


# --------------------------------------------------------------------------- #
# B1 -> B3: news / safety intent
# --------------------------------------------------------------------------- #
def test_news_request_calls_search_destination_news(news_run):
    _output, trace = news_run
    _require_execution_evidence(trace)
    assert _tool_names(trace) == [NEWS_TOOL], (
        f"Expected exactly one '{NEWS_TOOL}' call, got {_tool_names(trace)}"
    )
    _assert_tool_call_evidence(trace["tool_calls"][0])


def test_news_tool_not_first_in_trajectory(news_run):
    _output, trace = news_run
    _require_execution_evidence(trace)
    assert NEWS_TOOL in trace["trajectory"], "search_destination_news not observed in trajectory"
    assert trace["trajectory"].index(NEWS_TOOL) > 0, (
        "search_destination_news executed before any interpretation step"
    )


def test_news_request_does_not_fetch_weather(news_run):
    _output, trace = news_run
    _require_trace(trace)
    assert WEATHER_TOOL not in _tool_names(trace)
    assert WEATHER_TOOL not in trace["trajectory"]


# --------------------------------------------------------------------------- #
# B1 -> B4: travel tips, answered directly
# --------------------------------------------------------------------------- #
def test_tips_request_invokes_no_tools(tips_run):
    _output, trace = tips_run
    _require_execution_evidence(trace)
    assert trace["tool_calls"] == [], f"B4 must not call tools, got {_tool_names(trace)}"
    assert not SPEC_TOOLS.intersection(trace["trajectory"]), (
        "Tool node present in trajectory for a direct-answer request"
    )


# --------------------------------------------------------------------------- #
# B5 -> B7: empty / blank request
# --------------------------------------------------------------------------- #
def test_empty_request_invokes_no_tools(empty_run):
    _output, trace = empty_run
    _require_trace(trace)
    assert trace["status"] == "ok", f"Agent raised on empty input: {trace.get('error')}"
    assert trace["tool_calls"] == [], f"Empty request must not call tools: {_tool_names(trace)}"
    assert not SPEC_TOOLS.intersection(trace["trajectory"]), (
        "Tool node present in trajectory for an empty request"
    )
