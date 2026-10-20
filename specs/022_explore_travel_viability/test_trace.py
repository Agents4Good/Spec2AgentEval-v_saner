"""Trace-based evaluation tests derived exclusively from spec.md.

Behavioral graph (oracle):
    User Request -> B5 (empty?) --Yes--> B7
                               --No --> B1 -> B2 (get_weather) -> B3 (search_destination_news) -> B4 -> B7
    B2/B3/B4 -.-> B6 (errors/timeouts) -> B7

Only trace evidence is evaluated; the output is never semantically validated.
"""
from datetime import datetime

import pytest
from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from trace_capture import traced
from agent import agent

WEATHER = "get_weather"
NEWS = "search_destination_news"
SPEC_TOOLS = {WEATHER, NEWS}

VALID_REQUESTS = [
    "Is it a good idea to travel to Lisbon this weekend?",
    "I'm planning a trip to Tokyo next week. Should I go?",
]
EMPTY_REQUESTS = ["", "   ", "\n\t  "]


# --------------------------------------------------------------------------- #
# Fixtures: each scenario invokes the agent exactly once.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module", params=VALID_REQUESTS, ids=["lisbon", "tokyo"])
def valid_run(request):
    output, trace = traced(agent)(request.param)
    return request.param, output, trace


@pytest.fixture(scope="module", params=EMPTY_REQUESTS, ids=["empty", "spaces", "whitespace"])
def empty_run(request):
    output, trace = traced(agent)(request.param)
    return request.param, output, trace


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _require_trace(trace):
    assert isinstance(trace, dict), "Trace missing: cannot evaluate behavior"
    for key in ("status", "trajectory", "tool_calls"):
        assert key in trace, f"Trace lacks '{key}': insufficient evidence"
    assert isinstance(trace["trajectory"], list)
    assert isinstance(trace["tool_calls"], list)


def _calls(trace, name):
    return [c for c in trace["tool_calls"] if c.get("name") == name]


def _ts(value):
    return datetime.fromisoformat(value)


def _first_start(calls):
    return min(_ts(c["start"]) for c in calls)


def _weather_failed(trace):
    calls = _calls(trace, WEATHER)
    return bool(calls) and all("error" in c for c in calls)


# --------------------------------------------------------------------------- #
# B6 / B7: no propagated exceptions
# --------------------------------------------------------------------------- #
def test_valid_request_does_not_propagate_exception(valid_run):
    _, _, trace = valid_run
    _require_trace(trace)
    assert trace["status"] == "ok", f"Agent raised (B6 violated): {trace.get('error')}"


def test_empty_request_does_not_propagate_exception(empty_run):
    _, _, trace = empty_run
    _require_trace(trace)
    assert trace["status"] == "ok", f"Agent raised on empty request: {trace.get('error')}"


# --------------------------------------------------------------------------- #
# B5: empty request must not proceed with tool execution
# --------------------------------------------------------------------------- #
def test_empty_request_executes_no_tools(empty_run):
    _, _, trace = empty_run
    _require_trace(trace)
    assert trace["tool_calls"] == [], (
        f"B5 violated: tools executed on empty request: "
        f"{[c.get('name') for c in trace['tool_calls']]}"
    )
    assert not SPEC_TOOLS & set(trace["trajectory"]), "B5 violated: tool in trajectory"


# --------------------------------------------------------------------------- #
# B2 / B3: required tools called (DeepEval ToolCorrectnessMetric)
# --------------------------------------------------------------------------- #
def test_required_tools_called(valid_run):
    user_request, output, trace = valid_run
    _require_trace(trace)

    expected = [ToolCall(name=WEATHER)]
    # Graph: B2 -.-> B6 — if weather retrieval failed, B3 is not required.
    if not _weather_failed(trace):
        expected.append(ToolCall(name=NEWS))

    called = [
        ToolCall(name=c["name"], input_parameters=c.get("input") or {})
        for c in trace["tool_calls"]
        if c.get("name") in SPEC_TOOLS
    ]
    test_case = LLMTestCase(
        input=user_request,
        actual_output=str(output),
        tools_called=called,
        expected_tools=expected,
    )
    assert_test(test_case, [ToolCorrectnessMetric(threshold=1.0, should_consider_ordering=False)])


def test_weather_tool_called(valid_run):
    _, _, trace = valid_run
    _require_trace(trace)
    assert _calls(trace, WEATHER), "B2 not evidenced: get_weather never called"


def test_news_tool_called_unless_weather_failed(valid_run):
    _, _, trace = valid_run
    _require_trace(trace)
    if _weather_failed(trace):
        pytest.skip("get_weather failed: graph allows B2 -> B6 without B3")
    assert _calls(trace, NEWS), "B3 not evidenced: search_destination_news never called"


# --------------------------------------------------------------------------- #
# Ordering: B2 (weather) -> B3 (news)
# --------------------------------------------------------------------------- #
def test_weather_precedes_news(valid_run):
    _, _, trace = valid_run
    _require_trace(trace)
    weather, news = _calls(trace, WEATHER), _calls(trace, NEWS)
    if not news:
        assert weather, "No tool evidence for B2/B3"
        return
    assert weather, "B3 occurred without preceding B2 (get_weather)"
    for c in weather + news:
        assert c.get("start"), f"Missing timestamp in {c.get('name')}: cannot verify order"
    assert _first_start(weather) <= _first_start(news), (
        "Ordering violated: search_destination_news started before get_weather"
    )


def test_trajectory_order_weather_before_news(valid_run):
    _, _, trace = valid_run
    _require_trace(trace)
    traj = trace["trajectory"]
    assert WEATHER in traj, "get_weather absent from trajectory: insufficient evidence"
    if NEWS in traj:
        assert traj.index(WEATHER) < traj.index(NEWS), "Trajectory: B3 before B2"


# --------------------------------------------------------------------------- #
# Evidence: tool calls actually executed with inputs and results/errors
# --------------------------------------------------------------------------- #
def test_tool_calls_have_execution_evidence(valid_run):
    _, _, trace = valid_run
    _require_trace(trace)
    spec_calls = [c for c in trace["tool_calls"] if c.get("name") in SPEC_TOOLS]
    assert spec_calls, "No spec tool calls recorded: insufficient evidence"
    for c in spec_calls:
        assert isinstance(c.get("input"), dict) and c["input"], (
            f"{c['name']} called without input arguments"
        )
        assert "output" in c or "error" in c, f"{c['name']} has no output nor error"
        assert c.get("start") and c.get("end"), f"{c['name']} lacks timestamps"
        assert _ts(c["start"]) <= _ts(c["end"]), f"{c['name']} has inconsistent timestamps"
        assert c["name"] in trace["trajectory"], f"{c['name']} missing from trajectory"


# --------------------------------------------------------------------------- #
# B6: maximum execution time of 30 seconds (observable via tool timestamps)
# --------------------------------------------------------------------------- #
def test_tool_execution_within_time_limit(valid_run):
    _, _, trace = valid_run
    _require_trace(trace)
    calls = [c for c in trace["tool_calls"] if c.get("start") and c.get("end")]
    assert calls, "No timestamped tool calls: cannot verify time limit"
    span = max(_ts(c["end"]) for c in calls) - min(_ts(c["start"]) for c in calls)
    assert span.total_seconds() <= 30, f"Tool execution span {span} exceeds 30s"
