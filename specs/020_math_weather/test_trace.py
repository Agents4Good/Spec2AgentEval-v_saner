"""Trace-based evaluation tests for the math/weather agent (spec 020).

Oracle: the behavioral graph in spec.md.
  User request -> D -> {B1 add | B2 multiply | B3 get_weather | B5 no tool}
  B1/B2/B3 -> tool result -> {response | B4 unavailable | B6 failure}

All evidence comes exclusively from the trace returned by
`traced(agent)(...)`. No semantic validation of the output is performed.
"""

from datetime import datetime

import pytest
from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from trace_capture import traced
from agent import agent

SPEC_TOOLS = {"add", "multiply", "get_weather"}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _run(user_request):
    output, trace = traced(agent)(user_request)
    assert isinstance(trace, dict), "No trace was captured"
    for key in ("status", "trajectory", "tool_calls"):
        assert key in trace, f"Trace lacks required field '{key}'"
    return output, trace


def _calls(trace, name=None):
    calls = trace.get("tool_calls") or []
    return [c for c in calls if name is None or c.get("name") == name]


def _spec_calls(trace):
    return [c for c in _calls(trace) if c.get("name") in SPEC_TOOLS]


def _numeric_args(call):
    values = []
    for v in (call.get("input") or {}).values():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            values.append(float(v))
        elif isinstance(v, str):
            try:
                values.append(float(v))
            except ValueError:
                pass
    return sorted(values)


def _string_args(call):
    return [str(v) for v in (call.get("input") or {}).values()]


def _assert_ok(trace):
    assert trace["status"] == "ok", f"Agent raised: {trace.get('error')}"


def _assert_tool_evidence(call):
    """A tool call must carry evidence that it actually executed."""
    assert isinstance(call.get("input"), dict) and call["input"], (
        f"Tool call {call.get('name')} has no recorded input"
    )
    assert "start" in call and "end" in call, "Tool call lacks timestamps"
    start = datetime.fromisoformat(call["start"])
    end = datetime.fromisoformat(call["end"])
    assert start <= end, "Tool call ends before it starts"
    assert "output" in call or "error" in call, (
        "Tool call has neither output nor error recorded"
    )


def _assert_in_trajectory(trace, tool_name):
    assert tool_name in trace["trajectory"], (
        f"Tool '{tool_name}' not observed in trajectory {trace['trajectory']}"
    )


def _deepeval_tool_check(user_request, output, trace, expected):
    test_case = LLMTestCase(
        input=user_request,
        actual_output=output if isinstance(output, str) else str(output),
        tools_called=[
            ToolCall(name=c["name"], input_parameters=c.get("input") or {})
            for c in _spec_calls(trace)
        ],
        expected_tools=[ToolCall(name=n) for n in expected],
    )
    assert_test(
        test_case,
        [ToolCorrectnessMetric(threshold=1.0, should_exact_match=True)],
    )


# --------------------------------------------------------------------------
# Fixtures (one agent invocation per scenario)
# --------------------------------------------------------------------------
ADD_REQUEST = "What is 12 plus 30?"
MUL_REQUEST = "Multiply 7 by 6."
WEATHER_REQUEST = "What is the weather like in Paris?"
DIGIT_WEATHER_REQUEST = "What is the weather in 98765?"


@pytest.fixture(scope="module")
def add_run():
    return _run(ADD_REQUEST)


@pytest.fixture(scope="module")
def mul_run():
    return _run(MUL_REQUEST)


@pytest.fixture(scope="module")
def weather_run():
    return _run(WEATHER_REQUEST)


@pytest.fixture(scope="module")
def digit_weather_run():
    return _run(DIGIT_WEATHER_REQUEST)


# --------------------------------------------------------------------------
# B1 — Add two numbers
# --------------------------------------------------------------------------
def test_b1_add_called_with_request_numbers(add_run):
    _, trace = add_run
    _assert_ok(trace)
    calls = _calls(trace, "add")
    assert calls, "B1 requires a call to 'add'; none observed in trace"
    for c in calls:
        _assert_tool_evidence(c)
    assert any(_numeric_args(c) == [12.0, 30.0] for c in calls), (
        f"'add' not called with 12 and 30: {[c['input'] for c in calls]}"
    )
    _assert_in_trajectory(trace, "add")


def test_b1_only_add_selected(add_run):
    output, trace = add_run
    names = {c["name"] for c in _spec_calls(trace)}
    assert names == {"add"}, f"Unexpected tools for addition: {names}"
    _deepeval_tool_check(ADD_REQUEST, output, trace, ["add"])


# --------------------------------------------------------------------------
# B2 — Multiply two numbers
# --------------------------------------------------------------------------
def test_b2_multiply_called_with_request_numbers(mul_run):
    _, trace = mul_run
    _assert_ok(trace)
    calls = _calls(trace, "multiply")
    assert calls, "B2 requires a call to 'multiply'; none observed in trace"
    for c in calls:
        _assert_tool_evidence(c)
    assert any(_numeric_args(c) == [6.0, 7.0] for c in calls), (
        f"'multiply' not called with 7 and 6: {[c['input'] for c in calls]}"
    )
    _assert_in_trajectory(trace, "multiply")


def test_b2_only_multiply_selected(mul_run):
    output, trace = mul_run
    names = {c["name"] for c in _spec_calls(trace)}
    assert names == {"multiply"}, f"Unexpected tools for multiplication: {names}"
    _deepeval_tool_check(MUL_REQUEST, output, trace, ["multiply"])


# --------------------------------------------------------------------------
# B3 — Query the weather
# --------------------------------------------------------------------------
def test_b3_get_weather_called_with_location(weather_run):
    _, trace = weather_run
    _assert_ok(trace)
    calls = _calls(trace, "get_weather")
    assert calls, "B3 requires a call to 'get_weather'; none observed in trace"
    for c in calls:
        _assert_tool_evidence(c)
    assert any(
        any("paris" in s.lower() for s in _string_args(c)) for c in calls
    ), f"'get_weather' not called with Paris: {[c['input'] for c in calls]}"
    _assert_in_trajectory(trace, "get_weather")


def test_b3_only_get_weather_selected(weather_run):
    output, trace = weather_run
    names = {c["name"] for c in _spec_calls(trace)}
    assert names == {"get_weather"}, f"Unexpected tools for weather: {names}"
    _deepeval_tool_check(WEATHER_REQUEST, output, trace, ["get_weather"])


# --------------------------------------------------------------------------
# B3 -> B4 — Digit-only location remains a valid weather query
# --------------------------------------------------------------------------
def test_b4_digit_location_queried_via_get_weather(digit_weather_run):
    output, trace = digit_weather_run
    _assert_ok(trace)
    calls = _calls(trace, "get_weather")
    assert calls, (
        "A digit-only location is a valid weather query; 'get_weather' must "
        "be called (no B5 path), but no call was observed"
    )
    for c in calls:
        _assert_tool_evidence(c)
    assert any(
        any("98765" in s for s in _string_args(c)) for c in calls
    ), f"'get_weather' not called with 98765: {[c['input'] for c in calls]}"
    names = {c["name"] for c in _spec_calls(trace)}
    assert names == {"get_weather"}, (
        f"Digit location must not be treated as arithmetic: {names}"
    )
    _deepeval_tool_check(DIGIT_WEATHER_REQUEST, output, trace, ["get_weather"])


# --------------------------------------------------------------------------
# B5 — Invalid or unsupported request: no tool invoked
# --------------------------------------------------------------------------
INVALID_REQUESTS = [
    pytest.param("", id="empty"),
    pytest.param("Can you help me with something?", id="vague"),
    pytest.param("Please add 5.", id="missing_second_number"),
    pytest.param("What is the weather like?", id="missing_location"),
    pytest.param("What is 10 minus 4?", id="unsupported_subtraction"),
    pytest.param("Divide 20 by 5.", id="unsupported_division"),
    pytest.param(
        "Add 2 and 3 and tell me the weather in London.", id="combined_types"
    ),
]


@pytest.mark.parametrize("user_request", INVALID_REQUESTS)
def test_b5_no_tool_invoked_for_invalid_request(user_request):
    _, trace = _run(user_request)
    _assert_ok(trace)
    invoked = [c["name"] for c in _spec_calls(trace)]
    assert not invoked, f"B5 forbids tool calls, but trace shows: {invoked}"
    assert not (SPEC_TOOLS & set(trace["trajectory"])), (
        f"Spec tools appear in trajectory: {trace['trajectory']}"
    )


# --------------------------------------------------------------------------
# Graph compatibility: at most one tool branch per request; tool calls
# recorded in trace are consistent with trajectory ordering.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fixture_name,tool",
    [
        ("add_run", "add"),
        ("mul_run", "multiply"),
        ("weather_run", "get_weather"),
        ("digit_weather_run", "get_weather"),
    ],
)
def test_graph_single_branch_and_no_substitute_tool(request, fixture_name, tool):
    _, trace = request.getfixturevalue(fixture_name)
    calls = _spec_calls(trace)
    assert calls, "No tool call evidence in trace"
    # The graph allows exactly one branch out of D; B6 forbids substitutes.
    assert {c["name"] for c in calls} == {tool}
    # Tool calls are ordered by start time consistently with trajectory.
    starts = [datetime.fromisoformat(c["start"]) for c in _calls(trace)]
    assert starts == sorted(starts), "tool_calls not in chronological order"
    traj_tools = [t for t in trace["trajectory"] if t in SPEC_TOOLS]
    assert traj_tools and set(traj_tools) == {tool}, (
        f"Trajectory tools {traj_tools} inconsistent with branch '{tool}'"
    )
