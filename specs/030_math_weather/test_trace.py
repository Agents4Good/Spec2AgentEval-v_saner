"""Trace-based evaluation tests for the math/weather trip-budget agent.

Oracle: behavioral graph of the specification
    request -> B1 multiply -> B2 add -> B3 select_location -> B4 get_weather
    invalid request -> B5 (no tools)
    any tool failure -> B6 (no subsequent tool invocation)
"""
import pytest
from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from trace_capture import traced
from agent import agent

CHAIN = ["multiply", "add", "select_location", "get_weather"]

VALID_SCENARIOS = {
    "with_extra": {
        "request": "I have a daily budget of 100 for a 5-day trip, plus 200 of reserve money. "
                   "What is my total budget, which destination fits it, and what is the weather there?",
        "daily": 100, "days": 5, "extra": 200,
    },
    "without_extra": {
        "request": "My daily budget is 300 and the trip lasts 5 days. Tell me the total budget, "
                   "the destination for it, and the weather at that destination.",
        "daily": 300, "days": 5, "extra": 0,
    },
    "low_budget": {
        "request": "Daily budget 50, 3 days, extra 20. Which destination can I afford with the total "
                   "and how is the weather there?",
        "daily": 50, "days": 3, "extra": 20,
    },
}

INVALID_SCENARIOS = {
    "empty": "",
    "missing_values": "I want to travel somewhere nice. What's the weather going to be like?",
    "off_task": "Write me a poem about the ocean.",
}


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module", params=list(VALID_SCENARIOS))
def valid_run(request):
    scenario = VALID_SCENARIOS[request.param]
    output, trace = traced(agent)(scenario["request"])
    return scenario, output, trace


@pytest.fixture(scope="module", params=list(INVALID_SCENARIOS))
def invalid_run(request):
    output, trace = traced(agent)(INVALID_SCENARIOS[request.param])
    return output, trace


# ---------------------------------------------------------------- helpers
def _calls(trace):
    assert isinstance(trace, dict), "Trace missing"
    calls = trace.get("tool_calls")
    assert isinstance(calls, list), "Trace lacks 'tool_calls' evidence"
    return calls


def _num(value):
    if isinstance(value, bool):
        raise AssertionError(f"Not numeric: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            pass
    raise AssertionError(f"Not numeric evidence in trace: {value!r}")


def _eq(a, b):
    return abs(_num(a) - _num(b)) < 1e-6


def _first(calls, name):
    found = [c for c in calls if c.get("name") == name]
    assert found, f"Trace provides no evidence of a '{name}' call"
    return found[0]


def _ok(call):
    assert "error" not in call, f"'{call['name']}' failed: {call.get('error')}"
    assert "output" in call, f"'{call['name']}' has no recorded output"
    return call["output"]


def _is_subsequence(seq, sub):
    it = iter(seq)
    return all(x in it for x in sub)


# ---------------------------------------------------------------- valid path (B1-B4)
def test_valid_run_completes(valid_run):
    _, _, trace = valid_run
    assert trace.get("status") == "ok", trace.get("error")
    assert trace.get("runs_collected", 0) > 0


def test_required_tools_called_in_order(valid_run):
    _, output, trace = valid_run
    names = [c.get("name") for c in _calls(trace)]
    assert _is_subsequence(names, CHAIN), f"Expected order {CHAIN}, got {names}"

    test_case = LLMTestCase(
        input=str(trace.get("query")),
        actual_output=output or "",
        tools_called=[ToolCall(name=n, input_parameters=c.get("input"))
                      for n, c in ((c.get("name"), c) for c in _calls(trace))],
        expected_tools=[ToolCall(name=n) for n in CHAIN],
    )
    assert_test(test_case, [ToolCorrectnessMetric(should_consider_ordering=True, threshold=1.0)])


def test_tool_calls_are_temporally_ordered(valid_run):
    _, _, trace = valid_run
    calls = _calls(trace)
    chain = [_first(calls, n) for n in CHAIN]
    for prev, nxt in zip(chain, chain[1:]):
        assert prev.get("end") and nxt.get("start"), "Missing timestamps in trace"
        assert prev["end"] <= nxt["start"], f"{nxt['name']} started before {prev['name']} ended"


def test_b1_multiply_receives_identified_values(valid_run):
    scenario, _, trace = valid_run
    call = _first(_calls(trace), "multiply")
    args = call.get("input") or {}
    assert "a" in args and "b" in args, f"multiply input lacks a/b: {args}"
    assert {_num(args["a"]), _num(args["b"])} == {float(scenario["daily"]), float(scenario["days"])}
    _ok(call)


def test_b2_add_uses_multiply_result_and_extra(valid_run):
    scenario, _, trace = valid_run
    calls = _calls(trace)
    base = _ok(_first(calls, "multiply"))
    args = _first(calls, "add").get("input") or {}
    assert "a" in args and "b" in args, f"add input lacks a/b: {args}"
    assert _eq(args["a"], base), f"add.a={args['a']} != multiply output {base}"
    assert _eq(args["b"], scenario["extra"]), f"add.b={args['b']} != extra {scenario['extra']}"


def test_b3_select_location_uses_add_result(valid_run):
    _, _, trace = valid_run
    calls = _calls(trace)
    total = _ok(_first(calls, "add"))
    args = _first(calls, "select_location").get("input") or {}
    assert "total_budget" in args, f"select_location input lacks total_budget: {args}"
    assert _eq(args["total_budget"], total)


def test_b4_get_weather_uses_selected_location(valid_run):
    _, _, trace = valid_run
    calls = _calls(trace)
    location = _ok(_first(calls, "select_location"))
    args = _first(calls, "get_weather").get("input") or {}
    assert "location" in args, f"get_weather input lacks location: {args}"
    assert str(args["location"]).strip().lower() == str(location).strip().lower()


def test_each_tool_called_once_in_valid_run(valid_run):
    _, _, trace = valid_run
    names = [c.get("name") for c in _calls(trace)]
    for n in CHAIN:
        assert names.count(n) == 1, f"'{n}' called {names.count(n)} times: {names}"


# ---------------------------------------------------------------- B6 (any run)
def test_b6_no_tool_after_failure(valid_run):
    _, _, trace = valid_run
    calls = _calls(trace)
    for i, c in enumerate(calls):
        if "error" in c:
            assert not calls[i + 1:], f"Tools invoked after '{c['name']}' failed: {calls[i + 1:]}"


# ---------------------------------------------------------------- B5 invalid requests
def test_b5_invalid_request_invokes_no_tool(invalid_run):
    _, trace = invalid_run
    assert trace.get("status") == "ok", trace.get("error")
    calls = _calls(trace)
    assert calls == [], f"Tools invoked for invalid request: {[c.get('name') for c in calls]}"
