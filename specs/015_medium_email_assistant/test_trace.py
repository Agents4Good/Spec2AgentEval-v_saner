"""Trace-based evaluation tests for the Email Assistant agent (spec 015).

Oracle: the Behavioral Graph of the specification
    B0 (valid input?) -> B1 (classify) -> {B2 (ignore/notify, no tools) |
    B3 (T1 get_calendar_events) -> B4 (T2 save_email_draft)} -> B5
with exception edges from B1/B3/B4 to the B0 failure node.

The agent is invoked ONLY through the benchmark wrapper `traced`, and all
behavioral evidence is taken from the captured trace. When the trace lacks
evidence for a required behavior, the test fails.

Run with:  deepeval test run test_trace.py   (or pytest test_trace.py)
"""

from datetime import datetime

import pytest
from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from trace_capture import traced
from agent import agent

T1 = "get_calendar_events"
T2 = "save_email_draft"
REGISTERED_TOOLS = {T1, T2}

# --------------------------------------------------------------------------- #
# Scenarios (one agent invocation each)
# --------------------------------------------------------------------------- #
SCENARIOS = {
    "empty": "",
    "whitespace": "   \n\t  ",
    "spam": (
        "CONGRATULATIONS!!! You have been selected to win a FREE cruise. "
        "Click here now to claim your prize before it expires! Unsubscribe below."
    ),
    "marketing": (
        "Hi there, our SaaS platform boosts sales by 300%. Get 50% off our "
        "premium plan this week only. Reply STOP to opt out."
    ),
    "urgent_finance": (
        "URGENT: The Q3 wire transfer of $2.4M to our main supplier failed. "
        "Finance team is investigating. FYI, no action required from you yet."
    ),
    "meeting_request": (
        "Hello, I'm Maria from Acme Corp. I'd like to schedule a 30-minute "
        "meeting next Tuesday afternoon to discuss our partnership proposal. "
        "Could you let me know your availability? Best regards, Maria"
    ),
    "customer_inquiry": (
        "Hi, I'm a customer and I'd like to set up a call this Thursday to "
        "discuss renewing our enterprise contract. What times work for you? "
        "Thanks, John"
    ),
}

NO_TOOL_SCENARIOS = ["empty", "whitespace", "spam", "marketing", "urgent_finance"]
RESPOND_SCENARIOS = ["meeting_request", "customer_inquiry"]

_CACHE = {}


def run_scenario(name):
    """Invoke the agent once per scenario via the benchmark wrapper."""
    if name not in _CACHE:
        output, trace = traced(agent)(SCENARIOS[name])
        _CACHE[name] = (output, trace)
    return _CACHE[name]


@pytest.fixture(params=list(SCENARIOS), scope="module")
def scenario(request):
    output, trace = run_scenario(request.param)
    return request.param, output, trace


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _tool_calls(trace):
    calls = trace.get("tool_calls")
    assert isinstance(calls, list), "Trace has no 'tool_calls' list: insufficient evidence."
    return calls


def _names(trace):
    return [c.get("name") for c in _tool_calls(trace)]


def _ts(value):
    return datetime.fromisoformat(value)


def _result_dict(output):
    """Entrypoint returns (compiled_graph, dict); extract the dict, if any."""
    if isinstance(output, (tuple, list)) and len(output) == 2 and isinstance(output[1], dict):
        return output[1]
    return None


def _successful(trace, name):
    return [c for c in _tool_calls(trace) if c.get("name") == name and "error" not in c]


# --------------------------------------------------------------------------- #
# General trace sanity / evidence
# --------------------------------------------------------------------------- #
def test_trace_is_well_formed(scenario):
    _, _, trace = scenario
    for key in ("run_id", "status", "trajectory", "tool_calls", "runs_collected"):
        assert key in trace, f"Trace missing '{key}'"
    assert isinstance(trace["trajectory"], list)
    for call in _tool_calls(trace):
        assert "name" in call and "input" in call and "start" in call and "end" in call
        assert ("output" in call) or ("error" in call), "Tool call lacks output/error evidence"


def test_no_exception_propagates(scenario):
    """C1 / B0: exceptions must not propagate; failure dict is returned instead."""
    name, _, trace = scenario
    assert trace["status"] == "ok", f"[{name}] agent raised: {trace.get('error')}"


def test_only_registered_tools_are_called(scenario):
    name, _, trace = scenario
    unknown = [n for n in _names(trace) if n not in REGISTERED_TOOLS]
    assert not unknown, f"[{name}] unregistered tools called: {unknown}"


def test_tool_calls_appear_in_trajectory_in_same_order(scenario):
    name, _, trace = scenario
    traj = trace["trajectory"]
    idx = 0
    for tool in _names(trace):
        try:
            idx = traj.index(tool, idx) + 1
        except ValueError:
            pytest.fail(f"[{name}] tool '{tool}' not found (in order) in trajectory {traj}")


# --------------------------------------------------------------------------- #
# Graph ordering: T2 only after T1 (B3 -> B4)
# --------------------------------------------------------------------------- #
def test_t2_never_precedes_t1(scenario):
    name, _, trace = scenario
    calls = _tool_calls(trace)
    for i, call in enumerate(calls):
        if call["name"] == T2:
            prior_t1 = [c for c in calls[:i] if c["name"] == T1 and "error" not in c]
            assert prior_t1, f"[{name}] {T2} invoked without a prior successful {T1}"
            assert _ts(prior_t1[-1]["end"]) <= _ts(call["start"]), (
                f"[{name}] {T2} started before {T1} finished"
            )


def test_no_tool_call_after_a_failed_tool(scenario):
    """Exception edges G/H -> F: after a tool fails, the flow goes to B0 failure."""
    name, _, trace = scenario
    calls = _tool_calls(trace)
    for i, call in enumerate(calls):
        if "error" in call:
            assert not calls[i + 1:], f"[{name}] tools invoked after failure of {call['name']}"


# --------------------------------------------------------------------------- #
# Consistency between declared result and trace evidence (I2, I3, QA4)
# --------------------------------------------------------------------------- #
def test_tool_routing_consistent_with_declared_classification(scenario):
    """QA4/C2: tools are only called when classification is respond."""
    name, output, trace = scenario
    result = _result_dict(output)
    assert result is not None, f"[{name}] no result dict available to correlate with trace"
    names = _names(trace)
    cls = result.get("classification")
    if cls in ("ignore", "notify"):
        assert names == [], f"[{name}] tools called for '{cls}': {names}"
    elif cls == "unknown" and names:
        # Only reachable via exception edges from B3/B4: a tool must have failed
        # or the flow must have been interrupted after starting the respond path.
        assert names[0] == T1, f"[{name}] failure path started with {names[0]}"


def test_actions_taken_backed_by_trace_evidence(scenario):
    """I2: every action declared must correspond to a successful tool call."""
    name, output, trace = scenario
    result = _result_dict(output)
    assert result is not None, f"[{name}] no result dict available to correlate with trace"
    for action in result.get("actions_taken") or []:
        assert _successful(trace, action), (
            f"[{name}] '{action}' declared in actions_taken without successful call in trace"
        )
    if result.get("status") == "success" and result.get("classification") == "respond":
        assert _successful(trace, T1) and _successful(trace, T2), (
            f"[{name}] success/respond without evidence of both {T1} and {T2}"
        )


# --------------------------------------------------------------------------- #
# Scenario-specific paths of the behavioral graph
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", NO_TOOL_SCENARIOS)
def test_no_tools_on_invalid_or_non_respond_paths(name):
    """B0 (empty input) and B2 (ignore/notify): no T1/T2 calls."""
    _, trace = run_scenario(name)
    assert trace["status"] == "ok", trace.get("error")
    assert _names(trace) == [], f"[{name}] unexpected tool calls: {_names(trace)}"


@pytest.mark.parametrize("name", RESPOND_SCENARIOS)
def test_respond_path_calls_t1_then_t2(name):
    """B3 -> B4: calendar query followed by draft persistence (DeepEval ToolCorrectness)."""
    output, trace = run_scenario(name)
    assert trace["status"] == "ok", trace.get("error")
    calls = _tool_calls(trace)
    assert calls, f"[{name}] no tool calls in trace: respond path not evidenced"
    assert not any("error" in c for c in calls), f"[{name}] tool failed: {calls}"

    test_case = LLMTestCase(
        input=SCENARIOS[name],
        actual_output=str(output),
        tools_called=[ToolCall(name=c["name"], input_parameters=c.get("input")) for c in calls],
        expected_tools=[ToolCall(name=T1), ToolCall(name=T2)],
    )
    metric = ToolCorrectnessMetric(threshold=1.0, should_consider_ordering=True)
    assert_test(test_case, [metric])

    # B3 constraint: T2 not invoked during B3; B4: T1 not invoked during B4.
    names = _names(trace)
    first_t2 = names.index(T2)
    assert T1 not in names[first_t2:], f"[{name}] {T1} invoked after {T2}: {names}"
