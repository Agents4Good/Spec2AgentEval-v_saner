"""Trace-based evaluation of the customer support agent against its specification.

Each scenario invokes the agent once through the benchmark wrapper,
``output, trace = traced(agent)(query)``. Assertions read the execution trace.
The output is only checked against what the trace recorded, for example that the
response carries the exact escalate confirmation shown in the trace.

The behavioral graph is the oracle. The observed tool-call sequence must be one of
its paths:

    no tool call                       B3 (rejected input) or B6 (analysis failure)
    handle_support fails               B4 -> B7
    handle_support ok, escalate ok     B4 -> B5 -> B2
    handle_support ok, escalate fails  B4 -> B5 -> B7

Evidence policy: a requirement the trace cannot evidence fails. The trace format has
no LLM-call events, so one analytical LLM call (B1), its no-retry rule (B6), the LLM
use inside handle_support, and the order of LLM calls against the B3 error cannot be
evaluated. Those tests fail on purpose.

Skip policy: a test whose trigger never appears in the trace is skipped, not passed.
The scenarios are input-only, so a failed tool call (B7) is not guaranteed.

Run with `deepeval test run test_trace.py`, serially. Each scenario's invocation is
cached per process, so parallel workers would invoke the agent more than once.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pytest

from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from trace_capture import traced
from agent import agent


SUPPORT_TOOL = "handle_support"
ESCALATE_TOOL = "escalate"
CATEGORIES = ("Technical", "Billing", "General")
SENTIMENTS = ("Positive", "Neutral", "Negative")
INVALID_INPUT_RESPONSE = "Error: could not classify the query"
SUPPORT_TOOL_ERROR = "Error: could not execute handle_support"
ESCALATE_TOOL_ERROR = "Error: could not execute escalate"

SUCCEEDED = "succeeded"
FAILED = "failed"

NO_TOOL_PATH = "no_tool_path"
SUPPORT_FAILED_PATH = "support_failed"
HANDOFF_PATH = "handoff"
ESCALATE_FAILED_PATH = "escalate_failed"

SUPPORT = "support"
UNRELATED = "unrelated"
INVALID = "invalid"


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str
    query: Any


@dataclass(frozen=True)
class Run:
    scenario: Scenario
    output: Any
    trace: dict


SCENARIOS = (
    Scenario("support_technical", SUPPORT,
             "The mobile app crashes every time I upload a photo. How can I fix this?"),
    Scenario("support_billing", SUPPORT,
             "I was charged twice for my subscription this month. Can you help me get the duplicate payment refunded?"),
    Scenario("support_general", SUPPORT,
             "How can I contact your support team about my account?"),
    Scenario("support_positive", SUPPORT,
             "Thank you, the new dashboard is excellent! How do I schedule a recurring report?"),
    Scenario("support_neutral", SUPPORT,
             "How do I change the language setting in the application?"),
    Scenario("support_negative", SUPPORT,
             "This is unacceptable. The service has been down for two days and nobody has answered my support tickets."),
    Scenario("unrelated_geography", UNRELATED, "What is the capital of France?"),
    Scenario("unrelated_creative", UNRELATED, "Write a short poem about autumn leaves."),
    Scenario("invalid_none", INVALID, None),
    Scenario("invalid_integer", INVALID, 42),
    Scenario("invalid_list", INVALID, ["How do I reset my password?"]),
    Scenario("invalid_empty", INVALID, ""),
    Scenario("invalid_whitespace", INVALID, " \t\n "),
)
SUPPORT_SCENARIOS = tuple(s for s in SCENARIOS if s.kind == SUPPORT)
UNRELATED_SCENARIOS = tuple(s for s in SCENARIOS if s.kind == UNRELATED)
INVALID_SCENARIOS = tuple(s for s in SCENARIOS if s.kind == INVALID)
REJECTED_SCENARIOS = UNRELATED_SCENARIOS + INVALID_SCENARIOS


def _scenario_id(scenario):
    return scenario.name


@pytest.fixture(scope="session")
def invoke():
    """Invoke each scenario's agent once through the benchmark wrapper and cache the run."""
    runs = {}

    def _invoke(scenario):
        if scenario.name not in runs:
            output, trace = traced(agent)(scenario.query)
            runs[scenario.name] = Run(scenario, output, trace)
        return runs[scenario.name]

    return _invoke


def _response(run):
    """state['response'] of a run. The output contract is a state dict with a str response."""
    assert isinstance(run.output, dict) and isinstance(run.output.get("response"), str), (
        f"output must be a state dict with a str 'response', observed {run.output!r}"
    )
    return run.output["response"]


def _tool_calls(trace):
    calls = trace.get("tool_calls")
    if not isinstance(calls, list):
        pytest.fail("insufficient trace evidence: tool_calls is not a list")
    return calls


def _parse_time(value):
    """Timezone-aware ISO-8601 timestamp, or None when it is missing or malformed."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _ordered_calls(trace):
    """Tool calls by start time. Ordering needs timestamps, so missing ones are insufficient evidence."""
    keyed = []
    for index, call in enumerate(_tool_calls(trace)):
        start, end = _parse_time(call.get("start")), _parse_time(call.get("end"))
        if start is None or end is None:
            pytest.fail(
                f"insufficient trace evidence: tool call {call.get('name')!r} "
                "lacks timezone-aware start/end timestamps"
            )
        keyed.append((start, index, call))
    return [call for _, _, call in sorted(keyed, key=lambda item: (item[0], item[1]))]


def _outcome(call):
    """B7 outcome: FAILED if the tool raised or returned a non-string, empty or whitespace-only value.

    None when the trace holds neither an output nor an error.
    """
    if call.get("error") is not None:
        return FAILED
    if "output" not in call:
        return None
    output = call["output"]
    return SUCCEEDED if isinstance(output, str) and output.strip() else FAILED


def _graph_path(calls):
    """The behavioral-graph path the tool calls follow, or None when they follow none."""
    steps = [(call.get("name"), _outcome(call)) for call in calls]
    if any(outcome is None for _, outcome in steps):
        pytest.fail("insufficient trace evidence: a tool call has neither an output nor an error")
    if not steps:
        return NO_TOOL_PATH
    if steps == [(SUPPORT_TOOL, FAILED)]:
        return SUPPORT_FAILED_PATH
    if steps == [(SUPPORT_TOOL, SUCCEEDED), (ESCALATE_TOOL, SUCCEEDED)]:
        return HANDOFF_PATH
    if steps == [(SUPPORT_TOOL, SUCCEEDED), (ESCALATE_TOOL, FAILED)]:
        return ESCALATE_FAILED_PATH
    return None


def _called(calls, name):
    return [call for call in calls if call.get("name") == name]


def _is_subsequence(needle, haystack):
    remaining = iter(haystack)
    return all(item in remaining for item in needle)


def _llm_call_events(trace):
    """LLM-call events the trace records. None exist.

    tool_calls lists tool invocations only, and trajectory holds implementation-defined
    node names that must not be read as LLM calls, so the trace cannot evidence one.
    """
    return []


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_scenario_id)
def test_trace_provides_required_fields(invoke, scenario):
    """The trace has the documented shape, so the evidence used below is well-formed."""
    trace = invoke(scenario).trace
    assert isinstance(trace, dict), "trace must be a dict"
    assert isinstance(trace.get("run_id"), str), "trace.run_id must be a str"
    assert isinstance(trace.get("query"), (list, dict)), "trace.query must be a list or dict"
    assert trace.get("status") in ("ok", "error"), (
        f"trace.status must be 'ok' or 'error', observed {trace.get('status')!r}"
    )
    assert trace.get("error") is None or isinstance(trace.get("error"), str), (
        "trace.error must be None or a str"
    )
    assert isinstance(trace.get("runs_collected"), int), "trace.runs_collected must be an int"
    trajectory = trace.get("trajectory")
    assert isinstance(trajectory, list) and all(isinstance(node, str) for node in trajectory), (
        "trace.trajectory must be a list of str"
    )
    for call in _tool_calls(trace):
        assert isinstance(call, dict), "each tool call must be a dict"
        assert isinstance(call.get("name"), str), "tool call name must be a str"
        assert isinstance(call.get("input"), dict), "tool call input must be a dict"
        assert _parse_time(call.get("start")) is not None and _parse_time(call.get("end")) is not None, (
            "tool call start and end must be timezone-aware ISO-8601 timestamps"
        )
        assert "output" in call or call.get("error") is not None, (
            "tool call must carry an output, or an error when it failed"
        )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_scenario_id)
def test_agent_returns_state_without_raising(invoke, scenario):
    """Every behavior returns a state dict with a str response and never raises (B2, B3, B6, B7)."""
    run = invoke(scenario)
    assert run.trace.get("status") == "ok", f"agent raised: {run.trace.get('error')!r}"
    _response(run)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_scenario_id)
def test_tool_calls_follow_behavioral_graph(invoke, scenario):
    """Observed tool calls form a graph path, and the trajectory records them in order."""
    run = invoke(scenario)
    calls = _ordered_calls(run.trace)
    names = [call.get("name") for call in calls]
    path = _graph_path(calls)
    assert path is not None, f"tool sequence {names} is not a path of the behavioral graph"
    if scenario.kind == SUPPORT:
        assert path != NO_TOOL_PATH, "support-related query never reached B4: handle_support was not called"
    else:
        assert path == NO_TOOL_PATH, f"{scenario.kind} input must not call any tool (B3), observed {names}"
    assert _is_subsequence(names, run.trace.get("trajectory", [])), (
        "trajectory does not record the observed tool calls in start-time order"
    )


@pytest.mark.parametrize("scenario", SUPPORT_SCENARIOS, ids=_scenario_id)
def test_handle_support_called_once_with_query_and_category(invoke, scenario):
    """B4: handle_support runs once, with the unchanged query and a Category from B1."""
    calls = _tool_calls(invoke(scenario).trace)
    handle_support_calls = _called(calls, SUPPORT_TOOL)
    assert len(handle_support_calls) == 1, (
        f"handle_support must be called exactly once, observed {len(handle_support_calls)} calls"
    )
    arguments = handle_support_calls[0].get("input", {})
    assert set(arguments) == {"query", "category"}, (
        f"handle_support must be called with query and category, observed {sorted(arguments)}"
    )
    assert arguments["query"] == scenario.query, "handle_support must receive the unchanged customer query"
    assert arguments["category"] in CATEGORIES, (
        f"category {arguments['category']!r} is not an allowed Category"
    )


@pytest.mark.parametrize("scenario", SUPPORT_SCENARIOS, ids=_scenario_id)
def test_escalate_receives_query_and_exact_suggestion_after_handle_support(invoke, scenario):
    """B5: escalate runs after handle_support returns, with the original query and the exact suggestion."""
    run = invoke(scenario)
    calls = _ordered_calls(run.trace)
    if _graph_path(calls) not in (HANDOFF_PATH, ESCALATE_FAILED_PATH):
        pytest.skip("tool sequence is not a success path, so B5 is not evaluated (see the graph test)")
    handle_support, escalate = calls
    arguments = escalate["input"]
    assert set(arguments) == {"query", "support_response"}, (
        f"escalate must be called with query and support_response, observed {sorted(arguments)}"
    )
    assert arguments["query"] == scenario.query, "escalate must receive the original query"
    assert arguments["support_response"] == handle_support["output"], (
        "escalate must receive the exact string returned by handle_support"
    )
    assert _parse_time(escalate["start"]) >= _parse_time(handle_support["end"]), (
        "escalate started before handle_support returned"
    )


@pytest.mark.parametrize("scenario", SUPPORT_SCENARIOS, ids=_scenario_id)
def test_successful_handoff_returns_formatted_state(invoke, scenario):
    """B2: after escalate confirms, state['response'] holds Category, Sentiment and the unchanged confirmation."""
    run = invoke(scenario)
    calls = _ordered_calls(run.trace)
    if _graph_path(calls) != HANDOFF_PATH:
        pytest.skip("escalate did not confirm forwarding in this run, so B2 is not triggered")
    handle_support, escalate = calls
    response = _response(run)
    prefix = f"Category: {handle_support['input']['category']}\nSentiment: "
    assert response.startswith(prefix), "response must start with the B1 Category, then the Sentiment"
    sentiment, newline, tail = response[len(prefix):].partition("\n")
    assert newline and sentiment in SENTIMENTS, f"Sentiment must be one of {SENTIMENTS} on its own line"
    assert tail == f"Response: {escalate['output']}", (
        "response must end with the unchanged escalate confirmation"
    )


@pytest.mark.parametrize("scenario", SUPPORT_SCENARIOS, ids=_scenario_id)
def test_tool_failure_halts_path_and_reports_that_tool(invoke, scenario):
    """B7: a failed handle_support stops the run; a failed escalate yields no handoff; no retries."""
    run = invoke(scenario)
    calls = _ordered_calls(run.trace)
    path = _graph_path(calls)
    if path == SUPPORT_FAILED_PATH:
        assert not _called(calls, ESCALATE_TOOL), "escalate must not be called after handle_support fails"
        assert _response(run) == SUPPORT_TOOL_ERROR, "the failed handle_support must be reported in state['response']"
    elif path == ESCALATE_FAILED_PATH:
        assert _response(run) == ESCALATE_TOOL_ERROR, (
            "a failed escalate must be reported in state['response'], with no handoff confirmation"
        )
    elif path is None and any(_outcome(call) == FAILED for call in calls):
        pytest.fail(f"a tool failed but the run continued with {[call.get('name') for call in calls]}")
    else:
        pytest.skip("no tool failed in this run, so B7 is not triggered")


@pytest.mark.parametrize("scenario", REJECTED_SCENARIOS, ids=_scenario_id)
def test_rejected_input_calls_no_tool_and_returns_b3_error(invoke, scenario):
    """B3: unrelated and invalid input call neither tool and return the B3 error in state['response']."""
    run = invoke(scenario)
    calls = _tool_calls(run.trace)
    assert not calls, (
        f"{scenario.kind} input must not call either tool, observed {[call.get('name') for call in calls]}"
    )
    assert _response(run) == INVALID_INPUT_RESPONSE, "rejected input must return the B3 error in state['response']"


@pytest.mark.parametrize("scenario", SUPPORT_SCENARIOS, ids=_scenario_id)
def test_required_tools_called_in_spec_order(invoke, scenario):
    """Tool correctness on the handoff path: handle_support, then escalate, once each."""
    run = invoke(scenario)
    calls = _ordered_calls(run.trace)
    if _graph_path(calls) != HANDOFF_PATH:
        pytest.skip("handoff path not observed in this run")
    test_case = LLMTestCase(
        input=scenario.query,
        actual_output=_response(run),
        tools_called=[
            ToolCall(name=call["name"], input_parameters=call["input"], output=call["output"])
            for call in calls
        ],
        expected_tools=[ToolCall(name=SUPPORT_TOOL), ToolCall(name=ESCALATE_TOOL)],
    )
    metric = ToolCorrectnessMetric(strict_mode=True, should_exact_match=True, should_consider_ordering=True)
    assert_test(test_case=test_case, metrics=[metric])


def test_single_analytical_llm_call_is_evidenced(invoke):
    """C1, B1, B6: one analytical LLM call, without retry, is evidenced for every string input."""
    missing = [s.name for s in SUPPORT_SCENARIOS + UNRELATED_SCENARIOS if not _llm_call_events(invoke(s).trace)]
    assert not missing, (
        f"insufficient trace evidence for {missing}: the trace records no LLM-call events, so one analytical "
        "call (C1), that call producing Category and Sentiment (B1), the absence of retries (B6) and the "
        "origin of the Sentiment in a successful result cannot be evaluated"
    )


def test_handle_support_llm_use_is_evidenced(invoke):
    """handle_support may make one LLM call but must not repeat the Category or Sentiment analysis."""
    missing = [s.name for s in SUPPORT_SCENARIOS if not _llm_call_events(invoke(s).trace)]
    assert not missing, (
        f"insufficient trace evidence for {missing}: the trace records no LLM-call events, so the optional "
        "LLM call inside handle_support and the absence of a repeated analysis cannot be evaluated"
    )


def test_invalid_input_rejected_before_llm_call_is_evidenced(invoke):
    """B3: None, non-string, empty and whitespace-only input fail before any LLM call."""
    missing = [s.name for s in INVALID_SCENARIOS if not _llm_call_events(invoke(s).trace)]
    assert not missing, (
        f"insufficient trace evidence for {missing}: the trace records no LLM-call events, so the error "
        "cannot be evaluated as returned before any LLM call"
    )
