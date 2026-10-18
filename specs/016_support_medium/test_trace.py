"""Trace-based evaluation of the customer support agent (spec 016_support_medium).

Each scenario is a module-scoped fixture that invokes the agent exactly once
through the benchmark wrapper, ``traced(agent)``. The tests evaluate the captured
trace only: tool invocations with their arguments, outcomes and timing, and the
trajectory. The agent's output is never read.

Routing follows the specification's Sentiment rules: Negative -> escalate;
Positive or Neutral -> handle_support. Invalid and unrelated input must not invoke
any tool. Scenario inputs are chosen so that the Sentiment is unambiguous.

Requirements the trace cannot evidence fail on purpose rather than pass by
inference. The trace format records tool invocations only, so LLM calls are not
observable:
- B1 (the analytical LLM call, its retries, and its ordering before the tool
  call), and handle_support not repeating the analysis.
- B3 "before any LLM call".
- B5 category equality with the Category produced by B1 (the B1 result is not
  recorded).
- B7 (tool failure): the scenario inputs cannot induce a tool failure, so the test
  fails unless the trace shows one.

The routing test uses DeepEval's ToolCorrectnessMetric. DeepEval 3.9.5 builds its
default judge model when a metric is constructed, so OPENAI_API_KEY must be set;
names-only matching makes no model call.
"""
from dataclasses import dataclass
from typing import Optional

import pytest
from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from trace_capture import traced
from agent import agent

ESCALATE = "escalate"
HANDLE_SUPPORT = "handle_support"
TOOLS = (ESCALATE, HANDLE_SUPPORT)

# Tools table: the arguments each tool is declared with.
DECLARED_ARGUMENTS = {ESCALATE: {"query"}, HANDLE_SUPPORT: {"query", "category"}}

# Policies/Invariants: the allowed Category values.
ALLOWED_CATEGORIES = ("Technical", "Billing", "General")

# Sentiment per the specification: Negative = dissatisfaction or frustration;
# Positive = satisfaction or appreciation; Neutral = neither.
NEGATIVE_BILLING = (
    "I was charged twice for my subscription this month and nobody has answered "
    "my emails about the refund. This is unacceptable and I am furious."
)
NEGATIVE_TECHNICAL = (
    "The dashboard crashes every time I open it, and I am really angry that this "
    "still has not been fixed after a month."
)
POSITIVE_TECHNICAL = (
    "Thank you! The steps you sent to reset my password worked perfectly, and I am "
    "very happy with the help."
)
NEUTRAL_BILLING = "How can I update the billing address on my next invoice?"
NEUTRAL_GENERAL = "What are your customer support hours?"
UNRELATED = "What is the capital of France?"


@dataclass(frozen=True)
class Scenario:
    query: object  # the entrypoint argument, sent unchanged
    expected_tool: Optional[str]  # tool the specification routes to; None = no tool
    trace: dict


def _run(query, expected_tool=None):
    """Invoke the agent once under benchmark instrumentation; only the trace is kept."""
    _output, trace = traced(agent)(query)
    return Scenario(query=query, expected_tool=expected_tool, trace=trace)


@pytest.fixture
def scenario(request):
    """Resolve a named scenario fixture; each one invokes the agent once per module."""
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="module")
def negative_billing():
    return _run(NEGATIVE_BILLING, ESCALATE)


@pytest.fixture(scope="module")
def negative_technical():
    return _run(NEGATIVE_TECHNICAL, ESCALATE)


@pytest.fixture(scope="module")
def positive_technical():
    return _run(POSITIVE_TECHNICAL, HANDLE_SUPPORT)


@pytest.fixture(scope="module")
def neutral_billing():
    return _run(NEUTRAL_BILLING, HANDLE_SUPPORT)


@pytest.fixture(scope="module")
def neutral_general():
    return _run(NEUTRAL_GENERAL, HANDLE_SUPPORT)


@pytest.fixture(scope="module")
def input_none():
    return _run(None)


@pytest.fixture(scope="module")
def input_non_string():
    return _run(12345)


@pytest.fixture(scope="module")
def input_empty():
    return _run("")


@pytest.fixture(scope="module")
def input_whitespace():
    return _run(" \n\t ")


@pytest.fixture(scope="module")
def input_unrelated():
    return _run(UNRELATED)


@pytest.fixture(scope="module")
def llm_unavailable():
    """B6 trigger: the specification names missing LLM configuration as an analysis failure.

    The variable is removed only while the agent runs, and restored afterwards.
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        return _run(NEUTRAL_BILLING)


HANDLE_SUPPORT_SCENARIOS = ["positive_technical", "neutral_billing", "neutral_general"]
SUPPORT_SCENARIOS = ["negative_billing", "negative_technical", *HANDLE_SUPPORT_SCENARIOS]
INVALID_SCENARIOS = [
    "input_none",
    "input_non_string",
    "input_empty",
    "input_whitespace",
    "input_unrelated",
]
PRE_LLM_INVALID_SCENARIOS = ["input_none", "input_non_string", "input_empty", "input_whitespace"]


def _tool_calls(trace):
    assert "tool_calls" in trace, "trace does not record tool_calls"
    return trace["tool_calls"]


def _tool_names(trace):
    return [call["name"] for call in _tool_calls(trace)]


def _trajectory(trace):
    assert "trajectory" in trace, "trace does not record trajectory"
    return trace["trajectory"]


def _tool_failed(call):
    """B7 trigger: the tool raised, or returned something other than a non-empty string."""
    if "error" in call:
        return True
    output = call.get("output")
    return not (isinstance(output, str) and output.strip())


def _not_evidenced(requirement, reason):
    """Fail rather than infer the behavior: the trace cannot show it."""
    pytest.fail(f"{requirement} is not evidenced by the trace: {reason}", pytrace=False)


@pytest.mark.parametrize("scenario", SUPPORT_SCENARIOS, indirect=True)
def test_support_query_invokes_routed_tool_exactly_once(scenario):
    """B4, B5 and Constraints: one call to the Sentiment-routed tool; no retry or substitution."""
    test_case = LLMTestCase(
        input=scenario.query,
        tools_called=[
            ToolCall(name=call["name"], input_parameters=call["input"])
            for call in _tool_calls(scenario.trace)
        ],
        expected_tools=[ToolCall(name=scenario.expected_tool)],
    )
    # Names and call count only; arguments are asserted explicitly in the next test.
    metric = ToolCorrectnessMetric(
        threshold=1.0,
        should_exact_match=True,
        evaluation_params=[],
        async_mode=False,
    )
    assert_test(test_case=test_case, metrics=[metric], run_async=False)


@pytest.mark.parametrize("scenario", SUPPORT_SCENARIOS, indirect=True)
def test_routed_tool_receives_declared_arguments_with_unchanged_query(scenario):
    """Tools table and Constraints: escalate(query); handle_support(query, category)."""
    calls = _tool_calls(scenario.trace)
    assert calls, "no tool call recorded"
    for call in calls:
        assert call["name"] in DECLARED_ARGUMENTS, f"undeclared tool {call['name']!r} called"
        assert set(call["input"]) == DECLARED_ARGUMENTS[call["name"]], (
            f"{call['name']} called with arguments {sorted(call['input'])}"
        )
        assert call["input"]["query"] == scenario.query, f"{call['name']} received a changed query"


@pytest.mark.parametrize("scenario", HANDLE_SUPPORT_SCENARIOS, indirect=True)
def test_handle_support_category_is_an_allowed_value(scenario):
    """Policies and B5: the category passed to handle_support is an allowed Category."""
    calls = [call for call in _tool_calls(scenario.trace) if call["name"] == HANDLE_SUPPORT]
    assert len(calls) == 1, f"expected one handle_support call, found {len(calls)}"
    category = calls[0]["input"].get("category")
    assert category in ALLOWED_CATEGORIES, f"category {category!r} is not an allowed Category"


def test_handle_support_category_is_the_b1_category(neutral_billing):
    """B5 Constraint: the category argument is the Category produced by B1."""
    trace = neutral_billing.trace
    calls = [call for call in _tool_calls(trace) if call["name"] == HANDLE_SUPPORT]
    assert len(calls) == 1, f"expected one handle_support call, found {len(calls)}"
    _not_evidenced(
        "category equals the Category produced by B1",
        "the trace records the category argument but not the B1 result it must equal",
    )


@pytest.mark.parametrize("scenario", SUPPORT_SCENARIOS, indirect=True)
def test_trajectory_contains_routed_tool_and_not_the_other(scenario):
    """Behavioral graph: B4 and B5 are exclusive branches; the trajectory follows one."""
    other = HANDLE_SUPPORT if scenario.expected_tool == ESCALATE else ESCALATE
    trajectory = _trajectory(scenario.trace)
    assert scenario.expected_tool in trajectory, f"{scenario.expected_tool} missing from trajectory"
    assert other not in trajectory, f"{other} appears in the trajectory"


@pytest.mark.parametrize("scenario", SUPPORT_SCENARIOS, indirect=True)
def test_tool_calls_record_complete_evidence(scenario):
    """Trace evidence: each tool call records its name, arguments, timing and outcome."""
    calls = _tool_calls(scenario.trace)
    assert calls, "no tool call recorded"
    for call in calls:
        missing = [key for key in ("name", "input", "start", "end") if key not in call]
        assert not missing, f"tool call lacks {missing}"
        assert "output" in call or "error" in call, f"{call['name']} has no output or error"


@pytest.mark.parametrize("scenario", SUPPORT_SCENARIOS + INVALID_SCENARIOS, indirect=True)
def test_agent_returns_instead_of_raising(scenario):
    """B2, B3 and B7: every path returns state; an exception is not a valid outcome."""
    assert scenario.trace["status"] == "ok", f"agent raised: {scenario.trace.get('error')}"


@pytest.mark.parametrize("scenario", SUPPORT_SCENARIOS, indirect=True)
def test_failed_tool_call_is_not_retried_or_substituted(scenario):
    """B7 Constraints: after a failed tool call there is no retry and no substitution."""
    if not any(_tool_failed(call) for call in _tool_calls(scenario.trace)):
        _not_evidenced(
            "B7 (tool failure handling)",
            "no tool call failed in this run, and the scenario inputs cannot induce a tool failure",
        )
    assert _tool_names(scenario.trace) == [scenario.expected_tool], (
        f"after a failed tool call the trace shows {_tool_names(scenario.trace)}"
    )


@pytest.mark.parametrize("scenario", INVALID_SCENARIOS, indirect=True)
def test_invalid_or_unrelated_input_invokes_no_tool(scenario):
    """B3 and B6 Constraints: for invalid or unrelated input, neither tool is called."""
    assert _tool_names(scenario.trace) == [], "a tool was called for input that must not reach one"
    assert not any(tool in _trajectory(scenario.trace) for tool in TOOLS), (
        "a tool appears in the trajectory"
    )


@pytest.mark.parametrize("scenario", PRE_LLM_INVALID_SCENARIOS, indirect=True)
def test_invalid_input_returns_before_any_llm_call(scenario):
    """B3: None, non-string, empty or whitespace-only input returns before any LLM call."""
    assert _tool_names(scenario.trace) == [], "a tool was called for invalid input"
    _not_evidenced(
        "B3 (no LLM call before returning)",
        "the trace records no LLM-call events, so the absence of an LLM call cannot be observed",
    )


def test_analysis_is_one_llm_call_and_is_never_retried():
    """B1 Constraints: one analytical LLM call per non-empty input, never retried."""
    _not_evidenced(
        "B1 (one analytical LLM call, no retry)",
        "the trace records tool invocations only; no event identifies the analytical LLM call",
    )


def test_routed_tool_call_follows_the_analysis():
    """Behavioral graph: B1 completes before B4 or B5 starts."""
    _not_evidenced(
        "B1 -> B4/B5 ordering",
        "the trace has no analysis event, so the order of analysis and tool call is not observable",
    )


def test_handle_support_does_not_repeat_the_analysis():
    """handle_support Constraint: the tool does not repeat the Category or Sentiment analysis."""
    _not_evidenced(
        "handle_support not repeating the Category/Sentiment analysis",
        "LLM calls made inside a tool are not recorded in the trace",
    )


def test_analysis_failure_invokes_no_tool(llm_unavailable):
    """B6 Constraints: a failed analysis calls neither tool, and the agent returns the error."""
    trace = llm_unavailable.trace
    assert trace["status"] == "ok", f"agent raised: {trace.get('error')}"
    called = _tool_names(trace)
    if called:
        pytest.fail(
            f"{called} called although the LLM configuration was removed: the induced failure "
            "did not occur or B6 is violated; B6 cannot be verified from this run",
            pytrace=False,
        )
    assert not any(tool in _trajectory(trace) for tool in TOOLS), "a tool appears in the trajectory"
