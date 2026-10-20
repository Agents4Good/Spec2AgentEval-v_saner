"""Trace-based evaluation of the essay assistant (spec 013).

The agent entrypoint is `agent(user_request) -> (output, trace)`. Every
behavioral assertion below is derived from the returned trace; the output is only
checked for the strict contract (keys / status / user_request).

The trace is treated as a sequence of event records (dicts or objects). Because
the spec does not impose a trace schema, events are classified by tolerant,
generic fields (type/event/kind/name/node/tool...). If the trace does not give
enough evidence for a requirement, the test FAILS instead of inferring it.

Behavior graph used as oracle:
  B1 -> B2 (empty)                         -> failure, no LLM/tool
  B1 -> B3 -> B4 (refusal)                 -> success, no tool
  B1 -> B3 -> B5 (fetch_document) -> B7    -> success
  B1 -> B3 -> B6 (search_about_topic) -> B7 -> success
  B5/B6 failure -> B8                      -> failure, no new LLM/tool call
"""
import time

import pytest

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from agent import agent

FETCH = "fetch_document"
SEARCH = "search_about_topic"
TOOLS = (FETCH, SEARCH)

TOOL_KEYS = ("tool", "tool_name", "name", "function", "action", "tool_call")
LABEL_KEYS = ("type", "event", "kind", "name", "node", "step", "phase",
              "behavior", "action", "tool", "tool_name", "role", "component")
END_MARKERS = ("end", "result", "output", "finish", "complete", "return", "done")
LLM_MARKERS = ("llm", "chat_model", "chatmodel", "chatopenai", "openai", "gpt")

DOC_URL = "https://www.gutenberg.org/files/1342/1342-0.txt"
DOC_REQUEST = f"Please read the essay at {DOC_URL} and give me feedback on its structure."
SEARCH_REQUEST = "Research the main arguments and definitions about the topic 'climate change policy' for my essay."
BAD_URL_REQUEST = "Evaluate the essay at https://this-domain-does-not-exist-essay-xyz.invalid/essay.txt and give feedback."


# --------------------------------------------------------------------------- helpers
def _get(event, key):
    if isinstance(event, dict):
        return event.get(key)
    return getattr(event, key, None)


def _events(trace):
    if isinstance(trace, dict):
        for key in ("events", "trace", "steps"):
            if isinstance(trace.get(key), (list, tuple)):
                trace = trace[key]
                break
    assert isinstance(trace, (list, tuple)), f"trace must be a sequence of events, got {type(trace)}"
    assert len(trace) > 0, "empty trace: no evidence that any behavior occurred"
    return list(trace)


def _label(event):
    parts = []
    for key in LABEL_KEYS:
        value = _get(event, key)
        if isinstance(value, str):
            parts.append(value.lower())
    return " ".join(parts)


def _tool_name(event):
    for key in TOOL_KEYS:
        value = _get(event, key)
        if isinstance(value, dict):
            value = value.get("name") or value.get("tool")
        if value in TOOLS:
            return value
    return None


def _is_end(event):
    return any(m in _label(event) for m in END_MARKERS)


def _is_llm(event):
    return _tool_name(event) is None and any(m in _label(event) for m in LLM_MARKERS)


def tool_invocations(trace):
    """Return [(index, tool_name)] — one entry per tool invocation in the trace."""
    events = _events(trace)
    tool_events = [(i, _tool_name(e), e) for i, e in enumerate(events) if _tool_name(e)]
    starts = [(i, n) for i, n, e in tool_events if not _is_end(e)]
    # If only completion-style events exist, fall back to them as the evidence.
    return starts if starts else [(i, n) for i, n, _ in tool_events]


def llm_indices(trace):
    return [i for i, e in enumerate(_events(trace)) if _is_llm(e)]


def run(request):
    start = time.monotonic()
    result = agent(request)
    elapsed = time.monotonic() - start
    assert isinstance(result, tuple) and len(result) == 2, "agent must return (output, trace)"
    output, trace = result
    return output, trace, elapsed


def assert_contract(output, request, status):
    assert isinstance(output, dict)
    assert set(output.keys()) == {"status", "user_request", "response"}
    assert all(isinstance(v, str) for v in output.values())
    assert output["status"] == status
    assert output["user_request"] == request


def assert_tool_correctness(trace, expected):
    """DeepEval ToolCorrectnessMetric (deterministic) on tools observed in the trace."""
    called = [ToolCall(name=n) for _, n in tool_invocations(trace)]
    test_case = LLMTestCase(
        input="trace-based tool check",
        actual_output="n/a",
        tools_called=called,
        expected_tools=[ToolCall(name=n) for n in expected],
    )
    metric = ToolCorrectnessMetric(threshold=1.0, should_exact_match=True,
                                   include_reason=False, async_mode=False)
    metric.measure(test_case)
    assert metric.is_successful(), f"tool mismatch: called={[c.name for c in called]} expected={expected}"


def assert_no_tools(trace):
    _events(trace)
    assert tool_invocations(trace) == [], "no tool may be invoked on this path"


# --------------------------------------------------------------------------- B1 -> B2
@pytest.mark.parametrize("request_text", ["", "   ", "\n\t  \n"])
def test_empty_input_b1_to_b2_no_llm_no_tools(request_text):
    output, trace, _ = run(request_text)
    assert_contract(output, request_text, "failure")
    assert_no_tools(trace)
    assert llm_indices(trace) == [], "empty input must end before B3 (no LLM call)"


# --------------------------------------------------------------------------- B3 -> B4
@pytest.mark.parametrize("request_text", [
    "Give me some generic tips for writing a good essay introduction.",
    "Please evaluate my essay, I forgot to include the link.",
    "Research the topic of renewable energy and also read and evaluate the essay at https://example.com/essay.txt",
    "Help me with something for my essay.",
])
def test_refusal_b3_to_b4_llm_decides_without_tools(request_text):
    output, trace, _ = run(request_text)
    assert_contract(output, request_text, "success")
    assert_no_tools(trace)
    assert llm_indices(trace), "refusal must be decided by the LLM (B3): no LLM event in trace"


# --------------------------------------------------------------------------- B3 -> B5 -> B7
def test_document_route_b3_b5_b7_order_and_single_tool():
    output, trace, _ = run(DOC_REQUEST)
    assert_contract(output, DOC_REQUEST, "success")
    invocations = tool_invocations(trace)
    assert [n for _, n in invocations] == [FETCH], f"expected exactly one fetch_document, got {invocations}"
    assert_tool_correctness(trace, [FETCH])

    tool_idx = invocations[0][0]
    llm = llm_indices(trace)
    assert any(i < tool_idx for i in llm), "B3 LLM interpretation must precede fetch_document"
    assert any(i > tool_idx for i in llm), "B7 LLM synthesis must follow fetch_document"


# --------------------------------------------------------------------------- B3 -> B6 -> B7
def test_research_route_b3_b6_b7_order_and_single_tool():
    output, trace, _ = run(SEARCH_REQUEST)
    assert_contract(output, SEARCH_REQUEST, "success")
    invocations = tool_invocations(trace)
    assert [n for _, n in invocations] == [SEARCH], f"expected exactly one search_about_topic, got {invocations}"
    assert_tool_correctness(trace, [SEARCH])

    tool_idx = invocations[0][0]
    llm = llm_indices(trace)
    assert any(i < tool_idx for i in llm), "B3 LLM interpretation must precede search_about_topic"
    assert any(i > tool_idx for i in llm), "B7 LLM synthesis must follow search_about_topic"


# --------------------------------------------------------------------------- B5 -> B8
def test_fetch_failure_b5_to_b8_no_retry_no_fallback_no_new_llm():
    output, trace, _ = run(BAD_URL_REQUEST)
    assert_contract(output, BAD_URL_REQUEST, "failure")
    invocations = tool_invocations(trace)
    assert [n for _, n in invocations] == [FETCH], (
        f"failure path must show exactly one fetch_document and no retry/fallback, got {invocations}"
    )
    tool_idx = invocations[0][0]
    assert not any(i > tool_idx for i in llm_indices(trace)), "B8 must not make a new LLM call"
    events = _events(trace)
    assert not any(_tool_name(e) for e in events[tool_idx + 1:] if not _is_end(e)), \
        "no further tool event may follow the failed tool"


# --------------------------------------------------------------------------- invariants
@pytest.mark.parametrize("request_text", [DOC_REQUEST, SEARCH_REQUEST, BAD_URL_REQUEST])
def test_at_most_one_tool_invocation_and_no_chaining(request_text):
    _, trace, _ = run(request_text)
    invocations = tool_invocations(trace)
    assert len(invocations) <= 1, f"at most one tool call per execution, got {invocations}"
    assert len({n for _, n in invocations}) <= 1


def test_total_time_under_30_seconds_and_trace_present():
    output, trace, elapsed = run(DOC_REQUEST)
    _events(trace)
    assert elapsed < 30, f"execution took {elapsed:.1f}s"
    assert isinstance(output, dict) and output.get("status") in ("success", "failure")
