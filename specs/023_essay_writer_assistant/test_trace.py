"""Trace-based evaluation of the Essay Writer Agent (spec 023).

The agent entrypoint is `agent(user_request) -> (output, trace)`. Every
behavioral check below is made on the trace; the output is only used for the
contract fields the spec defines (status / user_request / response) and is
never used to infer that a behavior happened.

Oracle: the behavior graph of the spec
    B1 -> B2 (fetch_topic) -> B3 (LLM writes essay) -> B4 (save_to_file) -> B5
    B1/B2/B3/B4 failure -> B6

The trace format is not fixed by the spec, so a tolerant normalizer accepts a
list of events (dicts / objects) or a dict wrapping such a list under
"events" / "trace" / "steps" / "messages". Tool calls are recognised from
events such as {"type": "tool_call", "name": "fetch_topic"}, or LangChain-style
`tool_calls` lists. If a requirement cannot be proven from the trace, the test
fails (no inference).

Set ESSAY_DOC_URL to a real public Google Doc URL to run the success path.
"""
import os
import re
import time

import pytest
from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from agent import agent

DOC_URL = os.getenv(
    "ESSAY_DOC_URL",
    "https://docs.google.com/document/d/REPLACE_WITH_PUBLIC_DOC_ID/edit",
)
SUCCESS_REQUEST = f"Write an essay which the theme is available at: {DOC_URL}"
TIME_LIMIT = 30
FETCH, SAVE = "fetch_topic", "save_to_file"
CONTRACT_KEYS = {"status", "user_request", "response"}


# --------------------------------------------------------------------------
# Trace helpers
# --------------------------------------------------------------------------
def _get(obj, *keys, default=None):
    for k in keys:
        if isinstance(obj, dict) and k in obj and obj[k] is not None:
            return obj[k]
        if not isinstance(obj, dict) and getattr(obj, k, None) is not None:
            return getattr(obj, k)
    return default


def normalize_trace(trace):
    """Return a flat list of events (dict-like or objects)."""
    if trace is None:
        return []
    if isinstance(trace, dict):
        for key in ("events", "trace", "steps", "messages", "spans"):
            if key in trace and isinstance(trace[key], (list, tuple)):
                return list(trace[key])
        return []
    if isinstance(trace, (list, tuple)):
        return list(trace)
    return []


def _kind(ev):
    return str(_get(ev, "type", "event", "kind", "event_type", default="")).lower()


def tool_names_of(ev):
    """Names of tools invoked/emitted by this event (may be several)."""
    names = []
    for tc in _get(ev, "tool_calls", default=[]) or []:
        n = _get(tc, "name", "tool", "tool_name")
        if n:
            names.append(str(n))
    kind = _kind(ev)
    if "tool" in kind or _get(ev, "tool", "tool_name") is not None:
        n = _get(ev, "tool", "tool_name", "name")
        if n:
            names.append(str(n))
    return names


def is_tool_execution(ev):
    """Event proving that a tool was actually executed (not only requested)."""
    kind = _kind(ev)
    if any(w in kind for w in ("result", "end", "output", "response", "executed", "finish")):
        return bool(tool_names_of(ev))
    if kind in ("tool", "tool_call", "tool_execution", "tool_message", "toolmessage"):
        return bool(tool_names_of(ev))
    return False


def is_llm_event(ev):
    kind = _kind(ev)
    return any(w in kind for w in ("llm", "model", "ai", "chat", "write", "essay")) and not tool_names_of(ev)


def tool_calls_in_order(events):
    """[(index, tool_name)] for executed/emitted tool calls, in trace order."""
    out = []
    for i, ev in enumerate(events):
        for n in tool_names_of(ev):
            out.append((i, n))
    return out


def first_index(events, tool, executed_only=False):
    for i, ev in enumerate(events):
        if tool in tool_names_of(ev) and (not executed_only or is_tool_execution(ev)):
            return i
    return None


def run(request):
    start = time.monotonic()
    result = agent(request)
    elapsed = time.monotonic() - start
    assert isinstance(result, tuple) and len(result) == 2, "agent must return (output, trace)"
    output, trace = result
    return output, normalize_trace(trace), elapsed


@pytest.fixture(scope="module")
def success_run():
    return run(SUCCESS_REQUEST)


# --------------------------------------------------------------------------
# B1 / B6: invalid input -> failure, no tools, LLM-driven flow not forced
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["", "   ", "\n\t  "])
def test_b1_blank_input_ends_in_failure_without_tool_calls(bad):
    output, events, elapsed = run(bad)
    assert isinstance(output, dict) and set(output) == CONTRACT_KEYS
    assert output["status"] == "failure"
    assert output["user_request"] == bad
    assert elapsed <= TIME_LIMIT
    # Trace must be present and show that no tool was called (B1 -> B6).
    assert events, "trace is empty: cannot prove B1 -> B6 path"
    assert tool_calls_in_order(events) == [], "tools must not be forced after invalid input"


def test_b1_request_without_url_ends_in_failure_without_tools():
    req = "Write an essay about climate change."
    output, events, elapsed = run(req)
    assert output["status"] == "failure"
    assert output["user_request"] == req
    assert events, "trace is empty: cannot prove B1 -> B6 path"
    assert first_index(events, FETCH) is None
    assert first_index(events, SAVE) is None


# --------------------------------------------------------------------------
# B2 -> B6: retrieval failure must not force further tool calls
# --------------------------------------------------------------------------
def test_b2_retrieval_failure_goes_to_b6_without_save():
    req = "Write an essay, theme at: https://docs.google.com/document/d/INVALID_ID_000000/edit"
    output, events, elapsed = run(req)
    assert set(output) == CONTRACT_KEYS
    assert output["status"] == "failure"
    assert output["user_request"] == req
    assert elapsed <= TIME_LIMIT
    assert first_index(events, FETCH) is not None, "trace must show the fetch_topic attempt"
    assert first_index(events, SAVE) is None, "save_to_file must not be called after retrieval failure"


# --------------------------------------------------------------------------
# Happy path B1 -> B2 -> B3 -> B4 -> B5
# --------------------------------------------------------------------------
def test_trace_is_present_for_success_path(success_run):
    _, events, _ = success_run
    assert events, "trace is empty: no evidence of execution"


def test_b2_fetch_topic_called_with_supplied_url(success_run):
    _, events, _ = success_run
    idx = first_index(events, FETCH)
    assert idx is not None, "trace has no fetch_topic call"
    ev = events[idx]
    args = _get(ev, "args", "input", "arguments", "tool_input", default=None)
    if args is None:
        for tc in _get(ev, "tool_calls", default=[]) or []:
            if _get(tc, "name") == FETCH:
                args = _get(tc, "args", "arguments")
    assert args is not None, "trace does not expose fetch_topic arguments"
    assert DOC_URL.split("/edit")[0] in str(args), "fetch_topic must use the URL from user_request"


def test_b2_fetch_topic_actually_executed(success_run):
    _, events, _ = success_run
    assert first_index(events, FETCH, executed_only=True) is not None, \
        "trace has no evidence that fetch_topic was executed"


def test_b4_save_to_file_called_and_executed(success_run):
    _, events, _ = success_run
    assert first_index(events, SAVE) is not None, "trace has no save_to_file call"
    assert first_index(events, SAVE, executed_only=True) is not None, \
        "trace has no evidence that save_to_file was executed"


def test_order_fetch_before_save(success_run):
    _, events, _ = success_run
    f = first_index(events, FETCH, executed_only=True)
    s = first_index(events, SAVE)
    assert f is not None and s is not None, "missing fetch_topic or save_to_file in trace"
    assert f < s, "fetch_topic must precede save_to_file"


def test_b3_essay_written_between_fetch_and_save(success_run):
    """An LLM step must be observable after retrieval and before saving."""
    _, events, _ = success_run
    f = first_index(events, FETCH, executed_only=True)
    s = first_index(events, SAVE)
    assert f is not None and s is not None
    between = events[f + 1:s]
    assert any(is_llm_event(e) or _get(e, "node", "name") in ("write", "writer", "generate")
               for e in between), "no LLM/writing event observed between B2 and B4"


def test_save_content_comes_from_llm_not_empty(success_run):
    _, events, _ = success_run
    s = first_index(events, SAVE)
    assert s is not None
    ev = events[s]
    args = _get(ev, "args", "input", "arguments", "tool_input", default=None)
    if args is None:
        for tc in _get(ev, "tool_calls", default=[]) or []:
            if _get(tc, "name") == SAVE:
                args = _get(tc, "args", "arguments")
    assert isinstance(args, dict), "trace does not expose save_to_file arguments"
    assert isinstance(args.get("content"), str) and len(args["content"].split()) > 50
    assert isinstance(args.get("topic"), str) and args["topic"].strip()


def test_tools_called_exactly_once_each_in_order(success_run):
    _, events, _ = success_run
    names = [n for _, n in tool_calls_in_order(events)]
    assert names, "no tool calls in trace"
    # Collapse request/result duplicates of the same call; order must be fetch -> save.
    collapsed = [n for i, n in enumerate(names) if i == 0 or names[i - 1] != n]
    assert collapsed == [FETCH, SAVE], f"unexpected tool sequence: {collapsed}"


def test_deepeval_tool_correctness_ordering(success_run):
    """DeepEval ToolCorrectnessMetric over the tools observed in the trace."""
    output, events, _ = success_run
    names = [n for _, n in tool_calls_in_order(events)]
    collapsed = [n for i, n in enumerate(names) if i == 0 or names[i - 1] != n]
    assert collapsed, "no tool calls observed in trace"
    case = LLMTestCase(
        input=SUCCESS_REQUEST,
        actual_output=str(_get(output, "response", default="")),
        tools_called=[ToolCall(name=n) for n in collapsed],
        expected_tools=[ToolCall(name=FETCH), ToolCall(name=SAVE)],
    )
    metric = ToolCorrectnessMetric(threshold=1.0, should_consider_ordering=True)
    assert_test(case, [metric])


def test_b5_success_only_after_save_in_trace(success_run):
    output, events, elapsed = success_run
    assert set(output) == CONTRACT_KEYS
    assert output["status"] == "success", output.get("response")
    assert output["user_request"] == SUCCESS_REQUEST
    assert first_index(events, SAVE, executed_only=True) is not None, \
        "success reported without trace evidence of a completed save"
    assert elapsed <= TIME_LIMIT


def test_b5_response_contains_path_from_save_event(success_run):
    output, events, _ = success_run
    s = first_index(events, SAVE, executed_only=True)
    assert s is not None
    result = _get(events[s], "output", "result", "content", "return", default=None)
    assert result is not None, "trace does not expose save_to_file result"
    path = str(result).strip()
    assert os.path.isabs(path), "save_to_file must return an absolute path"
    assert path in output["response"]
    assert os.path.isfile(path)
    assert re.search(r"\w", open(path, encoding="utf-8").read())


def test_no_save_before_retrieval_in_any_prefix(success_run):
    """Ordering constraint: no save_to_file event may appear before fetch_topic."""
    _, events, _ = success_run
    f = first_index(events, FETCH)
    assert f is not None
    assert all(not (SAVE in tool_names_of(e)) for e in events[: f + 1])
