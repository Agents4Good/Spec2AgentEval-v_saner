"""Trace-based evaluation of the Personal Trainer Agent (spec.md).

The agent entrypoint is expected to return `(output, trace)`. The trace is the
primary evidence: every behavioral claim (B1..B7) is checked against trace
events, never inferred from the output. When the trace has no evidence for a
required behavior, the test fails.

Trace reading is deliberately tolerant of shape: the trace may be a list of
event dicts/objects, or a dict holding such a list under "events"/"trace"/
"steps". A tool call is recognised by an event naming `retrieve_restrictions`
or `search_exercises` (fields "tool", "tool_name", "name", or nested
"tool_calls"). Failure is recognised by an "error"/"exception" field, or a
status/type containing "error"/"fail".

Valid-input tests make real Notion/Tavily/OpenAI calls (spec requires it).
"""
import re

import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from agent import agent, build_graph

RETRIEVE = "retrieve_restrictions"
SEARCH = "search_exercises"
TOOLS = (RETRIEVE, SEARCH)

SECTION_TITLES = [
    "Assessment:",
    "Weekly Plan:",
    "Intensity & Progression:",
    "Nutrition Guidance:",
    "Recovery Protocol:",
    "Important Notes:",
]

VALID_PROFILE = (
    "I am 34 years old, intermediate fitness level. My goal is to build muscle. "
    "I have a pair of dumbbells and a pull-up bar at home. I have a mild "
    "lower-back strain from last month."
)

INVALID_PROFILES = {
    "empty": "",
    "whitespace": "   \n\t  ",
    "meaningless": "asdf ;;; 12#$ qwerty ???",
    "missing_minimum": "I have dumbbells at home and like training in the morning.",
    "injury_but_invalid": "My knee hurts a lot and I have a dumbbell.",
    "contradiction": (
        "I am 30 years old and my goal is to lose fat. I am a complete "
        "beginner and also an elite athlete. I have a shoulder injury but "
        "my shoulders are perfectly healthy."
    ),
}


# --------------------------------------------------------------------------
# Trace helpers
# --------------------------------------------------------------------------
def _get(ev, key, default=None):
    if isinstance(ev, dict):
        return ev.get(key, default)
    return getattr(ev, key, default)


def _events(trace):
    if trace is None:
        return []
    if isinstance(trace, dict):
        for k in ("events", "trace", "steps", "spans"):
            if isinstance(trace.get(k), (list, tuple)):
                return list(trace[k])
        return []
    if isinstance(trace, (list, tuple)):
        return list(trace)
    return []


def _tool_names(ev):
    """Tool names an event refers to (may be several for nested tool_calls)."""
    names = []
    for key in ("tool", "tool_name", "name"):
        v = _get(ev, key)
        if isinstance(v, str) and v in TOOLS:
            names.append(v)
    for call in _get(ev, "tool_calls", None) or []:
        n = _get(call, "name") or _get(call, "tool")
        if n in TOOLS:
            names.append(n)
    return names


def _kind(ev):
    return str(_get(ev, "type") or _get(ev, "event") or _get(ev, "kind") or "").lower()


def _is_failure(ev):
    if _get(ev, "error") or _get(ev, "exception"):
        return True
    blob = f"{_get(ev, 'status', '')} {_kind(ev)}".lower()
    return "error" in blob or "fail" in blob or "exception" in blob


def _is_result(ev):
    """Event that reports the outcome of a tool call (not merely the request)."""
    k = _kind(ev)
    return (
        any(t in k for t in ("result", "end", "output", "return", "response", "error", "fail"))
        or _get(ev, "output") is not None
        or _get(ev, "result") is not None
        or _get(ev, "status") is not None
        or bool(_get(ev, "error") or _get(ev, "exception"))
    )


def tool_events(trace):
    """[(index, tool_name, event)] in trace order."""
    out = []
    for i, ev in enumerate(_events(trace)):
        for name in _tool_names(ev):
            out.append((i, name, ev))
    return out


def tool_sequence(trace):
    return [name for _, name, _ in tool_events(trace)]


def first_index(trace, tool):
    for i, n, _ in tool_events(trace):
        if n == tool:
            return i
    return None


def successful_result_index(trace, tool):
    """Index of the first non-failing result event for a tool, else None."""
    for i, n, ev in tool_events(trace):
        if n == tool and _is_result(ev) and not _is_failure(ev):
            return i
    return None


def has_failure(trace, tool):
    return any(n == tool and _is_failure(ev) for _, n, ev in tool_events(trace))


def run(profile):
    result = agent(profile)
    assert isinstance(result, tuple) and len(result) == 2, (
        "agent(...) must return (output, trace)"
    )
    output, trace = result
    assert isinstance(output, str), "output must be a string"
    return output, trace


def require_trace(trace):
    assert _events(trace), "Trace is empty/unreadable: no evidence of execution"


def _search_query(trace):
    for _, n, ev in tool_events(trace):
        if n == SEARCH:
            for key in ("input", "args", "arguments", "tool_input", "params"):
                v = _get(ev, key)
                if isinstance(v, dict) and isinstance(v.get("query"), str):
                    return v["query"]
                if isinstance(v, str):
                    return v
            for call in _get(ev, "tool_calls", None) or []:
                a = _get(call, "args") or {}
                if isinstance(a, dict) and a.get("query"):
                    return a["query"]
    return None


def _tool_correctness(trace, expected):
    """DeepEval ToolCorrectnessMetric over the tools observed in the trace."""
    seq = tool_sequence(trace)
    tc = LLMTestCase(
        input="trace",
        actual_output="trace",
        tools_called=[ToolCall(name=n) for n in seq],
        expected_tools=[ToolCall(name=n) for n in expected],
    )
    metric = ToolCorrectnessMetric(threshold=1.0, should_consider_ordering=True)
    metric.measure(tc)
    return metric


# --------------------------------------------------------------------------
# B1 / B2: invalid input -> Error:, zero tool calls
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(INVALID_PROFILES))
def test_invalid_input_zero_tool_calls_and_error(name):
    output, trace = run(INVALID_PROFILES[name])
    # Trace must exist and positively show that no tool ran.
    assert trace is not None, "No trace returned"
    assert tool_events(trace) == [], (
        f"Invalid input must produce zero tool calls, got {tool_sequence(trace)}"
    )
    assert output.startswith("Error:"), output
    assert not any(t in output for t in SECTION_TITLES[1:2]), "Error must contain no plan"


def test_invalid_input_with_injury_still_no_tools():
    output, trace = run(INVALID_PROFILES["injury_but_invalid"])
    assert tool_sequence(trace) == []
    assert output.startswith("Error:")


def test_contradiction_error_explains_clarification():
    output, trace = run(INVALID_PROFILES["contradiction"])
    assert tool_sequence(trace) == []
    assert output.startswith("Error:")
    assert len(output) > len("Error:") + 20, "Error should explain what is unclear"


# --------------------------------------------------------------------------
# B3 / B4 / B5 / B6: valid input, dependent sequence
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def valid_run():
    return run(VALID_PROFILE)


def test_valid_trace_has_evidence(valid_run):
    _, trace = valid_run
    require_trace(trace)


def test_retrieve_called_for_valid_input(valid_run):
    _, trace = valid_run
    require_trace(trace)
    assert RETRIEVE in tool_sequence(trace), "B3: retrieve_restrictions not observed"


def test_search_called_for_valid_input(valid_run):
    _, trace = valid_run
    require_trace(trace)
    assert SEARCH in tool_sequence(trace), "B5: search_exercises not observed"


def test_tool_order_matches_behavior_graph(valid_run):
    """DeepEval ToolCorrectness with ordering: retrieve -> search."""
    _, trace = valid_run
    require_trace(trace)
    # Collapse request/result event pairs of the same call into one call.
    calls = []
    for _, n, ev in tool_events(trace):
        if calls and calls[-1][0] == n and _is_result(ev) and not calls[-1][1]:
            calls[-1][1] = True
        elif not calls or calls[-1][0] != n or calls[-1][1]:
            calls.append([n, _is_result(ev)])
    seq = [c[0] for c in calls]
    assert RETRIEVE in seq and SEARCH in seq
    assert seq.index(RETRIEVE) < seq.index(SEARCH)
    metric = _tool_correctness(trace, [RETRIEVE, SEARCH]) if seq == [RETRIEVE, SEARCH] else None
    if metric is not None:
        assert metric.is_successful(), metric.reason


def test_search_starts_only_after_successful_retrieval(valid_run):
    _, trace = valid_run
    require_trace(trace)
    ok = successful_result_index(trace, RETRIEVE)
    s = first_index(trace, SEARCH)
    assert ok is not None, "No evidence of a successful retrieve_restrictions result"
    assert s is not None, "No search_exercises event"
    assert ok < s, "search_exercises started before retrieval succeeded"


def test_tools_not_parallel(valid_run):
    """Tool events of the two tools must not interleave (no parallel calls)."""
    _, trace = valid_run
    require_trace(trace)
    r_done = successful_result_index(trace, RETRIEVE)
    s_first = first_index(trace, SEARCH)
    assert r_done is not None and s_first is not None
    assert r_done < s_first
    # No event may name both tools at once.
    for ev in _events(trace):
        assert len(set(_tool_names(ev))) <= 1, "Both tools in a single step (parallel)"


def test_each_tool_called_without_failure(valid_run):
    _, trace = valid_run
    require_trace(trace)
    assert not has_failure(trace, RETRIEVE)
    assert not has_failure(trace, SEARCH)


def test_consolidation_between_retrieval_and_search(valid_run):
    """B4: an LLM step must be observable between retrieval result and search."""
    _, trace = valid_run
    require_trace(trace)
    r = successful_result_index(trace, RETRIEVE)
    s = first_index(trace, SEARCH)
    assert r is not None and s is not None and r < s
    events = _events(trace)
    between = events[r + 1 : s]
    llm_between = [
        e
        for e in between
        if not _tool_names(e)
        and any(w in (_kind(e) + str(_get(e, "name", "")) + str(_get(e, "node", ""))).lower()
                for w in ("llm", "model", "chat", "agent", "assistant", "ai"))
    ]
    assert llm_between, "No LLM step observed between retrieval and search (B4 consolidation)"


def test_search_query_is_llm_generated_with_goal_and_equipment(valid_run):
    _, trace = valid_run
    require_trace(trace)
    q = _search_query(trace)
    assert q, "Search query not observable in trace"
    assert q != VALID_PROFILE, "Query must be formulated by the LLM, not the raw profile"
    ql = q.lower()
    assert any(w in ql for w in ("muscle", "hypertrophy", "strength")), "goal missing from query"
    assert "dumbbell" in ql or "pull-up" in ql or "pull up" in ql, "equipment missing from query"


def test_search_query_includes_user_restriction(valid_run):
    _, trace = valid_run
    q = _search_query(valid_run[1])
    assert q, "Search query not observable in trace"
    assert re.search(r"back|lumbar|spine", q.lower()), "user restriction missing from query"


def test_no_events_after_search_call_tools_again_before_synthesis(valid_run):
    """B6 follows B5: no retrieve call after search began."""
    _, trace = valid_run
    require_trace(trace)
    s = first_index(trace, SEARCH)
    assert s is not None
    later = [n for i, n, _ in tool_events(trace) if i > s]
    assert RETRIEVE not in later, "retrieve_restrictions called after search began"


def test_output_has_six_sections_in_order(valid_run):
    output, trace = valid_run
    require_trace(trace)
    assert tool_sequence(trace), "Routine returned with no tool evidence in trace"
    pos = []
    for t in SECTION_TITLES:
        assert output.count(t) == 1, f"Section {t!r} must appear exactly once"
        pos.append(output.index(t))
    assert pos == sorted(pos)
    assert "```" not in output
    assert not output.startswith("Error:")


# --------------------------------------------------------------------------
# B7: tool failure -> Error:, chain stops
# --------------------------------------------------------------------------
def test_notion_failure_blocks_search_and_plan(monkeypatch):
    monkeypatch.setenv("NOTION_API_KEY", "")
    output, trace = run(VALID_PROFILE)
    require_trace(trace)
    assert RETRIEVE in tool_sequence(trace), "retrieve_restrictions attempt not observed"
    assert has_failure(trace, RETRIEVE), "Retrieval failure not recorded in trace"
    assert SEARCH not in tool_sequence(trace), "Search must not run after Notion failure"
    assert output.startswith("Error:")
    assert "Weekly Plan:" not in output


def test_tavily_failure_blocks_synthesis(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "")
    output, trace = run(VALID_PROFILE)
    require_trace(trace)
    assert successful_result_index(trace, RETRIEVE) is not None
    assert SEARCH in tool_sequence(trace), "search_exercises attempt not observed"
    assert has_failure(trace, SEARCH), "Search failure not recorded in trace"
    assert output.startswith("Error:")
    assert "Weekly Plan:" not in output


# --------------------------------------------------------------------------
# Interface: build_graph
# --------------------------------------------------------------------------
def test_build_graph_returns_compiled_state_graph():
    from langgraph.graph.state import CompiledStateGraph

    assert isinstance(build_graph(), CompiledStateGraph)
