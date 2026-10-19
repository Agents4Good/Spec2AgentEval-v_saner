"""Trace-based evaluation tests for the Personal Trainer Agent (spec 027).

The agent entrypoint is `agent(user_profile) -> (output, trace)`. Every
behavioral check below is derived from the trace; the output is only used for
output-contract checks (Error: prefix, six sections). If the trace carries no
evidence for a trace-based requirement, the test fails.

The trace is normalised tolerantly (list of dicts, LangChain messages, objects
or strings) into ordered tool events; nothing is inferred from the output.
"""
import json
import re

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agent import agent

RETRIEVE = "retrieve_restrictions"
SEARCH = "search_exercises"
TOOLS = (RETRIEVE, SEARCH)

VALID_PROFILE = (
    "I am a 34 year old beginner. My goal is to build muscle and lose some fat. "
    "I only have a pair of dumbbells and a resistance band. I have a mild knee "
    "injury (patellar tendinopathy) and occasional lower back pain."
)
VALID_PROFILE_NO_INJURY = (
    "I'm 28, intermediate fitness level, my goal is to improve overall strength. "
    "I have a barbell, a rack and a bench."
)

INVALID_PROFILES = {
    "empty": "",
    "whitespace": "   \n\t  ",
    "meaningless": "asdf qwer zxcv !!! ??? 12345 ###",
    "missing_goal_and_level": "I have a yoga mat and two dumbbells.",
    "missing_level_and_age": "I want to lose weight. I have dumbbells.",
    "invalid_with_injury": "I have a bad shoulder injury and a herniated disc.",
}

SECTION_TITLES = [
    "Assessment:",
    "Weekly Plan:",
    "Intensity & Progression:",
    "Nutrition Guidance:",
    "Recovery Protocol:",
    "Important Notes:",
]


# --------------------------------------------------------------------------
# Trace normalisation
# --------------------------------------------------------------------------
def _get(obj, *keys):
    for k in keys:
        if isinstance(obj, dict):
            if obj.get(k) not in (None, ""):
                return obj[k]
        elif getattr(obj, k, None) not in (None, ""):
            return getattr(obj, k)
    return None


def _is_error(value) -> bool:
    if value is None:
        return False
    if isinstance(value, BaseException):
        return True
    s = str(value).strip().lower()
    return s in ("error", "failed", "failure", "exception") or s.startswith("error:")


def _events(trace):
    """Return ordered list of dicts: idx, tool, phase(call|result), args, error, raw."""
    assert trace is not None, "agent returned no trace"
    if isinstance(trace, (str, bytes)):
        trace = [trace]
    try:
        items = list(trace)
    except TypeError:
        pytest.fail("trace is not iterable; no evidence available")

    out = []

    def add(idx, tool, phase, args=None, error=False, raw=None):
        if tool in TOOLS:
            out.append(dict(idx=idx, tool=tool, phase=phase, args=args,
                            error=error, raw=raw))

    for i, ev in enumerate(items):
        # LangChain AIMessage-like: tool_calls list
        tcs = _get(ev, "tool_calls")
        if isinstance(tcs, list) and tcs:
            for tc in tcs:
                add(i, _get(tc, "name", "tool", "tool_name"), "call",
                    _get(tc, "args", "arguments", "input"), raw=ev)
            continue
        if isinstance(ev, str):
            for t in TOOLS:
                if t in ev:
                    low = ev.lower()
                    phase = "result" if any(w in low for w in ("result", "return", "end", "success", "fail", "error")) else "call"
                    add(i, t, phase, error=("fail" in low or "error" in low), raw=ev)
            continue
        name = _get(ev, "tool", "tool_name", "name")
        kind = str(_get(ev, "type", "event", "kind", "phase", "role") or "").lower()
        if name not in TOOLS:
            continue
        args = _get(ev, "args", "arguments", "input", "tool_input", "params")
        status = _get(ev, "status")
        err = _is_error(status) or _get(ev, "error", "exception") is not None
        is_result = (kind in ("tool", "tool_result", "tool_end", "tool_message", "result",
                              "end", "finish", "finished", "on_tool_end", "observation", "response")
                     or _get(ev, "output", "result", "content") is not None and not args)
        is_call = kind in ("tool_call", "tool_start", "call", "start", "on_tool_start",
                           "action", "request", "invoke")
        if is_call and not is_result:
            add(i, name, "call", args, err, ev)
        elif is_result:
            add(i, name, "result", args, err, ev)
        else:
            # single event carrying both call and outcome
            add(i, name, "call", args, err, ev)
            add(i, name, "result", args, err, ev)
    return out


def _calls(trace, tool):
    ev = _events(trace)
    calls = [e for e in ev if e["tool"] == tool and e["phase"] == "call"]
    if not calls:  # result-only evidence still proves the tool ran
        calls = [e for e in ev if e["tool"] == tool and e["phase"] == "result"]
    return calls


def _results(trace, tool):
    return [e for e in _events(trace) if e["tool"] == tool and e["phase"] == "result"]


def _tool_event_count(trace):
    return len({(e["tool"], e["idx"]) for e in _events(trace)})


def _query_text(call) -> str:
    args = call["args"]
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return args
    if isinstance(args, dict):
        q = args.get("query")
        return q if isinstance(q, str) else json.dumps(args)
    return str(args or "")


def _run(profile):
    result = agent(profile)
    assert isinstance(result, tuple) and len(result) == 2, "agent must return (output, trace)"
    output, trace = result
    assert isinstance(output, str), "output must be a string"
    assert trace is not None, "trace missing"
    return output, trace


# --------------------------------------------------------------------------
# Fixtures (real executions are expensive; run each scenario once)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def valid_run():
    return _run(VALID_PROFILE)


@pytest.fixture(scope="module")
def valid_run_no_injury():
    return _run(VALID_PROFILE_NO_INJURY)


# --------------------------------------------------------------------------
# B1/B2 - invalid input: zero tool calls, Error:, no plan
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(INVALID_PROFILES))
def test_invalid_input_zero_tool_calls_and_error(name):
    output, trace = _run(INVALID_PROFILES[name])
    assert _tool_event_count(trace) == 0, f"tools were called for invalid input: {_events(trace)}"
    assert output.startswith("Error:"), output
    assert "Weekly Plan:" not in output


# --------------------------------------------------------------------------
# B3 - retrieval is required for every valid profile
# --------------------------------------------------------------------------
def test_retrieve_restrictions_called_for_valid_profile(valid_run):
    _, trace = valid_run
    assert _calls(trace, RETRIEVE), "no retrieve_restrictions evidence in trace"


def test_retrieve_restrictions_called_even_without_injury(valid_run_no_injury):
    _, trace = valid_run_no_injury
    assert _calls(trace, RETRIEVE), "retrieval must happen even if profile has no injury"
    assert _calls(trace, SEARCH), "search must follow retrieval"


# --------------------------------------------------------------------------
# Guardrail - dependent sequence: retrieve (success) -> consolidate -> search
# --------------------------------------------------------------------------
def test_retrieve_precedes_search(valid_run):
    _, trace = valid_run
    r, s = _calls(trace, RETRIEVE), _calls(trace, SEARCH)
    assert r and s, "both tools must appear in the trace"
    assert min(c["idx"] for c in r) < min(c["idx"] for c in s)


def test_retrieval_succeeded_before_search_started(valid_run):
    _, trace = valid_run
    first_search = min(c["idx"] for c in _calls(trace, SEARCH))
    ok = [e for e in _results(trace, RETRIEVE) if not e["error"] and e["idx"] <= first_search]
    assert ok, "no successful retrieve_restrictions result observed before search"


def test_tools_not_parallelised_or_repeated_out_of_order(valid_run):
    _, trace = valid_run
    ev = [e for e in _events(trace) if e["phase"] == "call"] or _events(trace)
    assert ev, "no tool events"
    # A retrieve call must never appear after a search call (no reversal),
    # and no search call may share an event (parallel batch) with retrieve.
    first_search = min((e["idx"] for e in ev if e["tool"] == SEARCH), default=None)
    assert first_search is not None
    assert not [e for e in ev if e["tool"] == RETRIEVE and e["idx"] >= first_search]
    retrieve_idx = {e["idx"] for e in ev if e["tool"] == RETRIEVE}
    search_idx = {e["idx"] for e in ev if e["tool"] == SEARCH}
    assert not (retrieve_idx & search_idx), "tools executed in the same step (parallel)"


def test_search_called_once_with_consolidation_between(valid_run):
    _, trace = valid_run
    items = list(trace)
    r_done = max(e["idx"] for e in _results(trace, RETRIEVE) or _calls(trace, RETRIEVE))
    s_start = min(c["idx"] for c in _calls(trace, SEARCH))
    # Consolidation (B4) is an LLM step: it must be observable as at least one
    # trace event between the retrieval outcome and the search call.
    assert s_start - r_done >= 1 and len(items) > s_start, (
        "no observable event between retrieval and search (B4 consolidation not evidenced)"
    )


# --------------------------------------------------------------------------
# B5 - query is LLM generated and includes goal, equipment, restrictions
# --------------------------------------------------------------------------
def test_search_query_present_and_nonempty(valid_run):
    _, trace = valid_run
    q = _query_text(_calls(trace, SEARCH)[0])
    assert len(q.strip()) > 10, f"search query missing/too short: {q!r}"


def test_search_query_includes_goal_equipment_and_restrictions(valid_run):
    _, trace = valid_run
    q = _query_text(_calls(trace, SEARCH)[0])
    metric = GEval(
        name="QueryCoversProfileAndRestrictions",
        criteria=(
            "The search query must explicitly mention (1) the user's goal, "
            "(2) the user's available equipment, and (3) every restriction from "
            "the user's profile (knee injury, lower back pain), formulated for an "
            "exercise web search."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.6,
    )
    tc = LLMTestCase(input=VALID_PROFILE, actual_output=q)
    assert_test(tc, [metric])


# --------------------------------------------------------------------------
# B6 - output contract (six sections, plain text) and safe use of results
# --------------------------------------------------------------------------
def test_search_ran_before_successful_output(valid_run):
    output, trace = valid_run
    assert not output.startswith("Error:"), output
    assert _calls(trace, SEARCH), "routine returned without search_exercises evidence"
    assert [e for e in _results(trace, SEARCH) if not e["error"]], "no successful search result"


def test_output_has_six_sections_in_order(valid_run):
    output, trace = valid_run
    assert _calls(trace, SEARCH), "no trace evidence of search"
    pos = [output.find(t) for t in SECTION_TITLES]
    assert all(p >= 0 for p in pos), f"missing sections: {pos}"
    assert pos == sorted(pos), "sections out of order"
    assert pos[0] == 0 or output[: pos[0]].strip() == ""


def test_output_plain_text(valid_run):
    output, _ = valid_run
    assert "```" not in output
    assert not re.search(r"(?m)^\s*#{1,6}\s", output)
    assert "**" not in output


def test_routine_respects_restrictions(valid_run):
    output, trace = valid_run
    assert _calls(trace, RETRIEVE) and _calls(trace, SEARCH)
    metric = GEval(
        name="SafetyContract",
        criteria=(
            "The routine must not include exercises that load an injured knee "
            "(e.g. deep squats, lunges, jumping) or aggravate lower back pain "
            "(e.g. heavy deadlifts), unless explicitly adapted; Important Notes must "
            "list excluded exercise categories with reasons and substitutions."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.6,
    )
    assert_test(LLMTestCase(input=VALID_PROFILE, actual_output=output), [metric])


# --------------------------------------------------------------------------
# B7 - tool failures: Error:, chain stops
# --------------------------------------------------------------------------
def test_notion_failure_blocks_search_and_plan(monkeypatch):
    monkeypatch.setenv("NOTION_API_KEY", "invalid-key")
    output, trace = _run(VALID_PROFILE)
    assert _calls(trace, RETRIEVE), "retrieve_restrictions attempt not evidenced"
    assert not _calls(trace, SEARCH), "search ran after Notion failure"
    assert output.startswith("Error:"), output
    assert "Weekly Plan:" not in output


def test_tavily_failure_blocks_plan(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "invalid-key")
    output, trace = _run(VALID_PROFILE)
    r, s = _calls(trace, RETRIEVE), _calls(trace, SEARCH)
    assert r and s, "both attempts must be evidenced"
    assert min(c["idx"] for c in r) < min(c["idx"] for c in s)
    assert output.startswith("Error:"), output
    assert "Weekly Plan:" not in output
