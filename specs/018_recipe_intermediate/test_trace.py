"""Trace-based evaluation tests for the personalized recipe agent (spec 018).

Oracle: the behavioral graph of the specification.

    B1 invalid           -> no tool is called
    B2 non-empty list    -> B3: search_recipes exactly once, retrieve_pantry never
    B2 empty list        -> B4: retrieve_pantry exactly once, search_recipes never
    B3 / B4 error        -> B6: no further tool calls, no unhandled exception

Evidence comes exclusively from the trace returned by the benchmark wrapper.
Tool selection is scored with DeepEval's deterministic ToolCorrectnessMetric.
No semantic validation of the agent output is performed.
"""

import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from trace_capture import traced
from agent import agent

SEARCH = "search_recipes"
PANTRY = "retrieve_pantry"
SPEC_TOOLS = {SEARCH, PANTRY}


# --------------------------------------------------------------------------- #
# Scenarios (each fixture invokes the agent exactly once)
# --------------------------------------------------------------------------- #

INVALID_QUERIES = {
    "empty": "",
    "unrelated": "Explain how to configure a BGP router on a Cisco switch.",
}

# Ingredients explicitly present and compatible -> B2 non-empty -> B3.
INGREDIENT_QUERIES = {
    "explicit_ingredients": (
        "I have chicken breast, rice, garlic and onions. "
        "Make me a dinner recipe."
    ),
    "with_restriction_compatible": (
        "I'm vegetarian. I have tomatoes, zucchini and pasta. "
        "Suggest a lunch recipe."
    ),
}

# No available compatible ingredient in the query -> B2 empty -> B4.
NO_INGREDIENT_QUERIES = {
    "no_ingredients": "Suggest a quick dinner recipe for tonight.",
    "negated_ingredients": (
        "I don't have eggs or milk. What dessert could I cook?"
    ),
    "all_incompatible": (
        "I'm vegan. I have bacon and cheese. Create a recipe for me."
    ),
}


def _run(query):
    output, trace = traced(agent)(query)
    return output, trace


@pytest.fixture(scope="module", params=list(INVALID_QUERIES), ids=list(INVALID_QUERIES))
def invalid_run(request):
    return _run(INVALID_QUERIES[request.param])


@pytest.fixture(scope="module", params=list(INGREDIENT_QUERIES), ids=list(INGREDIENT_QUERIES))
def ingredient_run(request):
    return _run(INGREDIENT_QUERIES[request.param])


@pytest.fixture(scope="module", params=list(NO_INGREDIENT_QUERIES), ids=list(NO_INGREDIENT_QUERIES))
def no_ingredient_run(request):
    return _run(NO_INGREDIENT_QUERIES[request.param])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _assert_trace_usable(trace):
    assert isinstance(trace, dict), "No trace was captured."
    for key in ("status", "trajectory", "tool_calls", "runs_collected"):
        assert key in trace, f"Trace lacks required field '{key}'."
    assert isinstance(trace["tool_calls"], list), "trace['tool_calls'] is not a list."
    assert isinstance(trace["trajectory"], list), "trace['trajectory'] is not a list."
    assert trace["runs_collected"] > 0, (
        "Trace collected no runs: insufficient evidence of execution."
    )


def _calls(trace, name=None):
    return [c for c in trace["tool_calls"] if name is None or c.get("name") == name]


def _assert_no_unhandled_exception(trace):
    # B6: the agent must not terminate with an unhandled exception.
    assert trace["status"] == "ok", (
        f"Agent terminated with an unhandled exception: {trace.get('error')}"
    )


def _tool_correctness(trace, expected):
    """Deterministic DeepEval check that the called tools equal `expected`."""
    test_case = LLMTestCase(
        input=str(trace.get("query")),
        actual_output="<not evaluated>",
        tools_called=[
            ToolCall(name=c["name"], input_parameters=c.get("input") or {})
            for c in _calls(trace)
            if c.get("name") in SPEC_TOOLS
        ],
        expected_tools=[ToolCall(name=n) for n in expected],
    )
    metric = ToolCorrectnessMetric(threshold=1.0, should_exact_match=True)
    metric.measure(test_case)
    return metric


def _tool_failed(call):
    if "error" in call and call["error"]:
        return True
    out = call.get("output")
    if isinstance(out, dict):
        out = out.get("content", out)
    return isinstance(out, str) and out.strip().lower().startswith(("erro:", "error:"))


def _assert_tool_evidence(trace, call):
    # The tool call must be backed by concrete trace evidence.
    assert call.get("start") and call.get("end"), (
        f"Tool call '{call['name']}' lacks timing evidence."
    )
    assert "output" in call or "error" in call, (
        f"Tool call '{call['name']}' has neither output nor error recorded."
    )
    assert call["name"] in trace["trajectory"], (
        f"Tool '{call['name']}' recorded in tool_calls but absent from trajectory."
    )


def _assert_stops_after_failure(trace):
    # B3/B4 -> B6: after a failing tool, no further spec tool is invoked.
    calls = sorted(_calls(trace), key=lambda c: c["start"])
    for i, call in enumerate(calls):
        if call.get("name") in SPEC_TOOLS and _tool_failed(call):
            later = [c["name"] for c in calls[i + 1:] if c.get("name") in SPEC_TOOLS]
            assert not later, (
                f"Tools {later} were called after '{call['name']}' failed (B6)."
            )


# --------------------------------------------------------------------------- #
# B1 — invalid query: no tool is called
# --------------------------------------------------------------------------- #

class TestB1InvalidQuery:
    def test_trace_captured(self, invalid_run):
        _, trace = invalid_run
        _assert_trace_usable(trace)

    def test_no_unhandled_exception(self, invalid_run):
        _, trace = invalid_run
        _assert_trace_usable(trace)
        _assert_no_unhandled_exception(trace)

    def test_no_tool_called(self, invalid_run):
        _, trace = invalid_run
        _assert_trace_usable(trace)
        called = [c["name"] for c in _calls(trace)]
        assert called == [], f"Invalid query must not call any tool, got {called}."
        for tool in SPEC_TOOLS:
            assert tool not in trace["trajectory"], (
                f"Tool '{tool}' appears in trajectory for an invalid query."
            )


# --------------------------------------------------------------------------- #
# B2 non-empty -> B3: search_recipes exactly once, retrieve_pantry never
# --------------------------------------------------------------------------- #

class TestB3ReferenceSearch:
    def test_no_unhandled_exception(self, ingredient_run):
        _, trace = ingredient_run
        _assert_trace_usable(trace)
        _assert_no_unhandled_exception(trace)

    def test_tool_correctness(self, ingredient_run):
        _, trace = ingredient_run
        _assert_trace_usable(trace)
        metric = _tool_correctness(trace, [SEARCH])
        assert metric.score >= metric.threshold, (
            f"Expected exactly [{SEARCH}]: {metric.reason}"
        )

    def test_search_called_exactly_once(self, ingredient_run):
        _, trace = ingredient_run
        _assert_trace_usable(trace)
        calls = _calls(trace, SEARCH)
        assert len(calls) == 1, f"{SEARCH} must be called exactly once, got {len(calls)}."
        _assert_tool_evidence(trace, calls[0])

    def test_pantry_not_called(self, ingredient_run):
        _, trace = ingredient_run
        _assert_trace_usable(trace)
        assert not _calls(trace, PANTRY), f"{PANTRY} must not be called in B3 path."

    def test_search_receives_ingredients(self, ingredient_run):
        _, trace = ingredient_run
        _assert_trace_usable(trace)
        calls = _calls(trace, SEARCH)
        assert len(calls) == 1, f"No single {SEARCH} call in trace."
        args = calls[0].get("input")
        assert isinstance(args, dict) and args, (
            f"{SEARCH} was called without the B2 ingredient list: {args!r}."
        )
        assert any(v not in (None, "", [], {}) for v in args.values()), (
            f"{SEARCH} received an empty ingredient argument: {args!r}."
        )

    def test_stops_after_tool_failure(self, ingredient_run):
        _, trace = ingredient_run
        _assert_trace_usable(trace)
        _assert_stops_after_failure(trace)


# --------------------------------------------------------------------------- #
# B2 empty -> B4: retrieve_pantry exactly once, search_recipes never
# --------------------------------------------------------------------------- #

class TestB4PantryLookup:
    def test_no_unhandled_exception(self, no_ingredient_run):
        _, trace = no_ingredient_run
        _assert_trace_usable(trace)
        _assert_no_unhandled_exception(trace)

    def test_tool_correctness(self, no_ingredient_run):
        _, trace = no_ingredient_run
        _assert_trace_usable(trace)
        metric = _tool_correctness(trace, [PANTRY])
        assert metric.score >= metric.threshold, (
            f"Expected exactly [{PANTRY}]: {metric.reason}"
        )

    def test_pantry_called_exactly_once(self, no_ingredient_run):
        _, trace = no_ingredient_run
        _assert_trace_usable(trace)
        calls = _calls(trace, PANTRY)
        assert len(calls) == 1, f"{PANTRY} must be called exactly once, got {len(calls)}."
        _assert_tool_evidence(trace, calls[0])

    def test_search_not_called(self, no_ingredient_run):
        _, trace = no_ingredient_run
        _assert_trace_usable(trace)
        assert not _calls(trace, SEARCH), f"{SEARCH} must not be called in B4 path."

    def test_stops_after_tool_failure(self, no_ingredient_run):
        _, trace = no_ingredient_run
        _assert_trace_usable(trace)
        _assert_stops_after_failure(trace)
