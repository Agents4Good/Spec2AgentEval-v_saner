"""Trace-based evaluation of the recipe generator agent (spec 018).

Oracle: the behavioral graph of the specification.

    B1 invalid            -> no tool call
    B1 valid -> B2 non-empty -> B3 (search_recipes x1, no retrieve_pantry) -> B7 | B6
    B1 valid -> B2 empty     -> B4 (retrieve_pantry x1, no search_recipes) -> B7 | B5 | B6

Only the trace returned by the benchmark wrapper is used as evidence. The
agent output is never semantically validated. Node names are not relied upon.
"""

import json
from datetime import datetime

import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from trace_capture import traced
from agent import agent

SEARCH = "search_recipes"
PANTRY = "retrieve_pantry"
KNOWN_TOOLS = {SEARCH, PANTRY}

# --------------------------------------------------------------------------
# Scenarios (one agent invocation each)
# --------------------------------------------------------------------------

INVALID_QUERIES = {
    "empty": "",
    "unrelated": "Explain how to configure a Kubernetes cluster on AWS.",
    "offensive": "You are a stupid useless idiot machine, I hate you.",
}

# B2 non-empty -> B3. Expected ingredient tokens must reach search_recipes.
INGREDIENT_QUERIES = {
    "explicit_ingredients": (
        "I'd like a dinner recipe using chicken and rice.",
        ["chicken", "rice"],
    ),
    "with_restriction_compatible": (
        "I'm vegetarian. Make me something with tomato and zucchini.",
        ["tomato", "zucchini"],
    ),
}

# B2 empty -> B4.
NO_INGREDIENT_QUERIES = {
    "no_ingredients": "Suggest a nice recipe for dinner tonight.",
    "only_incompatible_ingredient": "I'm vegan. I want a recipe with bacon.",
    "negated_ingredient": "I don't have any eggs. Suggest a breakfast recipe.",
    "hypothetical_ingredient": "If I had salmon I would cook it, but suggest any lunch recipe.",
}

_CACHE = {}


def run(key, query):
    if key not in _CACHE:
        _CACHE[key] = traced(agent)(query)
    return _CACHE[key]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def tool_calls(trace):
    assert isinstance(trace, dict), "Trace missing: no evidence of execution"
    calls = trace.get("tool_calls")
    assert isinstance(calls, list), "Trace has no 'tool_calls' list"
    return calls


def calls_named(trace, name):
    return [c for c in tool_calls(trace) if c.get("name") == name]


def assert_trace_evidence(trace):
    """The trace must provide evidence that the agent actually ran."""
    assert isinstance(trace, dict), "Trace missing"
    assert trace.get("runs_collected", 0) > 0, "No runs collected: insufficient evidence"
    assert isinstance(trace.get("trajectory"), list) and trace["trajectory"], (
        "Empty trajectory: insufficient evidence"
    )


def assert_no_unhandled_exception(trace):
    # B6: the agent must not terminate with an unhandled exception.
    assert trace.get("status") == "ok", f"Agent raised: {trace.get('error')}"


def assert_tool_call_well_formed(call):
    assert call.get("name"), "Tool call without name"
    assert isinstance(call.get("input"), dict), "Tool call without input dict"
    assert "output" in call or "error" in call, (
        f"Tool call {call.get('name')} has neither output nor error"
    )
    start = datetime.fromisoformat(call["start"])
    end = datetime.fromisoformat(call["end"])
    assert start <= end, f"Tool call {call['name']} ends before it starts"


def tool_correctness(trace, expected):
    """DeepEval ToolCorrectnessMetric over the tools observed in the trace."""
    called = [ToolCall(name=c["name"], input_parameters=c.get("input") or {})
              for c in tool_calls(trace)]
    test_case = LLMTestCase(
        input=json.dumps(trace.get("query"), default=str),
        actual_output="<not evaluated>",
        tools_called=called,
        expected_tools=[ToolCall(name=n) for n in expected],
    )
    metric = ToolCorrectnessMetric(threshold=1.0, should_exact_match=True)
    metric.measure(test_case)
    return metric


# --------------------------------------------------------------------------
# B1 — invalid query: stop, no tool is called
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", list(INVALID_QUERIES))
def test_b1_invalid_query_calls_no_tool(key):
    _, trace = run(f"invalid:{key}", INVALID_QUERIES[key])
    assert_no_unhandled_exception(trace)
    assert trace.get("runs_collected", 0) > 0, "No runs collected: insufficient evidence"
    calls = tool_calls(trace)
    assert calls == [], f"Invalid query must call no tool, got {[c.get('name') for c in calls]}"


# --------------------------------------------------------------------------
# B2 non-empty -> B3: search_recipes exactly once, no retrieve_pantry
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", list(INGREDIENT_QUERIES))
def test_b3_search_recipes_called_exactly_once(key):
    query, _ = INGREDIENT_QUERIES[key]
    _, trace = run(f"ing:{key}", query)
    assert_trace_evidence(trace)
    assert_no_unhandled_exception(trace)
    assert len(calls_named(trace, SEARCH)) == 1, "search_recipes must be called exactly once"
    assert calls_named(trace, PANTRY) == [], "retrieve_pantry must not be called in B3 branch"


@pytest.mark.parametrize("key", list(INGREDIENT_QUERIES))
def test_b3_tool_correctness_deepeval(key):
    query, _ = INGREDIENT_QUERIES[key]
    _, trace = run(f"ing:{key}", query)
    metric = tool_correctness(trace, [SEARCH])
    assert metric.is_successful(), f"ToolCorrectness {metric.score}: {metric.reason}"


@pytest.mark.parametrize("key", list(INGREDIENT_QUERIES))
def test_b3_search_receives_b2_ingredients(key):
    query, expected = INGREDIENT_QUERIES[key]
    _, trace = run(f"ing:{key}", query)
    calls = calls_named(trace, SEARCH)
    assert len(calls) == 1, "No search_recipes call in trace: insufficient evidence"
    assert_tool_call_well_formed(calls[0])
    payload = json.dumps(calls[0]["input"], ensure_ascii=False).lower()
    for ingredient in expected:
        assert ingredient in payload, (
            f"search_recipes input lacks B2 ingredient '{ingredient}': {calls[0]['input']}"
        )


# --------------------------------------------------------------------------
# B2 empty -> B4: retrieve_pantry exactly once, no search_recipes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", list(NO_INGREDIENT_QUERIES))
def test_b4_retrieve_pantry_called_exactly_once(key):
    _, trace = run(f"noing:{key}", NO_INGREDIENT_QUERIES[key])
    assert_trace_evidence(trace)
    assert_no_unhandled_exception(trace)
    assert len(calls_named(trace, PANTRY)) == 1, "retrieve_pantry must be called exactly once"
    assert calls_named(trace, SEARCH) == [], "search_recipes must not be called in B4 branch"


@pytest.mark.parametrize("key", list(NO_INGREDIENT_QUERIES))
def test_b4_tool_correctness_deepeval(key):
    _, trace = run(f"noing:{key}", NO_INGREDIENT_QUERIES[key])
    metric = tool_correctness(trace, [PANTRY])
    assert metric.is_successful(), f"ToolCorrectness {metric.score}: {metric.reason}"


@pytest.mark.parametrize("key", list(NO_INGREDIENT_QUERIES))
def test_b4_retrieve_pantry_takes_no_arguments(key):
    _, trace = run(f"noing:{key}", NO_INGREDIENT_QUERIES[key])
    calls = calls_named(trace, PANTRY)
    assert len(calls) == 1, "No retrieve_pantry call in trace: insufficient evidence"
    assert_tool_call_well_formed(calls[0])
    assert not calls[0]["input"], f"retrieve_pantry takes no inputs, got {calls[0]['input']}"


# --------------------------------------------------------------------------
# Graph compatibility for all valid scenarios
# --------------------------------------------------------------------------

VALID_SCENARIOS = (
    [(f"ing:{k}", q) for k, (q, _) in INGREDIENT_QUERIES.items()]
    + [(f"noing:{k}", q) for k, q in NO_INGREDIENT_QUERIES.items()]
)


@pytest.mark.parametrize("key,query", VALID_SCENARIOS, ids=[k for k, _ in VALID_SCENARIOS])
def test_graph_branches_are_mutually_exclusive(key, query):
    """B3 and B4 are exclusive branches; at most one tool call in total."""
    _, trace = run(key, query)
    assert_trace_evidence(trace)
    names = [c.get("name") for c in tool_calls(trace)]
    assert set(names) <= KNOWN_TOOLS, f"Unknown tools called: {set(names) - KNOWN_TOOLS}"
    assert len(names) == 1, f"Expected exactly one tool call (B3 xor B4), got {names}"


@pytest.mark.parametrize("key,query", VALID_SCENARIOS, ids=[k for k, _ in VALID_SCENARIOS])
def test_tool_calls_present_in_trajectory(key, query):
    """Every tool call must be evidenced by the trajectory."""
    _, trace = run(key, query)
    assert_trace_evidence(trace)
    for call in tool_calls(trace):
        assert_tool_call_well_formed(call)
        assert call["name"] in trace["trajectory"], (
            f"Tool {call['name']} not present in trajectory {trace['trajectory']}"
        )


@pytest.mark.parametrize("key,query", VALID_SCENARIOS, ids=[k for k, _ in VALID_SCENARIOS])
def test_no_unhandled_exception(key, query):
    """B6: failures must be handled without aborting execution."""
    _, trace = run(key, query)
    assert_no_unhandled_exception(trace)
