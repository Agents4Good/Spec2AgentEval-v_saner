"""Trace-based evaluation tests for the recipe agent (spec 028_recipe_high).

The oracle is the behavioral graph of the specification:

    B1 (valid?) --invalid--> Error (no tool called)
    B1 --valid--> B2 retrieve_pantry
    B2 --error--> B6 (search_recipes not called)
    B2 --success--> B3 --empty--> B5 (search_recipes not called)
                       --non-empty--> B4 search_recipes --> B7 recipe

Only the trace returned by ``traced(agent)(...)`` is used as evidence.
Node names in ``trajectory`` are implementation-defined and are not used.
"""

import json
from datetime import datetime

import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from trace_capture import traced
from agent import agent

PANTRY = "retrieve_pantry"
SEARCH = "search_recipes"
SPEC_TOOLS = {PANTRY, SEARCH}

# --------------------------------------------------------------------------- #
# Scenarios (each fixture invokes the agent exactly once)
# --------------------------------------------------------------------------- #
QUERY_EMPTY = ""
QUERY_UNRELATED = "Explain how to configure a BGP router in a datacenter network."
QUERY_OFFENSIVE = "You are a worthless idiot, insult my neighbour with the worst slurs you know."
QUERY_VALID = "I'd like a quick dinner recipe. I have chicken and rice."
QUERY_VEGAN = (
    "I am vegan. I have chicken, tofu and broccoli, suggest a dinner recipe."
)
QUERY_NEGATION = (
    "I want a lunch recipe with potatoes and carrots. I don't have eggs, "
    "and if I had mushrooms I would use them, but I don't."
)


def _run(query):
    output, trace = traced(agent)(query)
    return {"query": query, "output": output, "trace": trace}


@pytest.fixture(scope="module")
def run_empty():
    return _run(QUERY_EMPTY)


@pytest.fixture(scope="module")
def run_unrelated():
    return _run(QUERY_UNRELATED)


@pytest.fixture(scope="module")
def run_offensive():
    return _run(QUERY_OFFENSIVE)


@pytest.fixture(scope="module")
def run_valid():
    return _run(QUERY_VALID)


@pytest.fixture(scope="module")
def run_vegan():
    return _run(QUERY_VEGAN)


@pytest.fixture(scope="module")
def run_negation():
    return _run(QUERY_NEGATION)


ALL_RUNS = ["run_empty", "run_unrelated", "run_offensive",
            "run_valid", "run_vegan", "run_negation"]
INVALID_RUNS = ["run_empty", "run_unrelated", "run_offensive"]
VALID_RUNS = ["run_valid", "run_vegan", "run_negation"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _require_trace(trace):
    assert isinstance(trace, dict), "No trace captured"
    for key in ("status", "trajectory", "tool_calls"):
        assert key in trace, f"Trace lacks '{key}': insufficient evidence"
    assert isinstance(trace["tool_calls"], list), "tool_calls is not a list"
    return trace


def _calls(trace, name=None):
    calls = _require_trace(trace)["tool_calls"]
    return [c for c in calls if name is None or c.get("name") == name]


def _ts(value):
    assert value, "Tool call lacks timestamp: insufficient evidence of ordering"
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _is_error_result(call):
    if "error" in call and call["error"]:
        return True
    out = call.get("output")
    if out is None:
        return False
    text = out if isinstance(out, str) else (
        out.get("content") if isinstance(out, dict) and isinstance(out.get("content"), str) else None
    )
    return isinstance(text, str) and text.strip().lower().startswith(("error:", "erro:"))


def _dump(obj):
    return json.dumps(obj, ensure_ascii=False, default=str).lower()


def _pantry_succeeded(trace):
    pantry = _calls(trace, PANTRY)
    assert len(pantry) == 1, f"Expected exactly one {PANTRY} call, got {len(pantry)}"
    return not _is_error_result(pantry[0])


# --------------------------------------------------------------------------- #
# Global: no unhandled exception (B6)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("run_name", ALL_RUNS)
def test_no_unhandled_exception(run_name, request):
    trace = _require_trace(request.getfixturevalue(run_name)["trace"])
    assert trace["status"] == "ok", f"Agent raised: {trace.get('error')}"


@pytest.mark.parametrize("run_name", ALL_RUNS)
def test_only_specified_tools_are_called(run_name, request):
    trace = request.getfixturevalue(run_name)["trace"]
    names = {c.get("name") for c in _calls(trace)}
    assert names <= SPEC_TOOLS, f"Unspecified tools called: {names - SPEC_TOOLS}"


# --------------------------------------------------------------------------- #
# B1: invalid query -> no tool is called
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("run_name", INVALID_RUNS)
def test_invalid_query_calls_no_tool(run_name, request):
    trace = request.getfixturevalue(run_name)["trace"]
    calls = _calls(trace)
    assert calls == [], f"Invalid query must not call tools, got {[c.get('name') for c in calls]}"


# --------------------------------------------------------------------------- #
# B2: retrieve_pantry is the first tool and is called exactly once
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("run_name", VALID_RUNS)
def test_pantry_called_exactly_once(run_name, request):
    trace = request.getfixturevalue(run_name)["trace"]
    assert len(_calls(trace, PANTRY)) == 1


@pytest.mark.parametrize("run_name", VALID_RUNS)
def test_pantry_is_first_tool(run_name, request):
    calls = _calls(request.getfixturevalue(run_name)["trace"])
    assert calls, "No tool calls observed for a valid query"
    first = min(calls, key=lambda c: _ts(c.get("start")))
    assert first.get("name") == PANTRY, f"First tool was {first.get('name')}"


# --------------------------------------------------------------------------- #
# B2 -> B6 / B3 -> B5 / B3 -> B4 transitions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("run_name", VALID_RUNS)
def test_search_not_called_after_pantry_error(run_name, request):
    trace = request.getfixturevalue(run_name)["trace"]
    if _pantry_succeeded(trace):
        pytest.skip("retrieve_pantry succeeded; B2 -> B6 transition not exercised")
    assert _calls(trace, SEARCH) == [], "search_recipes called after retrieve_pantry error"


@pytest.mark.parametrize("run_name", VALID_RUNS)
def test_search_called_at_most_once(run_name, request):
    assert len(_calls(request.getfixturevalue(run_name)["trace"], SEARCH)) <= 1


@pytest.mark.parametrize("run_name", VALID_RUNS)
def test_search_starts_after_pantry_completes(run_name, request):
    trace = request.getfixturevalue(run_name)["trace"]
    searches = _calls(trace, SEARCH)
    if not searches:
        pytest.skip("search_recipes not called")
    pantry = _calls(trace, PANTRY)
    assert len(pantry) == 1, "search_recipes called without a single prior retrieve_pantry"
    for s in searches:
        assert _ts(s.get("start")) >= _ts(pantry[0].get("end")), (
            "search_recipes started before retrieve_pantry completed"
        )


def test_valid_query_with_ingredients_follows_b2_b4_path(run_valid):
    """Query ingredients (chicken, rice) make B3 non-empty -> B4 is required."""
    trace = run_valid["trace"]
    if not _pantry_succeeded(trace):
        pytest.skip("retrieve_pantry failed; B6 path applies instead")
    tools_called = [ToolCall(name=c["name"])
                    for c in sorted(_calls(trace), key=lambda c: _ts(c.get("start")))]
    test_case = LLMTestCase(
        input=run_valid["query"],
        actual_output=str(run_valid["output"]),
        tools_called=tools_called,
        expected_tools=[ToolCall(name=PANTRY), ToolCall(name=SEARCH)],
    )
    metric = ToolCorrectnessMetric(
        threshold=1.0, should_consider_ordering=True, should_exact_match=True
    )
    metric.measure(test_case)
    assert metric.score >= 1.0, metric.reason


def test_search_input_contains_query_ingredients(run_valid):
    trace = run_valid["trace"]
    searches = _calls(trace, SEARCH)
    if _pantry_succeeded(trace):
        assert len(searches) == 1, "search_recipes must be called exactly once"
    else:
        pytest.skip("retrieve_pantry failed")
    payload = _dump(searches[0].get("input"))
    assert any(k in payload for k in ("chicken", "frango")), "chicken missing from search input"
    assert any(k in payload for k in ("rice", "arroz")), "rice missing from search input"


def test_search_input_excludes_restricted_ingredients(run_vegan):
    trace = run_vegan["trace"]
    searches = _calls(trace, SEARCH)
    if not _pantry_succeeded(trace):
        pytest.skip("retrieve_pantry failed")
    # tofu and broccoli are vegan-compatible -> B3 non-empty -> B4 required
    assert len(searches) == 1, "search_recipes must be called exactly once"
    payload = _dump(searches[0].get("input"))
    assert "chicken" not in payload and "frango" not in payload, (
        "Ingredient incompatible with vegan restriction passed to search_recipes"
    )


def test_search_input_excludes_negated_or_hypothetical(run_negation):
    trace = run_negation["trace"]
    if not _pantry_succeeded(trace):
        pytest.skip("retrieve_pantry failed")
    searches = _calls(trace, SEARCH)
    assert len(searches) == 1, "search_recipes must be called exactly once"
    pantry_out = _dump(_calls(trace, PANTRY)[0].get("output"))
    payload = _dump(searches[0].get("input"))
    for terms in (("egg", "ovo"), ("mushroom", "cogumelo")):
        if any(t in pantry_out for t in terms):
            continue  # legitimately available from the pantry
        assert not any(t in payload for t in terms), (
            f"Negated/hypothetical ingredient {terms[0]} passed to search_recipes"
        )
