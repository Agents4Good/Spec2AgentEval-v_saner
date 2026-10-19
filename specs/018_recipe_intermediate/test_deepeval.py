"""Black-box behavioral evaluation of the personalized recipe agent (spec 018).

Entrypoint contract: ``agent(query: str) -> dict`` (LangGraph state).
Run with: deepeval test run test_deepeval.py
"""
import json
from typing import Any, List

import pytest

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall

from agent import agent

EVAL_MODEL = "gpt-4o"
IO = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]


# --------------------------------------------------------------------------- #
# Observation helpers (based only on the output contract: a state dict)
# --------------------------------------------------------------------------- #
def _get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def run_agent(query: str) -> dict:
    state = agent(query)
    assert isinstance(state, dict), "Output must be a state dict"
    return state


def tools_called(state: dict) -> List[ToolCall]:
    calls = []
    for msg in state.get("messages", []) or []:
        for tc in _get(msg, "tool_calls", None) or []:
            calls.append(ToolCall(name=_get(tc, "name"), input_parameters=_get(tc, "args") or {}))
    return calls


def tool_names(state: dict) -> List[str]:
    return [c.name for c in tools_called(state)]


def final_text(state: dict) -> str:
    """Final observable answer: the last message content, plus the remaining
    state serialized so structured recipe/error fields are visible to the judge."""
    parts = []
    msgs = state.get("messages", []) or []
    if msgs:
        content = _get(msgs[-1], "content", "")
        if isinstance(content, str) and content.strip():
            parts.append(content)
    rest = {k: v for k, v in state.items() if k != "messages"}
    if rest:
        parts.append(json.dumps(rest, default=str, ensure_ascii=False))
    return "\n".join(parts) if parts else json.dumps(state, default=str)


def error_strings(state: dict) -> List[str]:
    """All string values in the state (and the last message) that look like errors."""
    found = []

    def walk(v):
        if isinstance(v, str):
            if v.strip().startswith("Error:"):
                found.append(v.strip())
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk({k: v for k, v in state.items() if k != "messages"})
    msgs = state.get("messages", []) or []
    if msgs:
        walk(_get(msgs[-1], "content", ""))
    return found


def make_case(query: str, state: dict) -> LLMTestCase:
    return LLMTestCase(input=query, actual_output=final_text(state), tools_called=tools_called(state))


# --------------------------------------------------------------------------- #
# Metrics (GEval custom criteria derived from the spec)
# --------------------------------------------------------------------------- #
recipe_completeness = GEval(
    name="Recipe Completeness (B7)",
    criteria=(
        "The actual output contains a complete recipe with ALL of these fields present and "
        "non-empty: a Title; Ingredients listed WITH quantities; Equipment list; step-by-step "
        "Instructions; Storage recommendations."
    ),
    evaluation_params=IO,
    threshold=0.7,
    model=EVAL_MODEL,
)

dietary_compliance = GEval(
    name="Dietary Restriction Compliance",
    criteria=(
        "Identify dietary restrictions declared in the input. A restriction implies all derived "
        "ingredients even if not mentioned (e.g. vegan excludes meat, fish, eggs, dairy, butter, "
        "cheese, honey, gelatin; lactose-free excludes milk, cream, butter, cheese; gluten-free "
        "excludes wheat flour, regular pasta, bread, soy sauce with wheat). The recipe in the "
        "actual output must contain NO ingredient violating those restrictions. Any violation "
        "means the lowest score."
    ),
    evaluation_params=IO,
    threshold=0.8,
    model=EVAL_MODEL,
)

ingredient_usage = GEval(
    name="Uses Query Ingredients (B2/B7)",
    criteria=(
        "The recipe in the actual output uses at least one of the ingredients the user stated "
        "as available in the input that is compatible with their dietary restrictions. "
        "Ingredients mentioned under negation, absence or hypothesis (e.g. 'I don't have eggs', "
        "'if I had chicken') are NOT available and must not be treated as the user's ingredients."
    ),
    evaluation_params=IO,
    threshold=0.7,
    model=EVAL_MODEL,
)

preference_adherence = GEval(
    name="Preference Adherence (Chef persona)",
    criteria=(
        "Acting as a chef, the recipe in the actual output takes into account the preferences "
        "expressed in the input (e.g. meal type, cuisine style, quick/simple, flavor) and is a "
        "coherent, personalized recipe."
    ),
    evaluation_params=IO,
    threshold=0.6,
    model=EVAL_MODEL,
)

RECIPE_METRICS = [recipe_completeness, dietary_compliance, ingredient_usage, preference_adherence]


def assert_no_recipe(state: dict):
    text = final_text(state).lower()
    assert not ("instructions" in text and "storage" in text and "equipment" in text), (
        "No recipe must be generated on error paths"
    )


# --------------------------------------------------------------------------- #
# B1 — Query validation
# --------------------------------------------------------------------------- #
INVALID_QUERIES = [
    "",
    "   ",
    "What is the capital of France?",
    "Write me a Python function that sorts a list.",
    "You are a useless idiot, go to hell.",
]


@pytest.mark.parametrize("query", INVALID_QUERIES)
def test_b1_invalid_query_returns_error_without_tools(query):
    state = run_agent(query)
    assert error_strings(state), f"Expected message starting with 'Error:'; got: {final_text(state)[:500]}"
    assert tool_names(state) == [], f"No tool may be called for invalid query; called: {tool_names(state)}"
    assert_no_recipe(state)


# --------------------------------------------------------------------------- #
# B2 + B3 + B7 — Query with ingredients: search once, no pantry, full recipe
# --------------------------------------------------------------------------- #
QUERIES_WITH_INGREDIENTS = [
    "I have chicken breast, rice and broccoli. I'd like a quick dinner recipe.",
    "I'm vegan. I have chickpeas, spinach, tomatoes and eggs. Suggest a lunch.",
    "I'm lactose intolerant. I have pasta, cheese, mushrooms and garlic. Make me something Italian.",
    "I have potatoes and onions but I don't have any eggs. What can I cook for breakfast?",
    "Tenho batatas, cenouras e frango. Quero um jantar simples.",
    "I'm celiac. I have rice, salmon, wheat flour and lemon. Something healthy please.",
]


@pytest.mark.parametrize("query", QUERIES_WITH_INGREDIENTS)
def test_b3_b7_query_with_ingredients(query):
    state = run_agent(query)
    names = tool_names(state)
    assert names.count("search_recipes") == 1, f"search_recipes must be called exactly once; called: {names}"
    assert "retrieve_pantry" not in names, f"retrieve_pantry must not be called; called: {names}"
    assert not error_strings(state), f"Unexpected error: {error_strings(state)}"
    assert_test(make_case(query, state), RECIPE_METRICS)


# Ingredients passed to search_recipes must exclude negated/hypothetical/incompatible ones.
SEARCH_ARG_CASES = [
    ("I'm vegan. I have tofu, broccoli and bacon. Make dinner.", ["tofu", "broccoli"], ["bacon"]),
    ("I have zucchini and garlic, but no eggs. If I had chicken I'd use it. Lunch idea?",
     ["zucchini", "garlic"], ["egg", "chicken"]),
    ("I'm lactose intolerant. I have milk, oats and bananas. Breakfast please.",
     ["oat", "banana"], ["milk"]),
]


@pytest.mark.parametrize("query,expected,forbidden", SEARCH_ARG_CASES)
def test_b2_b3_search_called_with_compatible_ingredients(query, expected, forbidden):
    state = run_agent(query)
    searches = [c for c in tools_called(state) if c.name == "search_recipes"]
    assert len(searches) == 1, f"search_recipes must be called exactly once; got {tool_names(state)}"
    args = json.dumps(searches[0].input_parameters, ensure_ascii=False).lower()
    for ing in expected:
        assert ing in args, f"Expected ingredient '{ing}' in search_recipes args: {args}"
    for ing in forbidden:
        assert ing not in args, f"Ingredient '{ing}' must be excluded from search_recipes args: {args}"


# --------------------------------------------------------------------------- #
# B4 + (B5 | B7) — Query without (compatible) ingredients: pantry lookup
# --------------------------------------------------------------------------- #
QUERIES_WITHOUT_INGREDIENTS = [
    "Suggest a dinner recipe for me.",
    "I'm vegetarian. What can I make for lunch today?",
    "I'm vegan and I have only cheese and honey. What can I cook?",
    "I don't have chicken or beef. Give me a gluten-free dinner idea.",
]


@pytest.mark.parametrize("query", QUERIES_WITHOUT_INGREDIENTS)
def test_b4_pantry_lookup(query):
    state = run_agent(query)
    names = tool_names(state)
    assert names.count("retrieve_pantry") == 1, f"retrieve_pantry must be called exactly once; called: {names}"
    assert "search_recipes" not in names, f"search_recipes must not be called; called: {names}"

    errors = error_strings(state)
    if errors:
        # B5 (no compatible pantry ingredients) or B6 (pantry failure)
        assert_no_recipe(state)
    else:
        # B7 from pantry ingredients
        case = make_case(query, state)
        assert_test(case, [recipe_completeness, dietary_compliance, preference_adherence])


# --------------------------------------------------------------------------- #
# B6 — Execution failure (missing credentials)
# --------------------------------------------------------------------------- #
def test_b6_search_failure_missing_tavily_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    query = "I have eggs, tomatoes and onions. Make me a breakfast recipe."
    state = run_agent(query)  # must not raise
    assert error_strings(state), f"Expected 'Error:' message; got: {final_text(state)[:500]}"
    assert_no_recipe(state)


def test_b6_pantry_failure_missing_notion_key(monkeypatch):
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    query = "Suggest a dinner recipe for me."
    state = run_agent(query)  # must not raise
    assert error_strings(state), f"Expected 'Error:' message; got: {final_text(state)[:500]}"
    assert_no_recipe(state)


def test_b6_model_failure_missing_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "invalid-key")
    query = "I have rice and beans. Make lunch."
    state = run_agent(query)  # must not raise an unhandled exception
    assert error_strings(state), f"Expected 'Error:' message; got: {final_text(state)[:500]}"
    assert_no_recipe(state)


# --------------------------------------------------------------------------- #
# Quality attribute — structured response
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("query", ["I have eggs and spinach. Breakfast idea?", ""])
def test_structured_response(query):
    state = run_agent(query)
    assert isinstance(state, dict) and len(state) > 0
    json.dumps(state, default=str)  # serializable for automated evaluation
