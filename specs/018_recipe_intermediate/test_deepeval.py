"""Black-box behavioral evaluation of the 018_recipe_intermediate recipe agent.

Requirements come only from specs/018_recipe_intermediate/spec.md. The agent is
driven through its entrypoint, agent(query) -> state (dict), and through the
environment variables the spec lists. Structural checks (error prefixes, tool
calls, tool arguments) are plain asserts on the returned state. Semantic checks
(recipe completeness, dietary compliance, goal fit, persona) are DeepEval GEval
metrics run with assert_test and judged by EVAL_JUDGE_MODEL.

Run:    deepeval test run test_deepeval.py
Needs:  OPENAI_API_KEY, TAVILY_API_KEY, NOTION_API_KEY and NOTION_PAGE_ID for the
        agent and the judge. SPEC_EMPTY_PANTRY_PAGE_ID (optional) enables B5.

Assumptions the spec does not pin down, all isolated in the helpers below:
  * agent(...) takes the user query as its single positional argument.
  * Tool calls appear in the returned state as LangChain-style tool_calls
    entries with a name and args. Without them the tool-usage checks fail.
  * Credentials are read when a tool or model is called, so the failure
    injections (monkeypatched environment variables) take effect.
"""

import json
import os
import re
from typing import Optional

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o")
THRESHOLD = 0.7
EMPTY_PANTRY_PAGE_ID = os.getenv("SPEC_EMPTY_PANTRY_PAGE_ID")

HAPPY_QUERY = "I have chicken thighs, rice and peas. I want a quick, mild dinner."
NO_INGREDIENTS_QUERY = "Suggest a vegetarian dinner idea. I don't have any particular ingredients in mind."
NO_RECIPE = (
    "Passes only if the output explains why it cannot proceed and contains no recipe: "
    "no recipe title, ingredient list, equipment list, preparation steps or storage advice."
)

INVALID_QUERIES = [
    pytest.param("", id="empty"),
    pytest.param("You are a stupid, worthless machine. Insult me back right now.", id="offensive"),
    pytest.param("What is the capital of Australia?", id="unrelated-to-food"),
]

# (query, ingredients that must reach search_recipes, ingredients that must not).
# A tuple lists alternatives such as regional variants. Covers B2 and guardrail GI2.
B2_CASES = [
    pytest.param(
        "I have 3 tomatoes, some potatoes, courgettes and aubergines. Make me a stew.",
        ["tomato", "potato", ("courgette", "zucchini"), ("aubergine", "eggplant")],
        [],
        id="inflections-and-regional-variants",
    ),
    pytest.param(
        "I don't have any shrimp, but I have chicken thighs and rice. Make me a dinner.",
        ["chicken", "rice"],
        ["shrimp"],
        id="negation",
    ),
    pytest.param(
        "I'm out of garlic. I have potatoes, onions and carrots. Make me a soup.",
        ["potato", "onion", "carrot"],
        ["garlic"],
        id="absence",
    ),
    pytest.param(
        "If I had some mushrooms I'd add them, but I only have spinach, lentils and tomatoes. Make me a stew.",
        ["spinach", "lentil", "tomato"],
        ["mushroom"],
        id="hypothesis",
    ),
    pytest.param(
        "I'm vegetarian. I have chicken breast, spinach and rice. Make me a dinner.",
        ["spinach", "rice"],
        ["chicken"],
        id="incompatible-with-restriction",
    ),
]

# Dietary restriction and the derived ingredients it implies (GI1).
GI1_CASES = [
    pytest.param(
        "I'm vegan. Make me a creamy pasta with tomatoes and garlic.",
        "vegan: no meat, fish, dairy, eggs, honey, gelatin or other animal-derived ingredients",
        id="vegan",
    ),
    pytest.param(
        "I'm gluten-free. I have rice, chickpeas and lemon. Make me a dinner.",
        "gluten-free: no wheat, barley, rye, malt, regular soy sauce, couscous, bulgur, regular pasta or bread crumbs",
        id="gluten-free",
    ),
]

RECIPE_FIELD_STEMS = ("title", "ingredient", "equipment", "instruction", "storage")


def run_agent(query: str) -> dict:
    """Call the agent entrypoint once and return the state it produces."""
    from agent import agent  # imported lazily so the suite collects before agent.py exists

    try:
        state = agent(query)
    except Exception as exc:  # B6: failures must be handled inside the agent
        pytest.fail(f"agent raised an unhandled exception: {exc!r}")
    assert isinstance(state, dict), f"output must be state: dict, got {type(state).__name__}"
    return state


def run_with_env(monkeypatch, query: str, name: str, value: Optional[str]) -> dict:
    """Run the agent with one environment variable overridden; None removes it."""
    with monkeypatch.context() as patch:
        if value is None:
            patch.delenv(name, raising=False)
        else:
            patch.setenv(name, value)
        return run_agent(query)


def _plain(node):
    """Return the state as plain dicts, lists and scalars; message objects are dumped via model_dump."""
    if isinstance(node, dict):
        return {str(key): _plain(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_plain(value) for value in node]
    if node is None or isinstance(node, (str, int, float, bool)):
        return node
    if hasattr(node, "model_dump"):
        return _plain(node.model_dump())
    return str(node)


def _call_name_and_args(call):
    if isinstance(call, str):
        return call, None
    if isinstance(call, dict):
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        return call.get("name") or function.get("name"), call.get("args", function.get("arguments"))
    return None, None


def tool_calls(state: dict) -> list:
    """(name, args) for every tool call in the state, in order. Tool results are not calls."""
    calls = []

    def walk(node):
        if isinstance(node, dict):
            # One message's calls. Do not descend, so nested copies (e.g. additional_kwargs) are not counted twice.
            if isinstance(node.get("tool_calls"), list):
                for call in node["tool_calls"]:
                    name, args = _call_name_and_args(call)
                    if name:
                        calls.append((name, args))
                return
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(_plain(state))
    return calls


def tool_names(state: dict) -> list:
    return [name for name, _ in tool_calls(state)]


_NON_AGENT_TYPES = {"tool", "human", "user", "system"}
_BOOKKEEPING_KEYS = {
    "tool_calls", "additional_kwargs", "response_metadata", "usage_metadata",
    "args", "arguments", "id", "tool_call_id", "type", "role",
}


def agent_texts(state: dict, query: str) -> list:
    """Strings the agent authored: no tool results, tool-call arguments or echoed user query."""
    texts = []

    def walk(node):
        if isinstance(node, dict):
            if str(node.get("type", node.get("role", ""))).lower() in _NON_AGENT_TYPES:
                return
            for key, value in node.items():
                if key not in _BOOKKEEPING_KEYS:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            text = node.strip()
            if text and text != query.strip():
                texts.append(text)

    walk(_plain(state))
    return texts


def begins_with_error(state: dict, query: str) -> bool:
    return any(text.startswith("Error:") for text in agent_texts(state, query))


def validity_flags(state: dict) -> list:
    """Boolean validity indicators, normalized so that True means 'valid'."""
    flags = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                lowered = str(key).lower()
                if isinstance(value, bool) and "invalid" in lowered:
                    flags.append(not value)
                elif isinstance(value, bool) and "valid" in lowered:
                    flags.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(_plain(state))
    return flags


def _keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield str(key)
            yield from _keys(value)
    elif isinstance(node, list):
        for value in node:
            yield from _keys(value)


def render(state: dict) -> str:
    return json.dumps(_plain(state), ensure_ascii=False, indent=2, default=str)


def assert_criteria(query: str, state: dict, name: str, criteria: str) -> None:
    """Judge the rendered output against one natural-language criterion with a DeepEval GEval metric."""
    metric = GEval(
        name=name,
        criteria=criteria,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
    )
    test_case = LLMTestCase(input=query or "(empty query)", actual_output=render(state))
    assert_test(test_case=test_case, metrics=[metric])


def test_precondition_tool_calls_are_observable():
    """Tool-usage checks below are only meaningful if tool calls appear in the returned state."""
    state = run_agent(HAPPY_QUERY)
    assert tool_names(state), "no tool-call trace in the returned state; tool-usage constraints cannot be verified"


@pytest.mark.parametrize("query", INVALID_QUERIES)
def test_b1_invalid_query_is_rejected_without_tools_or_recipe(query):
    """B1: an empty, offensive, or non-food query is invalid, stops execution, and yields no recipe."""
    state = run_agent(query)
    assert all(flag is False for flag in validity_flags(state)), "validity indicator must be False"
    assert begins_with_error(state, query), "output must begin with 'Error:'"
    assert tool_names(state) == [], "an invalid query must not call any tool"
    assert_criteria(query, state, "B1 rejection with explanation", (
        "Passes only if the output explains why the query is rejected (it is empty, offensive, or unrelated "
        "to food and cooking) and contains no recipe: no recipe title, ingredient list, equipment list, "
        "preparation steps or storage advice."
    ))


def test_b1_food_query_with_ingredients_is_valid():
    """B1: a food query that meets no invalidity criterion is valid and proceeds to B2 and B3."""
    query = "I have eggs, spinach and feta. Suggest a quick breakfast."
    state = run_agent(query)
    assert all(flag is True for flag in validity_flags(state)), "validity indicator must be True"
    assert tool_names(state).count("search_recipes") == 1, "a valid query with ingredients must reach B3 once"


@pytest.mark.parametrize("query, expected, excluded", B2_CASES)
def test_b2_identified_ingredients_match_expected_set(query, expected, excluded):
    """B2 and GI2: identified ingredients reach search_recipes; negated, absent, hypothetical or
    restriction-incompatible ones do not. The search_recipes argument is the observable B2 list."""
    state = run_agent(query)
    searches = [args for name, args in tool_calls(state) if name == "search_recipes"]
    assert len(searches) == 1, "a non-empty B2 list must trigger exactly one search_recipes call"
    identified = json.dumps(searches[0], ensure_ascii=False, default=str).lower()
    for option in expected:
        alternatives = option if isinstance(option, tuple) else (option,)
        assert any(alt in identified for alt in alternatives), f"expected one of {alternatives} in: {identified}"
    for ingredient in excluded:
        assert ingredient not in identified, f"'{ingredient}' must not be identified: {identified}"


def test_b3_searches_recipes_once_with_identified_ingredients_and_skips_pantry():
    """B3: search_recipes runs exactly once with the B2 ingredients, and retrieve_pantry is not called."""
    state = run_agent(HAPPY_QUERY)
    calls = tool_calls(state)
    names = [name for name, _ in calls]
    assert names.count("search_recipes") == 1, names
    assert "retrieve_pantry" not in names, names
    searched = json.dumps(next(args for name, args in calls if name == "search_recipes"), default=str).lower()
    for ingredient in ("chicken", "rice", "peas"):
        assert ingredient in searched, searched


def test_b3_search_results_are_reference_only():
    """B3: found recipes are a preparation reference; they are neither the ingredient source nor the final recipe."""
    state = run_agent(HAPPY_QUERY)
    assert_criteria(HAPPY_QUERY, state, "B3 search results are reference only", (
        "Passes only if the recipe is the agent's own adaptation built from the user's ingredients: it does not "
        "reproduce a web recipe found by the search, and its main ingredients come from the user's query rather "
        "than only from the search results. Common seasonings and staples (salt, pepper, oil, water) are acceptable."
    ))


@pytest.mark.parametrize(
    "query",
    [
        pytest.param(NO_INGREDIENTS_QUERY, id="no-ingredients"),
        pytest.param("I'm vegan. I only have butter, honey and cheese. Make me a snack.", id="no-compatible-ingredients"),
    ],
)
def test_b4_pantry_lookup_when_no_ingredient_is_identified(query):
    """B4: an empty B2 list (no ingredients, or none compatible) triggers one pantry lookup and no search."""
    state = run_agent(query)
    names = tool_names(state)
    assert names.count("retrieve_pantry") == 1, names
    assert "search_recipes" not in names, names
    assert_criteria(query, state, "B4 pantry-based outcome", (
        "Passes only if the output is either a complete recipe built from the user's pantry that respects the "
        "dietary restriction stated in the query, or a message beginning with 'Error:' explaining that no "
        "compatible pantry ingredients are available."
    ))


@pytest.mark.skipif(
    not EMPTY_PANTRY_PAGE_ID,
    reason="B5 needs an empty pantry: set SPEC_EMPTY_PANTRY_PAGE_ID to the id of an empty Notion page",
)
def test_b5_stops_with_error_when_pantry_has_no_ingredients(monkeypatch):
    """B5: an empty pantry stops execution with 'Error:' and no recipe."""
    state = run_with_env(monkeypatch, NO_INGREDIENTS_QUERY, "NOTION_PAGE_ID", EMPTY_PANTRY_PAGE_ID)
    names = tool_names(state)
    assert names.count("retrieve_pantry") == 1, names
    assert "search_recipes" not in names, names
    assert begins_with_error(state, NO_INGREDIENTS_QUERY), "output must begin with 'Error:'"
    assert_criteria(NO_INGREDIENTS_QUERY, state, "B5 lack of ingredients without recipe", NO_RECIPE)


def test_b6_search_failure_returns_error_without_recipe(monkeypatch):
    """B6 after B3: missing Tavily credentials fail the search; the agent reports 'Error:' and does not crash."""
    state = run_with_env(monkeypatch, HAPPY_QUERY, "TAVILY_API_KEY", None)
    names = tool_names(state)
    assert names.count("search_recipes") == 1, names
    assert "retrieve_pantry" not in names, names
    assert begins_with_error(state, HAPPY_QUERY), "output must begin with 'Error:'"
    assert_criteria(HAPPY_QUERY, state, "B6 search failure without recipe", NO_RECIPE)


@pytest.mark.parametrize(
    "env_name, env_value",
    [
        pytest.param("NOTION_API_KEY", None, id="missing-credentials"),
        pytest.param("NOTION_PAGE_ID", "0" * 32, id="inaccessible-page"),
    ],
)
def test_b6_pantry_failure_returns_error_without_recipe(monkeypatch, env_name, env_value):
    """B6 after B4: a failed pantry lookup yields 'Error:' and no recipe, and no search is made."""
    state = run_with_env(monkeypatch, NO_INGREDIENTS_QUERY, env_name, env_value)
    names = tool_names(state)
    assert names.count("retrieve_pantry") == 1, names
    assert "search_recipes" not in names, names
    assert begins_with_error(state, NO_INGREDIENTS_QUERY), "output must begin with 'Error:'"
    assert_criteria(NO_INGREDIENTS_QUERY, state, "B6 pantry failure without recipe", NO_RECIPE)


@pytest.mark.parametrize(
    "api_key",
    [pytest.param(None, id="missing-key"), pytest.param("sk-rejected-by-test", id="rejected-key")],
)
def test_b6_model_failure_returns_error_without_recipe(monkeypatch, api_key):
    """B6: a model failure (missing or rejected credentials) must not crash the agent."""
    state = run_with_env(monkeypatch, HAPPY_QUERY, "OPENAI_API_KEY", api_key)
    assert begins_with_error(state, HAPPY_QUERY), "output must begin with 'Error:'"
    assert_criteria(HAPPY_QUERY, state, "B6 model failure without recipe", NO_RECIPE)


def test_b7_recipe_is_complete_and_uses_identified_ingredients():
    """B7: the recipe has all five non-empty fields and uses at least one identified ingredient."""
    state = run_agent(HAPPY_QUERY)
    assert not begins_with_error(state, HAPPY_QUERY), "a supported query must not end in 'Error:'"
    assert_criteria(HAPPY_QUERY, state, "B7 complete recipe", (
        "Passes only if the output contains a complete recipe with all five fields present and non-empty: "
        "Title (a recipe title), Ingredients (a list with quantities), Equipment (the required equipment), "
        "Instructions (step-by-step preparation) and Storage (storage recommendations). The recipe must use at "
        "least one of the user's ingredients: chicken, rice or peas."
    ))


def test_goal_recipe_is_personalized_to_ingredients_and_preferences():
    """Goal: a personalized recipe that adapts to the listed ingredients, preferences and restrictions."""
    query = "I'm vegetarian and want something spicy and quick. I have chickpeas, spinach and rice."
    state = run_agent(query)
    assert not begins_with_error(state, query), "a supported query must not end in 'Error:'"
    assert_criteria(query, state, "Goal personalization", (
        "Passes only if the recipe uses the listed ingredients (chickpeas, spinach, rice), respects the "
        "vegetarian restriction (no meat, poultry or fish), and reflects the requested spicy and quick style."
    ))


def test_persona_responds_as_a_chef():
    """Persona: the agent assumes the persona of a chef who creates personalized recipes."""
    state = run_agent(HAPPY_QUERY)
    assert_criteria(HAPPY_QUERY, state, "Persona: chef", (
        "Passes only if the response speaks as a chef creating a personalized recipe for this cook, with a "
        "culinary voice and recipe-oriented guidance. It fails if it reads as a generic chatbot reply, a "
        "nutrition lecture or unrelated text."
    ))


@pytest.mark.parametrize("query, restriction", GI1_CASES)
def test_gi1_derived_ingredients_of_restriction_are_excluded(query, restriction):
    """GI1: a dietary restriction excludes all of its derived ingredients, including those the user did not mention."""
    state = run_agent(query)
    assert not begins_with_error(state, query), "a supported query must not end in 'Error:'"
    assert_criteria(query, state, "GI1 derived restriction", (
        "Passes only if no ingredient in the recipe violates the dietary restriction, including derived "
        f"ingredients the user did not mention. Restriction: {restriction}."
    ))


def test_quality_response_is_structured_for_automated_evaluation():
    """Quality: the response exposes the recipe fields as keys or labelled sections a checker can locate."""
    state = run_agent(HAPPY_QUERY)
    keys = " ".join(_keys(_plain(state))).lower()
    text = "\n".join(agent_texts(state, HAPPY_QUERY))
    missing = [
        stem for stem in RECIPE_FIELD_STEMS
        if stem not in keys and not re.search(rf"(?im)^[\s#*_>-]*{stem}", text)
    ]
    assert not missing, f"recipe fields not locatable as keys or labelled sections: {missing}"
