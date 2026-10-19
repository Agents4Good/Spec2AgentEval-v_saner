"""Black-box behavioral evaluation of the recipe-generator agent (spec 028).

Derived exclusively from spec.md. The agent is exercised only through its
entrypoint `agent(query) -> str`; only the returned string is evaluated.

Run with:  deepeval test run test_deepeval.py
"""

import os
import re
from importlib import import_module

import pytest

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

agent = import_module("agent").agent

JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", "gpt-4o")


def run_agent(query: str) -> str:
    # B6: the agent must never terminate with an unhandled exception.
    try:
        output = agent(query)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"Agent raised an unhandled exception: {exc!r}")
    assert isinstance(output, str), "Output contract: agent must return a string"
    return output


def is_error(output: str) -> bool:
    return output.strip().startswith("Error:")


RECIPE_SECTIONS = ["title", "ingredients", "equipment", "instructions", "storage"]


def has_recipe_sections(output: str) -> bool:
    low = output.lower()
    return all(s in low for s in RECIPE_SECTIONS[1:])


# --------------------------------------------------------------------------
# Metrics (GEval, judged on input + actual_output only)
# --------------------------------------------------------------------------

recipe_completeness = GEval(
    name="Recipe Completeness (B7)",
    criteria=(
        "The actual output must be a complete recipe containing ALL of these "
        "non-empty fields: a Title; an Ingredients list where each ingredient "
        "has a quantity; an Equipment list; step-by-step Instructions; and "
        "Storage recommendations. Penalize heavily any missing or empty field."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=JUDGE_MODEL,
    threshold=0.7,
)

dietary_compliance = GEval(
    name="Dietary Restriction Compliance (B3/B7 invariant)",
    criteria=(
        "Identify every dietary restriction declared in the input (e.g. vegan, "
        "lactose-free, gluten-free, allergies). A restriction implies all of its "
        "derived ingredients even if not named by the user (e.g. vegan excludes "
        "meat, fish, eggs, dairy, honey, butter, cheese; gluten-free excludes "
        "wheat flour, regular pasta, bread, soy sauce with wheat; lactose-free "
        "excludes milk, cream, butter, cheese unless lactose-free). The recipe's "
        "ingredients must contain NO ingredient violating any declared "
        "restriction. Any single violation means the lowest score."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=JUDGE_MODEL,
    threshold=0.8,
)

negated_ingredients = GEval(
    name="Negated/Hypothetical Ingredients Excluded (invariant)",
    criteria=(
        "Ingredients the user mentions under negation, absence or hypothesis "
        "(e.g. 'I don't have eggs', 'I ran out of milk', 'if I had bacon') are "
        "NOT available ingredients. The recipe must not list them as "
        "ingredients. Penalize heavily if any such ingredient appears in the "
        "recipe's ingredient list."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=JUDGE_MODEL,
    threshold=0.8,
)

uses_query_ingredients = GEval(
    name="Uses Identified Ingredients and Preferences (B3/B7)",
    criteria=(
        "The recipe must use at least one of the available ingredients mentioned "
        "in the input that is compatible with the declared restrictions "
        "(accounting for regional variants and inflected forms), and must take "
        "the user's stated preferences (meal type, cuisine, style) into account."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=JUDGE_MODEL,
    threshold=0.6,
)

chef_persona = GEval(
    name="Chef Persona",
    criteria=(
        "The output should read as produced by a chef creating a personalized "
        "recipe for the user: culinary, coherent and practical guidance."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=JUDGE_MODEL,
    threshold=0.5,
)

invalid_query_error = GEval(
    name="Invalid Query Rejection (B1)",
    criteria=(
        "The input is empty, offensive, or unrelated to food/cooking. The output "
        "must be an error message beginning with 'Error:' that explains why the "
        "query is invalid, and must NOT contain any recipe (no ingredient list, "
        "instructions, etc.)."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=JUDGE_MODEL,
    threshold=0.7,
)

failure_error = GEval(
    name="Execution Failure Handling (B6)",
    criteria=(
        "An execution failure occurred (e.g. missing credentials). The output "
        "must begin with 'Error:', explain the failure, and must NOT contain a "
        "recipe."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=JUDGE_MODEL,
    threshold=0.7,
)


# --------------------------------------------------------------------------
# B1 — Query validation: invalid queries
# --------------------------------------------------------------------------

INVALID_QUERIES = [
    pytest.param("", id="empty"),
    pytest.param("   ", id="whitespace-only"),
    pytest.param("What is the capital of France?", id="unrelated-geography"),
    pytest.param("Write me a Python function to sort a list.", id="unrelated-code"),
    pytest.param("Who won the 2018 football World Cup?", id="unrelated-sports"),
    pytest.param("You are a worthless idiot, go to hell.", id="offensive"),
]


@pytest.mark.parametrize("query", INVALID_QUERIES)
def test_b1_invalid_query_rejected(query):
    output = run_agent(query)
    assert is_error(output), f"Invalid query must yield 'Error:' message, got: {output[:300]}"
    assert not has_recipe_sections(output), "No recipe may be generated for an invalid query"
    if query.strip():
        assert_test(
            test_case=LLMTestCase(input=query, actual_output=output),
            metrics=[invalid_query_error],
        )


# --------------------------------------------------------------------------
# B1 (valid) + B2/B3/B4/B7 — valid queries produce a complete recipe
# --------------------------------------------------------------------------

VALID_QUERIES = [
    pytest.param(
        "I want a quick dinner using chicken breast, rice and broccoli.",
        id="simple-ingredients",
    ),
    pytest.param(
        "Suggest a dessert I can make with bananas and oats.",
        id="dessert",
    ),
    pytest.param(
        "I'd like an Italian-style lunch with tomatoes, garlic and basil.",
        id="cuisine-preference",
    ),
    pytest.param(
        "Make me a breakfast with potatoes, onions and bell peppers.",
        id="breakfast",
    ),
]


@pytest.mark.parametrize("query", VALID_QUERIES)
def test_b7_valid_query_generates_complete_recipe(query):
    output = run_agent(query)
    assert not is_error(output), f"Valid query with ingredients must not error: {output[:300]}"
    assert_test(
        test_case=LLMTestCase(input=query, actual_output=output),
        metrics=[recipe_completeness, uses_query_ingredients, chef_persona],
    )


# --------------------------------------------------------------------------
# Invariant: dietary restriction implies derived ingredients (B3/B7)
# --------------------------------------------------------------------------

RESTRICTED_QUERIES = [
    pytest.param(
        "I'm vegan. I have chickpeas, spinach and coconut milk. Make me a dinner.",
        id="vegan",
    ),
    pytest.param(
        "I'm lactose intolerant. Give me a pasta recipe with zucchini and garlic.",
        id="lactose-free",
    ),
    pytest.param(
        "I have celiac disease. I want a lunch with rice, salmon and carrots.",
        id="gluten-free",
    ),
    pytest.param(
        "I'm vegetarian and allergic to peanuts. Make a stir-fry with tofu and broccoli.",
        id="vegetarian-peanut-allergy",
    ),
]


@pytest.mark.parametrize("query", RESTRICTED_QUERIES)
def test_dietary_restrictions_respected(query):
    output = run_agent(query)
    assert not is_error(output), f"Compatible ingredients exist; expected a recipe: {output[:300]}"
    assert_test(
        test_case=LLMTestCase(input=query, actual_output=output),
        metrics=[dietary_compliance, recipe_completeness, uses_query_ingredients],
    )


# --------------------------------------------------------------------------
# Invariant: negated / absent / hypothetical ingredients are not available
# --------------------------------------------------------------------------

NEGATION_QUERIES = [
    pytest.param(
        "I want a breakfast with flour and bananas, but I don't have eggs.",
        id="negation",
    ),
    pytest.param(
        "I ran out of milk. Make something with oats and apples.",
        id="absence",
    ),
    pytest.param(
        "If I had bacon I'd use it, but let's cook something with pasta and mushrooms.",
        id="hypothesis",
    ),
]


@pytest.mark.parametrize("query", NEGATION_QUERIES)
def test_negated_ingredients_not_used(query):
    output = run_agent(query)
    assert not is_error(output), f"Available ingredients exist; expected a recipe: {output[:300]}"
    assert_test(
        test_case=LLMTestCase(input=query, actual_output=output),
        metrics=[negated_ingredients, recipe_completeness],
    )


# --------------------------------------------------------------------------
# B5 — Lack of compatible ingredients (only checkable when a recipe would be
# impossible regardless of pantry: any recipe returned must still be compliant)
# --------------------------------------------------------------------------

def test_b5_only_incompatible_query_ingredients():
    query = "I'm vegan. I only have eggs, cheese and bacon. What can I cook?"
    output = run_agent(query)
    if is_error(output):
        assert not has_recipe_sections(output), "Error output must not contain a recipe"
    else:
        # Pantry may supply compatible ingredients; the recipe must then comply.
        assert_test(
            test_case=LLMTestCase(input=query, actual_output=output),
            metrics=[dietary_compliance, recipe_completeness],
        )


# --------------------------------------------------------------------------
# B6 — Execution failure: missing credentials
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "missing_key",
    ["NOTION_API_KEY", "NOTION_PAGE_ID", "TAVILY_API_KEY", "OPENAI_API_KEY"],
)
def test_b6_missing_credentials_handled(monkeypatch, missing_key):
    monkeypatch.delenv(missing_key, raising=False)
    query = "Make me a dinner with chicken, rice and broccoli."
    output = run_agent(query)
    assert is_error(output), (
        f"With {missing_key} missing, output must begin with 'Error:', got: {output[:300]}"
    )
    assert not has_recipe_sections(output), "No recipe may be generated on failure"
    if missing_key != "OPENAI_API_KEY":
        assert_test(
            test_case=LLMTestCase(input=query, actual_output=output),
            metrics=[failure_error],
        )


def test_b6_invalid_notion_page_handled(monkeypatch):
    monkeypatch.setenv("NOTION_PAGE_ID", "00000000000000000000000000000000")
    output = run_agent("Make me a dinner with chicken, rice and broccoli.")
    assert is_error(output), f"Inaccessible page must yield 'Error:', got: {output[:300]}"
    assert not has_recipe_sections(output)


# --------------------------------------------------------------------------
# Error message format (all Error paths)
# --------------------------------------------------------------------------

def test_error_message_has_explanation():
    output = run_agent("Tell me a joke about politicians.")
    assert is_error(output)
    assert re.sub(r"^\s*Error:\s*", "", output).strip(), "Error must include an explanation"
