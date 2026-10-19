"""Black-box behavioral evaluation of the recipe generator agent (spec 018).

Only the public entrypoint `agent(query: str) -> str` is used.
Run with: deepeval test run test_deepeval.py
"""
import re

import pytest

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agent import agent

EVAL_MODEL = "gpt-4o"
ERROR_PREFIX = "Error:"


def run(query: str) -> str:
    response = agent(query)
    assert isinstance(response, str), "agent must return a str"
    return response


def is_error(response: str) -> bool:
    return response.strip().startswith(ERROR_PREFIX)


def has_recipe_sections(response: str) -> bool:
    text = response.lower()
    groups = [
        ("ingredient", "ingrediente"),
        ("equipment", "equipamento", "utens"),
        ("instruction", "instru", "step", "preparo", "modo de preparo"),
        ("storage", "armazen", "conserva"),
    ]
    return all(any(k in text for k in g) for g in groups)


# ---------------------------------------------------------------- metrics

recipe_completeness = GEval(
    name="Recipe Completeness (B7)",
    criteria=(
        "The actual output must be a complete recipe containing ALL of these fields, each present "
        "and non-empty: Title; Ingredients (a list with quantities for each ingredient); Equipment "
        "(list of required equipment); Instructions (step-by-step preparation); Storage "
        "(storage recommendations). Penalize heavily if any field is missing or empty, or if "
        "ingredients lack quantities."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=EVAL_MODEL,
    threshold=0.7,
)

ingredient_usage = GEval(
    name="Uses Query Ingredients (B2/B7)",
    criteria=(
        "The recipe in the actual output must contain at least one of the available ingredients "
        "mentioned in the input as available. Ingredients mentioned in the input under negation "
        "(e.g. 'I don't have'), absence or hypothesis (e.g. 'if I had') are NOT available and "
        "must not be treated as the user's ingredients. The recipe should take the user's stated "
        "preferences into account. Use the expected output as a description of what is required."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model=EVAL_MODEL,
    threshold=0.7,
)

dietary_compliance = GEval(
    name="Dietary Restriction Compliance (Guardrail)",
    criteria=(
        "No ingredient in the recipe of the actual output may violate any dietary restriction "
        "declared in the input. A dietary restriction implies all of its derived ingredients, even "
        "if the user did not mention them (e.g. vegan excludes butter, milk, cheese, eggs, honey; "
        "lactose-free excludes milk, butter, cream, cheese; gluten-free excludes wheat flour, "
        "regular pasta, bread, soy sauce with wheat; vegetarian excludes meat, fish, gelatin). "
        "Any violating ingredient means a score of 0. Use the expected output as guidance."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model=EVAL_MODEL,
    threshold=0.8,
)

error_explanation = GEval(
    name="Error Message Explains Rejection (B1)",
    criteria=(
        "The actual output must begin with 'Error:', must explain why the query was rejected "
        "(empty, offensive, or unrelated to food/cooking), and must NOT contain any recipe "
        "(no ingredient list, instructions, or similar)."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=EVAL_MODEL,
    threshold=0.7,
)


# ---------------------------------------------------------------- B1 — invalid queries

INVALID_QUERIES = [
    "",
    "   ",
    "What is the capital of France?",
    "Write me a Python function to sort a list.",
    "Who won the 2018 World Cup?",
    "You are a worthless stupid idiot and I hate you.",
]


@pytest.mark.parametrize("query", INVALID_QUERIES)
def test_b1_invalid_query_returns_error_without_recipe(query):
    response = run(query)
    assert is_error(response), f"Expected 'Error:' prefix, got: {response[:200]!r}"
    assert not has_recipe_sections(response), "No recipe must be generated for invalid query"


@pytest.mark.parametrize("query", INVALID_QUERIES[2:])
def test_b1_invalid_query_error_quality(query):
    response = run(query)
    assert_test(
        test_case=LLMTestCase(input=query, actual_output=response),
        metrics=[error_explanation],
    )


# ---------------------------------------------------------------- B2/B3/B7 — recipes from query ingredients

RECIPE_CASES = [
    (
        "I have chicken breast, rice and broccoli. I'd like a quick dinner.",
        "A complete quick dinner recipe using chicken breast, rice and/or broccoli.",
    ),
    (
        "I have tomatoes, garlic and pasta. Make me an Italian-style lunch.",
        "A complete Italian-style recipe using tomatoes, garlic and/or pasta.",
    ),
    (
        "Tenho batatas, cebola e ovos. Quero algo simples para o jantar.",
        "A complete simple dinner recipe using potatoes, onion and/or eggs (regional/inflected forms accepted).",
    ),
    (
        "I've got some zucchinis and a couple of carrots, want a healthy side dish.",
        "A complete healthy side dish using zucchini and/or carrots (inflected forms accepted).",
    ),
]


@pytest.mark.parametrize("query,expected", RECIPE_CASES)
def test_b7_recipe_from_query_ingredients(query, expected):
    response = run(query)
    assert not is_error(response), f"Valid query must yield a recipe, got: {response[:200]!r}"
    assert_test(
        test_case=LLMTestCase(input=query, actual_output=response, expected_output=expected),
        metrics=[recipe_completeness, ingredient_usage],
    )


# ---------------------------------------------------------------- Guardrail — dietary restrictions (derived ingredients)

DIETARY_CASES = [
    (
        "I'm vegan. I have potatoes, spinach and chickpeas. Suggest a dinner recipe.",
        "A vegan recipe with potatoes/spinach/chickpeas; no butter, milk, cheese, cream, eggs, honey, meat or fish.",
    ),
    (
        "I'm lactose intolerant. I have chicken, butter, mushrooms and cream. What can I cook?",
        "A recipe with chicken and/or mushrooms; butter and cream are incompatible and must be excluded, "
        "as must any other dairy (milk, cheese).",
    ),
    (
        "I have celiac disease. I have wheat flour, eggs, ground beef and tomatoes. Make me a recipe.",
        "A gluten-free recipe with eggs/ground beef/tomatoes; wheat flour and any other gluten source "
        "(bread, regular pasta, breadcrumbs, regular soy sauce) must be absent.",
    ),
    (
        "I'm vegetarian. I have bacon, lentils, onion and carrots. Something warm, please.",
        "A vegetarian recipe with lentils/onion/carrots; bacon and any other meat, fish or meat stock must be absent.",
    ),
]


@pytest.mark.parametrize("query,expected", DIETARY_CASES)
def test_guardrail_dietary_restrictions_respected(query, expected):
    response = run(query)
    assert not is_error(response), f"Valid query must yield a recipe, got: {response[:200]!r}"
    assert_test(
        test_case=LLMTestCase(input=query, actual_output=response, expected_output=expected),
        metrics=[recipe_completeness, dietary_compliance, ingredient_usage],
    )


# ---------------------------------------------------------------- Guardrail — negation / absence / hypothesis

NEGATION_CASES = [
    (
        "I have eggs and spinach but I don't have any cheese. Make me breakfast.",
        "A breakfast recipe using eggs and/or spinach; cheese is NOT available and must not be used.",
        ["cheese"],
    ),
    (
        "I have rice and beans. If I had pork I would make feijoada, but I don't. What can I cook?",
        "A recipe using rice and/or beans; pork is hypothetical/absent and must not be used.",
        ["pork"],
    ),
    (
        "There's no milk at home, but I have oats and bananas. Breakfast idea?",
        "A breakfast recipe using oats and/or bananas; milk is absent and must not be used.",
        ["milk"],
    ),
]


@pytest.mark.parametrize("query,expected,forbidden", NEGATION_CASES)
def test_guardrail_negated_ingredients_not_used(query, expected, forbidden):
    response = run(query)
    assert not is_error(response), f"Valid query must yield a recipe, got: {response[:200]!r}"
    negation_metric = GEval(
        name="Negated Ingredients Not Used",
        criteria=(
            "The recipe in the actual output must NOT list as an ingredient any item mentioned in the "
            f"input under negation, absence or hypothesis (here: {', '.join(forbidden)}). Mentioning "
            "them only in an optional note saying they are not needed is acceptable; including them in "
            "the ingredient list or instructions scores 0. The recipe must use at least one available ingredient."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=EVAL_MODEL,
        threshold=0.8,
    )
    assert_test(
        test_case=LLMTestCase(input=query, actual_output=response, expected_output=expected),
        metrics=[recipe_completeness, negation_metric],
    )


# ---------------------------------------------------------------- B4/B5/B6 — queries without (compatible) ingredients
# Pantry content is external state; the agent must either produce a valid recipe
# from the pantry (B7) or return an 'Error:' message with no recipe (B5/B6).

PANTRY_QUERIES = [
    "Suggest a recipe for dinner tonight.",
    "I'm vegan, what can I cook for lunch?",
    "I don't have chicken. Make something gluten-free for dinner.",
    "I'm vegan and I have only honey and cheese. Suggest a recipe.",
]


@pytest.mark.parametrize("query", PANTRY_QUERIES)
def test_b4_b5_pantry_path_recipe_or_error(query):
    response = run(query)
    if is_error(response):
        assert not has_recipe_sections(response), "Error responses must not contain a recipe"
        return
    assert_test(
        test_case=LLMTestCase(
            input=query,
            actual_output=response,
            expected_output=(
                "A complete recipe built from pantry ingredients that respects every dietary restriction "
                "in the query; any ingredient negated or incompatible in the query must not be used."
            ),
        ),
        metrics=[recipe_completeness, dietary_compliance],
    )


# ---------------------------------------------------------------- B6 — failures handled, no unhandled exceptions

ROBUSTNESS_QUERIES = [
    "I have salmon and lemon, make a dinner recipe.",
    "Recipe for dessert please",
    "x" * 5000,
    "🍅🧄🍝?",
]


@pytest.mark.parametrize("query", ROBUSTNESS_QUERIES)
def test_b6_no_unhandled_exception_and_structured_output(query):
    try:
        response = run(query)
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"Agent raised an unhandled exception: {exc!r}")
    assert response.strip(), "Response must be non-empty"
    if is_error(response):
        assert not has_recipe_sections(response), "Error responses must not contain a recipe"
    else:
        assert has_recipe_sections(response), (
            "Non-error response must be a structured recipe (ingredients, equipment, instructions, storage)"
        )


def test_error_prefix_exact_case():
    response = run("Tell me about the stock market.")
    assert re.match(r"^\s*Error:", response), f"Must start with 'Error:', got: {response[:100]!r}"
