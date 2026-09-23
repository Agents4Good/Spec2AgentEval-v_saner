"""
Tests for the Personalized Recipe Generator Agent.

The unit tests mock the LLM (``get_llm``) and the two tools
(``retrieve_pantry`` / ``search_recipes``) so they run offline and
deterministically, covering behaviors B1-B8 from the specification.

Run: ``pytest tests.py -v``
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

import agent as agent_module
from agent import (
    DietaryCompatibility,
    QueryAnalysis,
    QueryValidation,
    Recipe,
    agent,
)

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeStructuredLLM:
    def __init__(self, parent: "FakeLLM", schema):
        self.parent = parent
        self.schema = schema

    def invoke(self, messages):
        name = self.schema.__name__
        self.parent.calls.append((name, messages))
        handler = self.parent.responses.get(name)
        if handler is None:
            raise AssertionError(f"unexpected structured call for {name}")
        if isinstance(handler, Exception):
            raise handler
        if isinstance(handler, list):
            if not handler:
                raise AssertionError(f"no more responses queued for {name}")
            return handler.pop(0)
        return handler


class FakeLLM:
    """Minimal stand-in for ChatOpenAI supporting ``with_structured_output``."""

    def __init__(self, responses: Dict[str, Any]):
        self.responses = responses
        self.calls: List[Any] = []

    def with_structured_output(self, schema):
        return FakeStructuredLLM(self, schema)


def make_recipe(**overrides) -> Recipe:
    base = dict(
        title="Frango grelhado com brócolis",
        ingredients=["300 g de peito de frango", "200 g de brócolis", "1 colher de sopa de azeite", "sal a gosto"],
        equipment=["frigideira", "faca", "tábua de corte"],
        instructions=["Tempere o frango.", "Grelhe por 6 minutos de cada lado.", "Salteie o brócolis."],
        storage="Conserve em recipiente fechado na geladeira por até 3 dias.",
    )
    base.update(overrides)
    return Recipe(**base)


VALID = QueryValidation(is_food_related=True, is_offensive=False, reason="food")
OFFENSIVE = QueryValidation(is_food_related=True, is_offensive=True, reason="offensive")
NOT_FOOD = QueryValidation(is_food_related=False, is_offensive=False, reason="programming question")


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    monkeypatch.setenv("NOTION_API_KEY", "test-notion-key")
    monkeypatch.setenv("NOTION_PAGE_ID", "test-page-id")


@pytest.fixture
def tools(monkeypatch):
    """Patches both tools and records how many times each one was called."""
    calls: Dict[str, List[Any]] = {"search_recipes": [], "retrieve_pantry": []}

    def fake_search(ingredients):
        calls["search_recipes"].append(list(ingredients))
        return [
            {"title": "Grilled chicken with broccoli", "url": "https://example.com/1", "content": "Season, grill, serve."},
        ]

    def fake_pantry():
        calls["retrieve_pantry"].append(())
        return ["ovos", "arroz", "tomate", "cebola"]

    monkeypatch.setattr(agent_module, "search_recipes", fake_search)
    monkeypatch.setattr(agent_module, "retrieve_pantry", fake_pantry)
    return calls


def use_llm(monkeypatch, responses: Dict[str, Any]) -> FakeLLM:
    fake = FakeLLM(responses)
    monkeypatch.setattr(agent_module, "get_llm", lambda: fake)
    return fake


# --------------------------------------------------------------------------- #
# Interface contract
# --------------------------------------------------------------------------- #


def test_agent_returns_state_dict(env, tools, monkeypatch):
    use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=["frango", "brócolis"]),
        "Recipe": make_recipe(),
    })
    state = agent("Receita com frango e brócolis")
    assert isinstance(state, dict)
    for key in ("query", "valid", "tool_called", "base_ingredients", "compatible_ingredients", "recipe", "error"):
        assert key in state
    assert state["error"] is None


def test_default_llm_is_chatopenai_gpt4o(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    llm = agent_module.get_llm()
    assert type(llm).__name__ == "ChatOpenAI"
    assert llm.model_name == "gpt-4o"


def test_missing_openai_key_returns_error_without_building_graph(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    state = agent("Receita com frango")
    assert state["error"].startswith("Error: ")
    assert state["recipe"] is None


# --------------------------------------------------------------------------- #
# B1: query validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("query", ["", "   ", None])
def test_empty_query_returns_error_without_calling_llm_or_tools(env, tools, monkeypatch, query):
    fake = use_llm(monkeypatch, {})
    state = agent(query)
    assert state["valid"] is False
    assert state["error"].startswith("Error:")
    assert fake.calls == []
    assert tools["search_recipes"] == [] and tools["retrieve_pantry"] == []
    assert state["recipe"] is None


def test_offensive_query_returns_error(env, tools, monkeypatch):
    use_llm(monkeypatch, {"QueryValidation": OFFENSIVE})
    state = agent("seu cozinheiro idiota, faz uma receita")
    assert state["valid"] is False
    assert state["error"].startswith("Error:")
    assert state["tool_called"] is None
    assert tools["search_recipes"] == [] and tools["retrieve_pantry"] == []
    assert state["recipe"] is None


def test_non_food_query_returns_error(env, tools, monkeypatch):
    use_llm(monkeypatch, {"QueryValidation": NOT_FOOD})
    state = agent("Como faço um loop em Python?")
    assert state["error"].startswith("Error:")
    assert state["recipe"] is None
    assert tools["search_recipes"] == [] and tools["retrieve_pantry"] == []


# --------------------------------------------------------------------------- #
# B2/B3/B4: exactly one tool is chosen based on identified ingredients
# --------------------------------------------------------------------------- #


def test_query_with_ingredients_calls_only_search_recipes(env, tools, monkeypatch):
    use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=["frango", "brócolis"]),
        "Recipe": make_recipe(),
    })
    state = agent("Quero uma receita com frango e brócolis")
    assert state["tool_called"] == "search_recipes"
    assert tools["search_recipes"] == [["frango", "brócolis"]]
    assert tools["retrieve_pantry"] == []
    assert state["base_ingredients"] == ["frango", "brócolis"]


def test_query_without_ingredients_calls_only_retrieve_pantry(env, tools, monkeypatch):
    use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=[], preferences=["jantar rápido"]),
        "Recipe": make_recipe(title="Omelete de tomate"),
    })
    state = agent("Me sugira um jantar rápido com o que tenho em casa")
    assert state["tool_called"] == "retrieve_pantry"
    assert len(tools["retrieve_pantry"]) == 1
    assert tools["search_recipes"] == []
    assert state["base_ingredients"] == ["ovos", "arroz", "tomate", "cebola"]


# --------------------------------------------------------------------------- #
# B5: pantry with no ingredients stops the execution
# --------------------------------------------------------------------------- #


def test_empty_pantry_returns_error_and_skips_recipe(env, monkeypatch):
    monkeypatch.setattr(agent_module, "retrieve_pantry", lambda: [])
    fake = use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=[]),
    })
    state = agent("Faz um almoço pra mim")
    assert state["error"].startswith("Error:")
    assert state["tool_called"] == "retrieve_pantry"
    assert state["recipe"] is None
    assert all(name != "Recipe" for name, _ in fake.calls), "recipe generation must be skipped"


# --------------------------------------------------------------------------- #
# B6: dietary incompatibility stops the execution
# --------------------------------------------------------------------------- #


def test_all_base_ingredients_incompatible_returns_error(env, tools, monkeypatch):
    fake = use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=["mel", "leite"], dietary_restrictions=["vegano"]),
        "DietaryCompatibility": DietaryCompatibility(compatible=[], incompatible=["mel", "leite"]),
    })
    state = agent("Receita vegana com mel e leite")
    assert state["error"].startswith("Error:")
    assert state["compatible_ingredients"] == []
    assert state["recipe"] is None
    assert all(name != "Recipe" for name, _ in fake.calls)


def test_dietary_check_is_skipped_when_no_restrictions(env, tools, monkeypatch):
    fake = use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=["frango", "brócolis"]),
        "Recipe": make_recipe(),
    })
    state = agent("Receita com frango e brócolis")
    assert state["compatible_ingredients"] == ["frango", "brócolis"]
    assert all(name != "DietaryCompatibility" for name, _ in fake.calls)


def test_partially_incompatible_ingredients_keeps_only_compatible_ones(env, tools, monkeypatch):
    use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=["grão-de-bico", "mel"], dietary_restrictions=["vegano"]),
        "DietaryCompatibility": DietaryCompatibility(compatible=["grão-de-bico"], incompatible=["mel"]),
        "Recipe": make_recipe(title="Curry de grão-de-bico"),
    })
    state = agent("Receita vegana com grão-de-bico e mel")
    assert state["error"] is None
    assert state["compatible_ingredients"] == ["grão-de-bico"]


# --------------------------------------------------------------------------- #
# B7: tool/model failures never raise and always produce an Error message
# --------------------------------------------------------------------------- #


def test_tool_erro_prefixed_string_becomes_error_state(env, monkeypatch):
    monkeypatch.setattr(agent_module, "retrieve_pantry", lambda: "erro: credenciais do Notion ausentes")
    use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=[]),
    })
    state = agent("O que posso cozinhar hoje?")
    assert state["error"].startswith("Error:")
    assert "credenciais do Notion ausentes" in state["error"]
    assert state["recipe"] is None


def test_search_recipes_tool_error_becomes_error_state(env, monkeypatch):
    monkeypatch.setattr(agent_module, "search_recipes", lambda ingredients: "erro: timeout ao buscar receitas no Tavily.")
    use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=["frango"]),
    })
    state = agent("Receita com frango")
    assert state["error"].startswith("Error:")
    assert state["recipe"] is None


def test_tool_exception_does_not_propagate(env, monkeypatch):
    def boom():
        raise RuntimeError("connection reset")

    monkeypatch.setattr(agent_module, "retrieve_pantry", boom)
    use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=[]),
    })
    state = agent("O que posso cozinhar hoje?")
    assert state["error"].startswith("Error:")
    assert state["recipe"] is None


def test_llm_exception_during_generation_does_not_propagate(env, tools, monkeypatch):
    use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=["frango"]),
        "Recipe": RuntimeError("model unavailable"),
    })
    state = agent("Receita com frango")
    assert state["error"].startswith("Error:")
    assert state["recipe"] is None


# --------------------------------------------------------------------------- #
# B8: recipe generation
# --------------------------------------------------------------------------- #


def test_successful_recipe_has_all_required_fields(env, tools, monkeypatch):
    use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=["frango", "brócolis"]),
        "Recipe": make_recipe(),
    })
    state = agent("Receita com frango e brócolis")
    assert state["error"] is None
    recipe = state["recipe"]
    for field in ("Título", "Ingredientes", "Equipamentos", "Instruções", "Armazenamento"):
        assert field in recipe
        assert recipe[field], f"{field} must not be empty"


def test_incomplete_recipe_from_model_is_rejected(env, tools, monkeypatch):
    use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=["frango"]),
        "Recipe": make_recipe(title=""),
    })
    state = agent("Receita com frango")
    assert state["error"].startswith("Error:")
    assert state["recipe"] is None


def test_reference_recipes_are_passed_as_context_not_as_final_recipe(env, tools, monkeypatch):
    fake = use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=["frango", "brócolis"]),
        "Recipe": make_recipe(),
    })
    agent("Receita com frango e brócolis")
    recipe_calls = [msgs for name, msgs in fake.calls if name == "Recipe"]
    assert len(recipe_calls) == 1
    prompt = "\n".join(m.content for m in recipe_calls[0])
    assert "Grilled chicken with broccoli" in prompt
    assert "frango" in prompt and "brócolis" in prompt
    assert "apenas inspiração de preparo" in prompt.lower() or "inspiração" in prompt.lower()


def test_pantry_ingredients_are_passed_to_recipe_generation(env, tools, monkeypatch):
    fake = use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=[]),
        "Recipe": make_recipe(title="Arroz com ovos"),
    })
    agent("O que posso cozinhar hoje?")
    prompt = "\n".join(m.content for name, msgs in fake.calls if name == "Recipe" for m in msgs)
    for item in ["ovos", "arroz", "tomate", "cebola"]:
        assert item in prompt
