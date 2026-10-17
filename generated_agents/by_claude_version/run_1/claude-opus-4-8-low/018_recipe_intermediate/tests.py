"""
Tests for the Personalized Recipe Generator Agent.

The unit tests mock the LLM (``get_llm``) and the tool HTTP layers
(``_notion_pantry`` / ``_tavily_search``) so they run offline and
deterministically. An optional integration test runs only when the real API
keys are present.

Run: ``pytest tests.py -v``
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest

import agent as agent_module
from agent import (
    CompatibilityFilter,
    QueryAnalysis,
    QueryValidation,
    Recipe,
    agent,
    format_recipe,
    retrieve_pantry,
    search_recipes,
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
        self.parent.calls.append(name)
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
        self.calls: List[str] = []

    def with_structured_output(self, schema):
        return FakeStructuredLLM(self, schema)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    monkeypatch.setenv("NOTION_API_KEY", "test-notion-key")
    monkeypatch.setenv("NOTION_PAGE_ID", "test-page-id")


def install_llm(monkeypatch, responses: Dict[str, Any]) -> FakeLLM:
    fake = FakeLLM(responses)
    monkeypatch.setattr(agent_module, "get_llm", lambda: fake)
    return fake


def track_tools(monkeypatch):
    """Records which tools are invoked (through the LangChain .invoke interface)."""
    calls: List[str] = []
    real_pantry = agent_module._notion_pantry
    real_search = agent_module._tavily_search

    def pantry(*a, **k):
        calls.append("retrieve_pantry")
        return real_pantry(*a, **k)

    def search(ingredients):
        calls.append("search_recipes")
        return real_search(ingredients)

    monkeypatch.setattr(agent_module, "_notion_pantry", pantry)
    monkeypatch.setattr(agent_module, "_tavily_search", search)
    return calls


def a_valid_query() -> QueryValidation:
    return QueryValidation(is_food_related=True, is_offensive=False, reason="ok")


def a_recipe() -> Recipe:
    return Recipe(
        title="Chickpea Spinach Stew",
        ingredients=["1 can chickpeas", "2 cups spinach"],
        equipment=["pot", "spoon"],
        instructions=["Heat the pot.", "Add chickpeas and spinach.", "Simmer 10 min."],
        storage="Keep refrigerated up to 3 days.",
    )


# --------------------------------------------------------------------------- #
# B1 - Query validation
# --------------------------------------------------------------------------- #


def test_b1_empty_query(monkeypatch):
    calls = track_tools(monkeypatch)
    install_llm(monkeypatch, {})  # LLM must not be needed
    result = agent("   ")
    assert result["is_valid"] is False
    assert result["output"].startswith("Error:")
    assert calls == []  # no tool called
    assert result.get("recipe") is None


def test_b1_non_food_query(monkeypatch):
    calls = track_tools(monkeypatch)
    install_llm(monkeypatch, {
        "QueryValidation": QueryValidation(
            is_food_related=False, is_offensive=False, reason="it is about programming"
        )
    })
    result = agent("How do I write a for loop in Python?")
    assert result["is_valid"] is False
    assert result["output"].startswith("Error:")
    assert calls == []
    assert result.get("recipe") is None


def test_b1_offensive_query(monkeypatch):
    calls = track_tools(monkeypatch)
    install_llm(monkeypatch, {
        "QueryValidation": QueryValidation(is_food_related=True, is_offensive=True, reason="rude")
    })
    result = agent("some offensive cooking rant")
    assert result["is_valid"] is False
    assert result["output"].startswith("Error:")
    assert calls == []


# --------------------------------------------------------------------------- #
# B2 + B3 - ingredients identified -> search_recipes only
# --------------------------------------------------------------------------- #


def test_b3_search_called_once_pantry_not_called(monkeypatch):
    calls = track_tools(monkeypatch)
    monkeypatch.setattr(
        agent_module, "_tavily_search",
        lambda ingredients: calls.append("search_recipes") or [
            {"title": "Stew", "url": "http://x", "content": "a stew"}
        ],
    )
    install_llm(monkeypatch, {
        "QueryValidation": a_valid_query(),
        "QueryAnalysis": QueryAnalysis(
            ingredients=["chickpeas", "spinach"], dietary_restrictions=["vegan"], preferences=[]
        ),
        "Recipe": a_recipe(),
    })
    result = agent("A vegan dinner with chickpeas and spinach")
    assert result["tool_called"] == "search_recipes"
    assert calls.count("search_recipes") == 1
    assert "retrieve_pantry" not in calls
    assert result["recipe"] is not None
    assert result["error"] is None


# --------------------------------------------------------------------------- #
# B4 - no ingredients -> retrieve_pantry only, with dietary filter
# --------------------------------------------------------------------------- #


def test_b4_pantry_called_once_search_not_called(monkeypatch):
    calls = track_tools(monkeypatch)
    monkeypatch.setattr(
        agent_module, "_notion_pantry",
        lambda: calls.append("retrieve_pantry") or ["rice", "milk", "beans"],
    )
    install_llm(monkeypatch, {
        "QueryValidation": a_valid_query(),
        "QueryAnalysis": QueryAnalysis(
            ingredients=[], dietary_restrictions=["vegan"], preferences=[]
        ),
        "CompatibilityFilter": CompatibilityFilter(compatible=["rice", "beans"]),
        "Recipe": a_recipe(),
    })
    result = agent("Cook me a vegan meal from my pantry")
    assert result["tool_called"] == "retrieve_pantry"
    assert calls.count("retrieve_pantry") == 1
    assert "search_recipes" not in calls
    assert result["base_ingredients"] == ["rice", "beans"]
    assert result["recipe"] is not None


# --------------------------------------------------------------------------- #
# B5 - pantry has no compatible ingredient -> Error, no recipe
# --------------------------------------------------------------------------- #


def test_b5_empty_pantry(monkeypatch):
    monkeypatch.setattr(agent_module, "_notion_pantry", lambda: [])
    install_llm(monkeypatch, {
        "QueryValidation": a_valid_query(),
        "QueryAnalysis": QueryAnalysis(ingredients=[], dietary_restrictions=[], preferences=[]),
    })
    result = agent("Make me something tasty")
    assert result["output"].startswith("Error:")
    assert result.get("recipe") is None


def test_b5_no_compatible_pantry_ingredient(monkeypatch):
    monkeypatch.setattr(agent_module, "_notion_pantry", lambda: ["milk", "cheese"])
    install_llm(monkeypatch, {
        "QueryValidation": a_valid_query(),
        "QueryAnalysis": QueryAnalysis(ingredients=[], dietary_restrictions=["vegan"], preferences=[]),
        "CompatibilityFilter": CompatibilityFilter(compatible=[]),
    })
    result = agent("A vegan dish from my pantry")
    assert result["output"].startswith("Error:")
    assert result.get("recipe") is None


# --------------------------------------------------------------------------- #
# B6 - tool / model failures handled gracefully
# --------------------------------------------------------------------------- #


def test_b6_search_tool_failure(monkeypatch):
    def boom(ingredients):
        raise RuntimeError("missing TAVILY_API_KEY")
    monkeypatch.setattr(agent_module, "_tavily_search", boom)
    install_llm(monkeypatch, {
        "QueryValidation": a_valid_query(),
        "QueryAnalysis": QueryAnalysis(ingredients=["eggs"], dietary_restrictions=[], preferences=[]),
    })
    result = agent("Omelette with eggs")
    assert result["output"].startswith("Error:")
    assert result.get("recipe") is None


def test_b6_pantry_tool_failure(monkeypatch):
    def boom():
        raise TimeoutError("timed out")
    monkeypatch.setattr(agent_module, "_notion_pantry", boom)
    install_llm(monkeypatch, {
        "QueryValidation": a_valid_query(),
        "QueryAnalysis": QueryAnalysis(ingredients=[], dietary_restrictions=[], preferences=[]),
    })
    result = agent("Cook from my pantry")
    assert result["output"].startswith("Error:")
    assert result.get("recipe") is None


def test_b6_model_failure(monkeypatch):
    install_llm(monkeypatch, {"QueryValidation": RuntimeError("model unavailable")})
    result = agent("A pasta recipe")
    assert result["output"].startswith("Error:")
    assert result.get("recipe") is None


def test_b6_missing_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = agent("A pasta recipe")
    assert result["output"].startswith("Error:")


# --------------------------------------------------------------------------- #
# B7 - recipe fields
# --------------------------------------------------------------------------- #


def test_b7_recipe_fields_present(monkeypatch):
    track_tools(monkeypatch)
    monkeypatch.setattr(agent_module, "_tavily_search", lambda ingredients: [])
    install_llm(monkeypatch, {
        "QueryValidation": a_valid_query(),
        "QueryAnalysis": QueryAnalysis(ingredients=["chickpeas"], dietary_restrictions=[], preferences=[]),
        "Recipe": a_recipe(),
    })
    result = agent("A dish with chickpeas")
    recipe = result["recipe"]
    for field in ("title", "ingredients", "equipment", "instructions", "storage"):
        assert recipe.get(field)
    out = result["output"]
    assert "Title:" in out and "Ingredients:" in out
    assert "Equipment:" in out and "Instructions:" in out and "Storage:" in out


# --------------------------------------------------------------------------- #
# Tool interface & failure-prefix contract
# --------------------------------------------------------------------------- #


def test_search_recipes_returns_erro_prefix_on_failure(monkeypatch):
    monkeypatch.setattr(
        agent_module, "_tavily_search",
        lambda ingredients: (_ for _ in ()).throw(RuntimeError("missing TAVILY_API_KEY")),
    )
    out = search_recipes.invoke({"ingredients": ["eggs"]})
    assert isinstance(out, str) and out.lower().startswith("erro:")


def test_retrieve_pantry_returns_error_prefix_on_failure(monkeypatch):
    monkeypatch.setattr(
        agent_module, "_notion_pantry",
        lambda: (_ for _ in ()).throw(RuntimeError("missing NOTION_API_KEY")),
    )
    out = retrieve_pantry.invoke({})
    assert isinstance(out, str) and out.startswith("Error:")


def test_retrieve_pantry_empty_list(monkeypatch):
    monkeypatch.setattr(agent_module, "_notion_pantry", lambda: [])
    assert retrieve_pantry.invoke({}) == []


def test_search_recipes_empty_list(monkeypatch):
    monkeypatch.setattr(agent_module, "_tavily_search", lambda ingredients: [])
    assert search_recipes.invoke({"ingredients": ["x"]}) == []


def test_format_recipe_renders_all_fields():
    text = format_recipe(a_recipe().model_dump())
    assert "Title: Chickpea Spinach Stew" in text
    assert "Storage: Keep refrigerated up to 3 days." in text


# --------------------------------------------------------------------------- #
# Optional integration test
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY", "").strip()
    or os.getenv("OPENAI_API_KEY") == "test-openai-key",
    reason="real OPENAI_API_KEY required",
)
def test_integration_live():  # pragma: no cover
    result = agent("A vegan dinner with chickpeas and spinach")
    assert result["output"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
