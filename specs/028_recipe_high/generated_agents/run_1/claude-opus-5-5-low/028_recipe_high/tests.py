"""Offline tests for the recipe agent (LLM and HTTP are mocked)."""

import json
from unittest.mock import MagicMock, patch

import pytest

import agent as A


def fake_llm(validation=None, ingredients=None, recipe=None):
    """Return a factory for a fake ChatOpenAI whose structured output depends on schema."""
    answers = {A.Validation: validation, A.Ingredients: ingredients, A.Recipe: recipe}

    def factory():
        llm = MagicMock()
        def wso(schema):
            m = MagicMock()
            ans = answers[schema]
            if isinstance(ans, Exception):
                m.invoke.side_effect = ans
            else:
                m.invoke.return_value = ans
            return m
        llm.with_structured_output.side_effect = wso
        return llm
    return factory


VALID = A.Validation(valid=True, reason="")
RECIPE = A.Recipe(
    title="Tomato Rice", ingredients=[A.IngredientItem(name="rice", quantity="1 cup")],
    equipment=["pot"], instructions=["Cook rice."], storage="Fridge up to 3 days.",
)


@pytest.fixture
def tools():
    calls = []
    pantry = MagicMock(); pantry.invoke.side_effect = lambda a: calls.append("pantry") or json.dumps(["rice", "tomato"])
    search = MagicMock(); search.invoke.side_effect = lambda a: calls.append(("search", a)) or json.dumps(
        [{"title": "t", "url": "u", "content": "c"}])
    with patch.object(A, "retrieve_pantry", pantry), patch.object(A, "search_recipes", search):
        yield calls, pantry, search


def test_b1_invalid_query(tools):
    calls, *_ = tools
    with patch.object(A, "_llm", fake_llm(validation=A.Validation(valid=False, reason="off-topic"))):
        out = A.agent("Who won the election?")
    assert out.startswith("Error:") and calls == []


def test_b1_empty_query(tools):
    calls, *_ = tools
    assert A.agent("   ").startswith("Error:") and calls == []


def test_full_flow(tools):
    calls, *_ = tools
    ing = A.Ingredients(restrictions=[], ingredients=["rice", "tomato"])
    with patch.object(A, "_llm", fake_llm(VALID, ing, RECIPE)):
        out = A.agent("Make me dinner with rice")
    assert calls[0] == "pantry" and calls.count("pantry") == 1
    assert calls[1] == ("search", {"ingredients": ["rice", "tomato"]}) and len(calls) == 2
    for field in ("Title:", "Ingredients:", "Equipment:", "Instructions:", "Storage:"):
        assert field in out


def test_b5_no_ingredients(tools):
    calls, *_ = tools
    ing = A.Ingredients(restrictions=["vegan"], ingredients=[])
    with patch.object(A, "_llm", fake_llm(VALID, ing, RECIPE)):
        out = A.agent("Vegan recipe please")
    assert out.startswith("Error:") and calls == ["pantry"]


def test_b6_pantry_error(tools):
    calls, pantry, _ = tools
    pantry.invoke.side_effect = lambda a: "Error: timeout"
    with patch.object(A, "_llm", fake_llm(VALID)):
        assert A.agent("A pasta recipe").startswith("Error:")


def test_b6_search_error(tools):
    _, _, search = tools
    search.invoke.side_effect = lambda a: "Error: missing credentials"
    ing = A.Ingredients(restrictions=[], ingredients=["rice"])
    with patch.object(A, "_llm", fake_llm(VALID, ing, RECIPE)):
        assert A.agent("Rice recipe").startswith("Error:")


def test_b6_model_exception(tools):
    with patch.object(A, "_llm", fake_llm(validation=RuntimeError("boom"))):
        assert A.agent("Rice recipe").startswith("Error:")


def test_retrieve_pantry_missing_creds(monkeypatch):
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    assert A.retrieve_pantry.invoke({}).startswith("Error:")


def test_retrieve_pantry_parses(monkeypatch):
    monkeypatch.setenv("NOTION_API_KEY", "k"); monkeypatch.setenv("NOTION_PAGE_ID", "p")
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"results": [
        {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"plain_text": "Eggs"}]}},
        {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "milk, flour"}]}},
    ], "has_more": False}
    with patch.object(A.requests, "get", return_value=resp):
        assert json.loads(A.retrieve_pantry.invoke({})) == ["Eggs", "milk", "flour"]


def test_search_recipes_missing_creds(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert A.search_recipes.invoke({"ingredients": ["rice"]}).startswith("Error:")


def test_search_recipes_limit(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"results": [{"title": str(i), "url": "u", "content": "c"} for i in range(8)]}
    with patch.object(A.requests, "post", return_value=resp):
        assert len(json.loads(A.search_recipes.invoke({"ingredients": ["rice"]}))) == 5
