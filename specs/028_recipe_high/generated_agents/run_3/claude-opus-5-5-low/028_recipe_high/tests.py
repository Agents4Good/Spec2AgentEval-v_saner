"""Offline tests: LLM and HTTP are mocked; tools are still executed via .invoke."""
import json
from unittest.mock import MagicMock, patch

import pytest
import requests

import agent as A


def fake_llm(validation=True, ingredients=("tomato",), restrictions=(), recipe=None):
    def structured(schema):
        m = MagicMock()
        if schema is A.Validation:
            m.invoke.return_value = A.Validation(is_valid=validation, reason="test reason")
        elif schema is A.Ingredients:
            m.invoke.return_value = A.Ingredients(restrictions=list(restrictions), ingredients=list(ingredients))
        else:
            m.invoke.return_value = recipe or A.Recipe(
                title="Tomato Salad", ingredients=["2 tomatoes"], equipment=["knife"],
                instructions=["Slice", "Serve"], storage="Fridge up to 1 day.")
        return m
    llm = MagicMock()
    llm.with_structured_output.side_effect = structured
    return llm


def resp(status=200, payload=None):
    r = MagicMock(status_code=status, text="x")
    r.json.return_value = payload or {}
    return r


NOTION_OK = resp(payload={"results": [
    {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"plain_text": "tomato"}]}},
    {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "rice, cheese"}]}},
], "has_more": False})
TAVILY_OK = resp(payload={"results": [{"title": "T", "url": "http://u", "content": "c"}] * 7})


@pytest.fixture
def env(monkeypatch):
    for k in ("NOTION_API_KEY", "NOTION_PAGE_ID", "TAVILY_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(k, "x")


def test_retrieve_pantry_ok(env):
    with patch.object(A.requests, "get", return_value=NOTION_OK):
        assert json.loads(A.retrieve_pantry.invoke({})) == ["tomato", "rice", "cheese"]


def test_retrieve_pantry_empty(env):
    with patch.object(A.requests, "get", return_value=resp(payload={"results": []})):
        assert json.loads(A.retrieve_pantry.invoke({})) == []


def test_retrieve_pantry_errors(env, monkeypatch):
    with patch.object(A.requests, "get", side_effect=requests.Timeout()):
        assert A.retrieve_pantry.invoke({}).startswith("Error:")
    with patch.object(A.requests, "get", return_value=resp(404)):
        assert A.retrieve_pantry.invoke({}).startswith("Error:")
    monkeypatch.delenv("NOTION_API_KEY")
    assert A.retrieve_pantry.invoke({}).startswith("Error:")


def test_search_recipes_limits_and_errors(env, monkeypatch):
    with patch.object(A.requests, "post", return_value=TAVILY_OK):
        out = json.loads(A.search_recipes.invoke({"ingredients": ["tomato"]}))
        assert len(out) == 5 and set(out[0]) == {"title", "url", "content"}
    with patch.object(A.requests, "post", side_effect=requests.Timeout()):
        assert A.search_recipes.invoke({"ingredients": ["tomato"]}).startswith("Error:")
    monkeypatch.delenv("TAVILY_API_KEY")
    assert A.search_recipes.invoke({"ingredients": ["tomato"]}).startswith("Error:")


def test_invalid_query_calls_no_tool(env):
    with patch.object(A, "_llm", return_value=fake_llm(validation=False)), \
         patch.object(A.requests, "get") as g, patch.object(A.requests, "post") as p:
        out = A.agent("tell me about football")
        assert out.startswith("Error:")
        g.assert_not_called(); p.assert_not_called()


def test_empty_query():
    assert A.agent("   ").startswith("Error:")


def test_happy_path(env):
    with patch.object(A, "_llm", return_value=fake_llm()), \
         patch.object(A.requests, "get", return_value=NOTION_OK) as g, \
         patch.object(A.requests, "post", return_value=TAVILY_OK) as p:
        out = A.agent("A vegan dinner with tomatoes")
    assert g.call_count == 1 and p.call_count == 1
    for f in ("Title:", "Ingredients:", "Equipment:", "Instructions:", "Storage:"):
        assert f in out


def test_no_compatible_ingredients(env):
    with patch.object(A, "_llm", return_value=fake_llm(ingredients=())), \
         patch.object(A.requests, "get", return_value=NOTION_OK), \
         patch.object(A.requests, "post") as p:
        out = A.agent("vegan recipe, I have no vegetables")
    assert out.startswith("Error:")
    p.assert_not_called()


def test_pantry_failure(env):
    with patch.object(A, "_llm", return_value=fake_llm()), \
         patch.object(A.requests, "get", return_value=resp(500)), \
         patch.object(A.requests, "post") as p:
        assert A.agent("pasta recipe").startswith("Error:")
    p.assert_not_called()


def test_search_failure(env):
    with patch.object(A, "_llm", return_value=fake_llm()), \
         patch.object(A.requests, "get", return_value=NOTION_OK), \
         patch.object(A.requests, "post", side_effect=requests.Timeout()):
        assert A.agent("pasta recipe").startswith("Error:")


def test_model_failure():
    llm = MagicMock()
    llm.with_structured_output.side_effect = RuntimeError("boom")
    with patch.object(A, "_llm", return_value=llm):
        assert A.agent("pasta recipe").startswith("Error:")
