"""Unit tests for the recipe agent. LLM and HTTP calls are mocked."""
import os
from unittest.mock import MagicMock, patch

import pytest
import requests

import agent as A

os.environ.setdefault("OPENAI_API_KEY", "test")

RECIPE = A.Recipe(
    title="Tomato Omelette",
    ingredients=[A.IngredientLine(name="tomato", quantity="2"), A.IngredientLine(name="egg", quantity="3")],
    equipment=["pan"], instructions=["Beat eggs.", "Cook."], storage="Fridge up to 1 day.",
)


def fake_llm(*outputs):
    seq = list(outputs)
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.side_effect = lambda *_a, **_k: seq.pop(0)
    return llm


def resp(status=200, data=None):
    r = MagicMock(status_code=status, text="x")
    r.json.return_value = data or {}
    return r


# ---------------- Tools ----------------
def test_pantry_missing_credentials(monkeypatch):
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    assert A.retrieve_pantry.invoke({}).startswith("Error:")


def test_pantry_parses_blocks(monkeypatch):
    monkeypatch.setenv("NOTION_API_KEY", "k"); monkeypatch.setenv("NOTION_PAGE_ID", "p")
    data = {"results": [
        {"type": "heading_1", "heading_1": {"rich_text": [{"plain_text": "Pantry"}]}},
        {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"plain_text": "rice"}]}},
        {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "beans, carrot"}]}},
    ], "has_more": False}
    with patch.object(A.requests, "get", return_value=resp(data=data)):
        assert A.retrieve_pantry.invoke({}) == ["rice", "beans", "carrot"]


def test_pantry_empty_and_timeout(monkeypatch):
    monkeypatch.setenv("NOTION_API_KEY", "k"); monkeypatch.setenv("NOTION_PAGE_ID", "p")
    with patch.object(A.requests, "get", return_value=resp(data={"results": []})):
        assert A.retrieve_pantry.invoke({}) == []
    with patch.object(A.requests, "get", side_effect=requests.Timeout()):
        assert A.retrieve_pantry.invoke({}).startswith("Error:")
    with patch.object(A.requests, "get", return_value=resp(404)):
        assert A.retrieve_pantry.invoke({}).startswith("Error:")


def test_search_recipes(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert A.search_recipes.invoke({"ingredients": ["egg"]}).startswith("erro:")
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    results = [{"title": f"t{i}", "url": f"u{i}", "content": "c"} for i in range(7)]
    with patch.object(A.requests, "post", return_value=resp(data={"results": results})):
        out = A.search_recipes.invoke({"ingredients": ["egg"]})
    assert len(out) == 5 and set(out[0]) == {"title", "url", "content"}
    with patch.object(A.requests, "post", side_effect=requests.Timeout()):
        assert A.search_recipes.invoke({"ingredients": ["egg"]}).startswith("erro:")


# ---------------- Behaviors ----------------
@pytest.fixture
def tools():
    with patch.object(A, "search_recipes") as s, patch.object(A, "retrieve_pantry") as p:
        yield s, p


def test_b1_empty_query(tools):
    s, p = tools
    assert A.agent("   ").startswith("Error:")
    s.invoke.assert_not_called(); p.invoke.assert_not_called()


def test_b1_unrelated_query(tools):
    s, p = tools
    with patch.object(A, "_llm", return_value=fake_llm(A.Analysis(is_valid=False, reason="not about food"))):
        assert A.agent("Who won the world cup?").startswith("Error:")
    s.invoke.assert_not_called(); p.invoke.assert_not_called()


def test_b3_b7_query_ingredients(tools):
    s, p = tools
    s.invoke.return_value = [{"title": "t", "url": "u", "content": "c"}]
    an = A.Analysis(is_valid=True, reason="ok", ingredients=["tomato", "egg"])
    with patch.object(A, "_llm", return_value=fake_llm(an, RECIPE)):
        out = A.agent("I have tomatoes and eggs")
    s.invoke.assert_called_once_with({"ingredients": ["tomato", "egg"]})
    p.invoke.assert_not_called()
    for f in ("Title:", "Ingredients:", "Equipment:", "Instructions:", "Storage:"):
        assert f in out


def test_b4_b7_pantry(tools):
    s, p = tools
    p.invoke.return_value = ["rice", "chicken"]
    an = A.Analysis(is_valid=True, reason="ok", dietary_restrictions=["vegetarian"], ingredients=[])
    r = RECIPE.model_copy(update={"ingredients": [A.IngredientLine(name="rice", quantity="1 cup")]})
    with patch.object(A, "_llm", return_value=fake_llm(an, A.PantryFilter(compatible_ingredients=["rice"]), r)):
        out = A.agent("Suggest a vegetarian dinner")
    p.invoke.assert_called_once(); s.invoke.assert_not_called()
    assert "rice" in out and "chicken" not in out


def test_b5_no_compatible_pantry(tools):
    s, p = tools
    p.invoke.return_value = ["chicken"]
    an = A.Analysis(is_valid=True, reason="ok", dietary_restrictions=["vegan"], ingredients=[])
    with patch.object(A, "_llm", return_value=fake_llm(an, A.PantryFilter(compatible_ingredients=[]))):
        assert A.agent("vegan lunch please").startswith("Error:")


def test_b6_tool_errors(tools):
    s, p = tools
    s.invoke.return_value = "erro: timeout"
    an = A.Analysis(is_valid=True, reason="ok", ingredients=["egg"])
    with patch.object(A, "_llm", return_value=fake_llm(an)):
        assert A.agent("I have eggs").startswith("Error:")
    p.invoke.return_value = "Error: missing credentials"
    an = A.Analysis(is_valid=True, reason="ok", ingredients=[])
    with patch.object(A, "_llm", return_value=fake_llm(an)):
        assert A.agent("something for dinner").startswith("Error:")


def test_b6_model_exception():
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("boom")
    with patch.object(A, "_llm", return_value=llm):
        assert A.agent("I have eggs").startswith("Error:")
