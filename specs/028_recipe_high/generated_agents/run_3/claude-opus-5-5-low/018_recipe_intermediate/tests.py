"""Offline unit tests (LLM and HTTP mocked). Run: pytest tests.py"""
from unittest.mock import MagicMock, patch

import pytest
import requests

import agent as A


class FakeLLM:
    """Returns a predefined object per structured-output schema."""

    def __init__(self, outputs):
        self.outputs = outputs

    def with_structured_output(self, schema):
        out = self.outputs[schema.__name__]
        m = MagicMock()
        if isinstance(out, Exception):
            m.invoke.side_effect = out
        else:
            m.invoke.return_value = out
        return m


RECIPE = A.Recipe(title="Tomato Rice", ingredients=["200 g rice", "2 tomatoes"],
                  equipment=["pot"], instructions=["Cook rice", "Add tomatoes"], storage="Fridge up to 3 days")


def run(outputs, query="I have rice and tomatoes"):
    with patch.object(A, "_llm", lambda: FakeLLM(outputs)):
        return A.agent(query)


@pytest.fixture
def tools():
    with patch.object(A.search_recipes, "func") as s, patch.object(A.retrieve_pantry, "func") as p:
        yield s, p


# B1
def test_empty_query(tools):
    s, p = tools
    assert run({}, "   ").startswith("Error:")
    s.assert_not_called(); p.assert_not_called()


def test_invalid_query(tools):
    s, p = tools
    out = run({"Validation": A.Validation(is_valid=False, reason="unrelated")}, "who won the game?")
    assert out.startswith("Error:")
    s.assert_not_called(); p.assert_not_called()


# B2/B3/B7
def test_search_path(tools):
    s, p = tools
    s.return_value = [{"title": "t", "url": "u", "content": "c"}]
    out = run({"Validation": A.Validation(is_valid=True, reason=""),
               "IngredientExtraction": A.IngredientExtraction(ingredients=["rice", "tomato"]),
               "Recipe": RECIPE})
    assert s.call_count == 1 and s.call_args.kwargs["ingredients"] == ["rice", "tomato"]
    p.assert_not_called()
    for f in ("Title:", "Ingredients:", "Equipment:", "Instructions:", "Storage:"):
        assert f in out


# B4/B7
def test_pantry_path(tools):
    s, p = tools
    p.return_value = ["rice", "cheese"]
    out = run({"Validation": A.Validation(is_valid=True, reason=""),
               "IngredientExtraction": A.IngredientExtraction(dietary_restrictions=["vegan"]),
               "PantryFilter": A.PantryFilter(compatible_ingredients=["rice"]),
               "Recipe": RECIPE}, "vegan dinner idea")
    assert p.call_count == 1
    s.assert_not_called()
    assert out.startswith("Title:")


# B5
@pytest.mark.parametrize("pantry,compat", [([], []), (["cheese"], [])])
def test_lack_of_ingredients(tools, pantry, compat):
    s, p = tools
    p.return_value = pantry
    out = run({"Validation": A.Validation(is_valid=True, reason=""),
               "IngredientExtraction": A.IngredientExtraction(),
               "PantryFilter": A.PantryFilter(compatible_ingredients=compat)}, "vegan dinner")
    assert out.startswith("Error:")


# B6
def test_search_error(tools):
    s, p = tools
    s.return_value = "erro: timeout"
    out = run({"Validation": A.Validation(is_valid=True, reason=""),
               "IngredientExtraction": A.IngredientExtraction(ingredients=["rice"])})
    assert out.startswith("Error:") and "Title:" not in out


def test_pantry_error(tools):
    s, p = tools
    p.return_value = "Error: missing credentials"
    out = run({"Validation": A.Validation(is_valid=True, reason=""),
               "IngredientExtraction": A.IngredientExtraction()}, "dinner idea")
    assert out.startswith("Error:")


def test_model_failure():
    out = run({"Validation": RuntimeError("boom")})
    assert out.startswith("Error:")


# Tools
def test_tools_missing_credentials(monkeypatch):
    for k in ("NOTION_API_KEY", "NOTION_PAGE_ID", "TAVILY_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert A.retrieve_pantry.invoke({}).startswith("Error:")
    assert A.search_recipes.invoke({"ingredients": ["rice"]}).startswith("erro:")


def test_tools_timeout(monkeypatch):
    monkeypatch.setenv("NOTION_API_KEY", "x"); monkeypatch.setenv("NOTION_PAGE_ID", "y")
    monkeypatch.setenv("TAVILY_API_KEY", "z")
    with patch("requests.get", side_effect=requests.Timeout), patch("requests.post", side_effect=requests.Timeout):
        assert A.retrieve_pantry.invoke({}).startswith("Error:")
        assert A.search_recipes.invoke({"ingredients": ["rice"]}).startswith("erro:")


def test_pantry_parsing(monkeypatch):
    monkeypatch.setenv("NOTION_API_KEY", "x"); monkeypatch.setenv("NOTION_PAGE_ID", "y")
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"has_more": False, "results": [
        {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"plain_text": "rice"}]}},
        {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "beans, onion"}]}},
    ]}
    with patch("requests.get", return_value=resp):
        assert A.retrieve_pantry.invoke({}) == ["rice", "beans", "onion"]


def test_search_limits_results(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "z")
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"results": [{"title": str(i), "url": "u", "content": "c"} for i in range(8)]}
    with patch("requests.post", return_value=resp):
        assert len(A.search_recipes.invoke({"ingredients": ["rice"]})) == 5
