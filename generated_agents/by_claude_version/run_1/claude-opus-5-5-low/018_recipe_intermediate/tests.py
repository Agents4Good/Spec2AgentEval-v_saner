"""Offline tests for the chef agent (LLM and HTTP are mocked)."""
from unittest.mock import MagicMock, patch

import pytest
import requests

import agent as ag


class FakeLLM:
    """Returns pre-built structured outputs keyed by schema class."""

    def __init__(self, outputs):
        self.outputs = outputs

    def with_structured_output(self, schema):
        m = MagicMock()
        out = self.outputs.get(schema)
        if isinstance(out, Exception):
            m.invoke.side_effect = out
        else:
            m.invoke.return_value = out
        return m


RECIPE = ag.Recipe(title="Tomato Rice", ingredients=["200 g rice", "2 tomatoes"], equipment=["pot"],
                   instructions=["Cook rice", "Add tomato"], storage="Fridge up to 3 days")


def run(query, outputs, search=None, pantry=None):
    with patch.object(ag, "_llm", return_value=FakeLLM(outputs)):
        s = MagicMock(return_value=search if search is not None else [])
        p = MagicMock(return_value=pantry if pantry is not None else [])
        with patch.object(ag.search_recipes, "func", s), patch.object(ag.retrieve_pantry, "func", p):
            return ag.agent(query), s, p


def test_empty_query():
    state, s, p = run("   ", {})
    assert state["error"].startswith("Error:") and state["recipe"] is None
    assert state["tool_calls"] == []


def test_invalid_query():
    state, s, p = run("who won the match?", {ag.Validation: ag.Validation(is_valid=False, reason="unrelated")})
    assert state["is_valid"] is False and state["error"].startswith("Error:")
    s.assert_not_called(); p.assert_not_called()


def test_ingredients_path_search_once():
    outs = {ag.Validation: ag.Validation(is_valid=True, reason=""),
            ag.Ingredients: ag.Ingredients(ingredients=["rice", "tomato"]), ag.Recipe: RECIPE}
    state, s, p = run("dinner with rice and tomato", outs, search=[{"title": "t", "url": "u", "content": "c"}])
    assert state["status"] == "success" and state["recipe"]["title"]
    assert state["tool_calls"] == ["search_recipes"]
    s.assert_called_once(); p.assert_not_called()


def test_search_error_b6():
    outs = {ag.Validation: ag.Validation(is_valid=True, reason=""),
            ag.Ingredients: ag.Ingredients(ingredients=["rice"])}
    state, s, p = run("rice dish", outs, search="erro: timeout")
    assert state["error"].startswith("Error:") and state["recipe"] is None


def test_pantry_path():
    outs = {ag.Validation: ag.Validation(is_valid=True, reason=""), ag.Ingredients: ag.Ingredients(),
            ag.FilteredIngredients: ag.FilteredIngredients(ingredients=["Rice"]), ag.Recipe: RECIPE}
    state, s, p = run("make me a vegan dinner", outs, pantry=["Rice", "Eggs"])
    assert state["pantry_ingredients"] == ["Rice"] and state["status"] == "success"
    assert state["tool_calls"] == ["retrieve_pantry"]
    s.assert_not_called(); p.assert_called_once()


def test_empty_pantry_b5():
    outs = {ag.Validation: ag.Validation(is_valid=True, reason=""), ag.Ingredients: ag.Ingredients()}
    state, _, _ = run("surprise me", outs, pantry=[])
    assert state["error"].startswith("Error:") and state["recipe"] is None


def test_pantry_error_b6():
    outs = {ag.Validation: ag.Validation(is_valid=True, reason=""), ag.Ingredients: ag.Ingredients()}
    state, _, _ = run("surprise me", outs, pantry="Error: inaccessible")
    assert state["error"].startswith("Error:")


def test_model_failure_b6():
    state, _, _ = run("pasta", {ag.Validation: RuntimeError("boom")})
    assert state["error"].startswith("Error:") and state["recipe"] is None


def test_tools_missing_credentials(monkeypatch):
    for k in ("NOTION_API_KEY", "NOTION_PAGE_ID", "TAVILY_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert ag.retrieve_pantry.invoke({}).startswith("Error:")
    assert ag.search_recipes.invoke({"ingredients": ["rice"]}).startswith("erro:")


def test_search_tool_timeout(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "x")
    with patch.object(ag.requests, "post", side_effect=requests.Timeout()):
        assert ag.search_recipes.invoke({"ingredients": ["rice"]}).startswith("erro:")


def test_pantry_tool_parses(monkeypatch):
    monkeypatch.setenv("NOTION_API_KEY", "x"); monkeypatch.setenv("NOTION_PAGE_ID", "p")
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"results": [{"type": "bulleted_list_item",
                                           "bulleted_list_item": {"rich_text": [{"plain_text": "Rice"}]}}],
                              "has_more": False}
    with patch.object(ag.requests, "get", return_value=resp):
        assert ag.retrieve_pantry.invoke({}) == ["Rice"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
