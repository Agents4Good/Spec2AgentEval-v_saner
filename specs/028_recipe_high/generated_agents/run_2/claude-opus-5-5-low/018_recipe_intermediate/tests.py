"""Offline tests: LLM and HTTP calls are mocked."""
import os
from unittest.mock import MagicMock, patch

import pytest
import requests

import agent as A

os.environ.setdefault("OPENAI_API_KEY", "test")


def fake_llm(responses):
    """Patch _llm so each with_structured_output(schema).invoke returns responses[schema]."""
    def make():
        llm = MagicMock()
        def wso(schema):
            runner = MagicMock()
            r = responses[schema]
            runner.invoke.side_effect = r if isinstance(r, Exception) else (lambda *_a, **_k: r)
            return runner
        llm.with_structured_output.side_effect = wso
        return llm
    return patch.object(A, "_llm", side_effect=make)


RECIPE = A.Recipe(
    title="Tomato Rice", ingredients=[A.RecipeIngredient(name="rice", quantity="1 cup")],
    equipment=["pot"], instructions=["Cook rice."], storage="Fridge up to 3 days.",
)
VALID = A.Validation(is_valid=True, reason="ok")



def test_b1_empty_query():
    assert A.agent("   ").startswith("Error:")


def test_b1_invalid_query_no_tools():
    with fake_llm({A.Validation: A.Validation(is_valid=False, reason="unrelated")}), \
         patch.object(A.requests, "get") as g, patch.object(A.requests, "post") as p:
        out = A.agent("Who won the world cup?")
    assert out.startswith("Error:")
    g.assert_not_called(); p.assert_not_called()


def test_b3_b7_search_path():
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"results": [{"title": "T", "url": "u", "content": "c"}] * 7}
    with fake_llm({A.Validation: VALID,
                   A.Identification: A.Identification(ingredients=["rice", "tomato"]),
                   A.Recipe: RECIPE}), \
         patch.dict(os.environ, {"TAVILY_API_KEY": "k"}), \
         patch.object(A.requests, "post", return_value=resp) as p, \
         patch.object(A.requests, "get") as g:
        out = A.agent("Recipe with rice and tomato")
    assert p.call_count == 1 and "rice, tomato" in p.call_args.kwargs["json"]["query"]
    g.assert_not_called()
    for f in ("Title:", "Ingredients:", "Equipment:", "Instructions:", "Storage:"):
        assert f in out


def test_b6_search_failure():
    with fake_llm({A.Validation: VALID, A.Identification: A.Identification(ingredients=["rice"])}), \
         patch.dict(os.environ, {"TAVILY_API_KEY": ""}):
        out = A.agent("Recipe with rice")
    assert out.startswith("Error:")


def test_b6_search_timeout():
    with fake_llm({A.Validation: VALID, A.Identification: A.Identification(ingredients=["rice"])}), \
         patch.dict(os.environ, {"TAVILY_API_KEY": "k"}), \
         patch.object(A.requests, "post", side_effect=requests.Timeout()):
        out = A.agent("Recipe with rice")
    assert out.startswith("Error:")


def _notion(items):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"has_more": False, "results": [
        {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"plain_text": i}]}} for i in items]}
    return resp


def test_b4_b7_pantry_path():
    with fake_llm({A.Validation: VALID, A.Identification: A.Identification(ingredients=[]),
                   A.PantryFilter: A.PantryFilter(compatible_ingredients=["rice"]), A.Recipe: RECIPE}), \
         patch.dict(os.environ, {"NOTION_API_KEY": "k", "NOTION_PAGE_ID": "p"}), \
         patch.object(A.requests, "get", return_value=_notion(["rice", "milk"])) as g, \
         patch.object(A.requests, "post") as p:
        out = A.agent("Suggest a vegan dinner")
    assert g.call_count == 1; p.assert_not_called()
    assert out.startswith("Title:")


def test_b5_empty_pantry():
    with fake_llm({A.Validation: VALID, A.Identification: A.Identification(ingredients=[])}), \
         patch.dict(os.environ, {"NOTION_API_KEY": "k", "NOTION_PAGE_ID": "p"}), \
         patch.object(A.requests, "get", return_value=_notion([])):
        assert A.agent("Make me dinner").startswith("Error:")


def test_b5_no_compatible():
    with fake_llm({A.Validation: VALID, A.Identification: A.Identification(ingredients=[]),
                   A.PantryFilter: A.PantryFilter(compatible_ingredients=[])}), \
         patch.dict(os.environ, {"NOTION_API_KEY": "k", "NOTION_PAGE_ID": "p"}), \
         patch.object(A.requests, "get", return_value=_notion(["milk"])):
        assert A.agent("Vegan dinner please").startswith("Error:")


def test_b6_pantry_missing_credentials():
    with fake_llm({A.Validation: VALID, A.Identification: A.Identification(ingredients=[])}), \
         patch.dict(os.environ, {"NOTION_API_KEY": "", "NOTION_PAGE_ID": ""}):
        assert A.agent("Make me dinner").startswith("Error:")


def test_b6_model_failure():
    with fake_llm({A.Validation: RuntimeError("boom")}):
        assert A.agent("Recipe with rice").startswith("Error:")


def test_tools_are_langchain_tools():
    from langchain_core.tools import BaseTool
    assert isinstance(A.retrieve_pantry, BaseTool) and isinstance(A.search_recipes, BaseTool)


@pytest.mark.skipif(not os.getenv("RUN_LIVE"), reason="live test")
def test_live():
    out = A.agent("I have chicken and rice, I don't have onions. Quick dinner?")
    assert out.startswith("Title:") or out.startswith("Error:")
