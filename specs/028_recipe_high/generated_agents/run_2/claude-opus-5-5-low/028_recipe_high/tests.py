"""Offline tests for the recipe agent (LLM and HTTP are mocked)."""

from unittest.mock import MagicMock, patch

import pytest
import requests

import agent as A


def _resp(status=200, payload=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload or {}
    r.text = str(payload)
    return r


class FakeLLM:
    """Returns a queued structured output per with_structured_output call."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def with_structured_output(self, schema):
        llm = self

        class _R:
            def invoke(self, messages):
                llm.calls.append((schema, messages))
                out = llm.outputs.pop(0)
                if isinstance(out, Exception):
                    raise out
                return out

        return _R()


RECIPE = A.Recipe(
    title="Tomato Rice",
    ingredients=["1 cup rice", "2 tomatoes"],
    equipment=["pot"],
    instructions=["Cook rice", "Add tomatoes"],
    storage="Refrigerate up to 3 days.",
)


@pytest.fixture
def env(monkeypatch):
    for k in ("NOTION_API_KEY", "NOTION_PAGE_ID", "TAVILY_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(k, "x")


# ---------------- Tools ----------------

def test_retrieve_pantry_missing_credentials(monkeypatch):
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    assert retrieve_out().startswith("Error:")


def retrieve_out():
    return A.retrieve_pantry.invoke({})


def test_retrieve_pantry_parses_blocks(env):
    payload = {
        "results": [
            {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"plain_text": "rice"}]}},
            {"type": "to_do", "to_do": {"rich_text": [{"plain_text": "tomato, onion"}]}},
            {"type": "divider", "divider": {}},
        ],
        "has_more": False,
    }
    with patch.object(A.requests, "get", return_value=_resp(200, payload)):
        assert retrieve_out() == ["rice", "tomato", "onion"]


def test_retrieve_pantry_empty(env):
    with patch.object(A.requests, "get", return_value=_resp(200, {"results": [], "has_more": False})):
        assert retrieve_out() == []


def test_retrieve_pantry_timeout(env):
    with patch.object(A.requests, "get", side_effect=requests.Timeout()):
        assert retrieve_out().startswith("Error:")


def test_retrieve_pantry_inaccessible(env):
    with patch.object(A.requests, "get", return_value=_resp(404, {"message": "not found"})):
        assert retrieve_out().startswith("Error:")


def test_search_recipes_limits_to_five(env):
    results = [{"title": f"t{i}", "url": f"u{i}", "content": f"c{i}"} for i in range(8)]
    with patch.object(A.requests, "post", return_value=_resp(200, {"results": results})):
        out = A.search_recipes.invoke({"ingredients": ["rice"]})
    assert len(out) == 5 and set(out[0]) == {"title", "url", "content"}


def test_search_recipes_missing_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert A.search_recipes.invoke({"ingredients": ["rice"]}).startswith("Error:")


def test_search_recipes_timeout(env):
    with patch.object(A.requests, "post", side_effect=requests.Timeout()):
        assert A.search_recipes.invoke({"ingredients": ["rice"]}).startswith("Error:")


# ---------------- Agent flow ----------------

def run(query, llm_outputs, pantry=None, search=None):
    fake = FakeLLM(llm_outputs)
    pantry_mock = MagicMock(return_value=pantry if pantry is not None else ["rice"])
    search_mock = MagicMock(return_value=search if search is not None else [])
    with patch.object(A, "_llm", return_value=fake), \
         patch.object(A, "retrieve_pantry", MagicMock(invoke=pantry_mock)), \
         patch.object(A, "search_recipes", MagicMock(invoke=search_mock)):
        out = A.agent(query)
    return out, pantry_mock, search_mock


def test_b1_empty_query():
    out, p, s = run("   ", [])
    assert out.startswith("Error:") and not p.called and not s.called


def test_b1_invalid_query():
    out, p, s = run("What is the capital of France?", [A.QueryValidation(is_valid=False, reason="unrelated")])
    assert out.startswith("Error:") and not p.called and not s.called


def test_b2_pantry_error_b6():
    out, p, s = run("dinner", [A.QueryValidation(is_valid=True, reason="ok")], pantry="Error: timeout")
    assert out.startswith("Error:") and p.call_count == 1 and not s.called


def test_b5_no_compatible_ingredients():
    out, p, s = run(
        "vegan dinner",
        [A.QueryValidation(is_valid=True, reason="ok"), A.IngredientIdentification(ingredients=[])],
        pantry=["chicken"],
    )
    assert out.startswith("Error:") and p.call_count == 1 and not s.called


def test_b4_search_error_b6():
    out, p, s = run(
        "dinner with rice",
        [A.QueryValidation(is_valid=True, reason="ok"), A.IngredientIdentification(ingredients=["rice"])],
        search="Error: timeout",
    )
    assert out.startswith("Error:") and s.call_count == 1


def test_b7_full_recipe():
    out, p, s = run(
        "vegan dinner with tomatoes",
        [
            A.QueryValidation(is_valid=True, reason="ok"),
            A.IngredientIdentification(dietary_restrictions=["vegan"], ingredients=["rice", "tomato"]),
            RECIPE,
        ],
        search=[{"title": "x", "url": "y", "content": "z"}],
    )
    assert p.call_count == 1 and s.call_count == 1
    assert s.call_args[0][0] == {"ingredients": ["rice", "tomato"]}
    for field in ("Title:", "Ingredients:", "Equipment:", "Instructions:", "Storage:"):
        assert field in out


def test_b6_model_exception():
    out, _, _ = run("dinner", [RuntimeError("model down")])
    assert out.startswith("Error:")
