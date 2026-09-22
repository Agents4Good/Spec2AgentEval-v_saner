"""
Tests for the Personalized Recipe Generator Agent.

The unit tests mock the LLM (``get_llm``) and the tool HTTP layers
(``_notion_pantry`` / ``_tavily_search``) so they run offline and
deterministically. Integration tests at the bottom run only when the real API
keys are present.

Run: ``pytest tests.py -v``
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List

import httpx
import pytest

import agent as agent_module
from agent import (
    ComplianceCheck,
    NutritionInfo,
    QueryAnalysis,
    QueryValidation,
    Recipe,
    agent,
    format_recipe,
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
        if callable(handler) and not isinstance(handler, list):
            return handler(messages)
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
        servings="2 porções",
        ingredients=["300 g de peito de frango", "200 g de brócolis", "1 colher de sopa de azeite", "sal a gosto"],
        equipment=["frigideira", "faca", "tábua de corte"],
        instructions=["Tempere o frango.", "Grelhe por 6 minutos de cada lado.", "Salteie o brócolis."],
        nutrition=NutritionInfo(
            calories="420 kcal",
            protein="45 g",
            carbohydrates="10 g",
            fat="20 g",
            health_benefits=["rico em proteínas", "fonte de fibras"],
        ),
        storage="Conserve em recipiente fechado na geladeira por até 3 dias.",
    )
    base.update(overrides)
    return Recipe(**base)


VALID = QueryValidation(is_food_related=True, is_offensive=False, reason="food")
OFFENSIVE = QueryValidation(is_food_related=True, is_offensive=True, reason="offensive")
NOT_FOOD = QueryValidation(is_food_related=False, is_offensive=False, reason="programming question")
COMPLIANT = ComplianceCheck(compliant=True, issues=[])


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    monkeypatch.setenv("NOTION_API_KEY", "test-notion-key")
    monkeypatch.setenv("NOTION_PANTRY_PAGE_ID", "test-page-id")


@pytest.fixture
def tools(monkeypatch):
    """Patches the HTTP layer of both tools and records which ones were called."""
    calls: Dict[str, List[Any]] = {"search_recipes": [], "retrieve_pantry": []}

    def fake_search(ingredients):
        calls["search_recipes"].append(list(ingredients))
        return [
            {"title": "Grilled chicken with broccoli", "url": "https://example.com/1", "content": "Season, grill, serve."},
        ]

    def fake_pantry():
        calls["retrieve_pantry"].append(())
        return ["ovos", "arroz", "tomate", "cebola"]

    monkeypatch.setattr(agent_module, "_tavily_search", fake_search)
    monkeypatch.setattr(agent_module, "_notion_pantry", fake_pantry)
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
    for key in ("query", "tool_called", "tool_result", "recipe", "output", "error"):
        assert key in state
    assert state["error"] is None


def test_default_llm_is_chatopenai_gpt4o(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    llm = agent_module.get_llm()
    assert type(llm).__name__ == "ChatOpenAI"
    assert llm.model_name == "gpt-4o"


# --------------------------------------------------------------------------- #
# BR-01: query validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("query", ["", "   ", None])
def test_empty_query_returns_error_without_calling_llm_or_tools(env, tools, monkeypatch, query):
    fake = use_llm(monkeypatch, {})
    state = agent(query)
    assert state["output"].startswith("Error: ")
    assert fake.calls == []
    assert tools["search_recipes"] == [] and tools["retrieve_pantry"] == []


def test_offensive_query_returns_error(env, tools, monkeypatch):
    use_llm(monkeypatch, {"QueryValidation": OFFENSIVE})
    state = agent("seu cozinheiro idiota, faz uma receita")
    assert state["output"].startswith("Error: ")
    assert state["is_valid"] is False
    assert state["tool_called"] is None
    assert tools["search_recipes"] == [] and tools["retrieve_pantry"] == []


def test_non_food_query_returns_error(env, tools, monkeypatch):
    use_llm(monkeypatch, {"QueryValidation": NOT_FOOD})
    state = agent("Como faço um loop em Python?")
    assert state["output"].startswith("Error: ")
    assert state["recipe"] is None
    assert tools["search_recipes"] == [] and tools["retrieve_pantry"] == []


def test_error_message_never_contains_offensive_words_from_query(env, tools, monkeypatch):
    use_llm(monkeypatch, {"QueryValidation": OFFENSIVE})
    state = agent("receita, seu imbecil")
    assert "imbecil" not in state["output"].lower()


# --------------------------------------------------------------------------- #
# BR-02 / BR-03 / BR-04: tool selection (exactly one tool)
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
# BR-05 / BR-06: tool result feeds recipe generation
# --------------------------------------------------------------------------- #


def test_tool_result_is_passed_to_recipe_generation(env, tools, monkeypatch):
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
    assert "search_recipes" in prompt


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
    assert "retrieve_pantry" in prompt


# --------------------------------------------------------------------------- #
# Expected output format
# --------------------------------------------------------------------------- #


def test_successful_output_has_required_sections(env, tools, monkeypatch):
    use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=["frango", "brócolis"]),
        "Recipe": make_recipe(),
    })
    out = agent("Receita com frango e brócolis")["output"]
    assert not out.startswith("Error:")
    for section in ("Título:", "Ingredientes:", "Equipamentos:", "Instruções:", "Nutrição:", "Armazenamento:"):
        assert section in out
    assert "Frango grelhado com brócolis" in out
    assert "300 g de peito de frango" in out
    assert "420 kcal" in out
    assert "Conserve em recipiente fechado" in out


def test_format_recipe_orders_sections():
    text = format_recipe(make_recipe().model_dump())
    positions = [text.index(s) for s in ("Título:", "Ingredientes:", "Instruções:", "Nutrição:", "Armazenamento:")]
    assert positions == sorted(positions)
    assert "1. Tempere o frango." in text


# --------------------------------------------------------------------------- #
# Exception rule: no identifiable ingredients
# --------------------------------------------------------------------------- #


def test_empty_pantry_and_no_query_ingredients_returns_error(env, monkeypatch):
    monkeypatch.setattr(agent_module, "_notion_pantry", lambda: [])
    fake = use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=[]),
        "Recipe": make_recipe(),
    })
    state = agent("Faz um almoço pra mim")
    assert state["output"].startswith("Error: ")
    assert state["tool_called"] == "retrieve_pantry"
    assert state["recipe"] is None
    assert all(name != "Recipe" for name, _ in fake.calls), "recipe generation must be skipped"


# --------------------------------------------------------------------------- #
# Constraint: dietary restrictions are always respected
# --------------------------------------------------------------------------- #


def test_restrictions_are_passed_and_verified(env, tools, monkeypatch):
    fake = use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=["grão-de-bico"], dietary_restrictions=["vegano", "sem glúten"]),
        "Recipe": make_recipe(title="Curry de grão-de-bico", ingredients=["400 g de grão-de-bico", "leite de coco"]),
        "ComplianceCheck": COMPLIANT,
    })
    state = agent("Receita vegana e sem glúten com grão-de-bico")
    assert state["error"] is None
    recipe_prompt = "\n".join(m.content for name, msgs in fake.calls if name == "Recipe" for m in msgs)
    assert "vegano" in recipe_prompt and "sem glúten" in recipe_prompt
    assert any(name == "ComplianceCheck" for name, _ in fake.calls)


def test_non_compliant_recipe_is_regenerated_once(env, tools, monkeypatch):
    fake = use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=["grão-de-bico"], dietary_restrictions=["vegano"]),
        "Recipe": [
            make_recipe(title="Curry com manteiga", ingredients=["grão-de-bico", "manteiga"]),
            make_recipe(title="Curry vegano", ingredients=["grão-de-bico", "azeite"]),
        ],
        "ComplianceCheck": [
            ComplianceCheck(compliant=False, issues=["manteiga não é vegana"]),
            COMPLIANT,
        ],
    })
    state = agent("Receita vegana com grão-de-bico")
    assert state["error"] is None
    assert state["recipe"]["title"] == "Curry vegano"
    assert state["compliance_attempts"] == 2
    second_prompt = "\n".join(m.content for name, msgs in fake.calls if name == "Recipe" for m in msgs)
    assert "manteiga não é vegana" in second_prompt


def test_persistently_non_compliant_recipe_returns_error(env, tools, monkeypatch):
    use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=["camarão"], dietary_restrictions=["alergia a frutos do mar"]),
        "Recipe": [make_recipe(title="Camarão 1"), make_recipe(title="Camarão 2")],
        "ComplianceCheck": [
            ComplianceCheck(compliant=False, issues=["contém camarão"]),
            ComplianceCheck(compliant=False, issues=["contém camarão"]),
        ],
    })
    state = agent("Receita com camarão, tenho alergia a frutos do mar")
    assert state["output"].startswith("Error: ")
    assert "camarão" in state["output"]


def test_no_restrictions_skips_compliance_check(env, tools, monkeypatch):
    fake = use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=["ovo"]),
        "Recipe": make_recipe(),
    })
    state = agent("Receita com ovo")
    assert state["error"] is None
    assert all(name != "ComplianceCheck" for name, _ in fake.calls)


# --------------------------------------------------------------------------- #
# BR-07: execution failures -> "Error: ..."
# --------------------------------------------------------------------------- #


def test_missing_openai_key_returns_error(monkeypatch, tools):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    fake = use_llm(monkeypatch, {})
    state = agent("Receita com frango")
    assert state["output"].startswith("Error: ")
    assert "OPENAI_API_KEY" in state["output"]
    assert fake.calls == []


def test_missing_tavily_key_returns_error(env, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY")
    use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=["frango"]),
    })
    state = agent("Receita com frango")
    assert state["output"].startswith("Error: ")
    assert "TAVILY_API_KEY" in state["output"]
    assert state["tool_called"] == "search_recipes"
    assert state["recipe"] is None


def test_missing_notion_config_returns_error(env, monkeypatch):
    monkeypatch.delenv("NOTION_PANTRY_PAGE_ID")
    use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=[]),
    })
    state = agent("O que cozinho hoje?")
    assert state["output"].startswith("Error: ")
    assert "NOTION_PANTRY_PAGE_ID" in state["output"]
    assert state["tool_called"] == "retrieve_pantry"


def test_tool_timeout_returns_error(env, monkeypatch):
    def timeout(_ingredients):
        raise httpx.ReadTimeout("read timed out")

    monkeypatch.setattr(agent_module, "_tavily_search", timeout)
    use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=["frango"]),
    })
    state = agent("Receita com frango")
    assert state["output"].startswith("Error: ")
    assert "timed out" in state["output"]


def test_tool_http_error_returns_error(env, monkeypatch):
    def http_error():
        request = httpx.Request("GET", "https://api.notion.com/v1/blocks/x/children")
        response = httpx.Response(401, request=request)
        raise httpx.HTTPStatusError("unauthorized", request=request, response=response)

    monkeypatch.setattr(agent_module, "_notion_pantry", http_error)
    use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=[]),
    })
    state = agent("Sugira um almoço")
    assert state["output"].startswith("Error: ")
    assert "401" in state["output"]


def test_llm_exception_during_validation_returns_error(env, tools, monkeypatch):
    def boom(_messages):
        raise RuntimeError("Incorrect API key provided")

    use_llm(monkeypatch, {"QueryValidation": boom})
    state = agent("Receita com frango")
    assert state["output"].startswith("Error: ")
    assert tools["search_recipes"] == [] and tools["retrieve_pantry"] == []


def test_llm_exception_during_generation_returns_error(env, tools, monkeypatch):
    def boom(_messages):
        raise TimeoutError("request timed out")

    use_llm(monkeypatch, {
        "QueryValidation": VALID,
        "QueryAnalysis": QueryAnalysis(ingredients=["frango"]),
        "Recipe": boom,
    })
    state = agent("Receita com frango")
    assert state["output"].startswith("Error: ")
    assert state["recipe"] is None


# --------------------------------------------------------------------------- #
# Tools: unit tests of the HTTP layer (transport mocked)
# --------------------------------------------------------------------------- #


def _patch_httpx_client(monkeypatch, handler: Callable[[httpx.Request], httpx.Response]):
    original = httpx.Client

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return original(*args, **kwargs)

    monkeypatch.setattr(agent_module.httpx, "Client", factory)


def test_retrieve_pantry_parses_notion_blocks(env, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-notion-key"
        assert request.headers["Notion-Version"] == agent_module.NOTION_API_VERSION
        assert "test-page-id" in str(request.url)
        return httpx.Response(200, json={
            "results": [
                {"type": "heading_2", "heading_2": {"rich_text": [{"plain_text": "Despensa"}]}},
                {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"plain_text": "arroz"}]}},
                {"type": "to_do", "to_do": {"rich_text": [{"plain_text": "feijão"}]}},
                {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "tomate, cebola"}]}},
                {"type": "paragraph", "paragraph": {"rich_text": []}},
                {"type": "divider", "divider": {}},
                {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"plain_text": "Arroz"}]}},
            ],
            "has_more": False,
        })

    _patch_httpx_client(monkeypatch, handler)
    result = agent_module.retrieve_pantry.invoke({})
    assert result == ["Despensa", "arroz", "feijão", "tomate", "cebola"]


def test_retrieve_pantry_follows_pagination(env, monkeypatch):
    pages = {
        None: {"results": [{"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "ovos"}]}}],
               "has_more": True, "next_cursor": "c2"},
        "c2": {"results": [{"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "leite"}]}}],
               "has_more": False},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("start_cursor")
        return httpx.Response(200, json=pages[cursor])

    _patch_httpx_client(monkeypatch, handler)
    assert agent_module.retrieve_pantry.invoke({}) == ["ovos", "leite"]


def test_search_recipes_calls_tavily_and_normalizes_results(env, monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        captured["url"] = str(request.url)
        return httpx.Response(200, json={
            "results": [
                {"title": "Chicken curry", "url": "https://example.com/a", "content": "x" * 3000, "score": 0.9},
                {"title": "Chicken soup", "url": "https://example.com/b", "content": "warm"},
            ]
        })

    _patch_httpx_client(monkeypatch, handler)
    result = agent_module.search_recipes.invoke({"ingredients": ["chicken", "rice"]})
    assert captured["url"] == agent_module.TAVILY_SEARCH_URL
    assert captured["body"]["api_key"] == "test-tavily-key"
    assert "chicken" in captured["body"]["query"] and "rice" in captured["body"]["query"]
    assert [r["title"] for r in result] == ["Chicken curry", "Chicken soup"]
    assert set(result[0].keys()) == {"title", "url", "content"}
    assert len(result[0]["content"]) == 1500


def test_search_recipes_raises_on_http_error(env, monkeypatch):
    _patch_httpx_client(monkeypatch, lambda request: httpx.Response(500, json={"error": "boom"}))
    with pytest.raises(httpx.HTTPStatusError):
        agent_module.search_recipes.invoke({"ingredients": ["chicken"]})


def test_tools_have_names_and_descriptions():
    names = {t.name for t in agent_module.TOOLS}
    assert names == {"retrieve_pantry", "search_recipes"}
    for t in agent_module.TOOLS:
        assert t.description


# --------------------------------------------------------------------------- #
# Integration (requires real keys; skipped otherwise)
# --------------------------------------------------------------------------- #

needs_openai = pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
needs_tavily = pytest.mark.skipif(not os.getenv("TAVILY_API_KEY"), reason="TAVILY_API_KEY not set")
needs_notion = pytest.mark.skipif(
    not (os.getenv("NOTION_API_KEY") and os.getenv("NOTION_PANTRY_PAGE_ID")),
    reason="NOTION_API_KEY / NOTION_PANTRY_PAGE_ID not set",
)


@needs_openai
def test_integration_non_food_query():
    state = agent("Explique o teorema de Pitágoras")
    assert state["output"].startswith("Error: ")


@needs_openai
@needs_tavily
def test_integration_query_with_ingredients():
    state = agent("Quero um jantar vegetariano com grão-de-bico e espinafre")
    assert state["tool_called"] == "search_recipes"
    assert not state["output"].startswith("Error:")
    for section in ("Título:", "Ingredientes:", "Instruções:", "Nutrição:", "Armazenamento:"):
        assert section in state["output"]


@needs_openai
@needs_notion
def test_integration_query_without_ingredients():
    state = agent("Me sugira um almoço rápido com o que tenho na despensa")
    assert state["tool_called"] == "retrieve_pantry"
    assert state["output"].startswith("Error:") or "Título:" in state["output"]
