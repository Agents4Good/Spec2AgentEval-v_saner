"""Tests for the personalized recipe generator agent.

The LLM and the external tools are mocked so the behavioral graph can be
exercised deterministically, without real OpenAI/Notion/Tavily access.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

import agent as agent_mod


def _fake_llm_message(payload):
    """Build a fake message whose .content is `payload` (dict -> JSON str)."""
    msg = MagicMock()
    msg.content = json.dumps(payload) if isinstance(payload, (dict, list)) else payload
    return msg


class FakeLLM:
    """Scripted ChatOpenAI replacement returning queued responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# Tool unit tests
# ---------------------------------------------------------------------------


class ToolInterfaceTests(unittest.TestCase):
    def test_tools_are_langchain_tools(self):
        self.assertTrue(hasattr(agent_mod.retrieve_pantry, "invoke"))
        self.assertTrue(hasattr(agent_mod.search_recipes, "invoke"))

    def test_retrieve_pantry_missing_credentials(self):
        with patch.dict("os.environ", {}, clear=True):
            result = agent_mod.retrieve_pantry.invoke({})
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("Error:"))

    def test_search_recipes_missing_credentials(self):
        with patch.dict("os.environ", {}, clear=True):
            result = agent_mod.search_recipes.invoke({"ingredients": ["egg"]})
        self.assertIsInstance(result, str)
        self.assertTrue(result.lower().startswith("erro:"))

    def test_retrieve_pantry_timeout(self):
        import requests

        with patch.dict(
            "os.environ",
            {"NOTION_API_KEY": "k", "NOTION_PAGE_ID": "p"},
            clear=True,
        ):
            with patch.object(
                agent_mod.requests,
                "get",
                side_effect=requests.exceptions.Timeout(),
            ):
                result = agent_mod.retrieve_pantry.invoke({})
        self.assertTrue(result.startswith("Error:"))

    def test_search_recipes_returns_up_to_five(self):
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {
            "results": [
                {"title": f"t{i}", "url": f"u{i}", "content": f"c{i}"}
                for i in range(8)
            ]
        }
        with patch.dict("os.environ", {"TAVILY_API_KEY": "k"}, clear=True):
            with patch.object(agent_mod.requests, "post", return_value=fake_resp):
                result = agent_mod.search_recipes.invoke({"ingredients": ["egg"]})
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 5)
        self.assertEqual(set(result[0].keys()), {"title", "url", "content"})


# ---------------------------------------------------------------------------
# Behavioral graph tests
# ---------------------------------------------------------------------------


def _fake_tool(return_value=None):
    """A stand-in for a LangChain tool exposing the .invoke interface."""
    mock = MagicMock()
    mock.invoke.return_value = return_value
    return mock


class BehaviorTests(unittest.TestCase):
    def setUp(self):
        agent_mod._GRAPH = None  # rebuild graph fresh each test

    # B1: invalid query stops execution, no tool called -------------------
    def test_b1_invalid_query(self):
        fake = FakeLLM([_fake_llm_message({"valid": False, "reason": "not food"})])
        s, r = _fake_tool([]), _fake_tool([])
        with patch.object(agent_mod, "_llm", return_value=fake), patch.object(
            agent_mod, "search_recipes", s
        ), patch.object(agent_mod, "retrieve_pantry", r):
            out = agent_mod.agent("tell me a joke")
        self.assertTrue(out.startswith("Error:"))
        s.invoke.assert_not_called()
        r.invoke.assert_not_called()

    def test_b1_empty_query(self):
        out = agent_mod.agent("   ")
        self.assertTrue(out.startswith("Error:"))

    # B2 + B3: ingredients present -> search_recipes once, no pantry ------
    def test_b3_search_called_once_no_pantry(self):
        fake = FakeLLM(
            [
                _fake_llm_message({"valid": True, "reason": "ok"}),
                _fake_llm_message({"ingredients": ["chicken", "rice"]}),
                _fake_llm_message(
                    "Title: X\nIngredients: chicken\nEquipment: pan\n"
                    "Instructions: cook\nStorage: fridge"
                ),
            ]
        )
        s, r = _fake_tool([]), _fake_tool([])
        with patch.object(agent_mod, "_llm", return_value=fake), patch.object(
            agent_mod, "search_recipes", s
        ), patch.object(agent_mod, "retrieve_pantry", r):
            out = agent_mod.agent("I have chicken and rice")
        s.invoke.assert_called_once_with({"ingredients": ["chicken", "rice"]})
        r.invoke.assert_not_called()
        self.assertIn("Title:", out)

    # B4: no ingredients -> pantry lookup once, no search -----------------
    def test_b4_pantry_called_once_no_search(self):
        fake = FakeLLM(
            [
                _fake_llm_message({"valid": True, "reason": "ok"}),
                _fake_llm_message({"ingredients": []}),
                _fake_llm_message({"ingredients": ["tomato"]}),  # pantry filter
                _fake_llm_message(
                    "Title: X\nIngredients: tomato\nEquipment: pot\n"
                    "Instructions: cook\nStorage: fridge"
                ),
            ]
        )
        s, r = _fake_tool([]), _fake_tool(["tomato"])
        with patch.object(agent_mod, "_llm", return_value=fake), patch.object(
            agent_mod, "retrieve_pantry", r
        ), patch.object(agent_mod, "search_recipes", s):
            out = agent_mod.agent("make me something")
        r.invoke.assert_called_once()
        s.invoke.assert_not_called()
        self.assertIn("Title:", out)

    # B5: empty pantry / no compatible -> Error, no recipe ----------------
    def test_b5_lack_of_ingredients(self):
        fake = FakeLLM(
            [
                _fake_llm_message({"valid": True, "reason": "ok"}),
                _fake_llm_message({"ingredients": []}),
            ]
        )
        with patch.object(agent_mod, "_llm", return_value=fake), patch.object(
            agent_mod, "retrieve_pantry", _fake_tool([])
        ):
            out = agent_mod.agent("cook for me")
        self.assertTrue(out.startswith("Error:"))

    # B6: tool returns 'erro:' -> handled, Error message ------------------
    def test_b6_search_tool_error(self):
        fake = FakeLLM(
            [
                _fake_llm_message({"valid": True, "reason": "ok"}),
                _fake_llm_message({"ingredients": ["egg"]}),
            ]
        )
        with patch.object(agent_mod, "_llm", return_value=fake), patch.object(
            agent_mod, "search_recipes", _fake_tool("erro: timeout")
        ):
            out = agent_mod.agent("I have eggs")
        self.assertTrue(out.startswith("Error:"))

    def test_b6_pantry_tool_error(self):
        fake = FakeLLM(
            [
                _fake_llm_message({"valid": True, "reason": "ok"}),
                _fake_llm_message({"ingredients": []}),
            ]
        )
        with patch.object(agent_mod, "_llm", return_value=fake), patch.object(
            agent_mod, "retrieve_pantry", _fake_tool("Error: no creds")
        ):
            out = agent_mod.agent("cook")
        self.assertTrue(out.startswith("Error:"))

    def test_b6_no_unhandled_exception(self):
        with patch.object(agent_mod, "_llm", side_effect=RuntimeError("boom")):
            out = agent_mod.agent("I have eggs")
        self.assertTrue(out.startswith("Error:"))

    # B7: recipe has all fields -------------------------------------------
    def test_b7_recipe_fields_present(self):
        recipe = (
            "Title: Omelette\nIngredients: 2 eggs\nEquipment: pan\n"
            "Instructions: beat and cook\nStorage: eat fresh"
        )
        fake = FakeLLM(
            [
                _fake_llm_message({"valid": True, "reason": "ok"}),
                _fake_llm_message({"ingredients": ["egg"]}),
                _fake_llm_message(recipe),
            ]
        )
        with patch.object(agent_mod, "_llm", return_value=fake), patch.object(
            agent_mod, "search_recipes", _fake_tool([])
        ):
            out = agent_mod.agent("I have eggs")
        for field in ("Title:", "Ingredients:", "Equipment:", "Instructions:", "Storage:"):
            self.assertIn(field, out)


if __name__ == "__main__":
    unittest.main()
