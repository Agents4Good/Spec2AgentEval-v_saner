"""Tests for the personalized recipe generator agent.

The ChatOpenAI model and the external HTTP calls are mocked so the tests run
offline and deterministically, exercising the B1-B7 behaviors of the graph.
"""

import json
import unittest
from unittest import mock

import agent


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Replays queued JSON responses, sharing one queue across instances."""

    def __init__(self, queue):
        self._queue = queue

    def invoke(self, _prompt):
        return FakeMessage(self._queue.pop(0))


def make_llm(responses):
    queue = list(responses)  # shared across every _get_llm() call

    def factory():
        return FakeLLM(queue)

    return factory


class ToolInterfaceTest(unittest.TestCase):
    def test_tools_are_langchain_tools(self):
        from langchain_core.tools import BaseTool

        self.assertIsInstance(agent.retrieve_pantry, BaseTool)
        self.assertIsInstance(agent.search_recipes, BaseTool)


class B1ValidationTest(unittest.TestCase):
    def test_empty_query_is_invalid(self):
        result = agent.agent("   ")
        self.assertTrue(result.startswith("Error:"))

    def test_offensive_or_unrelated_query_stops(self):
        with mock.patch.object(
            agent, "_get_llm",
            make_llm(['{"valid": false, "reason": "not about cooking"}']),
        ):
            with mock.patch.object(agent, "search_recipes") as srch, \
                 mock.patch.object(agent, "retrieve_pantry") as pan:
                result = agent.agent("tell me about cars")
        self.assertTrue(result.startswith("Error:"))
        srch.invoke.assert_not_called()
        pan.invoke.assert_not_called()


class B3SearchPathTest(unittest.TestCase):
    def test_valid_with_ingredients_calls_search_only(self):
        responses = [
            '{"valid": true}',                              # B1
            '{"ingredients": ["chicken", "rice"]}',         # B2
            json.dumps({                                    # B7
                "Title": "Chicken Rice",
                "Ingredients": ["200g chicken", "1 cup rice"],
                "Equipment": ["pan"],
                "Instructions": ["cook"],
                "Storage": "fridge 3 days",
            }),
        ]
        search_ret = [{"title": "t", "url": "u", "content": "c"}]
        with mock.patch.object(agent, "_get_llm", make_llm(responses)), \
             mock.patch.object(agent, "search_recipes", mock.Mock(invoke=mock.Mock(return_value=search_ret))), \
             mock.patch.object(agent, "retrieve_pantry", mock.Mock()) as pantry_tool:
            result = agent.agent("make me a dish with chicken and rice")

        self.assertIn("Title: Chicken Rice", result)
        self.assertIn("Ingredients:", result)
        self.assertIn("Equipment:", result)
        self.assertIn("Instructions:", result)
        self.assertIn("Storage:", result)
        pantry_tool.invoke.assert_not_called()

    def test_search_error_goes_to_b6(self):
        responses = [
            '{"valid": true}',
            '{"ingredients": ["chicken"]}',
        ]
        with mock.patch.object(agent, "_get_llm", make_llm(responses)), \
             mock.patch.object(agent.search_recipes, "func",
                               return_value="erro: timeout while searching for recipes."):
            result = agent.agent("dish with chicken")
        self.assertTrue(result.startswith("Error:"))
        self.assertNotIn("Title:", result)


class B4PantryPathTest(unittest.TestCase):
    def test_no_ingredients_uses_pantry(self):
        responses = [
            '{"valid": true}',                      # B1
            '{"ingredients": []}',                  # B2 -> empty
            '{"ingredients": ["eggs", "flour"]}',   # B4 filter
            json.dumps({                            # B7
                "Title": "Pancakes",
                "Ingredients": ["2 eggs", "1 cup flour"],
                "Equipment": ["bowl"],
                "Instructions": ["mix"],
                "Storage": "airtight",
            }),
        ]
        with mock.patch.object(agent, "_get_llm", make_llm(responses)), \
             mock.patch.object(agent, "retrieve_pantry", mock.Mock(invoke=mock.Mock(return_value=["eggs", "flour"]))), \
             mock.patch.object(agent, "search_recipes", mock.Mock()) as search_tool:
            result = agent.agent("make me something tasty")
        self.assertIn("Title: Pancakes", result)
        search_tool.invoke.assert_not_called()

    def test_empty_pantry_stops_b5(self):
        responses = [
            '{"valid": true}',
            '{"ingredients": []}',
        ]
        with mock.patch.object(agent, "_get_llm", make_llm(responses)), \
             mock.patch.object(agent, "retrieve_pantry", mock.Mock(invoke=mock.Mock(return_value=[]))):
            result = agent.agent("make me something")
        self.assertTrue(result.startswith("Error:"))
        self.assertNotIn("Title:", result)

    def test_pantry_error_goes_to_b6(self):
        responses = [
            '{"valid": true}',
            '{"ingredients": []}',
        ]
        with mock.patch.object(agent, "_get_llm", make_llm(responses)), \
             mock.patch.object(agent, "retrieve_pantry", mock.Mock(invoke=mock.Mock(return_value="Error: missing NOTION_API_KEY credential."))):
            result = agent.agent("make me something")
        self.assertTrue(result.startswith("Error:"))


class ToolFailureTest(unittest.TestCase):
    def test_retrieve_pantry_missing_credentials(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            out = agent.retrieve_pantry.invoke({})
        self.assertTrue(out.startswith("Error:"))

    def test_search_recipes_missing_credentials(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            out = agent.search_recipes.invoke({"ingredients": ["egg"]})
        self.assertTrue(out.startswith("erro:"))


if __name__ == "__main__":
    unittest.main()
