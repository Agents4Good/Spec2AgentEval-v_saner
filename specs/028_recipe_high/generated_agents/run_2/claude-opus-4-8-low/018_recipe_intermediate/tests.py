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

    def test_unrelated_query_stops_without_tools(self):
        with mock.patch.object(
            agent, "_get_llm",
            make_llm(['{"valid": false, "reason": "not about cooking"}']),
        ), mock.patch.object(agent, "search_recipes") as srch, \
           mock.patch.object(agent, "retrieve_pantry") as pan:
            result = agent.agent("tell me about cars")
        self.assertTrue(result.startswith("Error:"))
        srch.invoke.assert_not_called()
        pan.invoke.assert_not_called()


class B3SearchPathTest(unittest.TestCase):
    def test_valid_with_ingredients_calls_search_only(self):
        responses = [
            '{"valid": true}',
            '{"ingredients": ["chicken", "rice"]}',
            json.dumps({
                "Title": "Chicken Rice",
                "Ingredients": ["200g chicken", "1 cup rice"],
                "Equipment": ["pan"],
                "Instructions": ["cook"],
                "Storage": "fridge 3 days",
            }),
        ]
        search_ret = [{"title": "t", "url": "u", "content": "c"}]
        search_tool = mock.Mock(invoke=mock.Mock(return_value=search_ret))
        pantry_tool = mock.Mock()
        with mock.patch.object(agent, "_get_llm", make_llm(responses)), \
             mock.patch.object(agent, "search_recipes", search_tool), \
             mock.patch.object(agent, "retrieve_pantry", pantry_tool):
            result = agent.agent("make me a dish with chicken and rice")

        self.assertIn("Title: Chicken Rice", result)
        self.assertIn("Ingredients:", result)
        self.assertIn("Equipment:", result)
        self.assertIn("Instructions:", result)
        self.assertIn("Storage:", result)
        search_tool.invoke.assert_called_once_with({"ingredients": ["chicken", "rice"]})
        pantry_tool.invoke.assert_not_called()

    def test_search_error_goes_to_b6(self):
        responses = [
            '{"valid": true}',
            '{"ingredients": ["chicken"]}',
        ]
        search_tool = mock.Mock(
            invoke=mock.Mock(return_value="erro: timeout while searching for recipes.")
        )
        with mock.patch.object(agent, "_get_llm", make_llm(responses)), \
             mock.patch.object(agent, "search_recipes", search_tool):
            result = agent.agent("dish with chicken")
        self.assertTrue(result.startswith("Error:"))
        self.assertNotIn("Title:", result)


class B4PantryPathTest(unittest.TestCase):
    def test_no_ingredients_uses_pantry(self):
        responses = [
            '{"valid": true}',
            '{"ingredients": []}',
            '{"ingredients": ["eggs", "flour"]}',
            json.dumps({
                "Title": "Pancakes",
                "Ingredients": ["2 eggs", "1 cup flour"],
                "Equipment": ["bowl"],
                "Instructions": ["mix"],
                "Storage": "airtight",
            }),
        ]
        pantry_tool = mock.Mock(invoke=mock.Mock(return_value=["eggs", "flour"]))
        search_tool = mock.Mock()
        with mock.patch.object(agent, "_get_llm", make_llm(responses)), \
             mock.patch.object(agent, "retrieve_pantry", pantry_tool), \
             mock.patch.object(agent, "search_recipes", search_tool):
            result = agent.agent("make me something tasty")
        self.assertIn("Title: Pancakes", result)
        pantry_tool.invoke.assert_called_once_with({})
        search_tool.invoke.assert_not_called()

    def test_empty_pantry_stops_b5(self):
        responses = [
            '{"valid": true}',
            '{"ingredients": []}',
        ]
        pantry_tool = mock.Mock(invoke=mock.Mock(return_value=[]))
        with mock.patch.object(agent, "_get_llm", make_llm(responses)), \
             mock.patch.object(agent, "retrieve_pantry", pantry_tool):
            result = agent.agent("make me something")
        self.assertTrue(result.startswith("Error:"))
        self.assertNotIn("Title:", result)

    def test_pantry_error_goes_to_b6(self):
        responses = [
            '{"valid": true}',
            '{"ingredients": []}',
        ]
        pantry_tool = mock.Mock(
            invoke=mock.Mock(return_value="Error: missing NOTION_API_KEY credential.")
        )
        with mock.patch.object(agent, "_get_llm", make_llm(responses)), \
             mock.patch.object(agent, "retrieve_pantry", pantry_tool):
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
