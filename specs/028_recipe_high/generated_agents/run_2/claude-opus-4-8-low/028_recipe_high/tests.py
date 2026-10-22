"""Tests for the personalized recipe generator agent (028 high spec).

The ChatOpenAI model and external HTTP calls are mocked so the tests run
offline and deterministically, exercising the B1-B7 behaviors and the
pantry-first flow of the graph.
"""

import json
import unittest
from unittest import mock

import agent


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Replays queued responses, sharing one queue across instances."""

    def __init__(self, queue):
        self._queue = queue

    def invoke(self, _prompt):
        return FakeMessage(self._queue.pop(0))


def make_llm(responses):
    queue = list(responses)  # shared across every _get_llm() call

    def factory():
        return FakeLLM(queue)

    return factory


RECIPE_JSON = json.dumps(
    {
        "Title": "Chicken Rice Bowl",
        "Ingredients": ["200g chicken", "1 cup rice"],
        "Equipment": ["pan", "pot"],
        "Instructions": ["cook rice", "sear chicken"],
        "Storage": "fridge up to 3 days",
    }
)


class ToolInterfaceTest(unittest.TestCase):
    def test_tools_are_langchain_tools(self):
        from langchain_core.tools import BaseTool

        self.assertIsInstance(agent.retrieve_pantry, BaseTool)
        self.assertIsInstance(agent.search_recipes, BaseTool)


class B1ValidationTest(unittest.TestCase):
    def test_empty_query_is_invalid(self):
        result = agent.agent("   ")
        self.assertTrue(result.startswith("Error:"))

    def test_invalid_query_stops_without_tools(self):
        pantry_tool = mock.Mock()
        search_tool = mock.Mock()
        with mock.patch.object(
            agent, "_get_llm",
            make_llm(['{"valid": false, "reason": "not about cooking"}']),
        ), mock.patch.object(agent, "retrieve_pantry", pantry_tool), \
           mock.patch.object(agent, "search_recipes", search_tool):
            result = agent.agent("tell me about cars")
        self.assertTrue(result.startswith("Error:"))
        pantry_tool.invoke.assert_not_called()
        search_tool.invoke.assert_not_called()


class HappyPathTest(unittest.TestCase):
    def test_pantry_first_then_search_then_recipe(self):
        responses = [
            '{"valid": true}',                               # B1
            '{"ingredients": ["chicken", "rice"]}',          # B3
            RECIPE_JSON,                                     # B7
        ]
        pantry_tool = mock.Mock(invoke=mock.Mock(return_value=["chicken"]))
        search_tool = mock.Mock(
            invoke=mock.Mock(return_value=[{"title": "t", "url": "u", "content": "c"}])
        )
        with mock.patch.object(agent, "_get_llm", make_llm(responses)), \
             mock.patch.object(agent, "retrieve_pantry", pantry_tool), \
             mock.patch.object(agent, "search_recipes", search_tool):
            result = agent.agent("make me a dish with rice")

        self.assertIn("Title: Chicken Rice Bowl", result)
        self.assertIn("Ingredients:", result)
        self.assertIn("Equipment:", result)
        self.assertIn("Instructions:", result)
        self.assertIn("Storage:", result)
        # B2: pantry is the first tool, called exactly once with {}.
        pantry_tool.invoke.assert_called_once_with({})
        # B4: search called exactly once with the B3 ingredient list.
        search_tool.invoke.assert_called_once_with(
            {"ingredients": ["chicken", "rice"]}
        )

    def test_search_with_no_results_still_generates(self):
        responses = [
            '{"valid": true}',
            '{"ingredients": ["eggs"]}',
            RECIPE_JSON,
        ]
        pantry_tool = mock.Mock(invoke=mock.Mock(return_value=["eggs"]))
        search_tool = mock.Mock(invoke=mock.Mock(return_value=[]))
        with mock.patch.object(agent, "_get_llm", make_llm(responses)), \
             mock.patch.object(agent, "retrieve_pantry", pantry_tool), \
             mock.patch.object(agent, "search_recipes", search_tool):
            result = agent.agent("something with eggs")
        self.assertIn("Title:", result)
        search_tool.invoke.assert_called_once()


class B5LackOfIngredientsTest(unittest.TestCase):
    def test_no_compatible_ingredients_stops_before_search(self):
        responses = [
            '{"valid": true}',
            '{"ingredients": []}',   # B3 empty
        ]
        pantry_tool = mock.Mock(invoke=mock.Mock(return_value=[]))
        search_tool = mock.Mock()
        with mock.patch.object(agent, "_get_llm", make_llm(responses)), \
             mock.patch.object(agent, "retrieve_pantry", pantry_tool), \
             mock.patch.object(agent, "search_recipes", search_tool):
            result = agent.agent("make me a vegan dish")
        self.assertTrue(result.startswith("Error:"))
        self.assertNotIn("Title:", result)
        pantry_tool.invoke.assert_called_once_with({})
        search_tool.invoke.assert_not_called()


class B6FailureTest(unittest.TestCase):
    def test_pantry_error_goes_to_b6(self):
        responses = ['{"valid": true}']
        pantry_tool = mock.Mock(
            invoke=mock.Mock(return_value="Error: missing NOTION_API_KEY credential.")
        )
        search_tool = mock.Mock()
        with mock.patch.object(agent, "_get_llm", make_llm(responses)), \
             mock.patch.object(agent, "retrieve_pantry", pantry_tool), \
             mock.patch.object(agent, "search_recipes", search_tool):
            result = agent.agent("dish with chicken")
        self.assertTrue(result.startswith("Error:"))
        self.assertNotIn("Title:", result)
        search_tool.invoke.assert_not_called()

    def test_search_error_goes_to_b6(self):
        responses = [
            '{"valid": true}',
            '{"ingredients": ["chicken"]}',
        ]
        pantry_tool = mock.Mock(invoke=mock.Mock(return_value=["chicken"]))
        search_tool = mock.Mock(
            invoke=mock.Mock(return_value="Error: timeout while searching for recipes.")
        )
        with mock.patch.object(agent, "_get_llm", make_llm(responses)), \
             mock.patch.object(agent, "retrieve_pantry", pantry_tool), \
             mock.patch.object(agent, "search_recipes", search_tool):
            result = agent.agent("dish with chicken")
        self.assertTrue(result.startswith("Error:"))
        self.assertNotIn("Title:", result)

    def test_unhandled_exception_is_caught(self):
        with mock.patch.object(agent, "_app", side_effect=RuntimeError("boom")):
            result = agent.agent("dish with chicken")
        self.assertTrue(result.startswith("Error:"))


class ToolFailureTest(unittest.TestCase):
    def test_retrieve_pantry_missing_credentials(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            out = agent.retrieve_pantry.invoke({})
        self.assertTrue(out.startswith("Error:"))

    def test_search_recipes_missing_credentials(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            out = agent.search_recipes.invoke({"ingredients": ["egg"]})
        self.assertTrue(out.startswith("Error:"))


if __name__ == "__main__":
    unittest.main()
