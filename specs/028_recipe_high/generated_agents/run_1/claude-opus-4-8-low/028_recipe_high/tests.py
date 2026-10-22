"""Tests for the high-complexity personalized recipe generator agent.

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


RECIPE_JSON = json.dumps(
    {
        "Title": "Chicken Rice Bowl",
        "Ingredients": ["200g chicken", "1 cup rice", "1 head broccoli"],
        "Equipment": ["pan", "pot"],
        "Instructions": ["cook the rice", "sear the chicken"],
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
        with mock.patch.object(
            agent, "_get_llm",
            make_llm(['{"valid": false, "reason": "not about cooking"}']),
        ), mock.patch.object(agent, "retrieve_pantry") as pantry, \
                mock.patch.object(agent, "search_recipes") as search:
            result = agent.agent("tell me about cars")
        self.assertTrue(result.startswith("Error:"))
        pantry.invoke.assert_not_called()
        search.invoke.assert_not_called()


class HappyPathTest(unittest.TestCase):
    def test_pantry_first_then_search_then_recipe(self):
        responses = [
            '{"valid": true}',                                  # B1
            '{"ingredients": ["chicken", "rice", "broccoli"]}',  # B3
            RECIPE_JSON,                                         # B7
        ]
        search_ret = [{"title": "t", "url": "u", "content": "c"}]
        pantry = mock.Mock(invoke=mock.Mock(return_value=["rice", "broccoli"]))
        search = mock.Mock(invoke=mock.Mock(return_value=search_ret))
        with mock.patch.object(agent, "_get_llm", make_llm(responses)), \
                mock.patch.object(agent, "retrieve_pantry", pantry), \
                mock.patch.object(agent, "search_recipes", search):
            result = agent.agent("make me a dish with chicken")

        self.assertIn("Title: Chicken Rice Bowl", result)
        self.assertIn("Ingredients:", result)
        self.assertIn("Equipment:", result)
        self.assertIn("Instructions:", result)
        self.assertIn("Storage:", result)
        pantry.invoke.assert_called_once()
        search.invoke.assert_called_once()
        # retrieve_pantry is called before search_recipes with B3 ingredients.
        search.invoke.assert_called_once_with(
            {"ingredients": ["chicken", "rice", "broccoli"]}
        )


class B5LackOfIngredientsTest(unittest.TestCase):
    def test_no_compatible_ingredient_stops_before_search(self):
        responses = [
            '{"valid": true}',        # B1
            '{"ingredients": []}',    # B3 -> empty
        ]
        pantry = mock.Mock(invoke=mock.Mock(return_value=[]))
        search = mock.Mock()
        with mock.patch.object(agent, "_get_llm", make_llm(responses)), \
                mock.patch.object(agent, "retrieve_pantry", pantry), \
                mock.patch.object(agent, "search_recipes", search):
            result = agent.agent("make me something vegan with only meat")
        self.assertTrue(result.startswith("Error:"))
        self.assertNotIn("Title:", result)
        search.invoke.assert_not_called()


class B6FailureTest(unittest.TestCase):
    def test_pantry_error_goes_to_b6(self):
        pantry = mock.Mock(
            invoke=mock.Mock(return_value="Error: missing NOTION_API_KEY credential.")
        )
        search = mock.Mock()
        with mock.patch.object(agent, "_get_llm", make_llm(['{"valid": true}'])), \
                mock.patch.object(agent, "retrieve_pantry", pantry), \
                mock.patch.object(agent, "search_recipes", search):
            result = agent.agent("make me a dish with chicken")
        self.assertTrue(result.startswith("Error:"))
        self.assertNotIn("Title:", result)
        search.invoke.assert_not_called()

    def test_search_error_goes_to_b6(self):
        responses = [
            '{"valid": true}',
            '{"ingredients": ["chicken"]}',
        ]
        pantry = mock.Mock(invoke=mock.Mock(return_value=["chicken"]))
        search = mock.Mock(
            invoke=mock.Mock(return_value="erro: timeout while searching for recipes.")
        )
        with mock.patch.object(agent, "_get_llm", make_llm(responses)), \
                mock.patch.object(agent, "retrieve_pantry", pantry), \
                mock.patch.object(agent, "search_recipes", search):
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
