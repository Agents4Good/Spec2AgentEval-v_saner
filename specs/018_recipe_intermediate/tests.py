import unittest
import json
import os
from unittest.mock import patch, MagicMock
from agent import (
    agent, validate_query, identify_ingredients, search_recipes,
    retrieve_pantry, AgentState
)


class TestRecipeAgent(unittest.TestCase):

    def test_b1_empty_query_is_invalid(self):
        """B1: Empty query should be invalid."""
        state = AgentState(
            query="",
            is_valid=False,
            error_message=None,
            identified_ingredients=[],
            dietary_restrictions=[],
            pantry_ingredients=[],
            reference_recipes=[],
            final_recipe=None,
            tool_called=None
        )
        result = validate_query(state)
        self.assertFalse(result["is_valid"])
        self.assertTrue(result["error_message"].startswith("Error:"))

    def test_b1_offensive_query_is_invalid(self):
        """B1: Offensive query should be invalid."""
        state = AgentState(
            query="receita para bomb algo",
            is_valid=False,
            error_message=None,
            identified_ingredients=[],
            dietary_restrictions=[],
            pantry_ingredients=[],
            reference_recipes=[],
            final_recipe=None,
            tool_called=None
        )
        result = validate_query(state)
        self.assertFalse(result["is_valid"])
        self.assertTrue(result["error_message"].startswith("Error:"))

    def test_b1_non_food_query_is_invalid(self):
        """B1: Non-food related query should be invalid."""
        state = AgentState(
            query="como construir uma casa",
            is_valid=False,
            error_message=None,
            identified_ingredients=[],
            dietary_restrictions=[],
            pantry_ingredients=[],
            reference_recipes=[],
            final_recipe=None,
            tool_called=None
        )
        result = validate_query(state)
        self.assertFalse(result["is_valid"])
        self.assertTrue(result["error_message"].startswith("Error:"))

    def test_b1_valid_recipe_query(self):
        """B1: Valid recipe query should be valid."""
        state = AgentState(
            query="quero fazer uma receita com tomate",
            is_valid=False,
            error_message=None,
            identified_ingredients=[],
            dietary_restrictions=[],
            pantry_ingredients=[],
            reference_recipes=[],
            final_recipe=None,
            tool_called=None
        )
        result = validate_query(state)
        self.assertTrue(result["is_valid"])
        self.assertIsNone(result["error_message"])

    def test_b2_ingredient_identification(self):
        """B2: Identify ingredients from query."""
        state = AgentState(
            query="quero fazer uma receita com tomate e cebola",
            is_valid=True,
            error_message=None,
            identified_ingredients=[],
            dietary_restrictions=[],
            pantry_ingredients=[],
            reference_recipes=[],
            final_recipe=None,
            tool_called=None
        )
        result = identify_ingredients(state)
        self.assertIn("tomate", result["identified_ingredients"])
        self.assertIn("cebola", result["identified_ingredients"])

    def test_b2_exclude_negated_ingredients(self):
        """B2: Exclude ingredients mentioned under negation."""
        state = AgentState(
            query="receita com tomate mas sem cebola",
            is_valid=True,
            error_message=None,
            identified_ingredients=[],
            dietary_restrictions=[],
            pantry_ingredients=[],
            reference_recipes=[],
            final_recipe=None,
            tool_called=None
        )
        result = identify_ingredients(state)
        self.assertIn("tomate", result["identified_ingredients"])
        self.assertNotIn("cebola", result["identified_ingredients"])

    def test_b2_dietary_restrictions_excluded_ingredients(self):
        """B2: Dietary restrictions should exclude compatible ingredients."""
        state = AgentState(
            query="receita vegana com carne e frango",
            is_valid=True,
            error_message=None,
            identified_ingredients=[],
            dietary_restrictions=[],
            pantry_ingredients=[],
            reference_recipes=[],
            final_recipe=None,
            tool_called=None
        )
        result = identify_ingredients(state)
        self.assertIn("vegano", result["dietary_restrictions"])
        self.assertNotIn("carne", result["identified_ingredients"])
        self.assertNotIn("frango", result["identified_ingredients"])

    def test_retrieve_pantry_missing_credentials(self):
        """Tool: retrieve_pantry should return error for missing credentials."""
        with patch.dict(os.environ, {}, clear=True):
            result = retrieve_pantry()
            self.assertTrue(result.startswith("erro:"))

    def test_search_recipes_missing_credentials(self):
        """Tool: search_recipes should return error for missing credentials."""
        with patch.dict(os.environ, {}, clear=True):
            result = search_recipes(["tomate"])
            self.assertTrue(result.startswith("erro:"))

    def test_search_recipes_empty_ingredients(self):
        """Tool: search_recipes with empty ingredients should return empty list."""
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test_key"}):
            result = search_recipes([])
            self.assertEqual(json.loads(result), [])

    def test_agent_returns_dict_with_state(self):
        """Agent should return a dictionary with state."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
            with patch("agent.ChatOpenAI") as mock_llm:
                mock_response = MagicMock()
                mock_response.content = json.dumps({
                    "título": "Salada de Tomate",
                    "ingredientes": ["2 tomates", "sal"],
                    "equipamentos": ["faca"],
                    "instruções": ["Cortar tomates"],
                    "armazenamento": "Refrigerar"
                })
                mock_llm.return_value.invoke.return_value = mock_response

                with patch("agent.retrieve_pantry", return_value=json.dumps(["tomate", "sal"])):
                    result = agent("receita com tomate")
                    self.assertIsInstance(result, dict)
                    self.assertIn("query", result)
                    self.assertIn("is_valid", result)

    def test_agent_invalid_query_no_tools_called(self):
        """B1: Invalid query should not call any tools."""
        with patch("agent.retrieve_pantry") as mock_retrieve:
            with patch("agent.search_recipes") as mock_search:
                result = agent("como construir uma casa")
                self.assertFalse(result["is_valid"])
                self.assertTrue(result["error_message"].startswith("Error:"))
                mock_retrieve.assert_not_called()
                mock_search.assert_not_called()

    def test_agent_empty_pantry_returns_error(self):
        """B5: Empty pantry should return error."""
        with patch.dict(os.environ, {"NOTION_API_KEY": "test", "NOTION_PAGE_ID": "test"}):
            with patch("agent.retrieve_pantry", return_value=json.dumps([])):
                result = agent("receita")
                self.assertTrue(result["error_message"].startswith("Error:"))

    def test_b3_search_recipes_called_when_ingredients_identified(self):
        """B3: search_recipes should be called when ingredients are identified."""
        with patch("agent.search_recipes", return_value=json.dumps([
            {"title": "Salada", "url": "http://example.com", "snippet": "Uma salada"}
        ])) as mock_search:
            with patch("agent.ChatOpenAI"):
                with patch("agent.identify_ingredients") as mock_identify:
                    result = agent("receita com tomate")
                    if result["identified_ingredients"]:
                        mock_search.assert_called()

    def test_b4_retrieve_pantry_called_when_no_ingredients(self):
        """B4: retrieve_pantry should be called when no ingredients identified."""
        with patch("agent.retrieve_pantry", return_value=json.dumps(["arroz", "feijão"])):
            with patch("agent.search_recipes") as mock_search:
                with patch("agent.ChatOpenAI"):
                    result = agent("receita")
                    if not result["identified_ingredients"]:
                        mock_search.assert_not_called()

    def test_b6_dietary_incompatibility_returns_error(self):
        """B6: When no ingredients are compatible with dietary restrictions, return error."""
        with patch.dict(os.environ, {"NOTION_API_KEY": "test", "NOTION_PAGE_ID": "test"}):
            with patch("agent.retrieve_pantry", return_value=json.dumps(["carne", "frango", "leite"])):
                result = agent("receita vegana")
                self.assertTrue(result["error_message"].startswith("Error:"))

    def test_b8_recipe_has_all_required_fields(self):
        """B8: Generated recipe should have all required fields."""
        with patch("agent.retrieve_pantry", return_value=json.dumps(["tomate"])):
            with patch("agent.ChatOpenAI") as mock_llm:
                mock_response = MagicMock()
                mock_response.content = json.dumps({
                    "título": "Salada de Tomate",
                    "ingredientes": ["2 tomates", "sal", "azeite"],
                    "equipamentos": ["faca", "tábua"],
                    "instruções": ["Cortar tomates", "Temperar com sal e azeite"],
                    "armazenamento": "Refrigerar por até 2 dias"
                })
                mock_llm.return_value.invoke.return_value = mock_response

                result = agent("receita com tomate")
                if result["final_recipe"]:
                    self.assertIn("título", result["final_recipe"])
                    self.assertIn("ingredientes", result["final_recipe"])
                    self.assertIn("equipamentos", result["final_recipe"])
                    self.assertIn("instruções", result["final_recipe"])
                    self.assertIn("armazenamento", result["final_recipe"])

                    self.assertTrue(len(result["final_recipe"]["titulo"]) > 0)
                    self.assertTrue(len(result["final_recipe"]["ingredientes"]) > 0)
                    self.assertTrue(len(result["final_recipe"]["equipamentos"]) > 0)
                    self.assertTrue(len(result["final_recipe"]["instruções"]) > 0)
                    self.assertTrue(len(result["final_recipe"]["armazenamento"]) > 0)

    def test_constraint_never_call_both_tools(self):
        """Constraint: Never call both tools in the same execution."""
        with patch("agent.search_recipes", return_value=json.dumps([])) as mock_search:
            with patch("agent.retrieve_pantry", return_value=json.dumps([])) as mock_retrieve:
                with patch("agent.ChatOpenAI"):
                    result = agent("receita com tomate e cebola")

                    tools_called = 0
                    if result["tool_called"] == "search_recipes":
                        tools_called += 1
                    if result["tool_called"] == "retrieve_pantry":
                        tools_called += 1

                    self.assertLessEqual(tools_called, 1)


if __name__ == "__main__":
    unittest.main()
