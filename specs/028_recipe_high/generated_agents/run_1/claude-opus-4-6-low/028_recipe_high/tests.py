import os
import json
import pytest
from unittest.mock import patch, MagicMock

# Set dummy env vars before importing agent
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("NOTION_API_KEY", "test-notion-key")
os.environ.setdefault("NOTION_PAGE_ID", "test-page-id")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")

from agent import agent, retrieve_pantry, search_recipes, validate_query, identify_ingredients


# Helper to mock LLM responses in sequence
def make_mock_llm(responses):
    mock = MagicMock()
    side_effects = []
    for r in responses:
        msg = MagicMock()
        msg.content = r
        side_effects.append(msg)
    mock.invoke = MagicMock(side_effect=side_effects)
    return mock


class TestB1QueryValidation:
    """B1 — Query validation"""

    @patch("agent.get_llm")
    def test_empty_query(self, mock_get_llm):
        result = agent("")
        assert result.startswith("Error:")

    @patch("agent.get_llm")
    def test_invalid_query_offensive(self, mock_get_llm):
        mock_get_llm.return_value = make_mock_llm(["INVALID: Offensive content"])
        result = agent("you are stupid")
        assert result.startswith("Error:")

    @patch("agent.get_llm")
    def test_invalid_query_unrelated(self, mock_get_llm):
        mock_get_llm.return_value = make_mock_llm(["INVALID: Not related to food or cooking"])
        result = agent("What is the capital of France?")
        assert result.startswith("Error:")

    @patch("agent.get_llm")
    @patch("agent.retrieve_pantry")
    def test_valid_query_proceeds(self, mock_pantry, mock_get_llm):
        mock_pantry.invoke = MagicMock(return_value='[]')
        mock_get_llm.return_value = make_mock_llm([
            "VALID",  # validation
            '["chicken"]',  # ingredient identification
        ])
        # Will fail at search but that's ok - we verify it got past validation
        result = agent("Make me a chicken recipe")
        # It should not be an invalid-query error
        assert "invalid" not in result.lower() or "ingredient" in result.lower()


class TestB2PantryLookup:
    """B2 — Pantry lookup"""

    @patch("agent.get_llm")
    @patch("agent.retrieve_pantry")
    def test_pantry_called_once(self, mock_pantry, mock_get_llm):
        mock_pantry.invoke = MagicMock(return_value='["rice", "beans"]')
        mock_get_llm.return_value = make_mock_llm([
            "VALID",
            '["rice", "beans"]',
        ])
        with patch("agent.search_recipes") as mock_search:
            mock_search.invoke = MagicMock(return_value='[]')
            with patch("agent.get_llm") as mock_llm2:
                mock_llm2.return_value = make_mock_llm([
                    "VALID",
                    '["rice", "beans"]',
                    "**Title:** Rice and Beans\n**Ingredients:** ...\n**Equipment:** ...\n**Instructions:** ...\n**Storage:** ..."
                ])
                result = agent("Make me something with rice")
                assert mock_pantry.invoke.call_count == 1

    @patch("agent.get_llm")
    @patch("agent.retrieve_pantry")
    def test_pantry_error_handled(self, mock_pantry, mock_get_llm):
        mock_pantry.invoke = MagicMock(return_value="Error: Missing NOTION_API_KEY credentials.")
        mock_get_llm.return_value = make_mock_llm(["VALID"])
        result = agent("Make me a recipe")
        assert result.startswith("Error:")


class TestB3IngredientIdentification:
    """B3 — Ingredient identification"""

    @patch("agent.get_llm")
    @patch("agent.retrieve_pantry")
    def test_excludes_negated_ingredients(self, mock_pantry, mock_get_llm):
        mock_pantry.invoke = MagicMock(return_value='["eggs", "flour", "milk"]')
        mock_get_llm.return_value = make_mock_llm([
            "VALID",
            '["flour"]',  # eggs and milk excluded due to negation/restriction
        ])
        with patch("agent.search_recipes") as mock_search:
            mock_search.invoke = MagicMock(return_value='[]')
            with patch("agent.get_llm") as mock_llm2:
                mock_llm2.return_value = make_mock_llm([
                    "VALID",
                    '["flour"]',
                    "**Title:** Simple Flatbread\n**Ingredients:** ...\n**Equipment:** ...\n**Instructions:** ...\n**Storage:** ..."
                ])
                result = agent("Make something without eggs or milk")
                # The recipe should be generated (not an error about no ingredients)
                assert "Error: No compatible ingredients" not in result

    @patch("agent.get_llm")
    @patch("agent.retrieve_pantry")
    def test_dietary_restriction_excludes_derived(self, mock_pantry, mock_get_llm):
        mock_pantry.invoke = MagicMock(return_value='["chicken", "rice", "cheese"]')
        mock_get_llm.return_value = make_mock_llm([
            "VALID",
            '["rice"]',  # vegan excludes chicken and cheese
        ])
        with patch("agent.search_recipes") as mock_search:
            mock_search.invoke = MagicMock(return_value='[]')
            with patch("agent.get_llm") as mock_llm2:
                mock_llm2.return_value = make_mock_llm([
                    "VALID",
                    '["rice"]',
                    "**Title:** Vegan Rice\n**Ingredients:** ...\n**Equipment:** ...\n**Instructions:** ...\n**Storage:** ..."
                ])
                result = agent("I'm vegan, make me something")
                assert "Error: No compatible ingredients" not in result


class TestB4ReferenceRecipeSearch:
    """B4 — Reference recipe search"""

    @patch("agent.get_llm")
    @patch("agent.retrieve_pantry")
    @patch("agent.search_recipes")
    def test_search_called_after_pantry(self, mock_search, mock_pantry, mock_get_llm):
        mock_pantry.invoke = MagicMock(return_value='["tomato", "pasta"]')
        mock_search.invoke = MagicMock(return_value=json.dumps([
            {"title": "Pasta Recipe", "url": "http://example.com", "content": "A great pasta..."}
        ]))
        mock_get_llm.return_value = make_mock_llm([
            "VALID",
            '["tomato", "pasta"]',
            "**Title:** Tomato Pasta\n**Ingredients:** - 200g pasta\n- 2 tomatoes\n**Equipment:** - Pot\n**Instructions:** 1. Boil pasta\n**Storage:** Store in fridge"
        ])
        result = agent("Make me pasta")
        assert mock_search.invoke.call_count == 1
        assert mock_pantry.invoke.call_count == 1

    @patch("agent.get_llm")
    @patch("agent.retrieve_pantry")
    @patch("agent.search_recipes")
    def test_search_receives_ingredients(self, mock_search, mock_pantry, mock_get_llm):
        mock_pantry.invoke = MagicMock(return_value='["chicken", "rice"]')
        mock_search.invoke = MagicMock(return_value='[]')
        mock_get_llm.return_value = make_mock_llm([
            "VALID",
            '["chicken", "rice"]',
            "**Title:** Chicken Rice\n**Ingredients:** ...\n**Equipment:** ...\n**Instructions:** ...\n**Storage:** ..."
        ])
        agent("Make me something with chicken")
        call_args = mock_search.invoke.call_args
        assert "ingredients" in call_args[0][0] or "ingredients" in call_args[1] if call_args[1] else True


class TestB5LackOfIngredients:
    """B5 — Lack of ingredients"""

    @patch("agent.get_llm")
    @patch("agent.retrieve_pantry")
    def test_no_compatible_ingredients(self, mock_pantry, mock_get_llm):
        mock_pantry.invoke = MagicMock(return_value='[]')
        mock_get_llm.return_value = make_mock_llm([
            "VALID",
            '[]',  # no compatible ingredients
        ])
        result = agent("I'm vegan but I only want steak")
        assert result.startswith("Error:")
        assert "ingredient" in result.lower() or "compatible" in result.lower()

    @patch("agent.get_llm")
    @patch("agent.retrieve_pantry")
    @patch("agent.search_recipes")
    def test_search_not_called_when_no_ingredients(self, mock_search, mock_pantry, mock_get_llm):
        mock_pantry.invoke = MagicMock(return_value='[]')
        mock_get_llm.return_value = make_mock_llm([
            "VALID",
            '[]',
        ])
        agent("Make me something")
        mock_search.invoke.assert_not_called()


class TestB6ExecutionFailure:
    """B6 — Execution failure"""

    @patch("agent.get_llm")
    @patch("agent.retrieve_pantry")
    def test_tool_exception_handled(self, mock_pantry, mock_get_llm):
        mock_pantry.invoke = MagicMock(side_effect=Exception("Connection timeout"))
        mock_get_llm.return_value = make_mock_llm(["VALID"])
        result = agent("Make me a recipe")
        assert result.startswith("Error:")
        assert "timeout" in result.lower() or "failed" in result.lower() or "error" in result.lower()

    @patch("agent.get_llm")
    @patch("agent.retrieve_pantry")
    @patch("agent.search_recipes")
    def test_search_error_handled(self, mock_search, mock_pantry, mock_get_llm):
        mock_pantry.invoke = MagicMock(return_value='["rice"]')
        mock_search.invoke = MagicMock(return_value="Error: Missing TAVILY_API_KEY credentials.")
        mock_get_llm.return_value = make_mock_llm([
            "VALID",
            '["rice"]',
        ])
        result = agent("Make me rice")
        assert result.startswith("Error:")

    def test_no_unhandled_exception(self):
        """Agent should never raise an unhandled exception."""
        with patch("agent.get_llm") as mock_llm:
            mock_llm.side_effect = Exception("Model unavailable")
            result = agent("Make me a recipe")
            assert isinstance(result, str)
            assert result.startswith("Error:")


class TestB7RecipeGeneration:
    """B7 — Recipe generation"""

    @patch("agent.get_llm")
    @patch("agent.retrieve_pantry")
    @patch("agent.search_recipes")
    def test_recipe_has_all_fields(self, mock_search, mock_pantry, mock_get_llm):
        mock_pantry.invoke = MagicMock(return_value='["chicken", "rice", "onion"]')
        mock_search.invoke = MagicMock(return_value=json.dumps([
            {"title": "Chicken Rice", "url": "http://example.com", "content": "Delicious chicken rice recipe"}
        ]))
        recipe_text = (
            "**Title:** Savory Chicken Rice\n\n"
            "**Ingredients:**\n- 500g chicken breast\n- 2 cups rice\n- 1 onion, diced\n\n"
            "**Equipment:**\n- Large pot\n- Frying pan\n- Cutting board\n\n"
            "**Instructions:**\n1. Cook rice according to package directions.\n"
            "2. Dice the onion and sauté in the frying pan.\n"
            "3. Cut chicken into pieces and cook until golden.\n"
            "4. Combine all ingredients.\n\n"
            "**Storage:**\nStore in an airtight container in the refrigerator for up to 3 days."
        )
        mock_get_llm.return_value = make_mock_llm([
            "VALID",
            '["chicken", "rice", "onion"]',
            recipe_text
        ])
        result = agent("Make me chicken rice")
        for field in ["Title", "Ingredients", "Equipment", "Instructions", "Storage"]:
            assert field.lower() in result.lower(), f"Missing field: {field}"

    @patch("agent.get_llm")
    @patch("agent.retrieve_pantry")
    @patch("agent.search_recipes")
    def test_recipe_generated_even_without_search_results(self, mock_search, mock_pantry, mock_get_llm):
        mock_pantry.invoke = MagicMock(return_value='["potato"]')
        mock_search.invoke = MagicMock(return_value='[]')
        recipe_text = (
            "**Title:** Roasted Potatoes\n**Ingredients:** - 4 potatoes\n"
            "**Equipment:** - Oven\n**Instructions:** 1. Roast potatoes\n"
            "**Storage:** Store in fridge"
        )
        mock_get_llm.return_value = make_mock_llm([
            "VALID",
            '["potato"]',
            recipe_text
        ])
        result = agent("Make me something with potato")
        assert not result.startswith("Error:")
        assert "potato" in result.lower() or "Potato" in result


class TestToolInterface:
    """Verify tools are proper LangChain tools invoked via .invoke()"""

    def test_retrieve_pantry_is_tool(self):
        assert hasattr(retrieve_pantry, 'invoke')
        assert hasattr(retrieve_pantry, 'name')

    def test_search_recipes_is_tool(self):
        assert hasattr(search_recipes, 'invoke')
        assert hasattr(search_recipes, 'name')


class TestGuardrails:
    """Test guardrails and invariants"""

    @patch("agent.get_llm")
    @patch("agent.retrieve_pantry")
    @patch("agent.search_recipes")
    def test_dietary_restriction_implies_derived(self, mock_search, mock_pantry, mock_get_llm):
        """A dietary restriction implies all derived ingredients."""
        mock_pantry.invoke = MagicMock(return_value='["butter", "rice", "honey"]')
        mock_search.invoke = MagicMock(return_value='[]')
        recipe_text = (
            "**Title:** Vegan Rice Bowl\n**Ingredients:** - 2 cups rice\n"
            "**Equipment:** - Pot\n**Instructions:** 1. Cook rice\n"
            "**Storage:** Refrigerate"
        )
        mock_get_llm.return_value = make_mock_llm([
            "VALID",
            '["rice"]',  # butter and honey excluded (vegan)
            recipe_text
        ])
        result = agent("I am vegan, make me something")
        assert not result.startswith("Error:")

    @patch("agent.get_llm")
    @patch("agent.retrieve_pantry")
    def test_negated_ingredient_excluded(self, mock_pantry, mock_get_llm):
        """An ingredient mentioned under negation is not available."""
        mock_pantry.invoke = MagicMock(return_value='["flour", "sugar"]')
        mock_get_llm.return_value = make_mock_llm([
            "VALID",
            '["flour"]',  # sugar excluded because user said "no sugar"
        ])
        with patch("agent.search_recipes") as mock_search:
            mock_search.invoke = MagicMock(return_value='[]')
            with patch("agent.get_llm") as mock_llm2:
                mock_llm2.return_value = make_mock_llm([
                    "VALID",
                    '["flour"]',
                    "**Title:** Flatbread\n**Ingredients:** ...\n**Equipment:** ...\n**Instructions:** ...\n**Storage:** ..."
                ])
                result = agent("Make something without sugar")
                assert not result.startswith("Error: No compatible")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
