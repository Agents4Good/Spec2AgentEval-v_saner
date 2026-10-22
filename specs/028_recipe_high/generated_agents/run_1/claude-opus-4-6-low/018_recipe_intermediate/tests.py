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


def mock_llm_response(content):
    resp = MagicMock()
    resp.content = content
    return resp


class TestB1QueryValidation:
    """B1 — Query validation"""

    @patch("agent.ChatOpenAI")
    def test_empty_query(self, mock_chat):
        result = agent("")
        assert result.startswith("Error:")

    @patch("agent.ChatOpenAI")
    def test_invalid_query_offensive(self, mock_chat):
        instance = mock_chat.return_value
        instance.invoke = MagicMock(side_effect=[
            mock_llm_response('{"valid": false, "reason": "Offensive content."}'),
        ])
        result = agent("you are stupid")
        assert result.startswith("Error:")

    @patch("agent.ChatOpenAI")
    def test_invalid_query_unrelated(self, mock_chat):
        instance = mock_chat.return_value
        instance.invoke = MagicMock(side_effect=[
            mock_llm_response('{"valid": false, "reason": "Unrelated to food."}'),
        ])
        result = agent("What is the capital of France?")
        assert result.startswith("Error:")

    @patch("agent.ChatOpenAI")
    @patch("agent.search_recipes")
    def test_valid_query(self, mock_search, mock_chat):
        instance = mock_chat.return_value
        instance.invoke = MagicMock(side_effect=[
            mock_llm_response('{"valid": true, "reason": ""}'),
            mock_llm_response('{"ingredients": ["chicken", "rice"], "restrictions": []}'),
            mock_llm_response("**Title:** Chicken Rice\n\n**Ingredients:**\n- 1 chicken\n\n**Equipment:**\n- pot\n\n**Instructions:**\n1. Cook\n\n**Storage:**\nFridge"),
        ])
        mock_search.invoke = MagicMock(return_value='[{"title": "Chicken Rice", "url": "http://example.com", "content": "A recipe"}]')
        result = agent("Make me a chicken and rice dish")
        assert not result.startswith("Error:")


class TestB2IngredientIdentification:
    """B2 — Ingredient identification"""

    @patch("agent.ChatOpenAI")
    @patch("agent.search_recipes")
    def test_identifies_ingredients(self, mock_search, mock_chat):
        instance = mock_chat.return_value
        instance.invoke = MagicMock(side_effect=[
            mock_llm_response('{"valid": true, "reason": ""}'),
            mock_llm_response('{"ingredients": ["tomato", "basil"], "restrictions": []}'),
            mock_llm_response("**Title:** T\n\n**Ingredients:**\n- tomato\n\n**Equipment:**\n- knife\n\n**Instructions:**\n1. Cut\n\n**Storage:**\nFridge"),
        ])
        mock_search.invoke = MagicMock(return_value='[]')
        result = agent("I have tomato and basil, make a salad")
        assert not result.startswith("Error:")

    @patch("agent.ChatOpenAI")
    @patch("agent.retrieve_pantry")
    def test_negated_ingredients_excluded(self, mock_pantry, mock_chat):
        instance = mock_chat.return_value
        instance.invoke = MagicMock(side_effect=[
            mock_llm_response('{"valid": true, "reason": ""}'),
            mock_llm_response('{"ingredients": [], "restrictions": ["no eggs"]}'),
            # pantry call
            mock_llm_response('["flour", "sugar"]'),
            # recipe generation
            mock_llm_response("**Title:** Cake\n\n**Ingredients:**\n- flour\n\n**Equipment:**\n- bowl\n\n**Instructions:**\n1. Mix\n\n**Storage:**\nRoom temp"),
        ])
        mock_pantry.invoke = MagicMock(return_value='["flour", "sugar", "eggs"]')
        result = agent("I don't have eggs, make something")
        assert not result.startswith("Error:")

    @patch("agent.ChatOpenAI")
    @patch("agent.retrieve_pantry")
    def test_dietary_restriction_filters_derived(self, mock_pantry, mock_chat):
        instance = mock_chat.return_value
        instance.invoke = MagicMock(side_effect=[
            mock_llm_response('{"valid": true, "reason": ""}'),
            mock_llm_response('{"ingredients": [], "restrictions": ["vegan"]}'),
            mock_llm_response('["rice", "beans"]'),
            mock_llm_response("**Title:** Vegan Bowl\n\n**Ingredients:**\n- rice\n\n**Equipment:**\n- pot\n\n**Instructions:**\n1. Cook\n\n**Storage:**\nFridge"),
        ])
        mock_pantry.invoke = MagicMock(return_value='["rice", "beans", "cheese", "butter"]')
        result = agent("I'm vegan, make me something")
        assert not result.startswith("Error:")


class TestB3SearchRecipes:
    """B3 — Reference recipe search"""

    @patch("agent.ChatOpenAI")
    @patch("agent.search_recipes")
    def test_search_called_with_ingredients(self, mock_search, mock_chat):
        instance = mock_chat.return_value
        instance.invoke = MagicMock(side_effect=[
            mock_llm_response('{"valid": true, "reason": ""}'),
            mock_llm_response('{"ingredients": ["pasta", "tomato"], "restrictions": []}'),
            mock_llm_response("**Title:** Pasta\n\n**Ingredients:**\n- pasta\n\n**Equipment:**\n- pot\n\n**Instructions:**\n1. Boil\n\n**Storage:**\nFridge"),
        ])
        mock_search.invoke = MagicMock(return_value='[{"title": "Pasta", "url": "http://ex.com", "content": "Good"}]')
        result = agent("Make pasta with tomato")
        mock_search.invoke.assert_called_once()


class TestB4PantryLookup:
    """B4 — Pantry lookup"""

    @patch("agent.ChatOpenAI")
    @patch("agent.retrieve_pantry")
    def test_pantry_called_when_no_ingredients(self, mock_pantry, mock_chat):
        instance = mock_chat.return_value
        instance.invoke = MagicMock(side_effect=[
            mock_llm_response('{"valid": true, "reason": ""}'),
            mock_llm_response('{"ingredients": [], "restrictions": []}'),
            mock_llm_response('["chicken", "onion"]'),
            mock_llm_response("**Title:** Soup\n\n**Ingredients:**\n- chicken\n\n**Equipment:**\n- pot\n\n**Instructions:**\n1. Boil\n\n**Storage:**\nFridge"),
        ])
        mock_pantry.invoke = MagicMock(return_value='["chicken", "onion"]')
        result = agent("Make me something for dinner")
        mock_pantry.invoke.assert_called_once()
        assert not result.startswith("Error:")


class TestB5LackOfIngredients:
    """B5 — Lack of ingredients"""

    @patch("agent.ChatOpenAI")
    @patch("agent.retrieve_pantry")
    def test_empty_pantry(self, mock_pantry, mock_chat):
        instance = mock_chat.return_value
        instance.invoke = MagicMock(side_effect=[
            mock_llm_response('{"valid": true, "reason": ""}'),
            mock_llm_response('{"ingredients": [], "restrictions": []}'),
        ])
        mock_pantry.invoke = MagicMock(return_value='[]')
        result = agent("Make me something")
        assert result.startswith("Error:")


class TestB6ExecutionFailure:
    """B6 — Execution failure"""

    @patch("agent.ChatOpenAI")
    @patch("agent.search_recipes")
    def test_search_error(self, mock_search, mock_chat):
        instance = mock_chat.return_value
        instance.invoke = MagicMock(side_effect=[
            mock_llm_response('{"valid": true, "reason": ""}'),
            mock_llm_response('{"ingredients": ["chicken"], "restrictions": []}'),
        ])
        mock_search.invoke = MagicMock(return_value="erro: Tavily API timeout")
        result = agent("Make chicken soup")
        assert result.startswith("Error:")

    @patch("agent.ChatOpenAI")
    @patch("agent.retrieve_pantry")
    def test_pantry_error(self, mock_pantry, mock_chat):
        instance = mock_chat.return_value
        instance.invoke = MagicMock(side_effect=[
            mock_llm_response('{"valid": true, "reason": ""}'),
            mock_llm_response('{"ingredients": [], "restrictions": []}'),
        ])
        mock_pantry.invoke = MagicMock(return_value="Error: Notion API returned status 401.")
        result = agent("Make me dinner")
        assert result.startswith("Error:")

    @patch("agent.ChatOpenAI")
    def test_llm_exception(self, mock_chat):
        mock_chat.return_value.invoke = MagicMock(side_effect=Exception("LLM down"))
        result = agent("Make me pasta")
        assert result.startswith("Error:")


class TestB7RecipeGeneration:
    """B7 — Recipe generation"""

    @patch("agent.ChatOpenAI")
    @patch("agent.search_recipes")
    def test_recipe_has_all_fields(self, mock_search, mock_chat):
        recipe_text = (
            "**Title:** Tomato Basil Pasta\n\n"
            "**Ingredients:**\n- 200g pasta\n- 2 tomatoes\n\n"
            "**Equipment:**\n- Large pot\n- Pan\n\n"
            "**Instructions:**\n1. Boil pasta\n2. Sauté tomatoes\n\n"
            "**Storage:**\nStore in airtight container in the fridge for up to 3 days."
        )
        instance = mock_chat.return_value
        instance.invoke = MagicMock(side_effect=[
            mock_llm_response('{"valid": true, "reason": ""}'),
            mock_llm_response('{"ingredients": ["pasta", "tomato"], "restrictions": []}'),
            mock_llm_response(recipe_text),
        ])
        mock_search.invoke = MagicMock(return_value='[]')
        result = agent("Make pasta with tomato")
        assert "**Title:**" in result
        assert "**Ingredients:**" in result
        assert "**Equipment:**" in result
        assert "**Instructions:**" in result
        assert "**Storage:**" in result


class TestTools:
    """Test tool interfaces directly"""

    @patch("requests.get")
    def test_retrieve_pantry_success(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"results": [
                {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "chicken"}]}},
                {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "rice"}]}},
            ]}
        )
        result = retrieve_pantry.invoke({})
        data = json.loads(result)
        assert "chicken" in data
        assert "rice" in data

    @patch("requests.get")
    def test_retrieve_pantry_missing_creds(self, mock_get):
        old = os.environ.pop("NOTION_API_KEY", None)
        try:
            result = retrieve_pantry.invoke({})
            assert result.startswith("Error:")
        finally:
            if old:
                os.environ["NOTION_API_KEY"] = old

    def test_search_recipes_missing_key(self):
        old = os.environ.pop("TAVILY_API_KEY", None)
        try:
            result = search_recipes.invoke({"ingredients": "chicken"})
            assert result.startswith("erro:")
        finally:
            if old:
                os.environ["TAVILY_API_KEY"] = old
