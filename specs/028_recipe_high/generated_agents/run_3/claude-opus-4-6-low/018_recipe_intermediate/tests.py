import json
import os
import pytest
from unittest.mock import patch, MagicMock

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily")
os.environ.setdefault("NOTION_API_KEY", "test-notion")
os.environ.setdefault("NOTION_PAGE_ID", "test-page-id")


def mock_llm_response(content):
    resp = MagicMock()
    resp.content = content
    return resp


# ---- B1: Query validation ----

class TestB1QueryValidation:
    @patch("agent.llm")
    def test_empty_query(self, mock_llm):
        from agent import agent
        result = agent("")
        assert result.startswith("Error:")

    @patch("agent.llm")
    def test_invalid_query_offensive(self, mock_llm):
        from agent import agent
        mock_llm.invoke = MagicMock(return_value=mock_llm_response("INVALID"))
        result = agent("you are stupid")
        assert result.startswith("Error:")

    @patch("agent.llm")
    def test_valid_query(self, mock_llm):
        from agent import agent

        def side_effect(prompt):
            p = str(prompt)
            if "VALID or INVALID" in p:
                return mock_llm_response("VALID")
            if "extract" in p.lower() or "ingredients" in p.lower():
                return mock_llm_response('{"ingredients": ["chicken", "rice"], "restrictions": [], "excluded_by_restriction": []}')
            if "chef" in p.lower() or "recipe" in p.lower():
                return mock_llm_response("**Title:** Chicken Rice\n\n**Ingredients:**\n- 500g chicken\n\n**Equipment:**\n- Pan\n\n**Instructions:**\n1. Cook\n\n**Storage:**\n- Refrigerate")
            return mock_llm_response("ok")

        mock_llm.invoke = MagicMock(side_effect=side_effect)

        with patch("agent.search_recipes") as mock_search:
            mock_search.invoke = MagicMock(return_value='[]')
            result = agent("Make me a chicken and rice dish")
            assert "Title" in result or "Error" not in result or result.startswith("Error") is False


# ---- B2: Ingredient identification ----

class TestB2IngredientIdentification:
    @patch("agent.llm")
    def test_identifies_ingredients(self, mock_llm):
        from agent import identify_ingredients
        mock_llm.invoke = MagicMock(return_value=mock_llm_response(
            '{"ingredients": ["tomato", "onion"], "restrictions": [], "excluded_by_restriction": []}'
        ))
        result = identify_ingredients({"query": "I have tomato and onion, make me something"})
        assert result["ingredients"] == ["tomato", "onion"]

    @patch("agent.llm")
    def test_negated_ingredients_excluded(self, mock_llm):
        from agent import identify_ingredients
        mock_llm.invoke = MagicMock(return_value=mock_llm_response(
            '{"ingredients": ["rice"], "restrictions": [], "excluded_by_restriction": ["eggs"]}'
        ))
        result = identify_ingredients({"query": "I have rice but no eggs"})
        assert "eggs" not in result["ingredients"]

    @patch("agent.llm")
    def test_no_ingredients_returns_empty(self, mock_llm):
        from agent import identify_ingredients
        mock_llm.invoke = MagicMock(return_value=mock_llm_response(
            '{"ingredients": [], "restrictions": ["vegan"], "excluded_by_restriction": []}'
        ))
        result = identify_ingredients({"query": "I want a vegan meal"})
        assert result["ingredients"] == []


# ---- B3: Reference recipe search ----

class TestB3SearchRecipes:
    @patch("agent.search_recipes")
    @patch("agent.llm")
    def test_search_called_with_ingredients(self, mock_llm, mock_search):
        from agent import search_recipes_node
        mock_search.invoke = MagicMock(return_value=json.dumps([
            {"title": "Recipe 1", "url": "http://example.com", "content": "Delicious recipe"}
        ]))
        result = search_recipes_node({"ingredients": ["chicken", "rice"], "search_results": [], "error": ""})
        assert len(result["search_results"]) == 1

    @patch("agent.search_recipes")
    def test_search_error_handled(self, mock_search):
        from agent import search_recipes_node
        mock_search.invoke = MagicMock(return_value="erro: connection timeout")
        result = search_recipes_node({"ingredients": ["chicken"], "search_results": [], "error": ""})
        assert result["error"].startswith("Error:")


# ---- B4: Pantry lookup ----

class TestB4PantryLookup:
    @patch("agent.llm")
    @patch("agent.retrieve_pantry")
    def test_pantry_retrieval(self, mock_pantry, mock_llm):
        from agent import pantry_lookup_node
        mock_pantry.invoke = MagicMock(return_value=json.dumps(["flour", "sugar", "eggs"]))
        mock_llm.invoke = MagicMock(return_value=mock_llm_response('["flour", "sugar", "eggs"]'))
        result = pantry_lookup_node({"query": "Make me something", "pantry_ingredients": [], "error": ""})
        assert len(result["pantry_ingredients"]) == 3

    @patch("agent.retrieve_pantry")
    def test_pantry_error(self, mock_pantry):
        from agent import pantry_lookup_node
        mock_pantry.invoke = MagicMock(return_value="Error: Missing Notion credentials.")
        result = pantry_lookup_node({"query": "Make me something", "pantry_ingredients": [], "error": ""})
        assert result["error"].startswith("Error:")

    @patch("agent.llm")
    @patch("agent.retrieve_pantry")
    def test_pantry_empty(self, mock_pantry, mock_llm):
        from agent import pantry_lookup_node
        mock_pantry.invoke = MagicMock(return_value="[]")
        result = pantry_lookup_node({"query": "Make me something", "pantry_ingredients": [], "error": ""})
        assert result["pantry_ingredients"] == []


# ---- B5: Lack of ingredients ----

class TestB5LackOfIngredients:
    def test_no_ingredients_error(self):
        from agent import no_ingredients_node
        result = no_ingredients_node({})
        assert result["error"].startswith("Error:")


# ---- B6: Execution failure ----

class TestB6ExecutionFailure:
    @patch("agent.llm")
    def test_agent_handles_exception(self, mock_llm):
        from agent import agent
        mock_llm.invoke = MagicMock(side_effect=Exception("Model failure"))
        result = agent("Make me pasta")
        assert result.startswith("Error:")


# ---- B7: Recipe generation ----

class TestB7RecipeGeneration:
    @patch("agent.llm")
    def test_recipe_has_all_fields(self, mock_llm):
        from agent import generate_recipe_node
        recipe_text = """**Title:** Tomato Pasta

**Ingredients:**
- 200g pasta
- 2 tomatoes

**Equipment:**
- Large pot
- Pan

**Instructions:**
1. Boil pasta
2. Make sauce

**Storage:**
- Store in airtight container in fridge for up to 3 days."""
        mock_llm.invoke = MagicMock(return_value=mock_llm_response(recipe_text))
        result = generate_recipe_node({
            "query": "pasta with tomatoes",
            "ingredients": ["pasta", "tomatoes"],
            "pantry_ingredients": [],
            "search_results": [],
        })
        assert "Title" in result["recipe"]
        assert "Ingredients" in result["recipe"]
        assert "Equipment" in result["recipe"]
        assert "Instructions" in result["recipe"]
        assert "Storage" in result["recipe"]

    @patch("agent.llm")
    def test_recipe_generation_failure(self, mock_llm):
        from agent import generate_recipe_node
        mock_llm.invoke = MagicMock(side_effect=Exception("API error"))
        result = generate_recipe_node({
            "query": "make pasta",
            "ingredients": ["pasta"],
            "pantry_ingredients": [],
            "search_results": [],
        })
        assert result["error"].startswith("Error:")


# ---- Integration-like tests ----

class TestRouting:
    def test_route_with_ingredients_goes_to_search(self):
        from agent import route_after_ingredients
        assert route_after_ingredients({"ingredients": ["chicken"]}) == "search_recipes_node"

    def test_route_without_ingredients_goes_to_pantry(self):
        from agent import route_after_ingredients
        assert route_after_ingredients({"ingredients": []}) == "pantry_lookup_node"

    def test_route_after_pantry_empty(self):
        from agent import route_after_pantry
        assert route_after_pantry({"pantry_ingredients": [], "error": ""}) == "no_ingredients_node"

    def test_route_after_pantry_with_ingredients(self):
        from agent import route_after_pantry
        assert route_after_pantry({"pantry_ingredients": ["flour"], "error": ""}) == "generate_recipe_node"

    def test_route_after_pantry_error(self):
        from agent import route_after_pantry
        assert route_after_pantry({"error": "Error: something"}) == "error_node"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
