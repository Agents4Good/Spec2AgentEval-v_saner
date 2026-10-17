import json
import os
import pytest
from unittest.mock import patch, MagicMock

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily")
os.environ.setdefault("NOTION_API_KEY", "test-notion")
os.environ.setdefault("NOTION_PAGE_ID", "test-page-id")

from agent import agent, retrieve_pantry, search_recipes, validate_query, identify_ingredients


def _mock_llm_response(content):
    mock = MagicMock()
    mock.content = content
    mock.tool_calls = []
    return mock


class TestB1QueryValidation:
    """B1: Query validation."""

    @patch("agent.ChatOpenAI")
    def test_empty_query(self, mock_cls):
        result = agent("")
        assert result["is_valid"] is False
        assert result["error_message"].startswith("Error:")
        assert result["recipe"] == {}

    @patch("agent.ChatOpenAI")
    def test_invalid_query(self, mock_cls):
        instance = mock_cls.return_value
        instance.invoke.return_value = _mock_llm_response('{"valid": false, "reason": "Not related to food."}')
        instance.bind_tools = MagicMock(return_value=instance)
        result = agent("What is the capital of France?")
        assert result["is_valid"] is False
        assert result["error_message"].startswith("Error:")

    @patch("agent.ChatOpenAI")
    def test_valid_query_proceeds(self, mock_cls):
        instance = mock_cls.return_value
        responses = [
            _mock_llm_response('{"valid": true}'),
            _mock_llm_response('{"ingredients": ["chicken", "rice"], "restrictions": [], "preferences": ""}'),
        ]
        # For search_recipes_node, we need bind_tools
        bound = MagicMock()
        bound_resp = _mock_llm_response("no tool call")
        bound_resp.tool_calls = []
        bound.invoke.return_value = bound_resp
        instance.bind_tools = MagicMock(return_value=bound)

        call_count = [0]
        def side_effect(*a, **k):
            idx = call_count[0]
            call_count[0] += 1
            if idx < len(responses):
                return responses[idx]
            return _mock_llm_response('{"title":"Test","ingredients":["1 cup rice"],"equipment":["pot"],"instructions":["Cook"],"storage":"Fridge"}')
        instance.invoke.side_effect = side_effect

        with patch("agent.search_recipes") as mock_tool:
            mock_tool.invoke.return_value = json.dumps([{"title": "Test Recipe", "url": "http://test.com", "content": "A recipe"}])
            result = agent("Make me a chicken and rice dish")

        assert result["is_valid"] is True


class TestB2IngredientIdentification:
    """B2: Ingredient identification."""

    @patch("agent.ChatOpenAI")
    def test_identifies_ingredients(self, mock_cls):
        instance = mock_cls.return_value
        instance.invoke.side_effect = [
            _mock_llm_response('{"valid": true}'),
            _mock_llm_response('{"ingredients": ["tomato", "basil"], "restrictions": [], "preferences": "Italian"}'),
            _mock_llm_response('{"title":"Pasta","ingredients":["2 tomatoes"],"equipment":["pan"],"instructions":["Cook"],"storage":"Fridge"}'),
        ]
        bound = MagicMock()
        bound.invoke.return_value = _mock_llm_response("[]")
        bound.invoke.return_value.tool_calls = []
        instance.bind_tools = MagicMock(return_value=bound)

        with patch("agent.search_recipes") as mock_tool:
            mock_tool.invoke.return_value = "[]"
            result = agent("I have tomato and basil, make something Italian")

        assert "tomato" in result["identified_ingredients"]
        assert "basil" in result["identified_ingredients"]

    @patch("agent.ChatOpenAI")
    def test_negated_ingredients_excluded(self, mock_cls):
        instance = mock_cls.return_value
        instance.invoke.side_effect = [
            _mock_llm_response('{"valid": true}'),
            _mock_llm_response('{"ingredients": [], "restrictions": ["vegan"], "preferences": ""}'),
        ]
        bound = MagicMock()
        pantry_resp = _mock_llm_response("no tool call")
        pantry_resp.tool_calls = []
        bound.invoke.return_value = pantry_resp
        instance.bind_tools = MagicMock(return_value=bound)

        with patch("agent.retrieve_pantry") as mock_tool:
            mock_tool.invoke.return_value = json.dumps(["tofu", "soy milk"])
            result = agent("I don't have any ingredients, vegan please")

        assert result["identified_ingredients"] == []


class TestB3SearchRecipes:
    """B3: Reference recipe search."""

    @patch("agent.ChatOpenAI")
    def test_search_called_with_ingredients(self, mock_cls):
        instance = mock_cls.return_value
        instance.invoke.side_effect = [
            _mock_llm_response('{"valid": true}'),
            _mock_llm_response('{"ingredients": ["chicken"], "restrictions": [], "preferences": ""}'),
            _mock_llm_response('{"title":"Chicken","ingredients":["1 chicken"],"equipment":["oven"],"instructions":["Bake"],"storage":"Fridge"}'),
        ]
        bound = MagicMock()
        bound_resp = _mock_llm_response("call tool")
        bound_resp.tool_calls = []
        bound.invoke.return_value = bound_resp
        instance.bind_tools = MagicMock(return_value=bound)

        with patch("agent.search_recipes") as mock_tool:
            mock_tool.invoke.return_value = json.dumps([{"title": "Roast Chicken", "url": "http://t.com", "content": "Roast it"}])
            result = agent("I have chicken, make something")

        assert "search_recipes" in result.get("tool_calls_made", [])
        assert "retrieve_pantry" not in result.get("tool_calls_made", [])


class TestB4PantryLookup:
    """B4: Pantry lookup when no ingredients in query."""

    @patch("agent.ChatOpenAI")
    def test_pantry_called_when_no_ingredients(self, mock_cls):
        instance = mock_cls.return_value
        instance.invoke.side_effect = [
            _mock_llm_response('{"valid": true}'),
            _mock_llm_response('{"ingredients": [], "restrictions": [], "preferences": "something tasty"}'),
            _mock_llm_response('{"title":"Pantry Meal","ingredients":["1 can beans"],"equipment":["pot"],"instructions":["Heat"],"storage":"Fridge"}'),
        ]
        bound = MagicMock()
        bound_resp = _mock_llm_response("call tool")
        bound_resp.tool_calls = []
        bound.invoke.return_value = bound_resp
        instance.bind_tools = MagicMock(return_value=bound)

        with patch("agent.retrieve_pantry") as mock_tool:
            mock_tool.invoke.return_value = json.dumps(["beans", "rice", "onion"])
            result = agent("Make me something tasty")

        assert "retrieve_pantry" in result.get("tool_calls_made", [])
        assert "search_recipes" not in result.get("tool_calls_made", [])


class TestB5LackOfIngredients:
    """B5: Error when pantry has no compatible ingredients."""

    @patch("agent.ChatOpenAI")
    def test_empty_pantry_error(self, mock_cls):
        instance = mock_cls.return_value
        instance.invoke.side_effect = [
            _mock_llm_response('{"valid": true}'),
            _mock_llm_response('{"ingredients": [], "restrictions": ["vegan"], "preferences": ""}'),
        ]
        bound = MagicMock()
        bound_resp = _mock_llm_response("call tool")
        bound_resp.tool_calls = []
        bound.invoke.return_value = bound_resp
        instance.bind_tools = MagicMock(return_value=bound)

        with patch("agent.retrieve_pantry") as mock_tool:
            mock_tool.invoke.return_value = "[]"
            result = agent("Make me a vegan dish")

        assert result["error_message"].startswith("Error:")
        assert result["recipe"] == {}


class TestB6ExecutionFailure:
    """B6: Handle tool/model failures gracefully."""

    @patch("agent.ChatOpenAI")
    def test_search_error_handled(self, mock_cls):
        instance = mock_cls.return_value
        instance.invoke.side_effect = [
            _mock_llm_response('{"valid": true}'),
            _mock_llm_response('{"ingredients": ["chicken"], "restrictions": [], "preferences": ""}'),
        ]
        bound = MagicMock()
        bound_resp = _mock_llm_response("call tool")
        bound_resp.tool_calls = []
        bound.invoke.return_value = bound_resp
        instance.bind_tools = MagicMock(return_value=bound)

        with patch("agent.search_recipes") as mock_tool:
            mock_tool.invoke.return_value = "erro: Connection timeout"
            result = agent("I have chicken")

        assert result["error_message"].startswith("Error:")
        assert result["recipe"] == {}

    def test_unhandled_exception_caught(self):
        with patch("agent._build_graph", side_effect=Exception("Unexpected crash")):
            with patch("agent._compiled_graph", None):
                result = agent("test query")
                assert result["error_message"].startswith("Error:")


class TestB7RecipeGeneration:
    """B7: Recipe has all required fields."""

    @patch("agent.ChatOpenAI")
    def test_recipe_has_all_fields(self, mock_cls):
        instance = mock_cls.return_value
        recipe_json = json.dumps({
            "title": "Chicken Stir Fry",
            "ingredients": ["200g chicken breast", "1 cup rice", "2 tbsp soy sauce"],
            "equipment": ["wok", "rice cooker"],
            "instructions": ["Step 1: Cook rice", "Step 2: Stir fry chicken"],
            "storage": "Refrigerate in sealed container for up to 3 days.",
        })
        instance.invoke.side_effect = [
            _mock_llm_response('{"valid": true}'),
            _mock_llm_response('{"ingredients": ["chicken", "rice"], "restrictions": [], "preferences": ""}'),
            _mock_llm_response(recipe_json),
        ]
        bound = MagicMock()
        bound_resp = _mock_llm_response("call")
        bound_resp.tool_calls = []
        bound.invoke.return_value = bound_resp
        instance.bind_tools = MagicMock(return_value=bound)

        with patch("agent.search_recipes") as mock_tool:
            mock_tool.invoke.return_value = "[]"
            result = agent("Make chicken and rice")

        recipe = result["recipe"]
        assert recipe.get("title")
        assert len(recipe.get("ingredients", [])) > 0
        assert len(recipe.get("equipment", [])) > 0
        assert len(recipe.get("instructions", [])) > 0
        assert recipe.get("storage")


class TestToolsDirectly:
    """Test tools in isolation."""

    @patch("requests.get")
    def test_retrieve_pantry_missing_creds(self, mock_get):
        with patch.dict(os.environ, {"NOTION_API_KEY": "", "NOTION_PAGE_ID": ""}, clear=False):
            result = retrieve_pantry.invoke({})
        assert result.startswith("Error:")

    @patch("requests.get")
    def test_retrieve_pantry_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"plain_text": "tomato"}]}},
                {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"plain_text": "onion"}]}},
            ]
        }
        mock_get.return_value = mock_resp

        with patch.dict(os.environ, {"NOTION_API_KEY": "key", "NOTION_PAGE_ID": "pid"}):
            result = retrieve_pantry.invoke({})
        data = json.loads(result)
        assert "tomato" in data
        assert "onion" in data

    def test_search_recipes_missing_key(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=False):
            result = search_recipes.invoke({"ingredients": "chicken"})
        assert result.startswith("erro:")
