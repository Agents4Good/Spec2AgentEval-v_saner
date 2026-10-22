import os
import pytest
from unittest.mock import patch, MagicMock
from agent import agent, validate_query, identify_ingredients, search_recipes_node, pantry_lookup_node, AgentState


def make_state(**kwargs) -> AgentState:
    defaults: AgentState = {
        "query": "",
        "is_valid": False,
        "error_message": "",
        "ingredients": [],
        "pantry_ingredients": [],
        "search_results": "",
        "recipe": "",
        "messages": [],
        "tools_called": [],
        "current_step": "start"
    }
    defaults.update(kwargs)
    return defaults


# --- B1: Query validation ---

class TestB1QueryValidation:
    def test_empty_query(self):
        result = agent("")
        assert result.startswith("Error:")

    @patch("agent.get_llm")
    def test_invalid_query_offensive(self, mock_llm):
        mock_resp = MagicMock()
        mock_resp.content = "invalid"
        mock_llm.return_value.invoke.return_value = mock_resp
        state = make_state(query="you are stupid")
        result = validate_query(state)
        assert result["is_valid"] is False
        assert result["error_message"].startswith("Error:")

    @patch("agent.get_llm")
    def test_valid_query(self, mock_llm):
        mock_resp = MagicMock()
        mock_resp.content = "valid"
        mock_llm.return_value.invoke.return_value = mock_resp
        state = make_state(query="I want a pasta recipe with tomatoes")
        result = validate_query(state)
        assert result["is_valid"] is True


# --- B2: Ingredient identification ---

class TestB2IngredientIdentification:
    @patch("agent.get_llm")
    def test_identifies_ingredients(self, mock_llm):
        mock_resp = MagicMock()
        mock_resp.content = "INGREDIENTS: tomato, basil, pasta\nRESTRICTIONS: NONE"
        mock_llm.return_value.invoke.return_value = mock_resp
        state = make_state(query="Make a recipe with tomato, basil, and pasta")
        result = identify_ingredients(state)
        assert "tomato" in result["ingredients"]
        assert "basil" in result["ingredients"]

    @patch("agent.get_llm")
    def test_no_ingredients(self, mock_llm):
        mock_resp = MagicMock()
        mock_resp.content = "INGREDIENTS: NONE\nRESTRICTIONS: vegan"
        mock_llm.return_value.invoke.return_value = mock_resp
        state = make_state(query="Make me a vegan recipe")
        result = identify_ingredients(state)
        assert result["ingredients"] == []

    @patch("agent.get_llm")
    def test_excludes_negated_ingredients(self, mock_llm):
        mock_resp = MagicMock()
        mock_resp.content = "INGREDIENTS: rice, vegetables\nRESTRICTIONS: NONE"
        mock_llm.return_value.invoke.return_value = mock_resp
        state = make_state(query="Recipe with rice and vegetables but without chicken")
        result = identify_ingredients(state)
        assert "chicken" not in result["ingredients"]


# --- B3: Reference recipe search ---

class TestB3SearchRecipes:
    @patch("agent.ToolNode")
    @patch("agent.get_llm")
    def test_search_called_with_ingredients(self, mock_llm, mock_tool_node):
        mock_tool_call = MagicMock()
        mock_tool_call.tool_calls = [{"name": "search_recipes", "args": {"ingredients": "tomato, basil"}, "id": "1"}]
        mock_tool_call.content = ""
        mock_llm.return_value.bind_tools.return_value.invoke.return_value = mock_tool_call

        tool_msg = MagicMock(spec=["content"])
        tool_msg.content = "[{'title': 'Test', 'url': 'http://test.com', 'content': 'test'}]"
        # Make isinstance check work for ToolMessage
        from langchain_core.messages import ToolMessage as TM
        tool_msg.__class__ = TM

        mock_tool_node.return_value.invoke.return_value = {"messages": [tool_msg]}

        state = make_state(query="recipe with tomato and basil", ingredients=["tomato", "basil"])
        result = search_recipes_node(state)
        assert "search_recipes" in result["tools_called"]
        assert result["current_step"] == "generate_recipe"

    @patch("agent.ToolNode")
    @patch("agent.get_llm")
    def test_search_error(self, mock_llm, mock_tool_node):
        mock_tool_call = MagicMock()
        mock_tool_call.tool_calls = [{"name": "search_recipes", "args": {"ingredients": "tomato"}, "id": "1"}]
        mock_llm.return_value.bind_tools.return_value.invoke.return_value = mock_tool_call

        from langchain_core.messages import ToolMessage as TM
        tool_msg = MagicMock(spec=["content"])
        tool_msg.content = "erro: Missing Tavily API key."
        tool_msg.__class__ = TM

        mock_tool_node.return_value.invoke.return_value = {"messages": [tool_msg]}

        state = make_state(query="recipe with tomato", ingredients=["tomato"])
        result = search_recipes_node(state)
        assert result["error_message"].startswith("Error:")


# --- B4: Pantry lookup ---

class TestB4PantryLookup:
    @patch("agent.ToolNode")
    @patch("agent.get_llm")
    def test_pantry_retrieval(self, mock_llm, mock_tool_node):
        mock_tool_call = MagicMock()
        mock_tool_call.tool_calls = [{"name": "retrieve_pantry", "args": {}, "id": "1"}]
        mock_llm.return_value.bind_tools.return_value.invoke.return_value = mock_tool_call

        from langchain_core.messages import ToolMessage as TM
        tool_msg = MagicMock(spec=["content"])
        tool_msg.content = "['rice', 'beans', 'onion']"
        tool_msg.__class__ = TM
        mock_tool_node.return_value.invoke.return_value = {"messages": [tool_msg]}

        filter_resp = MagicMock()
        filter_resp.content = "rice, beans, onion"
        mock_llm.return_value.invoke.return_value = filter_resp

        state = make_state(query="Make me a recipe")
        result = pantry_lookup_node(state)
        assert "retrieve_pantry" in result["tools_called"]
        assert len(result["pantry_ingredients"]) > 0


# --- B5: Lack of ingredients ---

class TestB5LackOfIngredients:
    @patch("agent.ToolNode")
    @patch("agent.get_llm")
    def test_empty_pantry(self, mock_llm, mock_tool_node):
        # Validate query
        validate_resp = MagicMock()
        validate_resp.content = "valid"
        # Identify ingredients
        identify_resp = MagicMock()
        identify_resp.content = "INGREDIENTS: NONE\nRESTRICTIONS: vegan"
        # Pantry filter
        filter_resp = MagicMock()
        filter_resp.content = "NONE"

        mock_llm.return_value.invoke.side_effect = [validate_resp, identify_resp, filter_resp]

        mock_tool_call = MagicMock()
        mock_tool_call.tool_calls = [{"name": "retrieve_pantry", "args": {}, "id": "1"}]
        mock_llm.return_value.bind_tools.return_value.invoke.return_value = mock_tool_call

        from langchain_core.messages import ToolMessage as TM
        tool_msg = MagicMock(spec=["content"])
        tool_msg.content = "[]"
        tool_msg.__class__ = TM
        mock_tool_node.return_value.invoke.return_value = {"messages": [tool_msg]}

        result = agent("Make me a vegan recipe")
        assert result.startswith("Error:")


# --- B6: Execution failure ---

class TestB6ExecutionFailure:
    def test_unhandled_exception(self):
        with patch("agent.build_graph", side_effect=Exception("Model failure")):
            result = agent("Make a pasta recipe")
            assert result.startswith("Error:")


# --- B7: Recipe generation ---

class TestB7RecipeGeneration:
    @patch("agent.ToolNode")
    @patch("agent.get_llm")
    def test_recipe_has_required_fields(self, mock_llm, mock_tool_node):
        validate_resp = MagicMock()
        validate_resp.content = "valid"
        identify_resp = MagicMock()
        identify_resp.content = "INGREDIENTS: chicken, rice\nRESTRICTIONS: NONE"
        recipe_resp = MagicMock()
        recipe_resp.content = """Title: Chicken Rice Bowl

Ingredients:
- 500g chicken breast
- 2 cups rice

Equipment:
- Large pot
- Frying pan

Instructions:
1. Cook rice in pot
2. Grill chicken

Storage:
- Refrigerate for up to 3 days"""

        mock_llm.return_value.invoke.side_effect = [validate_resp, identify_resp, recipe_resp]

        mock_tool_call = MagicMock()
        mock_tool_call.tool_calls = [{"name": "search_recipes", "args": {"ingredients": "chicken, rice"}, "id": "1"}]
        mock_llm.return_value.bind_tools.return_value.invoke.return_value = mock_tool_call

        from langchain_core.messages import ToolMessage as TM
        tool_msg = MagicMock(spec=["content"])
        tool_msg.content = "[{'title': 'Test', 'url': 'http://test.com', 'content': 'test'}]"
        tool_msg.__class__ = TM
        mock_tool_node.return_value.invoke.return_value = {"messages": [tool_msg]}

        result = agent("Make a recipe with chicken and rice")
        assert "Title:" in result
        assert "Ingredients:" in result
        assert "Equipment:" in result
        assert "Instructions:" in result
        assert "Storage:" in result


# --- Tool-level tests ---

class TestTools:
    def test_retrieve_pantry_missing_credentials(self):
        with patch.dict(os.environ, {"NOTION_API_KEY": "", "NOTION_PAGE_ID": ""}, clear=False):
            result = retrieve_pantry.invoke({})
            assert result.startswith("Error:")

    def test_search_recipes_missing_credentials(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=False):
            result = search_recipes.invoke({"ingredients": "tomato"})
            assert result.startswith("erro:")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
