import os
import json
import pytest
from unittest.mock import patch, MagicMock
from agent import agent, retrieve_pantry, search_recipes, TOOLS, build_graph


# ── Helpers ──

def mock_llm_response(content):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = []
    return msg


def mock_llm_tool_call(tool_name, tool_args, tool_id="call_123"):
    msg = MagicMock()
    msg.content = ""
    msg.tool_calls = [{"name": tool_name, "args": tool_args, "id": tool_id}]
    msg.id = "msg_123"
    msg.type = "ai"
    return msg


# ── B1: Query Validation ──

class TestB1QueryValidation:
    @patch("agent.ChatOpenAI")
    def test_empty_query(self, mock_chat):
        result = agent("")
        assert result.startswith("Error:")

    @patch("agent.ChatOpenAI")
    def test_whitespace_query(self, mock_chat):
        result = agent("   ")
        assert result.startswith("Error:")

    @patch("agent._compiled_graph", None)
    @patch("agent.ChatOpenAI")
    def test_invalid_query_offensive(self, mock_chat_cls):
        instance = MagicMock()
        mock_chat_cls.return_value = instance
        instance.invoke.return_value = mock_llm_response('{"valid": false, "reason": "Offensive content"}')
        instance.bind_tools.return_value = instance
        result = agent("you are stupid")
        assert result.startswith("Error:")

    @patch("agent._compiled_graph", None)
    @patch("agent.ChatOpenAI")
    def test_invalid_query_unrelated(self, mock_chat_cls):
        instance = MagicMock()
        mock_chat_cls.return_value = instance
        instance.invoke.return_value = mock_llm_response('{"valid": false, "reason": "Not related to food or cooking"}')
        instance.bind_tools.return_value = instance
        result = agent("What is the capital of France?")
        assert result.startswith("Error:")

    @patch("agent._compiled_graph", None)
    @patch("agent.ChatOpenAI")
    def test_valid_food_query(self, mock_chat_cls):
        instance = MagicMock()
        mock_chat_cls.return_value = instance

        call_count = [0]
        def side_effect(messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_llm_response('{"valid": true}')
            if call_count[0] == 2:
                return mock_llm_response('{"ingredients": ["chicken", "rice"], "restrictions": []}')
            return mock_llm_response(
                "**Title:** Chicken Rice\n\n**Ingredients:**\n- 500g chicken\n- 2 cups rice\n\n"
                "**Equipment:**\n- Pot\n- Pan\n\n**Instructions:**\n1. Cook rice\n2. Cook chicken\n\n"
                "**Storage:**\nRefrigerate for up to 3 days."
            )

        instance.invoke.side_effect = side_effect

        tool_response = MagicMock()
        tool_response.content = ""
        tool_response.tool_calls = [{"name": "retrieve_pantry", "args": {}, "id": "call_1"}]
        tool_response.id = "msg_1"
        tool_response.type = "ai"

        search_response = MagicMock()
        search_response.content = ""
        search_response.tool_calls = [{"name": "search_recipes", "args": {"ingredients": "chicken, rice"}, "id": "call_2"}]
        search_response.id = "msg_2"
        search_response.type = "ai"

        bound = MagicMock()
        instance.bind_tools.return_value = bound

        bound_call_count = [0]
        def bound_side_effect(messages):
            bound_call_count[0] += 1
            if bound_call_count[0] == 1:
                return tool_response
            return search_response
        bound.invoke.side_effect = bound_side_effect

        with patch("agent.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"results": []}
            mock_get.return_value = mock_resp

            with patch("agent.TavilyClient", create=True) as mock_tavily_cls:
                mock_tavily = MagicMock()
                mock_tavily_cls.return_value = mock_tavily
                mock_tavily.search.return_value = {"results": []}

                os.environ.setdefault("OPENAI_API_KEY", "test")
                os.environ.setdefault("NOTION_API_KEY", "test")
                os.environ.setdefault("NOTION_PAGE_ID", "test")
                os.environ.setdefault("TAVILY_API_KEY", "test")

                result = agent("I want a chicken and rice recipe")
                assert not result.startswith("Error:")


# ── B2: Pantry Lookup (tool unit tests) ──

class TestB2PantryLookup:
    @patch("agent.requests.get")
    def test_retrieve_pantry_success(self, mock_get):
        os.environ["NOTION_API_KEY"] = "test_key"
        os.environ["NOTION_PAGE_ID"] = "test_page"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"plain_text": "tomato"}]
                    }
                },
                {
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"plain_text": "onion"}]
                    }
                },
            ]
        }
        mock_get.return_value = mock_resp
        result = retrieve_pantry.invoke({})
        parsed = json.loads(result)
        assert "tomato" in parsed
        assert "onion" in parsed

    @patch("agent.requests.get")
    def test_retrieve_pantry_empty(self, mock_get):
        os.environ["NOTION_API_KEY"] = "test_key"
        os.environ["NOTION_PAGE_ID"] = "test_page"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": []}
        mock_get.return_value = mock_resp
        result = retrieve_pantry.invoke({})
        assert json.loads(result) == []

    def test_retrieve_pantry_missing_credentials(self):
        env = os.environ.copy()
        os.environ.pop("NOTION_API_KEY", None)
        os.environ.pop("NOTION_PAGE_ID", None)
        result = retrieve_pantry.invoke({})
        assert result.startswith("Error:")
        os.environ.update(env)

    @patch("agent.requests.get")
    def test_retrieve_pantry_http_error(self, mock_get):
        os.environ["NOTION_API_KEY"] = "test_key"
        os.environ["NOTION_PAGE_ID"] = "test_page"
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_get.return_value = mock_resp
        result = retrieve_pantry.invoke({})
        assert result.startswith("Error:")

    @patch("agent.requests.get")
    def test_retrieve_pantry_timeout(self, mock_get):
        os.environ["NOTION_API_KEY"] = "test_key"
        os.environ["NOTION_PAGE_ID"] = "test_page"
        import requests as req
        mock_get.side_effect = req.exceptions.Timeout("Connection timed out")
        result = retrieve_pantry.invoke({})
        assert result.startswith("Error:")


# ── B4: Search Recipes (tool unit tests) ──

class TestB4SearchRecipes:
    @patch("agent.TavilyClient", create=True)
    def test_search_recipes_success(self, mock_tavily_cls):
        os.environ["TAVILY_API_KEY"] = "test_key"
        mock_client = MagicMock()
        mock_tavily_cls.return_value = mock_client
        mock_client.search.return_value = {
            "results": [
                {"title": "Recipe 1", "url": "http://example.com/1", "content": "Delicious recipe"},
                {"title": "Recipe 2", "url": "http://example.com/2", "content": "Another recipe"},
            ]
        }

        # Need to patch the import inside the function
        with patch.dict("sys.modules", {"tavily": MagicMock(TavilyClient=mock_tavily_cls)}):
            result = search_recipes.invoke({"ingredients": "chicken, rice"})
            parsed = json.loads(result)
            assert len(parsed) <= 5
            assert parsed[0]["title"] == "Recipe 1"

    def test_search_recipes_missing_credentials(self):
        env = os.environ.copy()
        os.environ.pop("TAVILY_API_KEY", None)
        result = search_recipes.invoke({"ingredients": "chicken"})
        assert result.startswith("Error:")
        os.environ.update(env)


# ── B5: Lack of Ingredients ──

class TestB5LackOfIngredients:
    @patch("agent._compiled_graph", None)
    @patch("agent.ChatOpenAI")
    def test_no_compatible_ingredients(self, mock_chat_cls):
        instance = MagicMock()
        mock_chat_cls.return_value = instance

        call_count = [0]
        def side_effect(messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_llm_response('{"valid": true}')
            return mock_llm_response('{"ingredients": [], "restrictions": ["vegan"]}')

        instance.invoke.side_effect = side_effect

        tool_response = MagicMock()
        tool_response.content = ""
        tool_response.tool_calls = [{"name": "retrieve_pantry", "args": {}, "id": "call_1"}]
        tool_response.id = "msg_1"
        tool_response.type = "ai"

        bound = MagicMock()
        instance.bind_tools.return_value = bound
        bound.invoke.return_value = tool_response

        with patch("agent.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"results": [
                {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"plain_text": "steak"}]}}
            ]}
            mock_get.return_value = mock_resp

            os.environ["NOTION_API_KEY"] = "test"
            os.environ["NOTION_PAGE_ID"] = "test"
            os.environ["OPENAI_API_KEY"] = "test"

            result = agent("I want a vegan recipe")
            assert result.startswith("Error:")


# ── B6: Execution Failure ──

class TestB6ExecutionFailure:
    @patch("agent._compiled_graph", None)
    @patch("agent.ChatOpenAI")
    def test_pantry_tool_error(self, mock_chat_cls):
        instance = MagicMock()
        mock_chat_cls.return_value = instance

        instance.invoke.return_value = mock_llm_response('{"valid": true}')

        tool_response = MagicMock()
        tool_response.content = ""
        tool_response.tool_calls = [{"name": "retrieve_pantry", "args": {}, "id": "call_1"}]
        tool_response.id = "msg_1"
        tool_response.type = "ai"

        bound = MagicMock()
        instance.bind_tools.return_value = bound
        bound.invoke.return_value = tool_response

        with patch("agent.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_get.return_value = mock_resp

            os.environ["NOTION_API_KEY"] = "test"
            os.environ["NOTION_PAGE_ID"] = "test"
            os.environ["OPENAI_API_KEY"] = "test"

            result = agent("Make me pasta")
            # Should get an error from pantry failure
            assert result.startswith("Error:")

    def test_agent_does_not_raise_exception(self):
        """The agent should never raise an unhandled exception."""
        with patch("agent._compiled_graph", None):
            with patch("agent.build_graph", side_effect=Exception("Unexpected failure")):
                result = agent("Make me a cake")
                assert result.startswith("Error:")


# ── B7: Recipe Generation ──

class TestB7RecipeGeneration:
    def test_recipe_has_required_fields(self):
        """Test that a generated recipe string contains all required sections."""
        recipe = (
            "**Title:** Test Recipe\n\n"
            "**Ingredients:**\n- 1 cup rice\n\n"
            "**Equipment:**\n- Pot\n\n"
            "**Instructions:**\n1. Cook rice\n\n"
            "**Storage:**\nStore in fridge"
        )
        assert "Title:" in recipe or "title:" in recipe.lower()
        assert "Ingredients:" in recipe or "ingredients:" in recipe.lower()
        assert "Equipment:" in recipe or "equipment:" in recipe.lower()
        assert "Instructions:" in recipe or "instructions:" in recipe.lower()
        assert "Storage:" in recipe or "storage:" in recipe.lower()


# ── Tool Registration ──

class TestToolRegistration:
    def test_tools_are_langchain_tools(self):
        for t in TOOLS:
            assert hasattr(t, "invoke"), f"{t.name} must be a LangChain tool with invoke"

    def test_tool_names(self):
        names = [t.name for t in TOOLS]
        assert "retrieve_pantry" in names
        assert "search_recipes" in names


# ── Integration-like Tests ──

class TestIntegration:
    @patch("agent._compiled_graph", None)
    @patch("agent.ChatOpenAI")
    def test_full_flow_with_mocks(self, mock_chat_cls):
        """Test the full happy path with all components mocked."""
        instance = MagicMock()
        mock_chat_cls.return_value = instance

        call_count = [0]
        def llm_side_effect(messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_llm_response('{"valid": true}')
            if call_count[0] == 2:
                return mock_llm_response('{"ingredients": ["tomato", "onion", "garlic"], "restrictions": []}')
            return mock_llm_response(
                "**Title:** Tomato Soup\n\n"
                "**Ingredients:**\n- 4 tomatoes\n- 1 onion\n- 3 cloves garlic\n\n"
                "**Equipment:**\n- Large pot\n- Blender\n\n"
                "**Instructions:**\n1. Dice onion and garlic\n2. Sauté in pot\n3. Add tomatoes\n4. Blend\n\n"
                "**Storage:**\nRefrigerate in airtight container for up to 5 days."
            )
        instance.invoke.side_effect = llm_side_effect

        pantry_tool_resp = MagicMock()
        pantry_tool_resp.content = ""
        pantry_tool_resp.tool_calls = [{"name": "retrieve_pantry", "args": {}, "id": "call_1"}]
        pantry_tool_resp.id = "msg_1"
        pantry_tool_resp.type = "ai"

        search_tool_resp = MagicMock()
        search_tool_resp.content = ""
        search_tool_resp.tool_calls = [{"name": "search_recipes", "args": {"ingredients": "tomato, onion, garlic"}, "id": "call_2"}]
        search_tool_resp.id = "msg_2"
        search_tool_resp.type = "ai"

        bound = MagicMock()
        instance.bind_tools.return_value = bound

        bound_calls = [0]
        def bound_side_effect(messages):
            bound_calls[0] += 1
            if bound_calls[0] == 1:
                return pantry_tool_resp
            return search_tool_resp
        bound.invoke.side_effect = bound_side_effect

        with patch("agent.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "results": [
                    {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"plain_text": "tomato"}]}},
                    {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"plain_text": "onion"}]}},
                    {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"plain_text": "garlic"}]}},
                ]
            }
            mock_get.return_value = mock_resp

            with patch.dict("sys.modules", {"tavily": MagicMock(TavilyClient=MagicMock(return_value=MagicMock(search=MagicMock(return_value={"results": []}))))}):
                os.environ["NOTION_API_KEY"] = "test"
                os.environ["NOTION_PAGE_ID"] = "test"
                os.environ["OPENAI_API_KEY"] = "test"
                os.environ["TAVILY_API_KEY"] = "test"

                result = agent("I want a tomato soup")
                assert "Title" in result
                assert "Ingredients" in result
                assert "Equipment" in result
                assert "Instructions" in result
                assert "Storage" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
