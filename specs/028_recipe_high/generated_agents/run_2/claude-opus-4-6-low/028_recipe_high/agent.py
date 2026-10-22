import os
import json
from typing import TypedDict, Annotated, Sequence
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import requests


# ── Tools ──

@tool
def retrieve_pantry() -> str:
    """Obtain the ingredients available in the user's pantry from a Notion page.
    Returns a JSON list of ingredient strings, or an empty list if the pantry is empty."""
    try:
        api_key = os.environ.get("NOTION_API_KEY")
        page_id = os.environ.get("NOTION_PAGE_ID")
        if not api_key or not page_id:
            return "Error: Missing Notion credentials (NOTION_API_KEY or NOTION_PAGE_ID)."

        # Fetch block children of the Notion page
        url = f"https://api.notion.com/v1/blocks/{page_id}/children?page_size=100"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Notion-Version": "2022-06-28",
        }
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            return f"Error: Failed to access Notion page (HTTP {resp.status_code})."

        data = resp.json()
        ingredients = []
        for block in data.get("results", []):
            block_type = block.get("type", "")
            type_data = block.get(block_type, {})
            rich_texts = type_data.get("rich_text", [])
            for rt in rich_texts:
                text = rt.get("plain_text", "").strip()
                if text:
                    ingredients.append(text)

            # Handle bulleted/numbered list items
            if block_type in ("bulleted_list_item", "numbered_list_item"):
                pass  # already handled above

        return json.dumps(ingredients)
    except requests.exceptions.Timeout:
        return "Error: Timeout while accessing Notion page."
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def search_recipes(ingredients: str) -> str:
    """Search for recipes on the web using Tavily, based on the provided ingredients.
    Args:
        ingredients: A comma-separated list of ingredients to search recipes for.
    Returns a JSON list of up to 5 results with title, url, and content excerpt."""
    try:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return "Error: Missing Tavily API key (TAVILY_API_KEY)."

        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        query = f"recipe with {ingredients}"
        response = client.search(query=query, max_results=5)

        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:500],
            })
        return json.dumps(results)
    except Exception as e:
        error_msg = str(e).lower()
        if "timeout" in error_msg:
            return "Error: Timeout while searching for recipes."
        return f"Error: {str(e)}"


# ── State ──

class AgentState(TypedDict):
    messages: Annotated[Sequence, lambda x, y: list(x) + list(y)]
    query: str
    is_valid: bool
    pantry_ingredients: list
    compatible_ingredients: list
    search_results: list
    final_output: str
    current_step: str
    error: str


TOOLS = [retrieve_pantry, search_recipes]

SYSTEM_PROMPT = """You are a professional chef who creates personalized recipes. Your role is to:
1. Validate user queries about food/cooking
2. Use pantry ingredients and query ingredients to create recipes
3. Respect ALL dietary restrictions, including derived ingredients (e.g., "vegan" means no meat, dairy, eggs, honey, etc.)
4. Ingredients mentioned under negation, absence, or hypothesis are NOT available
5. Generate complete recipes with Title, Ingredients, Equipment, Instructions, and Storage sections

IMPORTANT RULES:
- A dietary restriction implies ALL derived ingredients, even those not explicitly mentioned
- An ingredient mentioned under negation ("I don't have X", "no X", "without X", "if I had X") is NOT available
- Never include ingredients that violate declared dietary restrictions
- Always include at least one ingredient from the identified compatible ingredients list
"""


def build_graph():
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    llm_with_tools = llm.bind_tools(TOOLS)
    tool_node = ToolNode(TOOLS)

    def validate_query(state: AgentState) -> dict:
        query = state["query"]
        if not query or not query.strip():
            return {
                "is_valid": False,
                "final_output": "Error: The query is empty. Please provide a food-related query.",
                "current_step": "done",
                "messages": [],
            }

        try:
            validation_msg = llm.invoke([
                SystemMessage(content=(
                    "You are a query validator for a recipe agent. "
                    "Classify the query as 'valid' or 'invalid'. "
                    "A query is INVALID if it is: empty, offensive/inappropriate, or completely unrelated to food/cooking/recipes/ingredients/diet. "
                    "A query is VALID if it relates in any way to food, cooking, recipes, ingredients, meals, or dietary preferences. "
                    "Respond with ONLY a JSON object: {\"valid\": true} or {\"valid\": false, \"reason\": \"explanation\"}"
                )),
                HumanMessage(content=f"Query: {query}"),
            ])
            resp = validation_msg.content.strip()
            # Extract JSON from response
            if "{" in resp:
                json_str = resp[resp.index("{"):resp.rindex("}") + 1]
                result = json.loads(json_str)
            else:
                result = {"valid": True}

            if not result.get("valid", True):
                reason = result.get("reason", "The query is not related to food or cooking.")
                return {
                    "is_valid": False,
                    "final_output": f"Error: {reason}",
                    "current_step": "done",
                    "messages": [],
                }
            return {"is_valid": True, "current_step": "pantry_lookup", "messages": []}
        except Exception as e:
            return {
                "is_valid": False,
                "final_output": f"Error: Failed to validate query: {str(e)}",
                "current_step": "done",
                "messages": [],
            }

    def pantry_lookup(state: AgentState) -> dict:
        messages = [
            SystemMessage(content="Call the retrieve_pantry tool to get the user's pantry ingredients."),
            HumanMessage(content="Please retrieve my pantry ingredients."),
        ]
        response = llm_with_tools.invoke(messages)
        return {"messages": messages + [response], "current_step": "pantry_tool_call"}

    def call_tool(state: AgentState) -> dict:
        last_msg = state["messages"][-1]
        if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
            return {"current_step": "identify_ingredients", "messages": []}
        result = tool_node.invoke({"messages": list(state["messages"])})
        new_messages = result["messages"]
        return {"messages": new_messages, "current_step": "process_tool_result"}

    def process_pantry_result(state: AgentState) -> dict:
        # Find the last tool message
        for msg in reversed(state["messages"]):
            if isinstance(msg, ToolMessage):
                content = msg.content
                if content.startswith("Error:"):
                    return {
                        "pantry_ingredients": [],
                        "final_output": content,
                        "current_step": "done",
                        "error": content,
                        "messages": [],
                    }
                try:
                    ingredients = json.loads(content)
                    if isinstance(ingredients, list):
                        return {
                            "pantry_ingredients": ingredients,
                            "current_step": "identify_ingredients",
                            "messages": [],
                        }
                except (json.JSONDecodeError, TypeError):
                    pass
                return {
                    "pantry_ingredients": [],
                    "current_step": "identify_ingredients",
                    "messages": [],
                }
        return {"pantry_ingredients": [], "current_step": "identify_ingredients", "messages": []}

    def identify_ingredients(state: AgentState) -> dict:
        query = state["query"]
        pantry = state.get("pantry_ingredients", [])
        try:
            id_msg = llm.invoke([
                SystemMessage(content=(
                    "You are an ingredient identifier for a recipe agent. Given a user query and pantry ingredients, "
                    "identify all usable ingredients and filter out those incompatible with declared dietary restrictions.\n\n"
                    "RULES:\n"
                    "1. Combine ingredients from the query AND the pantry.\n"
                    "2. EXCLUDE any ingredient mentioned under negation, absence, or hypothesis "
                    "(e.g., 'I don't have X', 'no X', 'without X', 'if I had X').\n"
                    "3. EXCLUDE any ingredient incompatible with declared dietary restrictions. "
                    "A dietary restriction implies ALL derived ingredients:\n"
                    "   - Vegan: no meat, poultry, fish, seafood, dairy, eggs, honey, gelatin, etc.\n"
                    "   - Vegetarian: no meat, poultry, fish, seafood\n"
                    "   - Gluten-free: no wheat, barley, rye, regular flour, bread, pasta (unless specified gluten-free)\n"
                    "   - Lactose-free/dairy-free: no milk, cheese, butter, cream, yogurt, etc.\n"
                    "   - Nut-free: no almonds, walnuts, cashews, peanuts, etc.\n"
                    "4. Consider regional variants and inflected forms as the same ingredient.\n"
                    "5. Respond with ONLY a JSON object: {\"ingredients\": [\"ing1\", \"ing2\", ...], \"restrictions\": [\"restriction1\", ...]}\n"
                    "   If no compatible ingredients, return {\"ingredients\": [], \"restrictions\": [...]}"
                )),
                HumanMessage(content=f"Query: {query}\nPantry ingredients: {json.dumps(pantry)}"),
            ])
            resp = id_msg.content.strip()
            if "{" in resp:
                json_str = resp[resp.index("{"):resp.rindex("}") + 1]
                result = json.loads(json_str)
            else:
                result = {"ingredients": [], "restrictions": []}

            compatible = result.get("ingredients", [])
            if not compatible:
                return {
                    "compatible_ingredients": [],
                    "final_output": "Error: No compatible ingredients found. There are no ingredients available that meet the declared dietary restrictions.",
                    "current_step": "done",
                    "messages": [],
                }
            return {
                "compatible_ingredients": compatible,
                "current_step": "search_recipes",
                "messages": [],
            }
        except Exception as e:
            return {
                "compatible_ingredients": [],
                "final_output": f"Error: Failed to identify ingredients: {str(e)}",
                "current_step": "done",
                "messages": [],
            }

    def recipe_search(state: AgentState) -> dict:
        ingredients = state.get("compatible_ingredients", [])
        ingredients_str = ", ".join(ingredients)
        messages = [
            SystemMessage(content="Call the search_recipes tool with the provided ingredients."),
            HumanMessage(content=f"Search for recipes with these ingredients: {ingredients_str}"),
        ]
        response = llm_with_tools.invoke(messages)
        return {"messages": messages + [response], "current_step": "search_tool_call"}

    def process_search_result(state: AgentState) -> dict:
        for msg in reversed(state["messages"]):
            if isinstance(msg, ToolMessage):
                content = msg.content
                if content.startswith("Error:"):
                    return {
                        "search_results": [],
                        "final_output": content,
                        "current_step": "done",
                        "error": content,
                        "messages": [],
                    }
                try:
                    results = json.loads(content)
                    if isinstance(results, list):
                        return {
                            "search_results": results,
                            "current_step": "generate_recipe",
                            "messages": [],
                        }
                except (json.JSONDecodeError, TypeError):
                    pass
                return {"search_results": [], "current_step": "generate_recipe", "messages": []}
        return {"search_results": [], "current_step": "generate_recipe", "messages": []}

    def generate_recipe(state: AgentState) -> dict:
        query = state["query"]
        ingredients = state.get("compatible_ingredients", [])
        search_results = state.get("search_results", [])

        reference_text = ""
        if search_results:
            reference_text = "\n\nReference recipes for inspiration (do NOT copy directly, use as guidance):\n"
            for i, r in enumerate(search_results, 1):
                reference_text += f"{i}. {r.get('title', 'Untitled')}: {r.get('content', '')[:200]}\n"

        try:
            recipe_msg = llm.invoke([
                SystemMessage(content=(
                    "You are a professional chef creating a personalized recipe. "
                    "Create a complete recipe using the available ingredients and respecting all dietary restrictions.\n\n"
                    "STRICT RULES:\n"
                    "- Use ONLY ingredients compatible with the dietary restrictions\n"
                    "- Include at least one ingredient from the provided compatible ingredients list\n"
                    "- Never add ingredients that violate the declared dietary restrictions\n"
                    "- A dietary restriction implies ALL derived ingredients\n\n"
                    "FORMAT your response EXACTLY as follows:\n"
                    "**Title:** [Recipe title]\n\n"
                    "**Ingredients:**\n- [ingredient 1 with quantity]\n- [ingredient 2 with quantity]\n...\n\n"
                    "**Equipment:**\n- [equipment 1]\n- [equipment 2]\n...\n\n"
                    "**Instructions:**\n1. [Step 1]\n2. [Step 2]\n...\n\n"
                    "**Storage:**\n[Storage recommendations]"
                )),
                HumanMessage(content=(
                    f"User query: {query}\n"
                    f"Compatible ingredients: {json.dumps(ingredients)}\n"
                    f"{reference_text}"
                )),
            ])
            return {
                "final_output": recipe_msg.content,
                "current_step": "done",
                "messages": [],
            }
        except Exception as e:
            return {
                "final_output": f"Error: Failed to generate recipe: {str(e)}",
                "current_step": "done",
                "messages": [],
            }

    def route_after_validate(state: AgentState) -> str:
        if not state.get("is_valid", False):
            return "end"
        return "pantry_lookup"

    def route_after_tool(state: AgentState) -> str:
        step = state.get("current_step", "")
        if step == "pantry_tool_call":
            return "call_pantry_tool"
        if step == "search_tool_call":
            return "call_search_tool"
        if step == "process_tool_result":
            # Determine which tool result to process based on context
            return "check_tool_type"
        return step

    def route_main(state: AgentState) -> str:
        step = state.get("current_step", "")
        if step == "done":
            return "end"
        return step

    # Build the graph
    graph = StateGraph(AgentState)

    graph.add_node("validate_query", validate_query)
    graph.add_node("pantry_lookup", pantry_lookup)
    graph.add_node("call_pantry_tool", call_tool)
    graph.add_node("process_pantry_result", process_pantry_result)
    graph.add_node("identify_ingredients", identify_ingredients)
    graph.add_node("recipe_search", recipe_search)
    graph.add_node("call_search_tool", call_tool)
    graph.add_node("process_search_result", process_search_result)
    graph.add_node("generate_recipe", generate_recipe)

    graph.set_entry_point("validate_query")

    graph.add_conditional_edges("validate_query", route_after_validate, {
        "end": END,
        "pantry_lookup": "pantry_lookup",
    })

    graph.add_edge("pantry_lookup", "call_pantry_tool")
    graph.add_edge("call_pantry_tool", "process_pantry_result")

    graph.add_conditional_edges("process_pantry_result", route_main, {
        "end": END,
        "identify_ingredients": "identify_ingredients",
    })

    graph.add_conditional_edges("identify_ingredients", route_main, {
        "end": END,
        "search_recipes": "recipe_search",
    })

    graph.add_edge("recipe_search", "call_search_tool")
    graph.add_edge("call_search_tool", "process_search_result")

    graph.add_conditional_edges("process_search_result", route_main, {
        "end": END,
        "generate_recipe": "generate_recipe",
    })

    graph.add_edge("generate_recipe", END)

    return graph.compile()


# ── Entrypoint ──

_compiled_graph = None

def agent(query: str) -> str:
    global _compiled_graph
    try:
        if _compiled_graph is None:
            _compiled_graph = build_graph()

        initial_state = {
            "messages": [],
            "query": query,
            "is_valid": False,
            "pantry_ingredients": [],
            "compatible_ingredients": [],
            "search_results": [],
            "final_output": "",
            "current_step": "",
            "error": "",
        }

        result = _compiled_graph.invoke(initial_state)
        return result.get("final_output", "Error: No output generated.")
    except Exception as e:
        return f"Error: {str(e)}"
