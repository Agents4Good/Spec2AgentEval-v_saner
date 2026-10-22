import os
import json
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator
import requests


@tool
def retrieve_pantry() -> str:
    """Obtain the ingredients available in the user's pantry from a Notion page.
    Returns a JSON list of pantry ingredients. An empty pantry returns an empty list.
    On failure returns a message beginning with 'Error:' followed by an explanation."""
    try:
        notion_api_key = os.environ.get("NOTION_API_KEY")
        notion_page_id = os.environ.get("NOTION_PAGE_ID")
        if not notion_api_key or not notion_page_id:
            return "Error: Missing Notion credentials (NOTION_API_KEY or NOTION_PAGE_ID)."
        url = f"https://api.notion.com/v1/blocks/{notion_page_id}/children?page_size=100"
        headers = {
            "Authorization": f"Bearer {notion_api_key}",
            "Notion-Version": "2022-06-28",
        }
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            return f"Error: Failed to access Notion page (HTTP {resp.status_code})."
        data = resp.json()
        ingredients = []
        for block in data.get("results", []):
            block_type = block.get("type", "")
            if block_type in ("bulleted_list_item", "numbered_list_item", "paragraph"):
                rich_texts = block.get(block_type, {}).get("rich_text", [])
                text = "".join(rt.get("plain_text", "") for rt in rich_texts).strip()
                if text:
                    ingredients.append(text)
        return json.dumps(ingredients, ensure_ascii=False)
    except requests.exceptions.Timeout:
        return "Error: Timeout while accessing Notion page."
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def search_recipes(ingredients: str) -> str:
    """Search for recipes on the web using Tavily, based on the given ingredients.
    Input: a comma-separated string of ingredients compatible with dietary restrictions.
    Returns a JSON list of up to 5 results, each with title, url, and content.
    On failure returns a message beginning with 'Error:' followed by an explanation."""
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
                "content": r.get("content", ""),
            })
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"Error: {str(e)}"


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


SYSTEM_PROMPT = """You are a professional chef who creates personalized recipes.

You follow this exact workflow:
1. VALIDATE the user's query: if it is empty, offensive, or unrelated to food/cooking, respond ONLY with a message starting with "Error:" explaining why it's invalid. Do NOT call any tools for invalid queries.
2. If valid, call retrieve_pantry to get the user's pantry ingredients. This MUST be the first tool you call.
3. Identify usable ingredients from both the query and the pantry. Apply dietary restrictions:
   - A dietary restriction implies ALL derived ingredients (e.g., "vegan" excludes all animal products).
   - Ingredients mentioned under negation, absence, or hypothesis (e.g., "I don't have eggs", "without milk", "if I had butter") are NOT available.
   - Remove any ingredient incompatible with declared dietary restrictions.
4. If no compatible ingredients remain, respond with a message starting with "Error:" explaining the absence of compatible ingredients. Do NOT call search_recipes.
5. If there are compatible ingredients, call search_recipes with a comma-separated list of ONLY the compatible ingredients.
6. Generate a complete recipe using the compatible ingredients, user preferences, and reference recipes (if any). The recipe MUST contain ALL of these sections:
   - **Title:** the recipe title
   - **Ingredients:** list of ingredients with quantities
   - **Equipment:** list of required equipment
   - **Instructions:** step-by-step preparation instructions
   - **Storage:** storage recommendations

CRITICAL RULES:
- No ingredient in the final recipe may violate the declared dietary restrictions.
- At least one ingredient from the identified compatible ingredients must appear in the recipe.
- retrieve_pantry must be called exactly once and before search_recipes.
- search_recipes must be called exactly once (only if there are compatible ingredients).
- If any tool returns a message starting with "Error:", respond with a message starting with "Error:" explaining the failure.
"""

tools = [retrieve_pantry, search_recipes]


def create_graph():
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def call_model(state: AgentState):
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def call_tools(state: AgentState):
        last = state["messages"][-1]
        tool_messages = []
        for tc in last.tool_calls:
            name = tc["name"]
            args = tc["args"]
            matching = [t for t in tools if t.name == name]
            if matching:
                try:
                    result = matching[0].invoke(args)
                    tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
                except Exception as e:
                    tool_messages.append(ToolMessage(content=f"Error: {e}", tool_call_id=tc["id"]))
            else:
                tool_messages.append(ToolMessage(content=f"Error: Unknown tool {name}", tool_call_id=tc["id"]))
        return {"messages": tool_messages}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", call_tools)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def agent(query: str) -> str:
    try:
        app = create_graph()
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=query if query else ""),
        ]
        result = app.invoke({"messages": messages})
        final = result["messages"][-1]
        return final.content
    except Exception as e:
        return f"Error: {str(e)}"
