import os
import json
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import re


class AgentState(TypedDict):
    query: str
    is_valid: bool
    error_message: Optional[str]
    identified_ingredients: List[str]
    dietary_restrictions: List[str]
    pantry_ingredients: List[str]
    reference_recipes: List[dict]
    final_recipe: Optional[dict]
    tool_called: Optional[str]


def retrieve_pantry() -> str:
    """Retrieve ingredients from user's pantry via Notion."""
    notion_api_key = os.getenv("NOTION_API_KEY")
    notion_page_id = os.getenv("NOTION_PAGE_ID")

    if not notion_api_key or not notion_page_id:
        return "erro: Credenciais do Notion ausentes (NOTION_API_KEY ou NOTION_PAGE_ID)"

    try:
        import requests
        headers = {
            "Authorization": f"Bearer {notion_api_key}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }

        response = requests.get(
            f"https://api.notion.com/v1/blocks/{notion_page_id}/children",
            headers=headers,
            timeout=5
        )

        if response.status_code == 401:
            return "erro: Credenciais inválidas para Notion"
        elif response.status_code == 404:
            return "erro: Página do Notion não encontrada"
        elif response.status_code != 200:
            return f"erro: Erro ao acessar Notion (HTTP {response.status_code})"

        data = response.json()
        ingredients = []

        for block in data.get("results", []):
            if block.get("type") == "paragraph":
                text = block.get("paragraph", {}).get("rich_text", [])
                if text:
                    ingredient = text[0].get("plain_text", "").strip()
                    if ingredient:
                        ingredients.append(ingredient)

        return json.dumps(ingredients)

    except requests.exceptions.Timeout:
        return "erro: Timeout ao acessar Notion"
    except requests.exceptions.RequestException as e:
        return f"erro: Falha ao conectar ao Notion: {str(e)}"
    except Exception as e:
        return f"erro: Erro inesperado ao recuperar despensa: {str(e)}"


def search_recipes(ingredients: List[str]) -> str:
    """Search for recipes using the identified ingredients via Tavily."""
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    if not tavily_api_key:
        return "erro: Credenciais do Tavily ausentes (TAVILY_API_KEY)"

    if not ingredients:
        return json.dumps([])

    try:
        import requests

        query = " ".join(ingredients[:3])

        response = requests.get(
            "https://api.tavily.com/search",
            params={
                "api_key": tavily_api_key,
                "query": f"receita {query}",
                "max_results": 5,
                "include_answer": False
            },
            timeout=5
        )

        if response.status_code == 401:
            return "erro: Credenciais inválidas para Tavily"
        elif response.status_code != 200:
            return f"erro: Erro ao acessar Tavily (HTTP {response.status_code})"

        data = response.json()
        results = []

        for result in data.get("results", [])[:5]:
            results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("content", "")
            })

        return json.dumps(results)

    except requests.exceptions.Timeout:
        return "erro: Timeout ao acessar Tavily"
    except requests.exceptions.RequestException as e:
        return f"erro: Falha ao conectar ao Tavily: {str(e)}"
    except Exception as e:
        return f"erro: Erro inesperado ao buscar receitas: {str(e)}"


def validate_query(state: AgentState) -> AgentState:
    """B1: Validate if the query is valid."""
    query = state["query"].strip()

    if not query:
        state["is_valid"] = False
        state["error_message"] = "Error: A query não pode estar vazia."
        return state

    if len(query) > 10000:
        state["is_valid"] = False
        state["error_message"] = "Error: A query é muito longa."
        return state

    offensive_words = ["bomb", "kill", "destroy", "hate"]
    if any(word in query.lower() for word in offensive_words):
        state["is_valid"] = False
        state["error_message"] = "Error: A query contém conteúdo ofensivo."
        return state

    food_keywords = ["receita", "ingrediente", "comida", "prato", "cozinha", "preparar", "fazer", "tempero", "alimento", "refeição"]
    if not any(keyword in query.lower() for keyword in food_keywords):
        state["is_valid"] = False
        state["error_message"] = "Error: A query não está relacionada a receitas, ingredientes ou culinária."
        return state

    state["is_valid"] = True
    return state


def identify_ingredients(state: AgentState) -> AgentState:
    """B2: Identify ingredients mentioned in the query."""
    if not state["is_valid"]:
        return state

    query_lower = state["query"].lower()

    common_ingredients = [
        "tomate", "alface", "cebola", "alho", "cenoura", "batata", "abóbora",
        "frango", "carne", "peixe", "ovos", "leite", "queijo", "manteiga",
        "arroz", "feijão", "macarrão", "pão", "açúcar", "sal", "óleo",
        "alecrim", "orégano", "salsinha", "coentro", "pimenta", "limão",
        "melancia", "maçã", "banana", "laranja", "morango", "uva",
        "brócolis", "espinafre", "repolho", "couve", "beterraba",
        "cogumelo", "abobrinha", "berinjela", "pimentão",
        "aipim", "macaxeira", "mandioca", "batata-doce", "inhame",
        "chocolate", "café", "chá", "vinho", "cerveja"
    ]

    identified = []
    for ingredient in common_ingredients:
        if ingredient in query_lower:
            pattern = r'\b' + ingredient + r'\b'
            if re.search(pattern, query_lower):
                identified.append(ingredient)

    negation_pattern = r'\b(?:não|sem|nenhum|nunca)\s+(?:de\s+)?(\w+)'
    negations = re.findall(negation_pattern, query_lower)
    identified = [ing for ing in identified if ing not in negations]

    state["identified_ingredients"] = list(set(identified))

    dietary_keywords = {
        "vegano": ["carne", "peixe", "frango", "leite", "queijo", "manteiga", "ovos", "mel", "gelatina", "whey"],
        "vegetariano": ["carne", "peixe", "frango"],
        "celíaco": ["cevada", "malte", "trigo", "shoyu"],
        "sem lactose": ["leite", "queijo", "manteiga"],
        "sem glúten": ["pão", "macarrão", "trigo"]
    }

    dietary_restrictions = []
    for restriction, excluded_ingredients in dietary_keywords.items():
        if restriction in query_lower:
            dietary_restrictions.append(restriction)
            identified = [ing for ing in identified if ing not in excluded_ingredients]

    state["dietary_restrictions"] = dietary_restrictions
    state["identified_ingredients"] = list(set(identified))

    return state


def route_to_tool(state: AgentState) -> str:
    """Route to appropriate tool based on identified ingredients."""
    if not state["is_valid"]:
        return "end"

    if state["identified_ingredients"]:
        return "search_recipes"
    else:
        return "retrieve_pantry"


def search_recipes_node(state: AgentState) -> AgentState:
    """B3: Search for reference recipes."""
    try:
        result = search_recipes(state["identified_ingredients"])

        if result.startswith("erro:"):
            state["error_message"] = result.replace("erro:", "Error:")
            return state

        state["reference_recipes"] = json.loads(result)
        state["tool_called"] = "search_recipes"
    except Exception as e:
        state["error_message"] = f"Error: Falha ao buscar receitas: {str(e)}"

    return state


def retrieve_pantry_node(state: AgentState) -> AgentState:
    """B4: Retrieve ingredients from pantry."""
    try:
        result = retrieve_pantry()

        if result.startswith("erro:"):
            state["error_message"] = result.replace("erro:", "Error:")
            return state

        state["pantry_ingredients"] = json.loads(result)
        state["tool_called"] = "retrieve_pantry"

        if not state["pantry_ingredients"]:
            state["error_message"] = "Error: A despensa está vazia e nenhum ingrediente foi fornecido na query."
    except Exception as e:
        state["error_message"] = f"Error: Falha ao recuperar despensa: {str(e)}"

    return state


def check_dietary_compatibility(state: AgentState) -> AgentState:
    """B6: Check dietary compatibility."""
    if state.get("error_message"):
        return state

    if not state["is_valid"]:
        return state

    ingredients_to_check = state["identified_ingredients"] or state["pantry_ingredients"]

    if not ingredients_to_check:
        state["error_message"] = "Error: Nenhum ingrediente disponível para criar uma receita."
        return state

    dietary_restrictions_lower = [r.lower() for r in state["dietary_restrictions"]]

    restricted_ingredients = {
        "vegano": ["carne", "peixe", "frango", "leite", "queijo", "manteiga", "ovos", "mel", "gelatina", "whey"],
        "vegetariano": ["carne", "peixe", "frango"],
        "celíaco": ["cevada", "malte", "trigo", "shoyu"],
        "sem lactose": ["leite", "queijo", "manteiga"],
        "sem glúten": ["pão", "macarrão", "trigo"]
    }

    all_restricted = set()
    for restriction in dietary_restrictions_lower:
        all_restricted.update(restricted_ingredients.get(restriction, []))

    compatible = [ing for ing in ingredients_to_check if ing not in all_restricted]

    if not compatible:
        state["error_message"] = "Error: Nenhum ingrediente é compatível com as restrições dietéticas declaradas."
        return state

    state["identified_ingredients"] = compatible

    return state


def generate_recipe(state: AgentState) -> AgentState:
    """B8: Generate the final recipe."""
    if state.get("error_message"):
        return state

    if not state["is_valid"]:
        return state

    ingredients = state["identified_ingredients"] or state["pantry_ingredients"]

    if not ingredients:
        state["error_message"] = "Error: Não há ingredientes para gerar uma receita."
        return state

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

        reference_info = ""
        if state["reference_recipes"]:
            titles = [r.get("title", "") for r in state["reference_recipes"]]
            reference_info = f"\n\nReceitas de referência encontradas: {', '.join(titles[:3])}"

        dietary_info = ""
        if state["dietary_restrictions"]:
            dietary_info = f"\n\nRestrições dietéticas: {', '.join(state['dietary_restrictions'])}"

        prompt = f"""Você é um chef de cozinha experiente. Crie uma receita completa e detalhada usando os seguintes ingredientes principais:

Ingredientes disponíveis: {', '.join(ingredients)}

Query original do usuário: {state['query']}{dietary_info}{reference_info}

Gere uma receita com a seguinte estrutura JSON (sem markdown, apenas JSON válido):
{{
    "título": "Nome da receita",
    "ingredientes": ["quantidade ingrediente 1", "quantidade ingrediente 2", ...],
    "equipamentos": ["equipamento 1", "equipamento 2", ...],
    "instruções": ["Passo 1", "Passo 2", ...],
    "armazenamento": "Recomendações de armazenamento"
}}

Regras importantes:
- A receita DEVE incluir todos os ingredientes principais fornecidos
- A receita DEVE respeitar as restrições dietéticas
- Todos os campos devem estar presentes e não vazios
- Retorne apenas o JSON, sem explicações adicionais"""

        response = llm.invoke(prompt)
        recipe_text = response.content

        json_match = re.search(r'\{.*\}', recipe_text, re.DOTALL)
        if json_match:
            recipe_json = json.loads(json_match.group())
            state["final_recipe"] = recipe_json
        else:
            state["error_message"] = "Error: Não foi possível processar a resposta do modelo."

    except json.JSONDecodeError:
        state["error_message"] = "Error: Não foi possível processar a receita gerada."
    except Exception as e:
        state["error_message"] = f"Error: Falha ao gerar receita: {str(e)}"

    return state


def agent(query: str) -> dict:
    """Main agent function."""
    initial_state = AgentState(
        query=query,
        is_valid=False,
        error_message=None,
        identified_ingredients=[],
        dietary_restrictions=[],
        pantry_ingredients=[],
        reference_recipes=[],
        final_recipe=None,
        tool_called=None
    )

    graph = StateGraph(AgentState)

    graph.add_node("validate", validate_query)
    graph.add_node("identify", identify_ingredients)
    graph.add_node("search_recipes", search_recipes_node)
    graph.add_node("retrieve_pantry", retrieve_pantry_node)
    graph.add_node("check_compatibility", check_dietary_compatibility)
    graph.add_node("generate", generate_recipe)

    graph.set_entry_point("validate")

    graph.add_edge("validate", "identify")

    graph.add_conditional_edges(
        "identify",
        route_to_tool,
        {
            "search_recipes": "search_recipes",
            "retrieve_pantry": "retrieve_pantry",
            "end": END
        }
    )

    graph.add_edge("search_recipes", "check_compatibility")
    graph.add_edge("retrieve_pantry", "check_compatibility")
    graph.add_edge("check_compatibility", "generate")
    graph.add_edge("generate", END)

    runnable = graph.compile()
    result = runnable.invoke(initial_state)

    return result
