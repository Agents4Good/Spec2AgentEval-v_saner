import os
import re
import sys
from typing import Any, Dict, Iterable, List

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal envs
    requests = None

try:
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError:  # pragma: no cover
    class ChatOpenAI:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

try:
    from langgraph.graph import END, StateGraph
except ModuleNotFoundError:  # pragma: no cover
    END = "__end__"

    class StateGraph:  # type: ignore[no-redef]
        def __init__(self, state_type: type) -> None:
            self.nodes = {}
            self.entry = None
            self.conditional = {}
            self.edges = {}

        def add_node(self, name: str, func: Any) -> None:
            self.nodes[name] = func

        def set_entry_point(self, name: str) -> None:
            self.entry = name

        def add_conditional_edges(self, source: str, route: Any, mapping: Dict[str, str]) -> None:
            self.conditional[source] = (route, mapping)

        def add_edge(self, source: str, target: str) -> None:
            self.edges[source] = target

        def compile(self) -> Any:
            graph = self

            class CompiledGraph:
                def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
                    current = graph.entry
                    while current and current != END:
                        state = graph.nodes[current](state)
                        if current in graph.conditional:
                            route, mapping = graph.conditional[current]
                            current = mapping[route(state)]
                        else:
                            current = graph.edges.get(current, END)
                    return state

            return CompiledGraph()


FOOD_TERMS = {
    "receita", "recipe", "cook", "cooking", "cozinhar", "culinaria",
    "culinaria", "meal", "dish", "prato", "food", "comida", "jantar",
    "almoco", "breakfast", "lunch", "dinner", "sobremesa", "salad",
    "soup", "bolo", "massa", "arroz", "feijao", "frango", "carne",
    "peixe", "ovo", "eggs", "tofu", "tomate", "batata", "queijo",
    "leite", "vegetariano", "vegano", "gluten", "lactose",
}

OFFENSIVE_TERMS = {
    "idiot", "stupid", "burro", "burra", "bosta", "merda", "hate",
}

COMMON_INGREDIENTS = {
    "rice", "arroz", "beans", "feijao", "chicken", "frango", "beef",
    "carne", "fish", "peixe", "egg", "eggs", "ovo", "ovos", "tofu",
    "tomato", "tomate", "potato", "batata", "onion", "cebola",
    "garlic", "alho", "cheese", "queijo", "milk", "leite", "flour",
    "farinha", "pasta", "massa", "lentil", "lentilha", "mushroom",
    "cogumelo", "spinach", "espinafre", "carrot", "cenoura",
    "broccoli", "brocolis", "banana", "apple", "maca", "oats", "aveia",
}

RESTRICTION_BLOCKS = {
    "vegan": {"chicken", "frango", "beef", "carne", "fish", "peixe", "egg", "eggs", "ovo", "ovos", "cheese", "queijo", "milk", "leite"},
    "vegano": {"chicken", "frango", "beef", "carne", "fish", "peixe", "egg", "eggs", "ovo", "ovos", "cheese", "queijo", "milk", "leite"},
    "vegetarian": {"chicken", "frango", "beef", "carne", "fish", "peixe"},
    "vegetariano": {"chicken", "frango", "beef", "carne", "fish", "peixe"},
    "sem lactose": {"cheese", "queijo", "milk", "leite"},
    "lactose-free": {"cheese", "queijo", "milk", "leite"},
    "gluten-free": {"flour", "farinha", "pasta", "massa"},
    "sem gluten": {"flour", "farinha", "pasta", "massa"},
    "sem glúten": {"flour", "farinha", "pasta", "massa"},
}


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def _tokens(text: str) -> List[str]:
    return re.findall(r"[\wÀ-ÿ-]+", _normalize(text))


def _extract_ingredients(text: str) -> List[str]:
    words = _tokens(text)
    found = [word for word in words if word in COMMON_INGREDIENTS]
    seen = set()
    return [item for item in found if not (item in seen or seen.add(item))]


def _extract_restrictions(text: str) -> List[str]:
    lowered = _normalize(text)
    return [restriction for restriction in RESTRICTION_BLOCKS if restriction in lowered]


def _ingredient_names(data: Any) -> List[str]:
    if isinstance(data, dict):
        values = []
        for key in ("ingredients", "ingredientes", "pantry", "items", "results"):
            values.extend(_ingredient_names(data.get(key, [])))
        if data.get("name"):
            values.append(str(data["name"]))
        if data.get("title"):
            values.append(str(data["title"]))
        return values
    if isinstance(data, (list, tuple, set)):
        values = []
        for item in data:
            values.extend(_ingredient_names(item))
        return values
    if isinstance(data, str):
        return _extract_ingredients(data) or [data]
    return []


def _is_incompatible(ingredients: Iterable[str], restrictions: Iterable[str]) -> bool:
    names = " ".join(str(item).lower() for item in ingredients)
    for restriction in restrictions:
        blocked = RESTRICTION_BLOCKS.get(restriction, set())
        if any(item in names for item in blocked):
            return True
    return False


def retrieve_pantry(page_id: str | None = None) -> List[str]:
    if requests is None:
        raise RuntimeError("requests package is not installed")
    page_id = page_id or os.getenv("NOTION_PAGE_ID")
    token = os.getenv("NOTION_API_KEY")
    if not page_id or not token:
        raise RuntimeError("missing NOTION_PAGE_ID or NOTION_API_KEY")

    response = requests.get(
        f"https://api.notion.com/v1/blocks/{page_id}/children",
        headers={
            "Authorization": f"Bearer {token}",
            "Notion-Version": "2022-06-28",
        },
        timeout=15,
    )
    response.raise_for_status()
    ingredients = []
    for block in response.json().get("results", []):
        rich_text = block.get(block.get("type", ""), {}).get("rich_text", [])
        text = " ".join(part.get("plain_text", "") for part in rich_text)
        ingredients.extend(_ingredient_names(text))
    return [item for item in ingredients if item]


def search_recipes(ingredients: List[str]) -> List[Dict[str, Any]]:
    if requests is None:
        raise RuntimeError("requests package is not installed")
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("missing TAVILY_API_KEY")

    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": f"recipe using {', '.join(ingredients)}",
            "search_depth": "basic",
            "max_results": 5,
        },
        timeout=15,
    )
    response.raise_for_status()
    return response.json().get("results", [])


def _format_recipe(query: str, base_data: Any, ingredients: List[str]) -> str:
    main = ingredients[:6]
    title_core = ", ".join(item.title() for item in main[:3]) or "Despensa"
    source_hint = ""
    if isinstance(base_data, list) and base_data and isinstance(base_data[0], dict):
        source_hint = str(base_data[0].get("title") or base_data[0].get("content") or "")[:120]

    ingredient_lines = "\n".join(f"- {item}: quantidade a gosto" for item in main)
    return (
        f"Título: Receita personalizada de {title_core}\n\n"
        f"Ingredientes:\n{ingredient_lines}\n- Sal, pimenta e azeite: a gosto\n\n"
        "Equipamentos:\n- Faca e tábua\n- Panela ou frigideira\n- Colher de preparo\n\n"
        "Instruções:\n"
        "1. Separe, higienize e corte os ingredientes principais.\n"
        "2. Aqueça a panela com azeite e refogue os aromáticos disponíveis.\n"
        "3. Adicione os ingredientes principais, respeitando o tempo de cozimento de cada um.\n"
        "4. Tempere aos poucos, ajuste textura e finalize quando tudo estiver cozido.\n"
        "5. Sirva ainda quente.\n\n"
        "Informações nutricionais:\n"
        "- Estimativa: refeição caseira com energia, proteínas e fibras variando conforme as quantidades usadas.\n\n"
        "Armazenamento:\n"
        "- Guarde em recipiente fechado por até 3 dias na geladeira. Reaqueça bem antes de servir."
        + (f"\n\nBase consultada: {source_hint}" if source_hint else "")
    )


def _validate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("query", "")
    lowered = _normalize(query)
    if not lowered:
        state["error"] = "Error: query vazia; informe um pedido relacionado a comida ou culinária."
        return state
    if any(term in lowered for term in OFFENSIVE_TERMS):
        state["error"] = "Error: query ofensiva; reformule o pedido de maneira respeitosa."
        return state
    if not any(term in lowered for term in FOOD_TERMS | COMMON_INGREDIENTS):
        state["error"] = "Error: query não relacionada a comida ou culinária."
        return state

    state["ingredients"] = _extract_ingredients(query)
    state["restrictions"] = _extract_restrictions(query)
    state["tool_to_call"] = "search_recipes" if state["ingredients"] else "retrieve_pantry"
    return state


def _tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("error"):
        return state
    try:
        if state["tool_to_call"] == "search_recipes":
            state["tool_result"] = search_recipes(state["ingredients"])
        else:
            state["tool_result"] = retrieve_pantry(state.get("page_id"))
    except Exception as exc:
        state["error"] = f"Error: falha ao executar ferramenta {state.get('tool_to_call')}: {exc}"
    return state


def _generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("error"):
        state["output"] = state["error"]
        return state

    tool_result = state.get("tool_result")
    if not tool_result:
        state["output"] = "Error: ferramenta não retornou informações úteis."
        return state

    names = _ingredient_names(tool_result)
    ingredients = state.get("ingredients") or _extract_ingredients(" ".join(names))
    if not ingredients:
        state["output"] = "Error: nenhum ingrediente identificável foi recuperado."
        return state
    if _is_incompatible(ingredients or names, state.get("restrictions", [])):
        state["output"] = "Error: ingredientes disponíveis incompatíveis com as restrições dietéticas informadas."
        return state

    state["llm"] = ChatOpenAI(model="gpt-4o", temperature=0)
    state["output"] = _format_recipe(state["query"], tool_result, ingredients)
    return state


def _route_after_validate(state: Dict[str, Any]) -> str:
    return "generate" if state.get("error") else "tool"


def _build_graph():
    graph = StateGraph(dict)
    graph.add_node("validate", _validate_node)
    graph.add_node("tool", _tool_node)
    graph.add_node("generate", _generate_node)
    graph.set_entry_point("validate")
    graph.add_conditional_edges("validate", _route_after_validate, {"tool": "tool", "generate": "generate"})
    graph.add_edge("tool", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def agent(query: str, **kwargs: Any) -> Dict[str, Any]:
    """Benchmark entrypoint. Returns a state dict containing the final output."""
    state = {"query": query, **kwargs}
    return _build_graph().invoke(state)


__all__ = ["agent", "retrieve_pantry", "search_recipes"]

def main():
    query = "Preciso de uma receita vegana sem gluten com ovo"
    result = agent(query)
    print(result.get("output", "No output generated."))


if __name__ == "__main__":
    main()
