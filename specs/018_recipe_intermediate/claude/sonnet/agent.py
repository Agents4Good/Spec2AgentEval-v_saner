"""
Personalized Recipe Generator Agent
====================================

LangGraph agent that assumes the persona of a *chefe de cozinha* (head chef)
and generates a personalized recipe, using either the ingredients mentioned
explicitly in the user's query or the ingredients available in the user's
pantry (a Notion page), while respecting the user's dietary restrictions and
preferences.

Interface contract (benchmark)
------------------------------
* Model      : gpt-4o
* Framework  : LangGraph
* GenAI Tool : ChatOpenAI
* Entrypoint : ``agent(query: str) -> dict``
* Output     : the final graph state (``dict``).
  - ``state["error"]``  is ``None`` on success, or a string starting with
    ``"Error: "`` explaining why the agent stopped.
  - ``state["recipe"]`` is ``None`` on failure, or a dict with the keys
    ``"Título"``, ``"Ingredientes"``, ``"Equipamentos"``, ``"Instruções"`` and
    ``"Armazenamento"`` on success.

Tools (generated together with the agent)
-----------------------------------------
* ``retrieve_pantry()``            -> reads the user's pantry from a Notion page.
* ``search_recipes(ingredients)``  -> searches reference recipes on Tavily.

Both tools return a string starting with ``"erro: "`` when they fail
(missing credentials, unreachable page/API, timeout), as required by the
tool contract. The agent converts that into ``state["error"]`` starting with
``"Error: "`` without raising.

Environment variables
----------------------
* ``OPENAI_API_KEY``          (required to reach the language model)
* ``NOTION_API_KEY``          (required by ``retrieve_pantry``)
* ``NOTION_PAGE_ID``          (required by ``retrieve_pantry``)
* ``TAVILY_API_KEY``          (required by ``search_recipes``)
* ``OPENAI_MODEL``            (optional, default ``gpt-4o``)
* ``REQUEST_TIMEOUT_SECONDS`` (optional, default ``20``)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional, TypedDict

import httpx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_MODEL = "gpt-4o"
NOTION_API_VERSION = "2022-06-28"
NOTION_BLOCKS_URL = "https://api.notion.com/v1/blocks/{block_id}/children"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

ERROR_PREFIX = "Error: "
TOOL_ERROR_PREFIX = "erro:"


def _request_timeout() -> float:
    try:
        return float(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))
    except ValueError:
        return 20.0


def get_llm() -> ChatOpenAI:
    """Builds the ChatOpenAI client used by every LLM node (patchable in tests)."""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        temperature=0,
        timeout=_request_timeout() * 2,
        max_retries=1,
    )


# --------------------------------------------------------------------------- #
# Persona
# --------------------------------------------------------------------------- #

CHEF_PERSONA = (
    "Você é um chefe de cozinha experiente e cuidadoso, cuja função é criar receitas "
    "personalizadas para o usuário. Você é sempre respeitoso e educado: nunca usa "
    "linguagem ofensiva, discriminatória ou rude, mesmo que o usuário utilize. Você "
    "leva restrições dietéticas extremamente a sério e jamais sugere um ingrediente "
    "que as viole, incluindo derivados não óbvios (por exemplo: mel, gelatina, "
    "manteiga e whey não são veganos; cevada, malte e shoyu comum não são para "
    "celíacos)."
)


# --------------------------------------------------------------------------- #
# Structured-output schemas (LLM -> Python)
# --------------------------------------------------------------------------- #


class QueryValidation(BaseModel):
    """B1: is the query acceptable for recipe generation?"""

    is_food_related: bool = Field(
        description="True se a query for sobre comida, culinária, receitas, ingredientes ou dietas."
    )
    is_offensive: bool = Field(
        description="True se a query contiver linguagem ofensiva, discriminatória ou abusiva."
    )
    reason: str = Field(description="Explicação curta e objetiva da decisão.")


class QueryAnalysis(BaseModel):
    """B2: what does the query tell us about ingredients, diet and preferences?"""

    ingredients: List[str] = Field(
        default_factory=list,
        description=(
            "Ingredientes DISPONÍVEIS mencionados explicitamente na query, já "
            "canonicalizados (forma-base singular, variantes regionais unificadas, "
            "ex.: 'macaxeira'/'aipim' -> 'mandioca'). NUNCA inclua ingredientes "
            "mencionados sob negação, ausência ou hipótese (ex.: 'não tenho ovos', "
            "'sem leite', 'se eu tivesse farinha')."
        ),
    )
    dietary_restrictions: List[str] = Field(
        default_factory=list,
        description="Restrições dietéticas mencionadas (ex.: vegano, vegetariano, celíaco, sem glúten, sem lactose, alergias).",
    )
    preferences: List[str] = Field(
        default_factory=list,
        description="Outras preferências mencionadas (tipo de cozinha, tempo de preparo, nível de dificuldade etc.).",
    )


class DietaryCompatibility(BaseModel):
    """B6: which base ingredients survive the user's dietary restrictions?"""

    compatible: List[str] = Field(
        default_factory=list,
        description="Subconjunto dos ingredientes-base que É compatível com TODAS as restrições dietéticas informadas.",
    )
    incompatible: List[str] = Field(
        default_factory=list,
        description="Subconjunto dos ingredientes-base que viola alguma restrição dietética.",
    )


class Recipe(BaseModel):
    """B8: the final generated recipe."""

    title: str = Field(description="Título da receita.")
    ingredients: List[str] = Field(description="Ingredientes com quantidades.")
    equipment: List[str] = Field(description="Equipamentos de cozinha necessários.")
    instructions: List[str] = Field(description="Instruções de preparo, passo a passo.")
    storage: str = Field(description="Recomendações de armazenamento.")


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


def _extract_block_text(block: Dict[str, Any]) -> List[str]:
    """Extracts plain text from the Notion block types that can hold pantry items."""
    block_type = block.get("type", "")
    payload = block.get(block_type, {}) or {}

    if block_type == "table_row":
        texts = []
        for cell in payload.get("cells", []):
            text = "".join(rt.get("plain_text", "") for rt in cell).strip()
            if text:
                texts.append(text)
        return [" - ".join(texts)] if texts else []

    rich_text = payload.get("rich_text")
    if rich_text is None:
        return []
    text = "".join(rt.get("plain_text", "") for rt in rich_text).strip()
    if not text:
        return []
    if block_type in {"paragraph", "quote", "callout"} and "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def retrieve_pantry() -> List[str] | str:
    """Obtém os ingredientes disponíveis na despensa do usuário, através de uma página do Notion.

    Retorna a lista completa de ingredientes (vazia se a despensa não tiver itens),
    ou uma string começando com ``"erro:"`` em caso de credenciais ausentes, página
    inacessível ou timeout.
    """
    api_key = os.getenv("NOTION_API_KEY", "").strip()
    page_id = os.getenv("NOTION_PAGE_ID", "").strip()
    if not api_key or not page_id:
        return f"{TOOL_ERROR_PREFIX} credenciais do Notion ausentes (NOTION_API_KEY/NOTION_PAGE_ID)."

    try:
        ingredients: List[str] = []
        cursor: Optional[str] = None
        with httpx.Client(timeout=_request_timeout()) as client:
            while True:
                params: Dict[str, Any] = {"page_size": 100}
                if cursor:
                    params["start_cursor"] = cursor
                resp = client.get(
                    NOTION_BLOCKS_URL.format(block_id=page_id),
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Notion-Version": NOTION_API_VERSION,
                    },
                    params=params,
                )
                resp.raise_for_status()
                data = resp.json()
                for block in data.get("results", []):
                    ingredients.extend(_extract_block_text(block))
                if not data.get("has_more"):
                    break
                cursor = data.get("next_cursor")
    except httpx.TimeoutException:
        return f"{TOOL_ERROR_PREFIX} timeout ao acessar a página do Notion."
    except httpx.HTTPStatusError as exc:
        return f"{TOOL_ERROR_PREFIX} página do Notion inacessível (HTTP {exc.response.status_code})."
    except Exception as exc:  # noqa: BLE001
        return f"{TOOL_ERROR_PREFIX} falha ao acessar a despensa ({exc})."

    seen = set()
    unique: List[str] = []
    for item in ingredients:
        key = item.lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def search_recipes(ingredients: List[str]) -> List[Dict[str, str]] | str:
    """Obtém receitas da web, através do Tavily, como referência de preparo.

    Recebe os ingredientes identificados na query e retorna até 5 resultados
    (cada um com título, URL e trecho do conteúdo), ou lista vazia se não
    houver resultados, ou uma string começando com ``"erro:"`` em caso de
    credenciais ausentes, página inacessível ou timeout.
    """
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return f"{TOOL_ERROR_PREFIX} credencial do Tavily ausente (TAVILY_API_KEY)."
    if not ingredients:
        return []

    query = "receita com " + ", ".join(ingredients)
    try:
        with httpx.Client(timeout=_request_timeout()) as client:
            resp = client.post(
                TAVILY_SEARCH_URL,
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": 5,
                    "include_answer": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        return f"{TOOL_ERROR_PREFIX} timeout ao buscar receitas no Tavily."
    except httpx.HTTPStatusError as exc:
        return f"{TOOL_ERROR_PREFIX} falha ao acessar o Tavily (HTTP {exc.response.status_code})."
    except Exception as exc:  # noqa: BLE001
        return f"{TOOL_ERROR_PREFIX} falha ao buscar receitas ({exc})."

    results = []
    for item in (data.get("results") or [])[:5]:
        results.append(
            {
                "title": str(item.get("title", "")).strip(),
                "url": str(item.get("url", "")).strip(),
                "content": str(item.get("content", "")).strip()[:1500],
            }
        )
    return results


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #

class AgentState(TypedDict, total=False):
    query: str
    valid: bool
    ingredients_query: List[str]
    dietary_restrictions: List[str]
    preferences: List[str]
    tool_called: Optional[Literal["search_recipes", "retrieve_pantry"]]
    search_results: List[Dict[str, str]]
    pantry_ingredients: List[str]
    base_ingredients: List[str]
    compatible_ingredients: List[str]
    recipe: Optional[Dict[str, Any]]
    error: Optional[str]


def _fail(message: str) -> Dict[str, Any]:
    """Helper that produces the error state update (``Error: [explicação]``)."""
    message = message.strip()
    if not message.startswith(ERROR_PREFIX):
        message = ERROR_PREFIX + message
    return {"error": message}


def _describe_exception(exc: Exception, context: str) -> str:
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return f"{context}: tempo limite excedido. Tente novamente mais tarde."
    if isinstance(exc, httpx.HTTPStatusError):
        return f"{context}: falhou com status HTTP {exc.response.status_code}."
    text = str(exc).strip() or exc.__class__.__name__
    if "api key" in text.lower() or "api_key" in text.lower() or "missing" in text.lower() or "ausente" in text.lower():
        return f"{context}: problema de configuração ({text})."
    return f"{context}: falhou ({text})."


def _strip_tool_error(message: str) -> str:
    return message[len(TOOL_ERROR_PREFIX):].strip()


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


def validate_query(state: AgentState) -> Dict[str, Any]:
    """B1: reject empty, offensive or non food/cooking related queries."""
    query = (state.get("query") or "").strip()
    if not query:
        return {"valid": False, **_fail("a consulta está vazia. Descreva o prato, ingredientes ou preferências desejadas.")}

    try:
        llm = get_llm().with_structured_output(QueryValidation)
        verdict: QueryValidation = llm.invoke(
            [
                SystemMessage(
                    content=CHEF_PERSONA
                    + "\n\nClassifique a query do usuário quanto a: (1) estar relacionada a comida, "
                    "culinária, receitas, ingredientes ou dietas; (2) conter linguagem ofensiva. Seja "
                    "rigoroso: saudações genéricas, perguntas de programação, matemática, política etc. "
                    "NÃO são relacionadas a comida."
                ),
                HumanMessage(content=query),
            ]
        )
    except Exception as exc:  # B7
        return {"valid": False, **_fail(_describe_exception(exc, "validação da query"))}

    if verdict.is_offensive:
        return {"valid": False, **_fail("a consulta contém linguagem ofensiva. Reformule de forma respeitosa e terei prazer em ajudar.")}
    if not verdict.is_food_related:
        return {"valid": False, **_fail(f"a consulta não está relacionada a comida ou culinária ({verdict.reason}).")}
    return {"valid": True, "error": None}


def identify_ingredients(state: AgentState) -> Dict[str, Any]:
    """B2: extract explicit ingredients, dietary restrictions and preferences."""
    try:
        llm = get_llm().with_structured_output(QueryAnalysis)
        analysis: QueryAnalysis = llm.invoke(
            [
                SystemMessage(
                    content=CHEF_PERSONA
                    + "\n\nExtraia da query do usuário: (1) ingredientes DISPONÍVEIS mencionados "
                    "explicitamente, canonicalizados (forma-base, singular, unificando variantes "
                    "regionais como 'macaxeira'/'aipim' -> 'mandioca', e formas flexionadas como "
                    "'dois tomates bem maduros' -> 'tomate'); NUNCA inclua ingredientes mencionados "
                    "sob negação, ausência ou hipótese (ex.: 'não tenho ovos', 'sem leite', 'se eu "
                    "tivesse farinha'); (2) restrições dietéticas mencionadas; (3) outras preferências. "
                    "Retorne listas vazias quando nada for mencionado."
                ),
                HumanMessage(content=state["query"]),
            ]
        )
    except Exception as exc:  # B7
        return _fail(_describe_exception(exc, "identificação de ingredientes"))

    return {
        "ingredients_query": [i.strip() for i in analysis.ingredients if i.strip()],
        "dietary_restrictions": [r.strip() for r in analysis.dietary_restrictions if r.strip()],
        "preferences": [p.strip() for p in analysis.preferences if p.strip()],
    }


def call_search_recipes(state: AgentState) -> Dict[str, Any]:
    """B3: the query mentions ingredients -> search reference recipes on Tavily."""
    ingredients = state.get("ingredients_query") or []
    try:
        result = search_recipes(ingredients)
    except Exception as exc:  # B7
        return {"tool_called": "search_recipes", **_fail(_describe_exception(exc, "busca de receitas de referência (search_recipes)"))}

    if isinstance(result, str) and result.startswith(TOOL_ERROR_PREFIX):
        return {"tool_called": "search_recipes", **_fail(_strip_tool_error(result))}

    return {
        "tool_called": "search_recipes",
        "search_results": result,
        "base_ingredients": list(ingredients),
    }


def call_retrieve_pantry(state: AgentState) -> Dict[str, Any]:
    """B4/B5: no ingredients in the query -> read the pantry from Notion."""
    try:
        result = retrieve_pantry()
    except Exception as exc:  # B7
        return {"tool_called": "retrieve_pantry", **_fail(_describe_exception(exc, "consulta à despensa (retrieve_pantry)"))}

    if isinstance(result, str) and result.startswith(TOOL_ERROR_PREFIX):
        return {"tool_called": "retrieve_pantry", **_fail(_strip_tool_error(result))}

    pantry = [str(i).strip() for i in (result or []) if str(i).strip()]
    if not pantry:  # B5
        return {
            "tool_called": "retrieve_pantry",
            "pantry_ingredients": [],
            **_fail("a despensa não possui ingredientes disponíveis."),
        }

    return {
        "tool_called": "retrieve_pantry",
        "pantry_ingredients": pantry,
        "base_ingredients": pantry,
    }


def check_dietary_compatibility(state: AgentState) -> Dict[str, Any]:
    """B6: keep only the base ingredients compatible with the declared restrictions."""
    base_ingredients = state.get("base_ingredients") or []
    restrictions = state.get("dietary_restrictions") or []

    if not restrictions:
        return {"compatible_ingredients": list(base_ingredients)}

    try:
        llm = get_llm().with_structured_output(DietaryCompatibility)
        result: DietaryCompatibility = llm.invoke(
            [
                SystemMessage(
                    content=CHEF_PERSONA
                    + "\n\nDados os ingredientes-base e as restrições dietéticas abaixo, classifique cada "
                    "ingrediente como compatível ou incompatível com TODAS as restrições. Uma restrição "
                    "dietética implica todos os seus ingredientes derivados, mesmo os não citados "
                    "explicitamente (ex.: 'vegano' exclui mel, gelatina, manteiga e whey; 'celíaco' exclui "
                    "cevada, malte e shoyu comum)."
                ),
                HumanMessage(
                    content=(
                        f"Ingredientes-base: {', '.join(base_ingredients)}\n"
                        f"Restrições dietéticas: {', '.join(restrictions)}"
                    )
                ),
            ]
        )
    except Exception as exc:  # B7
        return _fail(_describe_exception(exc, "avaliação de compatibilidade dietética"))

    compatible = [i.strip() for i in result.compatible if i.strip()]
    if not compatible:  # B6
        return {
            "compatible_ingredients": [],
            **_fail(
                "nenhum ingrediente disponível é compatível com as restrições dietéticas informadas "
                f"({', '.join(restrictions)})."
            ),
        }

    return {"compatible_ingredients": compatible}


def generate_recipe(state: AgentState) -> Dict[str, Any]:
    """B8: turn the compatible base ingredients into a complete recipe."""
    compatible = state.get("compatible_ingredients") or []
    restrictions = state.get("dietary_restrictions") or []
    preferences = state.get("preferences") or []
    tool_called = state.get("tool_called")

    if tool_called == "search_recipes":
        coverage_rule = (
            "A receita DEVE conter TODOS os ingredientes-base listados abaixo (a lista já exclui "
            "eventuais itens incompatíveis com as restrições dietéticas)."
        )
    else:
        coverage_rule = (
            "A receita DEVE conter PELO MENOS UM dos ingredientes-base listados abaixo, que vêm da "
            "despensa do usuário; use mais de um sempre que fizer sentido."
        )

    system = (
        CHEF_PERSONA
        + "\n\nCrie UMA receita completa seguindo estas regras:\n"
        "1. " + coverage_rule + "\n"
        "2. Você pode adicionar ingredientes extras apenas se for estritamente necessário "
        "(ex.: sal, água, óleo); mantenha os extras ao mínimo.\n"
        "3. A receita DEVE respeitar TODAS as restrições dietéticas informadas. Nunca inclua um "
        "ingrediente que as viole, nem mesmo como item opcional.\n"
        "4. Siga as preferências do usuário sempre que possível.\n"
        "5. Forneça ingredientes com quantidades, equipamentos necessários, instruções de preparo "
        "passo a passo e recomendações de armazenamento.\n"
        "6. Use as receitas de referência (quando houver) apenas como inspiração de preparo; elas "
        "NÃO são fonte de ingredientes-base nem a receita final.\n"
        "7. Escreva a receita no mesmo idioma da query do usuário.\n"
        "8. Nunca use linguagem ofensiva."
    )

    human_parts = [
        f"Query do usuário: {state['query']}",
        f"Ingredientes-base disponíveis: {', '.join(compatible)}",
        f"Restrições dietéticas: {', '.join(restrictions) if restrictions else 'nenhuma'}",
        f"Preferências: {', '.join(preferences) if preferences else 'nenhuma'}",
        f"Origem dos ingredientes-base: {tool_called}",
    ]
    if tool_called == "search_recipes":
        results = state.get("search_results") or []
        if results:
            refs = "\n".join(f"- {r.get('title', '')} ({r.get('url', '')}): {r.get('content', '')}" for r in results)
            human_parts.append("Receitas de referência (apenas inspiração de preparo):\n" + refs)
        else:
            human_parts.append("Nenhuma receita de referência foi encontrada; baseie-se no seu conhecimento culinário.")

    try:
        llm = get_llm().with_structured_output(Recipe)
        recipe: Recipe = llm.invoke([SystemMessage(content=system), HumanMessage(content="\n\n".join(human_parts))])
    except Exception as exc:  # B7
        return _fail(_describe_exception(exc, "geração da receita"))

    recipe_dict = {
        "Título": recipe.title.strip(),
        "Ingredientes": [str(i).strip() for i in recipe.ingredients if str(i).strip()],
        "Equipamentos": [str(e).strip() for e in recipe.equipment if str(e).strip()],
        "Instruções": [str(s).strip() for s in recipe.instructions if str(s).strip()],
        "Armazenamento": recipe.storage.strip(),
    }
    if not all(recipe_dict.values()):
        return _fail("o modelo gerou uma receita incompleta.")

    return {"recipe": recipe_dict}


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def _has_error(state: AgentState) -> bool:
    return bool(state.get("error"))


def route_after_validation(state: AgentState) -> str:
    return "error" if _has_error(state) or not state.get("valid") else "identify_ingredients"


def route_tool(state: AgentState) -> str:
    """B2/B3/B4: exactly ONE tool is chosen, based on the presence of ingredients."""
    if _has_error(state):
        return "error"
    return "call_search_recipes" if state.get("ingredients_query") else "call_retrieve_pantry"


def route_after_tool(state: AgentState) -> str:
    return "error" if _has_error(state) else "check_dietary_compatibility"


def route_after_dietary(state: AgentState) -> str:
    return "error" if _has_error(state) else "generate_recipe"


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #


def build_graph():
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("validate_query", validate_query)
    workflow.add_node("identify_ingredients", identify_ingredients)
    workflow.add_node("call_search_recipes", call_search_recipes)
    workflow.add_node("call_retrieve_pantry", call_retrieve_pantry)
    workflow.add_node("check_dietary_compatibility", check_dietary_compatibility)
    workflow.add_node("generate_recipe", generate_recipe)

    workflow.set_entry_point("validate_query")
    workflow.add_conditional_edges(
        "validate_query", route_after_validation, {"identify_ingredients": "identify_ingredients", "error": END}
    )
    workflow.add_conditional_edges(
        "identify_ingredients",
        route_tool,
        {"call_search_recipes": "call_search_recipes", "call_retrieve_pantry": "call_retrieve_pantry", "error": END},
    )
    workflow.add_conditional_edges(
        "call_search_recipes", route_after_tool, {"check_dietary_compatibility": "check_dietary_compatibility", "error": END}
    )
    workflow.add_conditional_edges(
        "call_retrieve_pantry", route_after_tool, {"check_dietary_compatibility": "check_dietary_compatibility", "error": END}
    )
    workflow.add_conditional_edges(
        "check_dietary_compatibility", route_after_dietary, {"generate_recipe": "generate_recipe", "error": END}
    )
    workflow.add_edge("generate_recipe", END)

    return workflow.compile()


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def _initial_state(query: Any) -> AgentState:
    return AgentState(
        query=query if isinstance(query, str) else ("" if query is None else str(query)),
        valid=False,
        ingredients_query=[],
        dietary_restrictions=[],
        preferences=[],
        tool_called=None,
        search_results=[],
        pantry_ingredients=[],
        base_ingredients=[],
        compatible_ingredients=[],
        recipe=None,
        error=None,
    )


def agent(query: str) -> dict:
    """
    Entrypoint. Runs the recipe-generation workflow for ``query`` and returns
    the final state as a ``dict``.

    ``state["recipe"]`` holds the generated recipe on success. ``state["error"]``
    is ``None`` on success, or a string starting with ``"Error: "`` explaining
    why no recipe was generated.
    """
    state = _initial_state(query)

    if not os.getenv("OPENAI_API_KEY", "").strip():
        state.update(_fail("a variável de ambiente OPENAI_API_KEY não está definida; não é possível acessar o modelo de linguagem."))
        return dict(state)

    try:
        graph = build_graph()
        final_state = graph.invoke(state)
    except Exception as exc:  # B7 - last line of defence
        state.update(_fail(_describe_exception(exc, "execução do agente")))
        return dict(state)

    return dict(final_state)


if __name__ == "__main__":
    import sys

    user_query = "Sou vegano. Tenho mel, gelatina e manteiga. Quero uma sobremesa."
    result = agent(user_query)
    if result.get("error"):
        print(result["error"])
    else:
        print(result["recipe"])
