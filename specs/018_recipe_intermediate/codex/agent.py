"""Personalized recipe agent with mutually exclusive pantry/web tools."""
from __future__ import annotations

import json
import os
import re
from typing import Any


def retrieve_pantry() -> list[str] | str:
    """Read pantry items from a Notion page."""
    try:
        page_id, token = os.getenv("NOTION_PAGE_ID"), os.getenv("NOTION_API_KEY")
        if not page_id or not token:
            return "erro: credenciais do Notion ausentes"
        import requests
        r = requests.get(f"https://api.notion.com/v1/blocks/{page_id}/children",
                         headers={"Authorization": f"Bearer {token}",
                                  "Notion-Version": "2022-06-28"}, timeout=10)
        r.raise_for_status()
        items: list[str] = []
        for block in r.json().get("results", []):
            typ = block.get("type", "")
            text = block.get(typ, {}).get("rich_text", [])
            value = "".join(x.get("plain_text", "") for x in text).strip()
            if value:
                items.extend(x.strip() for x in re.split(r"[,;\n]", value) if x.strip())
        return items
    except Exception as exc:
        return f"erro: falha ao acessar a despensa ({exc})"


def search_recipes(ingredients: list[str]) -> list[dict[str, str]] | str:
    """Search Tavily for preparation references; never supplies base ingredients."""
    try:
        key = os.getenv("TAVILY_API_KEY")
        if not key:
            return "erro: credencial do Tavily ausente"
        from tavily import TavilyClient
        response = TavilyClient(api_key=key).search(
            query="receitas e técnicas culinárias: " + ", ".join(ingredients),
            max_results=5)
        return [{"title": x.get("title", ""), "url": x.get("url", ""),
                 "snippet": x.get("content", "")} for x in response.get("results", [])[:5]]
    except Exception as exc:
        return f"erro: falha na busca de receitas ({exc})"


_DIET = {
    "vegano": {"carne", "frango", "peixe", "ovo", "leite", "queijo", "manteiga", "mel", "gelatina", "whey"},
    "vegetariano": {"carne", "frango", "peixe", "presunto", "bacon", "gelatina"},
    "celíaco": {"trigo", "cevada", "malte", "shoyu comum", "farinha de trigo"},
    "sem glúten": {"trigo", "cevada", "malte", "shoyu comum", "farinha de trigo"},
    "sem lactose": {"leite", "queijo", "manteiga", "creme de leite", "whey"},
}


def _valid(q: str) -> bool:
    return bool(q and q.strip() and not re.search(r"\b(idiota|matar| estupra)\b", q.lower())
                and re.search(r"(receita|cozin|comida|prato|ingrediente|bolo|sopa|molho|arroz|macarr|frango|carne|tomate|batata)", q.lower()))


def _ingredients(q: str) -> list[str]:
    # Conservative extraction: only positive, food-like noun phrases.
    neg = re.compile(r"(?:sem|não tenho|nao tenho|não há|nao ha|sem a|sem o|sem um)\s+([\wÀ-ÿ -]+)", re.I)
    excluded = {m.group(1).strip().split()[0].lower() for m in neg.finditer(q)}
    words = re.findall(r"\b(?:arroz|feijão|feijao|macarrão|macarrao|tomate|batata|mandioca|aipim|macaxeira|frango|carne|peixe|ovo|leite|queijo|farinha|cenoura|abobrinha|cogumelo|grão-de-bico|grao-de-bico|lentilha|banana|maçã|maca|chocolate|mel)\b", q.lower())
    out = []
    for w in words:
        w = {"aipim": "mandioca", "macaxeira": "mandioca", "feijao": "feijão", "macarrao": "macarrão", "maca": "maçã", "grao-de-bico": "grão-de-bico"}.get(w, w)
        if w not in excluded and w not in out:
            out.append(w)
    return out


def _compatible(items: list[str], q: str) -> list[str]:
    low = q.lower()
    forbidden = set().union(*(v for k, v in _DIET.items() if k in low))
    return [i for i in items if not any(x in i.lower() or i.lower() in x for x in forbidden)]


def _recipe(q: str, items: list[str], refs: list[dict[str, str]]) -> dict[str, Any]:
    title = "Receita personalizada de " + " com ".join(items[:3])
    return {"Título": title, "Ingredientes": [f"{('1 porção de' if i else '')} {i}".strip() for i in items] + ["sal a gosto", "azeite a gosto"],
            "Equipamentos": ["panela ou frigideira", "faca", "tábua de corte", "colher"],
            "Instruções": ["Higienize e corte os ingredientes.", "Aqueça o azeite e refogue os ingredientes mais firmes.", "Adicione os demais ingredientes e cozinhe até ficarem macios.", "Ajuste o sal e sirva."],
            "Armazenamento": "Guarde em recipiente fechado na geladeira por até 3 dias; reaqueça completamente.",
            "referências": refs}


def agent(query: str) -> dict[str, Any]:
    """Entrypoint required by the benchmark; always returns a state dictionary."""
    state: dict[str, Any] = {"query": query, "valid": False, "ingredients": [], "recipe": None, "message": ""}
    try:
        if not _valid(query):
            state["message"] = "Error: a consulta deve ser válida e relacionada a comida ou culinária."
            return state
        found = _ingredients(query)
        state["ingredients"] = found
        if found:
            bases, refs = found, search_recipes(found)  # exactly one tool
            if isinstance(refs, str) and refs.startswith("erro:"):
                state["message"] = "Error: " + refs[5:].strip(); return state
        else:
            bases, pantry = retrieve_pantry(), None  # exactly one tool
            pantry = bases
            if isinstance(pantry, str) and pantry.startswith("erro:"):
                state["message"] = "Error: " + pantry[5:].strip(); return state
            if not pantry:
                state["message"] = "Error: a despensa não contém ingredientes."; return state
            bases, refs = pantry, []
        compatible = _compatible(list(bases), query)
        if not compatible:
            state["message"] = "Error: nenhum ingrediente é compatível com as restrições dietéticas."; return state
        state["valid"] = True
        state["recipe"] = _recipe(query, compatible, refs)
        return state
    except Exception as exc:
        state["message"] = f"Error: falha durante a execução ({exc})"
        return state


__all__ = ["agent", "retrieve_pantry", "search_recipes"]
