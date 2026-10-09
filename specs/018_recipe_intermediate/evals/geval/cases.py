"""Um caso por fluxo da spec 018_recipe_intermediate.

Cada caso define a query e o que cada tool deve retornar no cenário
(`tool_returns`). Tools ausentes em `tool_returns` usam DEFAULT_TOOL_RETURNS,
assim nenhum caso depende de Notion ou Tavily reais.
"""

DEFAULT_SEARCH_RESULTS = [
    {
        "title": "Receita caseira simples",
        "url": "https://example.com/receita-caseira",
        "content": "Refogue os ingredientes em fogo médio com azeite, alho e cebola; ajuste o sal e sirva.",
    },
    {
        "title": "Preparo rápido na frigideira",
        "url": "https://example.com/preparo-rapido",
        "content": "Corte os ingredientes em pedaços pequenos e salteie em frigideira bem quente por poucos minutos.",
    },
]

DEFAULT_PANTRY = ["arroz", "feijão", "cebola", "alho"]

DEFAULT_TOOL_RETURNS = {
    "search_recipes": DEFAULT_SEARCH_RESULTS,
    "retrieve_pantry": DEFAULT_PANTRY,
}

CASES = [
    {
        "id": "B1-1",
        "query": "Quero uma receita de jantar leve com abobrinha e queijo.",
        "tool_returns": {},
    },
    {
        "id": "B1-2",
        "query": "Qual é a capital da Austrália e quantos habitantes ela tem?",
        "tool_returns": {},
    },
    {
        "id": "B2-1",
        "query": "Tenho macaxeira, dois tomates bem maduros e um resto de frango. O que faço pro almoço?",
        "tool_returns": {},
    },
    {
        "id": "B2-2",
        "query": "Quero uma sobremesa rápida para hoje à noite.",
        "tool_returns": {"retrieve_pantry": ["banana", "aveia", "canela"]},
    },
    {
        "id": "B2-3",
        "query": "Não tenho ovo e, se eu tivesse leite, faria um bolo. Me sugere algo para o café da manhã.",
        "tool_returns": {"retrieve_pantry": ["pão integral", "banana", "pasta de amendoim"]},
    },
    {
        "id": "B3-1",
        "query": "Quero um prato com berinjela e grão-de-bico.",
        "tool_returns": {
            "search_recipes": [
                {"title": f"Berinjela com grão-de-bico {i}", "url": f"https://example.com/berinjela-grao-{i}",
                 "content": "Asse a berinjela em cubos, misture ao grão-de-bico cozido e tempere com cominho e limão."}
                for i in range(1, 6)
            ]
        },
    },
    {
        "id": "B3-2",
        "query": "Faço o que com quiabo e camarão?",
        "tool_returns": {"search_recipes": []},
    },
    {
        "id": "B3-3",
        "query": "Tenho arroz e brócolis, e pode usar o que tiver na minha despensa também.",
        "tool_returns": {},
    },
    {
        "id": "B3-4",
        "query": "Quero uma receita com batata-doce.",
        "tool_returns": {
            "search_recipes": [
                {"title": "Gratinado de batata-doce com bacon", "url": "https://example.com/gratinado-batata-doce",
                 "content": "Ingredientes: batata-doce, bacon, creme de leite e queijo parmesão. Monte camadas e leve ao forno por 40 minutos."},
                {"title": "Purê cremoso de batata-doce", "url": "https://example.com/pure-batata-doce",
                 "content": "Cozinhe a batata-doce, amasse com creme de leite e finalize com parmesão ralado."},
            ]
        },
    },
    {
        "id": "B4-1",
        "query": "Me sugere uma receita para o jantar.",
        "tool_returns": {"retrieve_pantry": ["lentilha", "cenoura", "cebola"]},
    },
    {
        "id": "B5-1",
        "query": "O que posso cozinhar hoje com o que tenho em casa?",
        "tool_returns": {"retrieve_pantry": []},
    },
    {
        "id": "B6-1",
        "query": "Sou vegano e quero uma receita usando mel e manteiga.",
        "tool_returns": {},
    },
    {
        "id": "B7-1",
        "query": "Quero uma receita com cogumelos e espinafre.",
        "tool_returns": {"search_recipes": "erro: timeout ao acessar a API do Tavily"},
    },
    {
        "id": "B8-1",
        "query": "Quero um jantar rápido e picante com frango, pimentão e arroz.",
        "tool_returns": {},
    },
    {
        "id": "B8-2",
        "query": "Quero um lanche saudável para a tarde.",
        "tool_returns": {"retrieve_pantry": ["iogurte natural", "morango", "granola"]},
    },
    {
        "id": "B8-3",
        "query": "Sou celíaco e quero uma receita com frango, cevada, shoyu e abobrinha.",
        "tool_returns": {},
    },
]


def tool_returns_for(case: dict) -> dict:
    return {**DEFAULT_TOOL_RETURNS, **case["tool_returns"]}
