"""Métricas GEval, uma por fluxo da spec 018_recipe_intermediate.

Todas avaliam:
- INPUT: a query do usuário;
- ACTUAL_OUTPUT: o `state` (dict) retornado por `agent(...)`, serializado em JSON;
- TOOLS_CALLED: as chamadas reais a `retrieve_pantry` / `search_recipes`
  (com argumentos e retorno), capturadas pelo teste.

O `input` de cada test case também traz, ao final, o cenário das tools
(o que cada tool foi configurada para retornar), para o juiz conseguir
verificar o uso das referências e da despensa.
"""

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

PARAMS = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
    LLMTestCaseParams.TOOLS_CALLED,
]
THRESHOLD = 0.7

RECEITA_COMPLETA = (
    "A saída contém uma receita com os 5 campos presentes e NÃO vazios: Título, "
    "Ingredientes (cada um com quantidade), Equipamentos, Instruções (passo a passo) "
    "e Armazenamento. A saída não é uma mensagem iniciada por 'Error:'."
)


def _geval(name: str, steps: list[str]) -> GEval:
    return GEval(
        name=name,
        evaluation_steps=steps,
        evaluation_params=PARAMS,
        threshold=THRESHOLD,
    )


GEVAL_METRICS: dict[str, GEval] = {
    "B1-1": _geval(
        "B1-1 Query válida",
        [
            "A query é não vazia, não ofensiva e relacionada a comida/culinária, portanto deve ser classificada como VÁLIDA.",
            "Verifique se o estado indica a query como válida (indicador booleano de validade verdadeiro, se presente).",
            "Verifique se a saída NÃO é uma mensagem iniciada por 'Error:' por motivo de validação.",
            "Verifique se o fluxo seguiu para a identificação de ingredientes (há lista de ingredientes identificados no estado e/ou ocorreu chamada de tool ou geração de receita).",
            "Penalize fortemente se a query foi rejeitada como inválida.",
        ],
    ),
    "B1-2": _geval(
        "B1-2 Query inválida",
        [
            "A query não tem relação com comida, culinária, receitas ou ingredientes, portanto deve ser classificada como INVÁLIDA.",
            "Verifique se o estado indica a query como inválida (indicador booleano de validade falso, se presente).",
            "Verifique se a resposta final começa exatamente com 'Error:' e explica o motivo da invalidez.",
            "Verifique se a lista de tools chamadas está VAZIA (nem retrieve_pantry nem search_recipes).",
            "Verifique se nenhuma receita foi gerada e se o agente não respondeu/inventou uma resposta para a pergunta fora do domínio.",
            "Qualquer violação de um desses pontos deve resultar em nota baixa.",
        ],
    ),
    "B2-1": _geval(
        "B2-1 Normalização de ingredientes",
        [
            "Ingredientes esperados na lista identificada: mandioca (de 'macaxeira'), tomate (de 'dois tomates bem maduros') e frango (de 'um resto de frango').",
            "Verifique se a lista de ingredientes identificados no estado (ou os argumentos passados a search_recipes) contém exatamente esses três ingredientes, normalizados para a forma canônica, sem quantidades, adjetivos ou duplicatas.",
            "Penalize se 'macaxeira' não foi normalizada para mandioca, se aparecem termos como 'dois tomates bem maduros' ou 'resto de frango' sem normalização, ou se surgem ingredientes não mencionados na query.",
            "Verifique se, por a lista não ser vazia, o fluxo seguiu para search_recipes e não para retrieve_pantry.",
        ],
    ),
    "B2-2": _geval(
        "B2-2 Query sem ingredientes",
        [
            "A query é válida mas não menciona nenhum ingrediente.",
            "Verifique se a lista de ingredientes identificados no estado é vazia (se presente).",
            "Verifique se retrieve_pantry foi chamada exatamente uma vez e search_recipes não foi chamada.",
            "Verifique se a resposta final usa itens da despensa (retorno de retrieve_pantry) como ingredientes-base.",
            "Penalize se o agente inventou ingredientes como se tivessem sido citados na query ou chamou search_recipes.",
        ],
    ),
    "B2-3": _geval(
        "B2-3 Ingredientes sob negação/hipótese",
        [
            "Na query, 'ovo' aparece sob negação ('não tenho ovo') e 'leite' sob hipótese ('se eu tivesse leite'); nenhum ingrediente é afirmado como disponível.",
            "Verifique se ovo e leite NÃO aparecem na lista de ingredientes identificados no estado e se a lista é vazia.",
            "Verifique se, por a lista ser vazia, retrieve_pantry foi chamada exatamente uma vez e search_recipes não foi chamada.",
            "Verifique se a receita final não usa ovo nem leite como ingredientes-base e usa ao menos um item retornado por retrieve_pantry.",
            "Penalize fortemente se ovo ou leite foram tratados como ingredientes disponíveis.",
        ],
    ),
    "B3-1": _geval(
        "B3-1 Busca de referências",
        [
            "A query menciona os ingredientes berinjela e grão-de-bico.",
            "Verifique se search_recipes foi chamada EXATAMENTE uma vez, com uma lista de ingredientes equivalente a [berinjela, grão-de-bico].",
            "Verifique se retrieve_pantry NÃO foi chamada.",
            "Verifique se os resultados de search_recipes (descritos no cenário do input) foram usados apenas como referência de preparo: a receita final não é cópia de nenhum resultado e é centrada em berinjela e grão-de-bico.",
            "Penalize chamadas repetidas a search_recipes, argumentos com ingredientes a mais/a menos ou qualquer chamada a retrieve_pantry.",
        ],
    ),
    "B3-2": _geval(
        "B3-2 Busca sem resultados",
        [
            "No cenário, search_recipes retorna lista vazia.",
            "Verifique se a resposta final NÃO é uma mensagem iniciada por 'Error:' — ausência de referências não é falha.",
            RECEITA_COMPLETA,
            "Verifique se a receita usa quiabo e camarão (ingredientes da query) e não cita receitas de referência inexistentes (títulos/URLs inventados).",
            "Verifique se retrieve_pantry NÃO foi chamada como substituta da busca vazia.",
        ],
    ),
    "B3-3": _geval(
        "B3-3 Nunca chamar as duas tools",
        [
            "A query cita ingredientes (arroz, brócolis) e também pede para usar a despensa.",
            "Verifique se search_recipes foi chamada exatamente uma vez com os ingredientes da query.",
            "Verifique se retrieve_pantry NÃO foi chamada em nenhum momento da execução, apesar do pedido do usuário.",
            "Qualquer chamada a retrieve_pantry deve resultar em nota mínima, pois viola a constraint de nunca chamar ambas as ferramentas na mesma execução.",
            "Verifique se a resposta final é uma receita que contém arroz e brócolis.",
        ],
    ),
    "B3-4": _geval(
        "B3-4 Referências não são ingredientes-base",
        [
            "A query cita apenas batata-doce. No cenário, os resultados de search_recipes trazem bacon, creme de leite e queijo parmesão, que NÃO foram citados pelo usuário.",
            "Verifique se a lista de ingredientes identificados/ingredientes-base no estado contém apenas batata-doce, sem bacon, creme de leite ou parmesão.",
            "Verifique se a receita final é centrada em batata-doce e NÃO reproduz um dos resultados de search_recipes (mesmo título, mesma lista de ingredientes e mesmos passos).",
            "Penalize se bacon, creme de leite ou parmesão aparecem como ingredientes principais da receita, como se tivessem vindo do usuário.",
            RECEITA_COMPLETA,
        ],
    ),
    "B4-1": _geval(
        "B4-1 Consulta à despensa",
        [
            "A query não cita ingredientes, logo a lista identificada é vazia.",
            "Verifique se retrieve_pantry foi chamada EXATAMENTE uma vez e search_recipes NÃO foi chamada.",
            "Verifique se a receita final usa como ingredientes-base itens retornados por retrieve_pantry (lentilha, cenoura, cebola no cenário).",
            RECEITA_COMPLETA,
        ],
    ),
    "B5-1": _geval(
        "B5-1 Despensa vazia",
        [
            "A query não cita ingredientes e, no cenário, retrieve_pantry retorna lista vazia.",
            "Verifique se retrieve_pantry foi chamada exatamente uma vez e search_recipes não foi chamada.",
            "Verifique se a resposta final começa exatamente com 'Error:' e explica que faltam ingredientes.",
            "Verifique se nenhuma receita foi gerada (sem título, ingredientes ou instruções de receita).",
            "Penalize fortemente se o agente inventou ingredientes ou gerou uma receita mesmo assim.",
        ],
    ),
    "B6-1": _geval(
        "B6-1 Incompatibilidade dietética total",
        [
            "O usuário declara ser vegano e cita apenas mel e manteiga. Pela policy de ingredientes derivados, 'vegano' exclui mel e manteiga, então nenhum ingrediente-base é compatível.",
            "Verifique se a resposta final começa exatamente com 'Error:' e explica a incompatibilidade entre os ingredientes e a restrição vegana.",
            "Verifique se nenhuma receita foi gerada.",
            "Verifique se retrieve_pantry NÃO foi chamada para buscar ingredientes substitutos.",
            "Penalize fortemente se o agente gerou uma receita (com ou sem mel/manteiga).",
        ],
    ),
    "B7-1": _geval(
        "B7-1 Falha de tool",
        [
            "No cenário, search_recipes retorna uma mensagem iniciada por 'erro:' (timeout).",
            "Verifique se a execução terminou sem exceção não tratada (há um estado/resposta final).",
            "Verifique se a resposta final começa exatamente com 'Error:' e explica qual falha ocorreu (falha/timeout na busca de receitas).",
            "Verifique se nenhuma receita foi gerada.",
            "Verifique se retrieve_pantry NÃO foi chamada como tentativa alternativa.",
        ],
    ),
    "B8-1": _geval(
        "B8-1 Receita com ingredientes da query",
        [
            "A query cita frango, pimentão e arroz, sem restrições dietéticas, e pede um jantar rápido e picante.",
            RECEITA_COMPLETA,
            "Verifique se a seção Ingredientes contém TODOS os ingredientes da query: frango, pimentão e arroz.",
            "Verifique se a receita considera as preferências do usuário: preparo rápido e sabor picante (por exemplo, pimenta ou ingrediente picante e tempo de preparo curto).",
            "Verifique se search_recipes foi chamada uma vez, retrieve_pantry não foi chamada, e se as referências do cenário foram usadas apenas como inspiração de preparo.",
        ],
    ),
    "B8-2": _geval(
        "B8-2 Receita com ingredientes da despensa",
        [
            "A query não cita ingredientes; no cenário, retrieve_pantry retorna iogurte natural, morango e granola.",
            "Verifique se retrieve_pantry foi chamada exatamente uma vez e search_recipes não foi chamada.",
            RECEITA_COMPLETA,
            "Verifique se a seção Ingredientes contém ao menos um item retornado por retrieve_pantry.",
            "Verifique se a receita respeita a preferência da query (lanche saudável para a tarde).",
        ],
    ),
    "B8-3": _geval(
        "B8-3 Receita sem ingredientes incompatíveis",
        [
            "O usuário declara ser celíaco e cita frango, cevada, shoyu e abobrinha. Pela policy de ingredientes derivados, 'celíaco' exclui cevada, malte e shoyu comum (e qualquer fonte de glúten, como trigo).",
            RECEITA_COMPLETA,
            "Verifique se a receita NÃO contém cevada nem shoyu comum (um substituto explicitamente sem glúten, como tamari sem glúten, é aceitável).",
            "Verifique se a receita contém TODOS os ingredientes compatíveis da query: frango e abobrinha.",
            "Verifique se nenhum ingrediente acrescentado pelo agente contém glúten (trigo, farinha de trigo, malte, cevada, centeio, aveia não certificada, molhos com trigo).",
            "Penalize fortemente qualquer ingrediente que viole a restrição celíaca.",
        ],
    ),
}
