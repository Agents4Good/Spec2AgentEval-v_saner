"""Métricas GEval por fluxo (B1–B8) do agente de receitas.

Cada métrica avalia um único fluxo e deve ser aplicada apenas aos goldens
daquele fluxo (ver FLOW_METRICS). Fluxos que dependem de chamadas de tool
avaliam TOOLS_CALLED, então o test case precisa trazer as tool calls
capturadas (nome, argumentos e retorno).
"""

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

INPUT = LLMTestCaseParams.INPUT
OUTPUT = LLMTestCaseParams.ACTUAL_OUTPUT
TOOLS = LLMTestCaseParams.TOOLS_CALLED

RECIPE_FIELDS = (
    "Título, Ingredientes (com quantidades), Equipamentos, "
    "Instruções passo a passo e Armazenamento"
)


# --- B1: validação da query -------------------------------------------------

b1_1_valid_query = GEval(
    name="B1-1 Query válida segue o fluxo",
    evaluation_steps=[
        "Confirme que o input não está vazio, não é ofensivo e tem relação com comida ou culinária.",
        "Verifique que o actual output NÃO rejeita a query como inválida (não há mensagem 'Error:' alegando query vazia, ofensiva ou fora do tema).",
        "Verifique em tools called que o agente prosseguiu no fluxo, chamando search_recipes ou retrieve_pantry.",
        "Penalize fortemente se o agente tratou a query válida como inválida ou encerrou sem chamar nenhuma tool por motivo de validação.",
    ],
    evaluation_params=[INPUT, OUTPUT, TOOLS],
)

b1_2_invalid_query = GEval(
    name="B1-2 Query inválida é rejeitada",
    evaluation_steps=[
        "Confirme que o input é vazio, ofensivo ou sem relação com comida, culinária, receitas ou ingredientes.",
        "Verifique que o actual output começa exatamente com 'Error:'.",
        "Verifique que a mensagem explica o motivo da rejeição (vazia, ofensiva ou fora do tema).",
        "Verifique que tools called está vazio: nenhuma chamada a search_recipes ou retrieve_pantry.",
        "Verifique que o actual output não contém receita nem resposta inventada ao pedido fora do tema.",
        "Qualquer tool chamada ou receita gerada deve levar à nota mínima.",
    ],
    evaluation_params=[INPUT, OUTPUT, TOOLS],
)


# --- B2: identificação de ingredientes --------------------------------------

b2_1_ingredient_normalization = GEval(
    name="B2-1 Ingredientes identificados e normalizados",
    evaluation_steps=[
        "Liste os ingredientes mencionados de forma afirmativa no input.",
        "Normalize cada um para a forma canônica: variantes regionais viram o nome padrão (ex.: 'macaxeira' e 'aipim' viram 'mandioca') e formas flexionadas ou com quantidade/adjetivo viram o substantivo singular (ex.: 'dois tomates bem maduros' vira 'tomate').",
        "Compare essa lista com os argumentos de ingredientes passados a search_recipes em tools called.",
        "Penalize ingredientes faltando, ingredientes não citados no input, duplicatas de variantes (ex.: 'aipim' e 'mandioca' juntos) e termos não normalizados (plural, quantidade, adjetivo).",
    ],
    evaluation_params=[INPUT, TOOLS],
)

b2_2_no_ingredients = GEval(
    name="B2-2 Query sem ingredientes leva à despensa",
    evaluation_steps=[
        "Confirme que o input é uma query culinária válida que não menciona nenhum ingrediente.",
        "Verifique em tools called que retrieve_pantry foi chamada.",
        "Verifique que search_recipes NÃO foi chamada (a lista de ingredientes da query está vazia).",
        "Penalize se o agente inventou ingredientes a partir da query ou chamou search_recipes.",
    ],
    evaluation_params=[INPUT, TOOLS],
)

b2_3_negated_ingredients = GEval(
    name="B2-3 Ingredientes negados ou hipotéticos são excluídos",
    evaluation_steps=[
        "Identifique no input os ingredientes citados sob negação, ausência ou hipótese (ex.: 'não tenho ovo', 'sem leite', 'se eu tivesse leite').",
        "Identifique os ingredientes citados de forma afirmativa, se houver.",
        "Se houver ingredientes afirmativos: verifique que search_recipes recebeu apenas eles e nenhum dos negados/hipotéticos.",
        "Se não houver ingredientes afirmativos: verifique que retrieve_pantry foi chamada e search_recipes não foi.",
        "Verifique que a receita no actual output não usa os ingredientes negados ou hipotéticos como disponíveis.",
        "Qualquer ingrediente negado/hipotético tratado como disponível deve levar à nota mínima.",
    ],
    evaluation_params=[INPUT, OUTPUT, TOOLS],
)


# --- B3: busca de receitas de referência ------------------------------------

b3_1_search_recipes_call = GEval(
    name="B3-1 search_recipes chamada uma vez com ingredientes da query",
    evaluation_steps=[
        "Confirme que o input menciona ao menos um ingrediente disponível.",
        "Verifique em tools called que search_recipes foi chamada exatamente uma vez.",
        "Verifique que os argumentos de search_recipes correspondem aos ingredientes identificados na query (normalizados).",
        "Verifique que retrieve_pantry NÃO foi chamada.",
        "Verifique que o actual output usa os resultados (até 5, com título, URL e trecho) apenas como referência de preparo, sem copiar uma receita encontrada como a receita final.",
    ],
    evaluation_params=[INPUT, OUTPUT, TOOLS],
)

b3_2_empty_search_results = GEval(
    name="B3-2 Busca sem resultados ainda gera receita",
    evaluation_steps=[
        "Confirme em tools called que search_recipes retornou uma lista vazia.",
        "Verifique que o actual output NÃO começa com 'Error:' e não trata a ausência de resultados como falha.",
        f"Verifique que o actual output contém uma receita completa com {RECIPE_FIELDS}, todos preenchidos.",
        "Verifique que a receita usa os ingredientes da query como base e não cita receitas de referência, títulos ou URLs inexistentes.",
        "Penalize erro emitido, receita incompleta ou referências inventadas.",
    ],
    evaluation_params=[INPUT, OUTPUT, TOOLS],
)

b3_3_never_both_tools = GEval(
    name="B3-3 Query com ingredientes e pedido de despensa usa só search_recipes",
    evaluation_steps=[
        "Confirme que o input menciona ingredientes e também pede para usar a despensa.",
        "Verifique em tools called que search_recipes foi chamada exatamente uma vez.",
        "Verifique que retrieve_pantry NÃO aparece em tools called.",
        "Qualquer chamada a retrieve_pantry na mesma execução deve levar à nota mínima.",
    ],
    evaluation_params=[INPUT, TOOLS],
)

b3_4_references_not_base = GEval(
    name="B3-4 Resultados da busca não viram ingredientes-base",
    evaluation_steps=[
        "Liste os ingredientes do input e os ingredientes que aparecem nos resultados de search_recipes em tools called.",
        "Identifique os ingredientes presentes nos resultados mas ausentes do input.",
        "Verifique que o actual output não trata esses ingredientes extras como ingredientes-base disponíveis do usuário (podem aparecer no máximo como complementos básicos, nunca como protagonistas trazidos da referência).",
        "Verifique que a receita final não é cópia de nenhuma receita encontrada (título, lista de ingredientes e passos não reproduzidos literalmente).",
        "Penalize receitas montadas em torno de ingredientes vindos só dos resultados ou que reproduzem uma receita de referência.",
    ],
    evaluation_params=[INPUT, OUTPUT, TOOLS],
)


# --- B4 / B5: despensa -------------------------------------------------------

b4_1_retrieve_pantry_call = GEval(
    name="B4-1 Lista vazia chama retrieve_pantry uma vez",
    evaluation_steps=[
        "Confirme que o input não contém ingredientes disponíveis (nenhum citado, ou só negados/hipotéticos).",
        "Verifique em tools called que retrieve_pantry foi chamada exatamente uma vez.",
        "Verifique que search_recipes NÃO foi chamada.",
        "Verifique que a receita do actual output, se houver, usa itens retornados pela despensa como ingredientes-base.",
    ],
    evaluation_params=[INPUT, OUTPUT, TOOLS],
)

b5_1_empty_pantry = GEval(
    name="B5-1 Despensa vazia interrompe com erro",
    evaluation_steps=[
        "Confirme em tools called que retrieve_pantry retornou uma lista vazia.",
        "Verifique que o actual output começa exatamente com 'Error:'.",
        "Verifique que a mensagem explica que faltam ingredientes para gerar a receita.",
        "Verifique que o actual output não contém receita (nem título, lista de ingredientes ou instruções).",
        "Receita gerada a partir de ingredientes inventados deve levar à nota mínima.",
    ],
    evaluation_params=[OUTPUT, TOOLS],
)


# --- B6: restrições dietéticas incompatíveis --------------------------------

b6_1_all_incompatible = GEval(
    name="B6-1 Nenhum ingrediente compatível interrompe com erro",
    evaluation_steps=[
        "Identifique as restrições dietéticas declaradas no input (ex.: vegano, celíaco, sem lactose).",
        "Identifique os ingredientes-base: os da query ou, se a query não tiver ingredientes, os retornados por retrieve_pantry em tools called.",
        "Considere ingredientes derivados implícitos: vegano exclui mel, leite, ovos, gelatina etc.; celíaco exclui trigo, cevada, centeio, malte etc.",
        "Confirme que nenhum ingrediente-base é compatível com as restrições.",
        "Verifique que o actual output começa exatamente com 'Error:' e explica a incompatibilidade entre os ingredientes e a restrição.",
        "Verifique que o actual output não contém receita.",
        "Receita gerada com ingredientes incompatíveis ou substituídos por ingredientes inventados deve levar à nota mínima.",
    ],
    evaluation_params=[INPUT, OUTPUT, TOOLS],
)


# --- B7: falhas ---------------------------------------------------------------

b7_1_failure_handling = GEval(
    name="B7-1 Falha de tool ou exceção vira mensagem Error:",
    evaluation_steps=[
        "Confirme que houve falha: uma tool em tools called retornou mensagem iniciada por 'erro:', ou ocorreu exceção (timeout, credenciais ausentes, página inacessível ou falha do modelo).",
        "Verifique que o actual output é uma mensagem tratada, sem stack trace ou exceção não tratada.",
        "Verifique que o actual output começa exatamente com 'Error:' e explica o que falhou.",
        "Verifique que o actual output não contém receita.",
        "Penalize saídas que ignoram a falha e geram receita, ou que expõem a exceção crua.",
    ],
    evaluation_params=[OUTPUT, TOOLS],
)


# --- B8: geração da receita --------------------------------------------------

b8_1_recipe_from_query = GEval(
    name="B8-1 Receita completa com todos os ingredientes da query",
    evaluation_steps=[
        "Identifique os ingredientes afirmativos do input, as restrições dietéticas (se houver) e as preferências do usuário (tempo, método de preparo, porções, estilo etc.).",
        f"Verifique que o actual output contém {RECIPE_FIELDS}, todos preenchidos e não vazios.",
        "Verifique que a seção Ingredientes traz quantidades para cada item.",
        "Verifique que todos os ingredientes da query aparecem na receita.",
        "Verifique que a receita respeita as preferências declaradas e que nenhum ingrediente viola as restrições.",
        "Se search_recipes retornou resultados em tools called, verifique que o preparo é coerente com essas referências sem copiá-las.",
        "Penalize cada campo ausente ou vazio, cada ingrediente da query omitido e cada preferência ignorada.",
    ],
    evaluation_params=[INPUT, OUTPUT, TOOLS],
)

b8_2_recipe_from_pantry = GEval(
    name="B8-2 Receita completa a partir da despensa",
    evaluation_steps=[
        "Confirme que o input não contém ingredientes e que retrieve_pantry, em tools called, retornou ao menos um ingrediente compatível com as restrições.",
        f"Verifique que o actual output contém {RECIPE_FIELDS}, todos preenchidos e não vazios.",
        "Verifique que a receita contém pelo menos um ingrediente retornado por retrieve_pantry.",
        "Verifique que nenhum ingrediente da receita viola as restrições declaradas no input.",
        "Receita sem nenhum ingrediente da despensa deve levar à nota mínima.",
    ],
    evaluation_params=[INPUT, OUTPUT, TOOLS],
)

b8_3_partial_incompatibility = GEval(
    name="B8-3 Ingredientes incompatíveis removidos, compatíveis mantidos",
    evaluation_steps=[
        "Identifique as restrições dietéticas do input e os ingredientes-base (da query ou de retrieve_pantry em tools called).",
        "Classifique cada ingrediente-base como compatível ou incompatível, considerando ingredientes derivados (ex.: vegano exclui mel e laticínios; celíaco exclui malte e trigo), inclusive quando o próprio usuário pede um ingrediente que viola a restrição.",
        f"Verifique que o actual output contém uma receita completa com {RECIPE_FIELDS}, todos preenchidos.",
        "Verifique que nenhum ingrediente incompatível aparece na receita.",
        "Verifique que todos os ingredientes compatíveis da query aparecem na receita.",
        "Verifique que nenhum ingrediente acrescentado pelo agente (complementos, temperos, substitutos) viola as restrições.",
        "Qualquer ingrediente que viole a restrição deve levar à nota mínima; cada ingrediente compatível omitido deve ser penalizado.",
    ],
    evaluation_params=[INPUT, OUTPUT, TOOLS],
)


# Listas importadas pelo arquivo de teste, uma por fluxo (aplicar só aos
# goldens daquele fluxo).
B1_1_VALID_QUERY_METRICS = [b1_1_valid_query]
B1_2_INVALID_QUERY_METRICS = [b1_2_invalid_query]
B2_1_INGREDIENT_NORMALIZATION_METRICS = [b2_1_ingredient_normalization]
B2_2_NO_INGREDIENTS_METRICS = [b2_2_no_ingredients]
B2_3_NEGATED_INGREDIENTS_METRICS = [b2_3_negated_ingredients]
B3_1_SEARCH_RECIPES_CALL_METRICS = [b3_1_search_recipes_call]
B3_2_EMPTY_SEARCH_RESULTS_METRICS = [b3_2_empty_search_results]
B3_3_NEVER_BOTH_TOOLS_METRICS = [b3_3_never_both_tools]
B3_4_REFERENCES_NOT_BASE_METRICS = [b3_4_references_not_base]
B4_1_RETRIEVE_PANTRY_CALL_METRICS = [b4_1_retrieve_pantry_call]
B5_1_EMPTY_PANTRY_METRICS = [b5_1_empty_pantry]
B6_1_ALL_INCOMPATIBLE_METRICS = [b6_1_all_incompatible]
B7_1_FAILURE_HANDLING_METRICS = [b7_1_failure_handling]
B8_1_RECIPE_FROM_QUERY_METRICS = [b8_1_recipe_from_query]
B8_2_RECIPE_FROM_PANTRY_METRICS = [b8_2_recipe_from_pantry]
B8_3_PARTIAL_INCOMPATIBILITY_METRICS = [b8_3_partial_incompatibility]

FLOW_METRICS = {
    "B1-1": B1_1_VALID_QUERY_METRICS,
    "B1-2": B1_2_INVALID_QUERY_METRICS,
    "B2-1": B2_1_INGREDIENT_NORMALIZATION_METRICS,
    "B2-2": B2_2_NO_INGREDIENTS_METRICS,
    "B2-3": B2_3_NEGATED_INGREDIENTS_METRICS,
    "B3-1": B3_1_SEARCH_RECIPES_CALL_METRICS,
    "B3-2": B3_2_EMPTY_SEARCH_RESULTS_METRICS,
    "B3-3": B3_3_NEVER_BOTH_TOOLS_METRICS,
    "B3-4": B3_4_REFERENCES_NOT_BASE_METRICS,
    "B4-1": B4_1_RETRIEVE_PANTRY_CALL_METRICS,
    "B5-1": B5_1_EMPTY_PANTRY_METRICS,
    "B6-1": B6_1_ALL_INCOMPATIBLE_METRICS,
    "B7-1": B7_1_FAILURE_HANDLING_METRICS,
    "B8-1": B8_1_RECIPE_FROM_QUERY_METRICS,
    "B8-2": B8_2_RECIPE_FROM_PANTRY_METRICS,
    "B8-3": B8_3_PARTIAL_INCOMPATIBILITY_METRICS,
}
