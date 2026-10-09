"""Métricas GEval da suíte de avaliação do agente de receitas (spec 018).

Uma métrica GEval por fluxo (B1-1 ... B8-3). O teste escolhe a métrica pelo
campo ``additional_metadata["flow"]`` de cada golden em ``.dataset.json``.
"""

import os

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", "gpt-4.1")
THRESHOLD = 0.6

EVALUATION_PARAMS = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
    LLMTestCaseParams.EXPECTED_OUTPUT,
    LLMTestCaseParams.TOOLS_CALLED,
]

# --------------------------------------------------------------------------- #
# Passos compartilhados
# --------------------------------------------------------------------------- #

READ_INPUT = "Leia input e identifique os ingredientes disponíveis e as restrições dietéticas declaradas, desconsiderando ingredientes citados sob negação, ausência ou hipótese."
READ_TOOLS = "Use tools_called para verificar quais tools foram chamadas, quantas vezes, com quais input_parameters e o que retornaram em output. Lista vazia significa que nenhuma tool foi chamada."
EXCEPTION_NOTE = "Se actual_output começar com 'EXCEPTION', o agente encerrou com exceção não tratada: isso é falha completa (0.0-0.2)."
FIELDS = "Verifique se actual_output contém receita com Título, Ingredientes com quantidades, Equipamentos, Instruções passo a passo e Armazenamento, todos não vazios."
BASICS = "Ingredientes básicos (sal, água, óleo, temperos secos) podem ser adicionados, desde que respeitem as restrições."
ERROR_CHECK = "Verifique se actual_output começa com 'Error:' e explica o motivo, e se nenhuma receita (título, ingredientes, instruções) foi gerada."
TRAJ = "Não penalize diferenças de trajetória que sejam válidas e eficazes: ordem interna de etapas, número de chamadas ao LLM, redação da mensagem, nomes dos campos, uso ou não das receitas de referência e substituições razoáveis. Penalize apenas o que viola a conduta esperada: tool proibida, ambas as tools na mesma execução, chamada repetida, receita gerada quando deveria haver erro, erro quando deveria haver receita, ingrediente que viola restrição."
RUBRIC_ERR = "Atribua a nota pelo espectro: 0.0-0.2 falha completa (gera receita, inventa resposta, encerra com exceção não tratada ou retorna saída vazia); 0.3-0.5 resolução parcial (interrompe sem receita, mas a mensagem não começa com 'Error:' ou não explica o motivo, ou chama tool proibida antes de interromper); 0.6-0.8 resolução correta com pequenas falhas (mensagem começa com 'Error:', sem receita e sem tool proibida, mas a explicação é vaga, imprecisa ou cita causa errada); 0.9-1.0 resolução correta e de alta qualidade (mensagem começa com 'Error:', explica com clareza o motivo real, nenhuma receita e trajetória de tools conforme o esperado)."
RUBRIC_RECIPE = "Atribua a nota pelo espectro: 0.0-0.2 falha completa (exceção não tratada, saída vazia, recusa indevida, ou violação grave como ingrediente proibido pela restrição ou chamada a ambas as tools); 0.3-0.5 resolução parcial (trajetória correta, mas receita ausente, ou receita sem vários campos obrigatórios, ou omite ingredientes centrais da query/despensa); 0.6-0.8 resolução correta com pequenas falhas (receita válida e segura, mas com um campo raso, quantidade faltante, ingrediente secundário omitido ou detalhe pouco claro); 0.9-1.0 resolução correta e de alta qualidade (todos os campos completos e coerentes, todos os ingredientes exigidos presentes, restrições respeitadas, instruções executáveis e preferências atendidas)."


def recipe_steps(*specific):
    return [READ_INPUT, READ_TOOLS, EXCEPTION_NOTE, *specific, FIELDS, BASICS, TRAJ, RUBRIC_RECIPE]


def error_steps(*specific):
    return [READ_TOOLS, EXCEPTION_NOTE, *specific, ERROR_CHECK, TRAJ, RUBRIC_ERR]


# --------------------------------------------------------------------------- #
# Passos por fluxo
# --------------------------------------------------------------------------- #

FLOW_STEPS = {
    # B1 — Validação da query
    "B1-1": [
        READ_TOOLS,
        EXCEPTION_NOTE,
        "Confirme que a query é válida: não está vazia, não é ofensiva e tem relação com comida/culinária, apesar do tom informal.",
        "Verifique que actual_output NÃO é uma mensagem 'Error:' de query inválida. Rejeitar esta query como inválida, ofensiva ou fora de escopo é falha completa (0.0-0.2).",
        "Verifique em tools_called que o fluxo seguiu para B2/B3: search_recipes foi chamada exatamente uma vez com abobrinha e retrieve_pantry não foi chamada.",
        "Verifique que a receita contém abobrinha e é adequada para jantar.",
        FIELDS,
        BASICS,
        TRAJ,
        RUBRIC_RECIPE,
    ],
    "B1-2": error_steps(
        "Confirme que a query é inválida: não tem relação com comida, culinária, receitas ou ingredientes.",
        "Verifique que tools_called está vazia. Qualquer chamada a retrieve_pantry ou search_recipes é violação.",
        "Verifique que actual_output não fornece nem inventa resposta ao pedido fora de escopo: redigir o e-mail, total ou parcialmente, é falha completa (0.0-0.2).",
    ),
    # B2 — Identificação de ingredientes
    "B2-1": recipe_steps(
        "Verifique em tools_called os input_parameters de search_recipes: devem corresponder a mandioca (ou macaxeira/aipim como sinônimo), tomate e frango. Argumentos com quantidades ou adjetivos ('dois tomates bem maduros', 'um resto de frango') indicam normalização incompleta e rebaixam para no máximo 0.6-0.8.",
        "A checagem de normalização vale SOMENTE para os input_parameters de search_recipes. Quantidades e descrições na lista de Ingredientes da receita final ('500g de mandioca', '2 tomates maduros') são exigidas pelo formato da receita e NÃO devem ser penalizadas.",
        "Ingrediente ausente da lista identificada ou ingrediente inventado na lista (não citado na query) rebaixa para no máximo 0.3-0.5.",
        "Verifique que search_recipes foi chamada exatamente uma vez e que retrieve_pantry não foi chamada. Chamar ambas é falha completa.",
        "Verifique que a receita contém mandioca/macaxeira, tomate e frango.",
    ),
    "B2-2": recipe_steps(
        "Confirme que a query não menciona nenhum ingrediente: 'café da manhã' é uma refeição, não um ingrediente.",
        "Verifique que retrieve_pantry foi chamada exatamente uma vez e que search_recipes não foi chamada. Chamar search_recipes (por exemplo, com ingredientes inventados) indica que a lista de B2 não ficou vazia e é violação grave; chamar ambas é falha completa.",
        "Verifique que a receita contém ao menos um ingrediente retornado por retrieve_pantry e é compatível com café da manhã rápido.",
    ),
    "B2-3": recipe_steps(
        "Confirme que ovo está sob negação ('não tenho ovo') e leite sob hipótese ('se eu tivesse leite'); nenhum dos dois é ingrediente disponível.",
        "Verifique que search_recipes NÃO foi chamada (em especial com ovo ou leite) e que retrieve_pantry foi chamada exatamente uma vez. Chamar search_recipes com ovo ou leite é violação grave (no máximo 0.3-0.5); chamar ambas é falha completa.",
        "Verifique que a receita não usa ovo nem leite como ingrediente, pois não estão disponíveis nem na query nem na despensa. Usar algum deles rebaixa para no máximo 0.3-0.5.",
        "Verifique que a receita contém ao menos um ingrediente retornado por retrieve_pantry e é uma sobremesa.",
    ),
    # B3 — Busca de receitas de referência
    "B3-1": recipe_steps(
        "Verifique que search_recipes foi chamada exatamente uma vez, com input_parameters correspondentes a lentilha, cenoura e cebola. Chamada repetida rebaixa para no máximo 0.3-0.5; argumentos que omitem ou inventam ingredientes rebaixam para no máximo 0.6-0.8.",
        "Verifique que retrieve_pantry não foi chamada. Chamar ambas as tools é falha completa (0.0-0.2).",
        "Verifique que a receita final não é cópia de um dos resultados de search_recipes: os resultados são referência de preparo, não a receita final.",
        "Verifique que a receita contém lentilha, cenoura e cebola e é uma sopa.",
    ),
    "B3-2": recipe_steps(
        "Verifique que search_recipes foi chamada exatamente uma vez com quiabo e camarão e retornou lista vazia, e que retrieve_pantry não foi chamada.",
        "Lista vazia de search_recipes NÃO é falha: responder com 'Error:', recusar ou interromper sem receita é falha completa (0.0-0.2).",
        "Verifique que o agente não cita nem inventa receitas de referência, URLs ou fontes que não vieram de tools_called. Inventar referências rebaixa para no máximo 0.6-0.8.",
        "Verifique que a receita contém quiabo e camarão.",
    ),
    "B3-3": recipe_steps(
        "Verifique que search_recipes foi chamada exatamente uma vez com batata-doce e frango.",
        "Verifique que retrieve_pantry NÃO foi chamada. Chamar retrieve_pantry na mesma execução (ambas as tools) é falha completa (0.0-0.2), mesmo que o usuário tenha pedido para usar a despensa.",
        "Verifique que a receita não usa itens que só existiriam na despensa (por exemplo, brócolis ou creme de leite) como se ela tivesse sido consultada; isso indica uso indevido da despensa.",
        "Verifique que a receita contém batata-doce e frango.",
    ),
    "B3-4": recipe_steps(
        "Verifique que search_recipes foi chamada exatamente uma vez com arroz e brócolis e que retrieve_pantry não foi chamada.",
        "Verifique se a receita incorpora como ingredientes principais itens que só aparecem nas referências: bacon, creme de leite, parmesão, camarão ou vinho branco. Usar algum deles como ingrediente-base rebaixa para no máximo 0.3-0.5; mencioná-los apenas como sugestão opcional rebaixa para no máximo 0.6-0.8.",
        "Verifique que a receita final não é cópia de uma das referências (mesmo título, mesma lista de ingredientes ou mesmo passo a passo). Copiar uma referência é no máximo 0.3-0.5.",
        "Verifique que a receita contém arroz e brócolis e é um acompanhamento simples.",
    ),
    # B4 — Consulta à despensa
    "B4-1": recipe_steps(
        "Verifique que retrieve_pantry foi chamada exatamente uma vez. Chamada repetida rebaixa para no máximo 0.3-0.5.",
        "Verifique que search_recipes NÃO foi chamada, nem antes nem depois da despensa (por exemplo, com os itens da despensa). Chamar ambas as tools é falha completa (0.0-0.2).",
        "Verifique que os ingredientes principais da receita vêm da despensa retornada; ingredientes centrais que não estão na despensa nem são básicos rebaixam para no máximo 0.6-0.8.",
    ),
    # B5 — Falta de insumos
    "B5-1": error_steps(
        "Verifique que retrieve_pantry foi chamada exatamente uma vez e retornou lista vazia, e que search_recipes não foi chamada.",
        "Verifique que o agente não inventa ingredientes para compensar a despensa vazia: gerar receita com ingredientes inventados é falha completa (0.0-0.2).",
        "A explicação deve indicar falta de ingredientes (despensa vazia / nenhum ingrediente disponível); atribuir a causa a outro motivo (query inválida, falha de tool) rebaixa para no máximo 0.6-0.8.",
    ),
    # B6 — Incompatibilidade dietética
    "B6-1": error_steps(
        "Confirme que a restrição vegana exclui mel, gelatina e manteiga, embora o usuário não os tenha declarado proibidos.",
        "Neste caso search_recipes NÃO é tool proibida: a query tem ingredientes, então chamá-la uma vez (com quaisquer ingredientes da query, inclusive os incompatíveis) ou não chamá-la são ambos corretos e não devem ser penalizados. Receitas retornadas por search_recipes são apenas referência e não contam como receita gerada.",
        "A única tool proibida é retrieve_pantry. Chamar retrieve_pantry, ou chamar search_recipes mais de uma vez, é violação.",
        "Gerar qualquer receita (com os ingredientes incompatíveis, com substitutos inventados ou com ingredientes que não vieram da query) é falha completa (0.0-0.2).",
        "A explicação deve citar a incompatibilidade com a dieta vegana; bônus se identificar que mel, gelatina e manteiga são de origem animal.",
    ),
    # B7 — Falha de execução
    "B7-1": error_steps(
        "Verifique que search_recipes foi chamada e retornou em output uma mensagem iniciada por 'erro:'.",
        "Verifique que retrieve_pantry não foi chamada como fallback (seria ambas as tools na mesma execução) e que search_recipes não foi chamada repetidamente.",
        "Verifique que o agente não ignora a falha gerando receita: o retorno 'erro:' não é lista vazia, e gerar receita é falha completa (0.0-0.2).",
        "A explicação deve indicar a falha na busca de receitas (timeout / Tavily); atribuir a causa a outro motivo rebaixa para no máximo 0.6-0.8.",
    ),
    # B8 — Geração da receita
    "B8-1": recipe_steps(
        "Verifique que search_recipes foi chamada exatamente uma vez com salmão, aspargos e limão e que retrieve_pantry não foi chamada.",
        "Verifique que a receita contém salmão, aspargos e limão. Omitir um deles rebaixa para no máximo 0.6-0.8; omitir dois ou mais rebaixa para no máximo 0.3-0.5.",
        "Verifique que as preferências são atendidas: preparo no forno, refeição leve e tempo total compatível com 30 minutos. Ignorar uma preferência rebaixa para no máximo 0.6-0.8.",
        "Verifique que cada ingrediente tem quantidade e que as instruções são executáveis e coerentes com os equipamentos listados.",
    ),
    "B8-2": recipe_steps(
        "Verifique que retrieve_pantry foi chamada exatamente uma vez e search_recipes não foi chamada.",
        "Verifique que a receita contém ao menos um ingrediente retornado por retrieve_pantry. Nenhum ingrediente da despensa é falha completa (0.0-0.2).",
        "Verifique que a receita é um lanche e que as quantidades são compatíveis com duas pessoas. Ignorar o número de porções rebaixa para no máximo 0.6-0.8.",
    ),
    "B8-3": recipe_steps(
        "Confirme que frango e arroz são compatíveis e que shoyu comum, cerveja preta e macarrão comum são incompatíveis com a doença celíaca (derivados de trigo, cevada ou malte).",
        "Verifique que search_recipes foi chamada exatamente uma vez e que retrieve_pantry não foi chamada. Aceite que os argumentos incluam ou não os ingredientes incompatíveis, desde que a receita final os exclua.",
        "Verifique cada ingrediente e passo da receita, inclusive os acrescentados pelo agente, contra a restrição celíaca: shoyu comum, cerveja, macarrão comum, farinha de trigo, cevada, malte, centeio, pão ou empanados. Qualquer um presente é falha completa (0.0-0.2). Versões explicitamente sem glúten (shoyu sem glúten/tamari, macarrão de arroz) são aceitáveis.",
        "Atender ao pedido de macarrão comum em detrimento da restrição é falha completa (0.0-0.2).",
        "Verifique que a receita contém frango e arroz. Omitir um deles rebaixa para no máximo 0.6-0.8.",
        "Bônus de qualidade: o agente informa que shoyu, cerveja e macarrão comum foram excluídos por conterem glúten ou sugere substitutos sem glúten.",
    ),
}


def flow_metrics(flow: str) -> list[GEval]:
    """Métricas do fluxo, criadas a cada teste para não compartilhar estado entre casos."""
    return [
        GEval(
            name=f"Conduta {flow}",
            evaluation_params=EVALUATION_PARAMS,
            evaluation_steps=FLOW_STEPS[flow],
            model=JUDGE_MODEL,
            threshold=THRESHOLD,
        )
    ]
