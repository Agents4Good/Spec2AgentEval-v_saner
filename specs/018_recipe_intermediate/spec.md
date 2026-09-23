**Task:** você é um gerador de agentes de IA e abaixo está uma especificação de um agente. Gere um agente conforme a especificação. Considere que as ferramentas do agente devem ser geradas junto ao código do agente.  
**Contexto:** Um agente de IA é uma entidade que usa LLM e ferramentas para executar tarefas específicas de forma autônoma.  
**Output:** agent.py, requirements.txt, tests.py

**Contrato de interface com o benchmark:**

| Requisito de interface | Exemplo |
| --- | --- |
| Modelo | GPT-4o |
| Framework | LangGraph |
| GenAI Tool | ChatOpenAI |
| Entrypoint | agent(...) |
| Output | state: dict |

**Requisitos gerais:**

| Requisito | Significado | Especificação do agente |
| --- | --- | --- |
| Goal | Objetivo e comportamento principal do agente na voz ativa | O agente atua como um gerador de receitas personalizadas, adaptadas para utilizar os ingredientes colocados na query do usuário ou na sua despensa e para respeitar preferências e restrições dietéticas do usuário. Retorna a receita completa com ingredientes, equipamentos, instruções e recomendações de armazenamento. |
| Persona (Opcional) | A personalidade que o agente assume ao realizar as tarefas | O agente assume a persona de um **chefe de cozinha**, cuja função é criar receitas personalizadas. |
| Inputs | Informação enviada para o agente | O agente recebe uma _query (em linguagem natural)_ do usuário, contendo preferências e opcionalmente ingredientes específicos e restrições dietéticas. |
| Constraints | Uma afirmação que restringe as ações que o agente (ou seus usuários/operadores) têm permissão de realizar. | • O agente não deve fornecer/inventar respostas quando a query não for relacionada a receitas ou ingredientes.  
• O agente nunca deve chamar ambas as ferramentas na mesma execução.  
• O agente nunca deve gerar receitas que não respeitem as restrições dietéticas do usuário (quando houverem).  
• A receita deve conter todos os ingredientes identificados na query, exceto os incompatíveis com as restrições dietéticas do usuário.  
• Quando a query não contiver ingredientes, a receita deve conter ao menos um ingrediente retornado por `retrieve_pantry`. |
| Policies/Invariants | Afirmações que descrevem o que é verdadeiro sobre o domínio ou sobre o estado do agente em um dado momento | • Uma restrição dietética implica todos os seus ingredientes derivados, mesmo os não citados pelo usuário. Exemplo: "vegano" exclui mel, gelatina, manteiga e whey; "celíaco" exclui cevada, malte e shoyu comum.  
• Um ingrediente mencionado sob negação, ausência ou hipótese não é um ingrediente disponível. |
| Quality attributes (non-functional requirements) | Características observáveis do comportamento do agente durante a execução, que não descrevem o que ele faz nem o que ele nunca deve fazer, mas quão bem ele faz. | A execução deve produzir uma resposta estruturada e adequada para avaliação automatizada. |
| Env | Chaves disponíveis através da configuração do ambiente | O agente deve acessar as variáveis de ambiente quando necessário.  
  
Chaves disponíveis: NOTION\_PAGE\_ID, TAVILY\_API\_KEY, NOTION\_API\_KEY |

**Tools:**

| Nome | Propósito | Entradas | Retorno | Falhas |
| --- | --- | --- | --- | --- |
| retrieve\_pantry | Obter os ingredientes disponíveis na despensa do usuário, através de uma página do notion. | Nenhuma. | A lista completa de ingredientes da despensa. Despensa sem itens retorna lista vazia. | Credenciais ausentes, página inacessível ou timeout devem retornar uma mensagem começando com `erro:` seguida por uma mensagem explicando a falha. |
| search\_recipes | Obter receitas da web, através do **tavily**, como **referência** de preparo para os ingredientes identificados na query. As receitas encontradas não são fonte de ingredientes-base, nem a receita final. | Ingredientes identificados na query. | Até 5 resultados, cada um com título, URL e trecho do conteúdo. Ausência de resultados retorna lista vazia. | Credenciais ausentes, página inacessível ou timeout devem retornar uma mensagem começando com `erro:` seguida por uma mensagem explicando a falha. |

**Comportamentos:**

| Comportamento | Trigger | Input | Tools | Saída | Efeito no ambiente | Condição de satisfação |
| --- | --- | --- | --- | --- | --- | --- |
| **B1 — Validação da query**  
O agente classifica a query como inválida (vazia, ofensiva ou sem relação com comida/culinária) ou válida (qualquer query que não satisfaça nenhum critério de invalidez), e interrompe a execução se for inválida. | Recebimento da query. | Query do usuário. | Nenhuma. | Indicador booleano de validade. Se inválida, mensagem iniciada por `Error:` com a explicação. | Nenhum. | O indicador corresponde à classificação esperada. Se inválida, a mensagem começa com `Error:`, nenhuma tool é chamada e nenhuma receita é gerada. |
| **B2 — Identificação de ingredientes**  
O agente identifica os ingredientes mencionados na query. | Query classificada como válida em B1. | Query do usuário. | Nenhuma. | Lista de ingredientes identificados, se nenhum ingrediente for identificado, lista vazia. | Nenhum. | A lista é igual ao conjunto esperado, considerando variantes regionais (por exemplo, "macaxeira" e "aipim" correspondem a mandioca), formas flexionadas (por exemplo, "dois tomates bem maduros" e "um resto de frango"), e excluindo ingredientes mencionados sob negação, ausência ou hipótese. |
| **B3 — Busca de receitas de referência**  
O agente busca receitas que sirvam de referência de preparo para os ingredientes identificados na query. | Lista de ingredientes identificados em B2 não vazia. | Lista de ingredientes identificados em B2. | `search_recipes` | Até cinco resultados, cada um com título, URL e trecho do conteúdo, possivelmente nenhum. | Nenhum. | `search_recipes` é chamada exatamente uma vez, com a lista de ingredientes identificados em B2, e `retrieve_pantry` não é chamada. |
| **B4 — Consulta à despensa**  
O agente recupera os ingredientes disponíveis na despensa do usuário. | Lista de ingredientes identificados em B2 vazia. | Nenhum. | `retrieve_pantry` | Lista de ingredientes da despensa, possivelmente vazia. | Nenhum. | `retrieve_pantry` é chamada exatamente uma vez, e `search_recipes` não é chamada. |
| **B5 — Falta de insumos**  
O agente interrompe a execução quando a despensa não fornece ingredientes. | `retrieve_pantry` retorna lista vazia. | Retorno de `retrieve_pantry`. | Nenhuma. | Mensagem iniciada por `Error:` explicando a ausência de ingredientes. | Nenhum. | A mensagem começa com `Error:` e nenhuma receita é gerada. |
| **B6 — Incompatibilidade dietética**  
O agente interrompe a execução quando nenhum ingrediente-base é compatível com as restrições dietéticas do usuário. | Nenhum ingrediente-base compatível com as restrições dietéticas declaradas na query. | Query do usuário e ingredientes-base. | Nenhuma. | Mensagem iniciada por `Error:` explicando a incompatibilidade. | Nenhum. | A mensagem começa com `Error:` e nenhuma receita é gerada. |
| **B7 — Falha de execução**  
O agente trata falhas das tools ou do modelo sem abortar a execução. | Uma tool retorna mensagem iniciada por `erro:`, ou ocorre exceção durante a execução (timeout, credenciais ausentes ou falha do modelo). | Mensagem de falha ou exceção recebida. | Nenhuma. | Mensagem iniciada por `Error:` explicando a falha. | Nenhum. | O agente não encerra com exceção não tratada, a mensagem começa com `Error:` e nenhuma receita é gerada. |
| **B8 — Geração da receita**  
O agente usa seu conhecimento culinário para transformar os ingredientes-base compatíveis em uma receita completa, considerando as preferências do usuário e usando as receitas de referência, quando houver. | Existe ao menos um ingrediente-base compatível com as restrições dietéticas declaradas na query. | Query do usuário, ingredientes-base e resultados de B3, quando houver. | Nenhuma. | Receita com os campos:  
• **Título:** título da receita;  
• **Ingredientes:** lista de ingredientes com suas quantidades;  
• **Equipamentos:** lista de equipamentos necessários;  
• **Instruções:** instruções de preparo, passo a passo;  
• **Armazenamento:** recomendações de armazenamento. | Nenhum. | Todos os campos estão presentes e não vazios; nenhum ingrediente da receita viola as restrições dietéticas declaradas na query; a receita satisfaz as Constraints sobre ingredientes da query e da despensa. |