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
| Persona (Opcional) | A personalidade que o agente assume ao realizar as tarefas | O agente assume a persona de um **chefe de cozinha**, cuja função é criar receitas personalizadas.  |
| Inputs | Informação enviada para o agente | O agente recebe uma _query (em linguagem natural)_ do usuário, contendo preferências e opcionalmente ingredientes específicos e restrições dietéticas. |
| Constraints | Uma afirmação que restringe as ações que o agente (ou seus usuários/operadores) têm permissão de realizar. | 
*   O agente não deve fornecer/ inventar respostas quando a query não for relacionada a receitas ou ingredientes.
*   O agente nunca deve chamar ambas as ferramentas na mesma execução.
*   O agente nunca deve gerar receitas que não respeitem as restrições dietéticas do usuário.

 |
| Policies/Invariants | Afirmações que descrevem o que é verdadeiro sobre o domínio ou sobre o estado do agente em um dado momento | 

*   As informações retornadas devem corresponder aos dados obtidos do GitHub. As issues retornadas pertencem • Uma restrição dietética implica todos os seus ingredientes derivados, mesmo os não citados pelo usuário. Exemplo: "vegano" exclui mel, gelatina, manteiga e whey; "celíaco" exclui cevada, malte e shoyu comum.
*   Um ingrediente mencionado sob negação, ausência ou hipótese não é um ingrediente disponível. repositório solicitado e satisfazem os critérios de filtragem especificados pelo usuário.

 |
| Quality attributes (non-functional requirements) | Características observáveis do comportamento do agente durante a execução, que não descrevem o que ele faz nem o que ele nunca deve fazer, mas quão bem ele faz. | A execução deve produzir uma resposta estruturada e adequada para avaliação automatizada. |
| Env | Chaves disponiveis através da configuração de ambiente | 

O agente deve acessar as variáveis de ambiente quando necessário.

Chaves disponíveis: NOTION\_PAGE\_ID, TAVILY\_API\_KEY, NOTION\_API\_KEY

 |

**Tools:**

<table><tbody><tr><td><strong>Nome</strong></td><td><strong>Propósito</strong></td><td><strong>Entradas</strong></td><td><strong>Retorno</strong></td><td><strong>Falhas</strong></td></tr><tr><td>retrieve_pantry</td><td>Obter os ingredientes disponíveis na despensa do usuário, através de uma página do notion.</td><td>Nenhuma.</td><td>A lista completa de ingredientes da despensa. Despensa sem itens retorna lista vazia.</td><td>Credenciais ausentes, página inacessível ou timeout são sinalizados de forma distinguível da lista vazia.</td></tr><tr><td>search_recipes</td><td>Obter receitas da web como referência de preparo para os ingredientes-base identificados na query. As receitas encontradas não são fonte de ingredientes-base.</td><td>ingredientes-base identificados na query.</td><td>Até 5 resultados, cada um com título, URL e trecho do conteúdo. Ausência de resultados retorna lista vazia.</td><td>Credenciais ausentes, timeout ou erro do serviço de busca são sinalizados de forma distinguível da lista vazia.</td></tr></tbody></table>

**Comportamentos:**

| ID | Comportamento | Input | Tools | Saída | Efeito no Ambiente | Condição de Satisfação |
| --- | --- | --- | --- | --- | --- | --- |
| **B1 — Validar query**  | **O agente classifica a query como inválida (vazia, ofensiva ou não relacionada a comida/culinária), ou válida (qualquer query que não satisfaça nenhum critério de invalidez), interrompendo a geração quando é inválida.** | Query do usuário | Nenhuma | 
Query inválida: string iniciada por `erro:`, seguida de explicação clara do motivo da rejeição. 

Query válida: nenhuma saída própria, a execução prossegue conforme B2 ou B3.

 | Nenhum | 

Query inválida: o agente retorna uma resposta começando com `erro:`; seguida por uma explicação clara do motivo. Nenhuma receita é gerada.

Query válida: A resposta não contém erro de query inválida e ao menos uma tool é chamada (`search_recipes` ou `retrieve_pantry`).

 |
| **B2 — Determinar se a query fornece ingredientes.**  |   | Query do usuário |   | Lista contendo as issues que correspondem aos critérios solicitados. | Nenhum. | As issues retornadas satisfazem os critérios de filtragem especificados na solicitação. |
| **B3 — Nenhuma issue corresponde ao filtro** |   | Repositório válido; solicitação de filtragem sem correspondências. | GitHub MCP. | Lista vazia e `status = success`. | Apenas leitura das informações do repositório. | A saída contém uma lista vazia e `status = success`. |
| **B4 — Repositório inválido ou inacessível** |   | Repositório inválido ou inacessível. | GitHub MCP. | `status = failure` e mensagem indicando a falha. | Nenhum. | A saída contém `status = failure` e uma mensagem indicando a impossibilidade de acesso ao repositório. |
| **B5 — Retornar resposta estruturada** |   | Resultado da execução do agente. | — | Objeto contendo `action`, `repository`, `filtered_issues`, `status` e `message`. Cada issue contém `issue_number`, `title` e `state`. | Nenhum. | A saída contém os campos definidos no contrato e os valores possuem os tipos e formatos especificados. |