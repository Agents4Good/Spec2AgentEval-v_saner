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

| Requisito | Significado | Especificação do agente | Critério de Corretude |
| --- | --- | --- | --- |
| Goal | Objetivo e comportamento principal do agente na voz ativa | O agente deve listar e filtrar issues de um repositório público do GitHub a partir de uma solicitação em linguagem natural | O comportamento principal está declarado definindo quem faz, o que faz e o resultado? |
| Persona (Opcional) | A personalidade que o agente assume ao realizar as tarefas | Desenvolvedor especialista no Github | A especificação define uma persona e sua área de atuação? A persona é adequada/coerente com o Goal? |
| Inputs | Informação enviada para o agente | Repositório: identificador no formato owner/repository. Solicitação do usuário: descrição em linguagem natural dos critérios de filtragem. Token do GitHub: credencial de acesso para leitura. | A especificação define qual é a entrada esperada? |
| Tools | Lista com as ferramentas disponíveis e para que elas servem | GitHub MCP para acesso a informações de issues do repositório, link: https://api.githubcopilot.com/mcp/ | A especificação define a ferramenta, seus parâmetros, o retorno e o **propósito de uso** dela estão definidos? |
| Constraints | Uma afirmação que restringe as ações que o agente (ou seus usuários/operadores) têm permissão de realizar. | O agente deve operar somente com dados reais obtidos por meio do MCP, sem fabricar ou simular issues. A execução não deve realizar tentativas indefinidas. O agente não deve modificar o repositório. | A especificação declara explicitamente uma ação que o agente deve, não deve, ou só pode sob certas condições? A restrição está expressa em termos de ação/comportamento permitido ou proibido — não em termos de tecnologia, ferramenta ou biblioteca usada para implementar o agente? Quando a restrição envolve papel/autorização ("só X pode fazer Y"), a especificação define quem ou o que está autorizado? A restrição é verificável: é possível observar, no comportamento do agente, se ela foi violada ou respeitada? |
| Policies/Invariants | Afirmações que descrevem o que é verdadeiro sobre o domínio ou sobre o estado do agente em um dado momento | As informações retornadas devem corresponder aos dados obtidos do GitHub. As issues retornadas pertencem ao repositório solicitado e satisfazem os critérios de filtragem especificados pelo usuário. | A especificação declara uma relação, propriedade ou condição do domínio que é (ou deve ser) sempre verdadeira, e não uma ação permitida/proibida? O invariante está conectado a uma entrada, saída, evento ou dado que o agente efetivamente manipula (não é conhecimento de domínio irrelevante ao escopo do agente)? É possível verificar, a qualquer momento durante a execução, se o invariante ainda é válido? |
| Quality attributes (non-functional requirements) | Características observáveis do comportamento do agente durante a execução, que não descrevem o que ele faz nem o que ele nunca deve fazer, mas quão bem ele faz. | A execução deve produzir uma resposta estruturada e adequada para avaliação automatizada. | O atributo está descrito de forma observável/mensurável no comportamento? Existe um critério de aceitação verificável? Havendo trade-off entre atributos, a especificação deixa explícita a prioridade? |

**Comportamentos:**

| Comportamento | Input | Tools | Saída | Efeito no Ambiente | Condição de Satisfação |
| --- | --- | --- | --- | --- | --- |
| **B1 — Listar issues de um repositório** | Repositório válido e público; solicitação de filtragem em linguagem natural. | GitHub MCP para recuperar as issues do repositório. | Resposta contendo as issues que correspondem à solicitação, com número, título e estado; `status = success`. | Apenas leitura das informações do repositório. | A execução retorna `status = success` e as issues retornadas correspondem ao repositório e à solicitação fornecidos. |
| **B2 — Interpretar e aplicar filtro** | Lista de issues recuperadas; critérios de filtragem fornecidos em linguagem natural. | Informações recuperadas pelo GitHub MCP. | Lista contendo as issues que correspondem aos critérios solicitados. | Nenhum. | As issues retornadas satisfazem os critérios de filtragem especificados na solicitação. |
| **B3 — Nenhuma issue corresponde ao filtro** | Repositório válido; solicitação de filtragem sem correspondências. | GitHub MCP. | Lista vazia e `status = success`. | Apenas leitura das informações do repositório. | A saída contém uma lista vazia e `status = success`. |
| **B4 — Repositório inválido ou inacessível** | Repositório inválido ou inacessível. | GitHub MCP. | `status = failure` e mensagem indicando a falha. | Nenhum. | A saída contém `status = failure` e uma mensagem indicando a impossibilidade de acesso ao repositório. |
| **B5 — Retornar resposta estruturada** | Resultado da execução do agente. | — | Objeto contendo `action`, `repository`, `filtered_issues`, `status` e `message`. Cada issue contém `issue_number`, `title` e `state`. | Nenhum. | A saída contém os campos definidos no contrato e os valores possuem os tipos e formatos especificados. |