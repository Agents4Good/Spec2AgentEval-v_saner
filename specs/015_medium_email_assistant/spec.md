**Task:** você é um gerador de agentes de IA e abaixo está uma especificação de um agente. Gere um agente conforme a especificação. Considere que as ferramentas do agente devem ser geradas junto ao código do agente.  
**Contexto:** Um agente de IA é uma entidade que usa LLM e ferramentas para executar tarefas específicas de forma autônoma.  
**Output:** agent.py, requirements.txt, tests.py
**Contrato de interface com o benchmark:**

| Requisito de interface | Exemplo |
| --- | --- |
| Modelo | GPT-4o |
| Framework | LangGraph |
| GenAI Tool | ChatOpenAI |
| Entrypoint | `agent(email_input: str)` |
| Output | `Tuple[CompiledGraph, dict]` |

**Detalhes de Implementação**
| Variáveis de ambiente | Descrição |
| --- | --- |
| OPENAI_API_KEY | Chave de acesso à API da OpenAI. |

**Requisitos gerais:**

| Requisito | Significado | Especificação do agente | Critério de Corretude |
| --- | --- | --- | --- |
| Goal | Objetivo e comportamento principal do agente na voz ativa | O agente deve atuar como um assistente executivo, fazendo a triagem de e-mails recebidos, classificando cada um como `ignore`, `notify` ou `respond`, executando autonomamente exatamente uma ferramenta correspondente à classificação, e retornando o resultado do processamento. | O comportamento principal está declarado definindo quem faz, o que faz e o resultado? |
| Persona (Opcional) | A personalidade que o agente assume ao realizar as tarefas | Assistente executivo de e-mails, com tom profissional, objetivo e direto. | A especificação define uma persona e sua área de atuação? A persona é adequada/coerente com o Goal? |
| Inputs | Informação enviada para o agente | Um texto contendo o corpo bruto do e-mail recebido. | A especificação define qual é a entrada esperada? |
| Tools | Lista com as ferramentas disponíveis e para que elas servem | (T1) `archive_email`: arquiva silenciosamente o e-mail; retorna uma string de confirmação; usada quando a classificação é `ignore`.(T2) `send_slack_alert`: envia um resumo urgente para o canal do Slack do CEO; retorna uma string de confirmação; usada quando a classificação é `notify`.(T3) `save_email_draft`: salva um rascunho de resposta gerado na base "Drafts" do Notion; retorna uma string de confirmação; usada quando a classificação é `respond`. | A especificação define a ferramenta, seus parâmetros, o retorno e o **propósito de uso** dela estão definidos? |
| Constraints | Uma afirmação que restringe as ações que o agente (ou seus usuários/operadores) têm permissão de realizar. | (1) O agente deve executar no máximo uma ferramenta por e-mail processado; não deve encadear nem executar mais de uma ferramenta na mesma execução. (2) A ferramenta executada deve corresponder exatamente à classificação atribuída ao e-mail (`ignore`→ T1, `notify`→ T2, `respond`→ T3); o agente não deve executar uma ferramenta diferente da mapeada para a classificação escolhida. | A especificação declara explicitamente uma ação que o agente deve, não deve, ou só pode sob certas condições? A restrição está expressa em termos de ação/comportamento permitido ou proibido — não em termos de tecnologia, ferramenta ou biblioteca usada para implementar o agente? Quando a restrição envolve papel/autorização ("só X pode fazer Y"), a especificação define quem ou o que está autorizado? A restrição é verificável: é possível observar, no comportamento do agente, se ela foi violada ou respeitada? |
| Policies/Invariants | Afirmações que descrevem o que é verdadeiro sobre o domínio ou sobre o estado do agente em um dado momento | (1) O campo `action` é sempre `"process_email"`. (2) `classification` pertence sempre a `ignore`, `notify`, `respond` ou `unknown`. (3) `classification` é `unknown` se `status` é `failure`. (4) `draft_response` é uma string não vazia se `classification` é `respond` e `status` é `success`; em qualquer outro caso é `null`. (5) `actions_taken` contém exatamente um elemento (o nome da ferramenta executada) quando `status` é `success`, e nenhum elemento quando `status` é `failure`. (6) O elemento de `actions_taken`, quando presente, corresponde à ferramenta mapeada para `classification` conforme a Constraint (3). | A especificação declara uma relação, propriedade ou condição do domínio que é (ou deve ser) sempre verdadeira, e não uma ação permitida/proibida? O invariante está conectado a uma entrada, saída, evento ou dado que o agente efetivamente manipula (não é conhecimento de domínio irrelevante ao escopo do agente)? É possível verificar, a qualquer momento durante a execução, se o invariante ainda é válido? |
| Quality attributes (non-functional requirements) | Características observáveis do comportamento do agente durante a execução, que não descrevem o que ele faz nem o que ele nunca deve fazer, mas quão bem ele faz. | Concisão do campo `message`: no máximo 2 frases, justificando a classificação e a ação tomada. | O atributo está descrito de forma observável/mensurável no comportamento? Existe um critério de aceitação verificável? Havendo trade-off entre atributos, a especificação deixa explícita a prioridade? |

**Comportamentos:**

| Comportamento | Trigger | Input | Tools | Saída | Efeito no Ambiente |
| --- | --- | --- | --- | --- | --- |
| **B1 — Classificar o e-mail como `ignore` (SPAM, marketing, cold e-mails irrelevantes) e executar `archive_email`.** | Recebimento de um e-mail válido (não vazio) cujo conteúdo é identificado como irrelevante. | Um texto contendo o corpo do e-mail. | T1. | `{"action": "process_email", "classification": "ignore", "actions_taken": ["archive_email"], "status": "success", "message": "string (≤ 2 frases)", "draft_response": null}` | O e-mail é arquivado (chamada real ou mock de `archive_email`). |
| **B2 — Classificar o e-mail como `notify` (atualizações financeiras urgentes, mensagens críticas de RH) e executar `send_slack_alert`.** | Recebimento de um e-mail válido (não vazio) cujo conteúdo é identificado como urgente/crítico mas não exige resposta direta. | Um texto contendo o corpo do e-mail. | T2. | `{"action": "process_email", "classification": "notify", "actions_taken": ["send_slack_alert"], "status": "success", "message": "string (≤ 2 frases)", "draft_response": null}` | Um alerta é enviado ao canal do Slack do CEO (chamada real ou mock de `send_slack_alert`). |
| **B3 — Classificar o e-mail como `respond` (solicitações de reunião, dúvidas de clientes, apresentações), gerar um rascunho de resposta e executar `save_email_draft`.** | Recebimento de um e-mail válido (não vazio) cujo conteúdo exige uma resposta direta. | Um texto contendo o corpo do e-mail. | T3. | `{"action": "process_email", "classification": "respond", "actions_taken": ["save_email_draft"], "status": "success", "message": "string (≤ 2 frases)", "draft_response": "string"}` | O rascunho gerado é salvo na base "Drafts" do Notion (chamada real ou mock de `save_email_draft`). |
| **B4 — Em caso de timeout ou de falha ao interpretar (parsear) a saída do LLM, retornar o schema de falha padrão.** | Timeout das APIs da LLM ou falha ao parsear a saída do LLM em resposta estruturada. | Um texto contendo o corpo do e-mail. | Nenhuma. | `{"action": "process_email", "classification": "unknown", "actions_taken": [], "status": "failure", "message": "string", "draft_response": null}` | Nenhum. |
| **B5 — Tratar entrada inválida retornando o schema de falha padrão.** | Entrada vazia, composta apenas por espaços em branco ou que não seja um texto. | Uma entrada vazia, apenas com espaços ou não textual. | Nenhuma. | `{"action": "process_email", "classification": "unknown", "actions_taken": [], "status": "failure", "message": "string informando que a entrada é inválida", "draft_response": null}` | Nenhum. |

| Campo | Critério de corretude |
| --- | --- |
| Comportamento | Cada comportamento está especificado como um requisito individual e identificável, separado do Goal? Cada requisito é atômico o suficiente para ser testado isoladamente (dá pra escrever um caso de teste "dado X, o agente deve fazer Y")? A especificação cobre explicitamente o comportamento esperado diante de: entrada inválida, falha de ferramenta/tool call, ambiguidade, ausência de informação necessária? Os requisitos, somados, são suficientes para realizar o Goal declarado (nada essencial ficou implícito)? |
| Trigger | A condição ou evento que inicia o comportamento está explicitamente definido e é observável? |
| Input | A especificação define qual é a entrada esperada? |
| Tools | As ferramentas necessárias para o comportamento estão identificadas e seu propósito de uso está definido? |
| Expected Outcome | A especificação define qual alteração deve ocorrer no ambiente? A alteração esperada é observável e verificável após a execução do comportamento? |
| Environment Effect | A especificação define qual ambiente o agente vai alterar?  A especificação define como será feita a alteração no ambiente? |