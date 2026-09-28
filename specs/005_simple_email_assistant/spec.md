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
| Output | `state: dict` |

**Detalhes de Implementação**
| Variáveis de ambiente | Descrição |
| --- | --- |
| OPENAI_API_KEY | Chave de acesso à API da Openai |

**Requisitos gerais:**

| Requisito | Significado | Especificação do agente | Critério de Corretude |
| --- | --- | --- | --- |
| Goal | Objetivo e comportamento principal do agente na voz ativa | O agente deve atuar como um assistente executivo para fazer a triagem, classificação (ignorar, notificar ou responder) e processamento de e-mails recebidos. | O comportamento principal está declarado definindo quem faz, o que faz e o resultado? |
| Persona (Opcional) | A personalidade que o agente assume ao realizar as tarefas | Assistente executivo com tom profissional. | A especificação define uma persona e sua área de atuação? A persona é adequada/coerente com o Goal? |
| Inputs | Informação enviada para o agente | Um texto contendo o texto bruto ou corpo do e-mail recebido. | A especificação define qual é a entrada esperada? |
| Tools | Lista com as ferramentas disponíveis e para que elas servem | Não se aplica. | A especificação define a ferramenta, seus parâmetros, o retorno e o **propósito de uso** dela estão definidos? |
| Constraints | Uma afirmação que restringe as ações que o agente (ou seus usuários/operadores) têm permissão de realizar. | (1) O rascunho de resposta deve ser gerado baseando-se **exclusivamente** no contexto fornecido pelo e-mail, sem inventar dados externos. (2) A saída não pode conter marcações markdown, nem no dicionário (como blocos de código ao redor) nem nos textos de `message` e `draft_response`. (3) A saída não pode ter campos ausentes: os cinco campos do schema devem sempre estar presentes. | A especificação declara explicitamente uma ação que o agente deve, não deve, ou só pode sob certas condições? A restrição está expressa em termos de ação/comportamento permitido ou proibido — não em termos de tecnologia, ferramenta ou biblioteca usada para implementar o agente? Quando a restrição envolve papel/autorização ("só X pode fazer Y"), a especificação define quem ou o que está autorizado? A restrição é verificável: é possível observar, no comportamento do agente, se ela foi violada ou respeitada? |
| Policies/Invariants | Afirmações que descrevem o que é verdadeiro sobre o domínio ou sobre o estado do agente em um dado momento | (1) O campo `classification` pertence sempre a `ignore`, `notify`, `respond` ou `unknown`. (2) `classification` é `unknown` se, e somente se, `status` é `failure`. (3) `draft_response` é uma string não vazia se, e somente se, `classification` é `respond`; em qualquer outro caso é `null`. (4) O campo `action` é sempre `process_email`. | A especificação declara uma relação, propriedade ou condição do domínio que é (ou deve ser) sempre verdadeira, e não uma ação permitida/proibida? O invariante está conectado a uma entrada, saída, evento ou dado que o agente efetivamente manipula (não é conhecimento de domínio irrelevante ao escopo do agente)? É possível verificar, a qualquer momento durante a execução, se o invariante ainda é válido? |
| Quality attributes (non-functional requirements) | Características observáveis do comportamento do agente durante a execução, que não descrevem o que ele faz nem o que ele nunca deve fazer, mas quão bem ele faz. | Concisão da justificativa: o campo `message` deve ter no máximo 2 frases. | O atributo está descrito de forma observável/mensurável no comportamento? Existe um critério de aceitação verificável? Havendo trade-off entre atributos, a especificação deixa explícita a prioridade? |

**Comportamentos:**

| Comportamento | Trigger | Input | Tools | Saída | Efeito no Ambiente |
| --- | --- | --- | --- | --- | --- |
| **B1 — Analisar o conteúdo do e-mail recebido e classificá-lo em uma das três categorias: `ignore` (SPAM, marketing, e-mails irrelevantes), `notify` (atualizações financeiras urgentes, mensagens jurídicas) ou `respond` (solicitações de reunião, dúvidas de clientes).** | Recebimento de uma entrada textual válida (não vazia) com o corpo de um e-mail. | Um texto contendo o corpo do e-mail ou um texto bruto. | Nenhuma. | Um dicionário no formato: `{"action": "process_email", "classification": "ignore OR notify OR respond", "status": "success", "message": "string", "draft_response": "string OR null"}` | Nenhum. |
| **B2 — Fornecer uma breve justificativa lógica para a classificação escolhida no campo `message`.** | Conclusão da classificação do e-mail com sucesso (B1). | Um texto contendo o corpo do e-mail ou um texto bruto. | Nenhuma. | Um texto de justificativa, com no máximo 2 frases, no campo `message` da saída final. | Nenhum. |
| **B3 — Gerar de forma autônoma um rascunho de resposta caso o e-mail seja classificado como `respond`.** | Classificação do e-mail como `respond`. | Um texto contendo o corpo do e-mail ou um texto bruto. | Nenhuma. | Um texto de resposta ao e-mail, baseado apenas no contexto do próprio e-mail, no campo `draft_response`. | Nenhum. |
| **B4 — Em caso de falha no processamento ou estouro de tempo, capturar o erro e retornar o schema de falha.** | Exceção durante o processamento, estouro de tempo de execução ou saída do LLM fora do schema ou das categorias permitidas. | Um texto contendo o corpo do e-mail ou um texto bruto. | Nenhuma. | O dicionário padrão com `action = process_email`, `classification = unknown`, `status = failure`, `message` com uma breve descrição do erro (string) e `draft_response = null`. | Nenhum. |
| **B5 — Tratar entrada inválida retornando o schema de falha.** | Entrada vazia, composta apenas por espaços em branco ou que não seja um texto. | Uma entrada vazia, apenas com espaços ou não textual. | Nenhuma. | O dicionário padrão com `action = process_email`, `classification = unknown`, `status = failure`, `message` informando que a entrada é inválida e `draft_response = null`. | Nenhum. |

| Campo | Critério de corretude |
| --- | --- |
| Comportamento | Cada comportamento está especificado como um requisito individual e identificável, separado do Goal? Cada requisito é atômico o suficiente para ser testado isoladamente (dá pra escrever um caso de teste "dado X, o agente deve fazer Y")? A especificação cobre explicitamente o comportamento esperado diante de: entrada inválida, falha de ferramenta/tool call, ambiguidade, ausência de informação necessária? Os requisitos, somados, são suficientes para realizar o Goal declarado (nada essencial ficou implícito)? |
| Trigger | A condição ou evento que inicia o comportamento está explicitamente definido e é observável? |
| Input | A especificação define qual é a entrada esperada? |
| Tools | As ferramentas necessárias para o comportamento estão identificadas e seu propósito de uso está definido? |
| Expected Outcome | A especificação define qual alteração deve ocorrer no ambiente? A alteração esperada é observável e verificável após a execução do comportamento? |
| Environment Effect | A especificação define qual ambiente o agente vai alterar?  A especificação define como será feita a alteração no ambiente? |