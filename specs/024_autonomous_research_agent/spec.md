**Task:** você é um gerador de agentes de IA e abaixo está uma especificação de um agente. Gere um agente conforme a especificação. Considere que as ferramentas do agente devem ser geradas junto ao código do agente.  
**Contexto:** Um agente de IA é uma entidade que usa LLM e ferramentas para executar tarefas específicas de forma autônoma.  
**Output:** agent.py, requirements.txt, tests.py (sem módulos, pacotes ou subdiretórios adicionais)  
**Contrato de interface com o benchmark:**

| Requisito de interface | Exemplo |
| --- | --- |
| Modelo | GPT-4o |
| Framework | LangGraph |
| GenAI Tool | ChatOpenAI |
| Entrypoint | `agent(user_request: str)` |
| Output | `str` (contendo exclusivamente um objeto JSON válido) |

**Detalhes de Implementação**
| Variáveis de ambiente | Descrição |
| --- | --- |
| OPENAI_API_KEY | Chave de acesso à API da OpenAI. |

**Requisitos gerais:**

| Requisito | Significado | Especificação do agente | Critério de Corretude |
| --- | --- | --- | --- |
| Goal | Objetivo e comportamento principal do agente  na voz ativa | O agente deve realizar uma pesquisa investigativa em múltiplas etapas: primeiro descobrir uma entidade específica (gene, cientista, teorema etc.) usando uma ferramenta de conhecimento geral, em seguida usar essa entidade para consultar a base acadêmica especializada apropriada, e retornar a síntese final . | O comportamento principal está declarado definindo quem faz, o que faz e o resultado? |
| Persona (Opcional) | A personalidade que o agente assume ao realizar as tarefas | Assistente de pesquisa investigativa, com tom profissional, objetivo e direto. Conduz a investigação em etapas de forma metódica e só reporta o resultado final estruturado, sem comentários, opiniões ou texto de conversa. | A especificação define uma persona e sua área de atuação? A persona é adequada/coerente com o Goal? |
| Inputs | Informação enviada para o agente | Um texto em linguagem natural complexo, que exige investigação em múltiplas etapas. | A especificação define qual é a entrada esperada? |
| Tools | Lista com as ferramentas disponíveis e para que elas servem | (T1) `get_wikipedia_summary`: obtém o resumo de um tópico na API REST da Wikipedia (`https://en.wikipedia.org/api/rest_v1/page/summary/{topic}`); retorna o texto de resumo do tópico, ou um erro/ausência de resultado se o tópico não for encontrado; usada na Etapa 1 (Descoberta), para identificar a entidade específica mencionada no pedido do usuário.(T2) `search_arxiv`: busca artigos no ArXiv (`http://export.arxiv.org/api/query`, via pacote `arxiv` ou `requests`); retorna uma lista de artigos, cada um com `title` e `summary`; usada na Etapa 2 (Aprofundamento), quando a entidade descoberta pertence ao domínio de Ciências Exatas/Tecnologia.(T3) `search_pubmed`: busca artigos no PubMed via API E-utilities do NCBI (`esearch.fcgi` para obter o `idlist` e `esummary.fcgi` para obter os resumos, via `requests`); retorna uma lista de artigos, cada um com `title` e `summary`; usada na Etapa 2 (Aprofundamento), quando a entidade descoberta pertence ao domínio de Ciências da Vida/Saúde. | A especificação define a ferramenta, seus parâmetros, o retorno e o **propósito de uso** dela estão definidos? |
| Constraints | Uma afirmação que restringe as ações que o agente (ou seus usuários/operadores) têm permissão de realizar. | (C1) O agente deve executar a investigação em duas etapas ordenadas: primeiro `get_wikipedia_summary` (Descoberta), depois `search_arxiv` ou `search_pubmed` (Aprofundamento); a etapa de Aprofundamento não deve ocorrer antes da etapa de Descoberta. (C2) Se `get_wikipedia_summary` retornar erro ou não encontrar o tópico, o agente não deve abortar a execução; deve inferir a entidade usando o conhecimento interno do LLM e prosseguir para a Etapa 2. (C3) A saída final não deve conter nenhum texto de preenchimento conversacional (ex.: "Aqui está o seu JSON:") nem marcações markdown (ex.: blocos de código com crases); deve ser exclusivamente a string do objeto JSON. | A especificação declara explicitamente uma ação que o agente deve, não deve, ou só pode sob certas condições? A restrição está expressa em termos de ação/comportamento permitido ou proibido — não em termos de tecnologia, ferramenta ou biblioteca usada para implementar o agente? Quando a restrição envolve papel/autorização ("só X pode fazer Y"), a especificação define quem ou o que está autorizado? A restrição é verificável: é possível observar, no comportamento do agente, se ela foi violada ou respeitada? |
| Policies/Invariants | Afirmações que descrevem o que é verdadeiro sobre o domínio ou sobre o estado do agente em um dado momento | (I1) O domínio da base acadêmica usada na Etapa 2 corresponde ao domínio da entidade descoberta na Etapa 1: entidades de Ciências da Vida/Saúde (ex.: genes, doenças) usam `search_pubmed`; entidades de Ciências Exatas/Tecnologia usam `search_arxiv`. | A especificação declara uma relação, propriedade ou condição do domínio que é (ou deve ser) sempre verdadeira, e não uma ação permitida/proibida? O invariante está conectado a uma entrada, saída, evento ou dado que o agente efetivamente manipula (não é conhecimento de domínio irrelevante ao escopo do agente)? É possível verificar, a qualquer momento durante a execução, se o invariante ainda é válido? |
| Quality attributes (non-functional requirements) | Características observáveis do comportamento do agente durante a execução, que não descrevem o que ele faz nem o que ele nunca deve fazer, mas quão bem ele faz. | (QA1) Precisão do encadeamento de dados: o termo de busca usado na Etapa 2 deve corresponder exatamente (ou de forma muito próxima) à entidade descoberta na Etapa 1, e não ao tópico genérico original do usuário. | O atributo está descrito de forma observável/mensurável no comportamento? Existe um critério de aceitação verificável? Havendo trade-off entre atributos, a especificação deixa explícita a prioridade? |

**Comportamentos:**

| Comportamento | Trigger | Input | Tools | Saída | Efeito no Ambiente |
| --- | --- | --- | --- | --- | --- |
| **B1 — Descobrir a entidade específica mencionada no pedido do usuário via Wikipedia (Etapa 1).** | Recebimento de uma solicitação válida (não vazia) que menciona ou implica uma entidade específica a ser descoberta. | Um texto em linguagem natural complexo. | T1. | A entidade descoberta, extraída do resumo retornado pela ferramenta, usada como entrada da Etapa 2. | Nenhum. |
| **B2 — Quando a descoberta via Wikipedia falhar ou não encontrar o tópico, inferir a entidade usando o conhecimento interno do LLM e prosseguir para a Etapa 2 sem abortar a execução.** | T1 retorna erro ou nenhum resultado para o tópico solicitado. | Um texto em linguagem natural complexo. | T1. | A entidade inferida pelo LLM (string), usada como entrada da Etapa 2, no lugar do resultado da ferramenta. | Nenhum. |
| **B3 — Usar a entidade descoberta para consultar a base acadêmica apropriada (ArXiv ou PubMed) e sintetizar o resultado final em JSON (Etapas 2 e 3).** | Conclusão da Etapa 1 ou da Etapa 2 (B1 ou B2) com uma entidade descoberta ou inferida. | A entidade descoberta/inferida e o pedido original do usuário. | T2 **ou** T3. | Uma string contendo exclusivamente um objeto JSON válido, sem texto conversacional nem markdown, com as chaves `topic`, `discovered_entity` e `papers`, onde o campo de artigos é uma lista de objetos com título e fonte/resumo. | Nenhum. |
| **B4 — Em caso de entrada vazia, erro de validação ou exceção inesperada durante a execução, retornar o JSON de erro padrão.** | Entrada vazia ou composta apenas por espaços em branco; erro de validação da entrada; ou exceção não tratada durante qualquer etapa da execução. | Uma entrada vazia, inválida, ou qualquer entrada durante a qual ocorra uma exceção. | Nenhuma (ou uma chamada de ferramenta que lançou exceção). | A string exata: `{"status": "error", "message": "Execution failed or invalid request."}` | Nenhum. |

| Campo | Critério de corretude |
| --- | --- |
| Comportamento | Cada comportamento está especificado como um requisito individual e identificável, separado do Goal? Cada requisito é atômico o suficiente para ser testado isoladamente (dá pra escrever um caso de teste "dado X, o agente deve fazer Y")? A especificação cobre explicitamente o comportamento esperado diante de: entrada inválida, falha de ferramenta/tool call, ambiguidade, ausência de informação necessária? Os requisitos, somados, são suficientes para realizar o Goal declarado (nada essencial ficou implícito)? |
| Trigger | A condição ou evento que inicia o comportamento está explicitamente definido e é observável? |
| Input | A especificação define qual é a entrada esperada? |
| Tools | As ferramentas necessárias para o comportamento estão identificadas e seu propósito de uso está definido? |
| Expected Outcome | A especificação define qual alteração deve ocorrer no ambiente? A alteração esperada é observável e verificável após a execução do comportamento? |
| Environment Effect | A especificação define qual ambiente o agente vai alterar?  A especificação define como será feita a alteração no ambiente? |