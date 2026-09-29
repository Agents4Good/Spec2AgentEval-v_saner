**Task:** você é um gerador de agentes de IA e abaixo está uma especificação de um agente. Gere um agente conforme a especificação. Considere que as ferramentas do agente devem ser geradas junto ao código do agente.  
**Contexto:** Um agente de IA é uma entidade que usa LLM e ferramentas para executar tarefas específicas de forma autônoma.  
**Output:** agent.py, requirements.txt, tests.py (toda a implementação do agente deve estar em um único arquivo `agent.py`, sem módulos, pacotes ou subdiretórios adicionais)  
**Contrato de interface com o benchmark:**

| Requisito de interface | Exemplo |
| --- | --- |
| Modelo | GPT-4o |
| Framework | LangGraph |
| GenAI Tool | ChatOpenAI |
| Entrypoint | `agent(user_request: str)` |
| Output | `str` |

**Detalhes de Implementação**
| Variáveis de ambiente | Descrição |
| --- | --- |
| OPENAI_API_KEY | Chave de acesso à API da Openai |

**Requisitos gerais:**

| Requisito | Significado | Especificação do agente | Critério de Corretude |
| --- | --- | --- | --- |
| Goal | Objetivo e comportamento principal do agente na voz ativa | O agente deve fazer a triagem de uma solicitação acadêmica do usuário, selecionando exatamente uma ferramenta apropriada ao domínio do assunto (Ciências Exatas/Tecnologia, Ciências da Vida/Saúde ou Conceitos Gerais) e retornando um resumo formatado dos resultados encontrados, utilizando um LLM como motor de raciocínio. | O comportamento principal está declarado definindo quem faz, o que faz e o resultado? |
| Persona (Opcional) | A personalidade que o agente assume ao realizar as tarefas | Não se aplica. | A especificação define uma persona e sua área de atuação? A persona é adequada/coerente com o Goal? |
| Inputs | Informação enviada para o agente | Um texto em linguagem natural solicitando artigos acadêmicos ou a explicação de um conceito. | A especificação define qual é a entrada esperada? |
| Tools | Lista com as ferramentas disponíveis e para que elas servem | (T1) search_arxiv: busca artigos no ArXiv; retorna os dados brutos dos artigos encontrados; usada para o domínio de Ciências Exatas/Tecnologia. (T2) search_pubmed: busca artigos no PubMed; retorna os dados brutos dos artigos encontrados; usada para o domínio de Ciências da Vida/Saúde. (T3) get_wikipedia_summary: obtém o resumo de um tópico na Wikipedia; retorna os dados brutos do resumo do tópico; usada para o domínio de Conceitos Gerais. | A especificação define a ferramenta, seus parâmetros, o retorno e o **propósito de uso** dela estão definidos? |
| Constraints | Uma afirmação que restringe as ações que o agente (ou seus usuários/operadores) têm permissão de realizar. | (1) Para cada solicitação válida, o agente deve invocar exatamente uma ferramenta; não deve invocar duas ou mais ferramentas na mesma execução. (2) O agente deve basear a resposta final exclusivamente nos dados retornados pela ferramenta invocada, sem inventar artigos, títulos ou informações. | A especificação declara explicitamente uma ação que o agente deve, não deve, ou só pode sob certas condições? A restrição está expressa em termos de ação/comportamento permitido ou proibido — não em termos de tecnologia, ferramenta ou biblioteca usada para implementar o agente? Quando a restrição envolve papel/autorização ("só X pode fazer Y"), a especificação define quem ou o que está autorizado? A restrição é verificável: é possível observar, no comportamento do agente, se ela foi violada ou respeitada? |
| Policies/Invariants | Afirmações que descrevem o que é verdadeiro sobre o domínio ou sobre o estado do agente em um dado momento | (1) A saída do agente é sempre uma string (`str`): ou a resposta sintetizada a partir da ferramenta, ou exatamente uma das quatro mensagens de erro padrão definidas nos comportamentos. (2) Em toda execução com solicitação válida, apenas uma ferramenta é executada. (3) O mapeamento entre domínio e ferramenta é fixo: uma consulta de Ciências Exatas/Tecnologia (ex.: inteligência artificial) usa `search_arxiv`; uma consulta de Ciências da Vida/Saúde (ex.: medicina) usa `search_pubmed`; uma consulta de Conceitos Gerais usa `get_wikipedia_summary`. | A especificação declara uma relação, propriedade ou condição do domínio que é (ou deve ser) sempre verdadeira, e não uma ação permitida/proibida? O invariante está conectado a uma entrada, saída, evento ou dado que o agente efetivamente manipula (não é conhecimento de domínio irrelevante ao escopo do agente)? É possível verificar, a qualquer momento durante a execução, se o invariante ainda é válido? |
| Quality attributes (non-functional requirements) | Características observáveis do comportamento do agente durante a execução, que não descrevem o que ele faz nem o que ele nunca deve fazer, mas quão bem ele faz. | Não se aplica. | O atributo está descrito de forma observável/mensurável no comportamento? Existe um critério de aceitação verificável? Havendo trade-off entre atributos, a especificação deixa explícita a prioridade? |

**Comportamentos:**

| Comportamento | Trigger | Input | Tools | Saída | Efeito no Ambiente |
| --- | --- | --- | --- | --- | --- |
| **B1 — Para solicitações do domínio de Ciências Exatas/Tecnologia, invocar exatamente T1 e sintetizar os artigos encontrados em uma resposta em linguagem natural.** | Recebimento de uma solicitação válida (não vazia, dentro do escopo acadêmico e de um único domínio) cujo assunto é de Ciências Exatas/Tecnologia (ex.: inteligência artificial). | Um texto em linguagem natural solicitando artigos acadêmicos. | T1 | Uma string com uma breve introdução e uma lista de artigos com título e resumo curto de cada um. | Nenhum. |
| **B2 — Para solicitações do domínio de Ciências da Vida/Saúde, invocar exatamente T2 e sintetizar os artigos encontrados em uma resposta em linguagem natural.** | Recebimento de uma solicitação válida (não vazia, dentro do escopo acadêmico e de um único domínio) cujo assunto é de Ciências da Vida/Saúde (ex.: melanoma). | Um texto em linguagem natural solicitando artigos acadêmicos. | T2 | Uma string com uma breve introdução e uma lista de artigos com título e resumo curto de cada um. | Nenhum. |
| **B3 — Para solicitações de explicação de conceitos gerais, invocar exatamente T3 e sintetizar o resumo em uma resposta em linguagem natural.** | Recebimento de uma solicitação válida (não vazia, dentro do escopo acadêmico e de um único domínio) que pede a explicação de um conceito geral. | Um texto em linguagem natural solicitando a explicação de um conceito. | T3 | Uma string contendo um parágrafo explicativo coeso sobre o conceito. | Nenhum. |
| **B4 — Tratar entrada vazia retornando a mensagem de erro padrão.** | Entrada vazia ou composta apenas por espaços em branco. | Uma string vazia ou em branco. | Nenhuma. | A string exata: `Error: The user request cannot be empty.` | Nenhum. |
| **B5 — Tratar solicitações que abrangem múltiplos domínios distintos retornando a mensagem de erro padrão.** | Solicitação que pede, no mesmo texto, informações de mais de um domínio distinto (ex.: artigos sobre inteligência artificial e melanoma no mesmo pedido). | Um texto em linguagem natural que abrange múltiplos domínios. | Nenhuma. | A string exata: `Error: The query spans multiple distinct domains. Please make one specific research request at a time.` | Nenhum. |
| **B6 — Tratar solicitações fora do escopo acadêmico retornando a mensagem de erro padrão.** | Solicitação não vazia cujo assunto está fora do escopo acadêmico. | Um texto em linguagem natural fora do escopo acadêmico. | Nenhuma. | A string exata: `Error: Query outside of academic scope.` | Nenhum. |
| **B7 — Em caso de falha ao produzir uma resposta ou de falha inesperada na execução, interceptar o erro e retornar a mensagem de erro padrão.** | Exceção durante a execução, falha ou ausência de resultados na chamada da ferramenta, ou ausência de resposta do agente. | Um texto em linguagem natural com a solicitação do usuário. | Nenhuma. | A string exata: `Error: Unable to get response from agent.` | Nenhum. |

| Campo | Critério de corretude |
| --- | --- |
| Comportamento | Cada comportamento está especificado como um requisito individual e identificável, separado do Goal? Cada requisito é atômico o suficiente para ser testado isoladamente (dá pra escrever um caso de teste "dado X, o agente deve fazer Y")? A especificação cobre explicitamente o comportamento esperado diante de: entrada inválida, falha de ferramenta/tool call, ambiguidade, ausência de informação necessária? Os requisitos, somados, são suficientes para realizar o Goal declarado (nada essencial ficou implícito)? |
| Trigger | A condição ou evento que inicia o comportamento está explicitamente definido e é observável? |
| Input | A especificação define qual é a entrada esperada? |
| Tools | As ferramentas necessárias para o comportamento estão identificadas e seu propósito de uso está definido? |
| Expected Outcome | A especificação define qual alteração deve ocorrer no ambiente? A alteração esperada é observável e verificável após a execução do comportamento? |
| Environment Effect | A especificação define qual ambiente o agente vai alterar?  A especificação define como será feita a alteração no ambiente? |