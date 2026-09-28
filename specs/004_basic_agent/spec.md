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

**Requisitos:**

| Requisito | Significado | Especificação do agente | Critério de Corretude |
| --- | --- | --- | --- |
| Goal | Objetivo e comportamento principal do agente na voz ativa | O agente deve responder de forma autônoma e diretamente a perguntas ou instruções do usuário com informações precisas, concisas e baseadas em fatos, utilizando um LLM como motor de raciocínio. | O comportamento principal está declarado definindo quem faz, o que faz e o resultado? |
| Persona (Opcional) | A personalidade que o agente assume ao realizar as tarefas | Um assistente útil (_helpful assistant_) de perguntas e respostas de conhecimento geral, com um tom respeitoso. | A especificação define uma persona e sua área de atuação? A persona é adequada/coerente com o Goal? |
| Inputs | Informação enviada para o agente | O agente recebe um texto representando a instrução ou pergunta do usuário. | A especificação define qual é a entrada esperada? |
| Tools | Lista com as ferramentas disponíveis e para que elas servem | Não se aplica. | A especificação define a ferramenta, seus parâmetros, o retorno e o **propósito de uso** dela estão definidos? |
| Constraints | Uma afirmação que restringe as ações que o agente (ou seus usuários/operadores) têm permissão de realizar. | (1) Quando o agente não souber a resposta, ele deve declarar explicitamente que não sabe, sem inventar informações. (2) O agente deve responder com base nas informações factuais mais recentes de que dispõe e, quando a pergunta depender de informação que ele não possa confirmar como atual, deve informar essa limitação em vez de apresentá-la como fato atual. | A especificação declara explicitamente uma ação que o agente deve, não deve, ou só pode sob certas condições? A restrição está expressa em termos de ação/comportamento permitido ou proibido — não em termos de tecnologia, ferramenta ou biblioteca usada para implementar o agente? Quando a restrição envolve papel/autorização ("só X pode fazer Y"), a especificação define quem ou o que está autorizado? A restrição é verificável: é possível observar, no comportamento do agente, se ela foi violada ou respeitada? |
| Policies/Invariants | Afirmações que descrevem o que é verdadeiro sobre o domínio ou sobre o estado do agente em um dado momento | A saída do agente é sempre um texto, e esse texto é ou uma resposta à entrada do usuário ou exatamente a string `Error: Unable to get response from agent.`. | A especificação declara uma relação, propriedade ou condição do domínio que é (ou deve ser) sempre verdadeira, e não uma ação permitida/proibida? O invariante está conectado a uma entrada, saída, evento ou dado que o agente efetivamente manipula (não é conhecimento de domínio irrelevante ao escopo do agente)? É possível verificar, a qualquer momento durante a execução, se o invariante ainda é válido? |
| Quality attributes (non-functional requirements) | Características observáveis do comportamento do agente durante a execução, que não descrevem o que ele faz nem o que ele nunca deve fazer, mas quão bem ele faz. | Precisão e concisão: a resposta deve ir direto ao ponto, sem preâmbulos nem repetição da pergunta, e conter apenas informações factuais. Em caso de conflito entre os dois atributos, a precisão prevalece sobre a concisão. | O atributo está descrito de forma observável/mensurável no comportamento? Existe um critério de aceitação verificável? Havendo trade-off entre atributos, a especificação deixa explícita a prioridade? |

**Comportamentos:**

| Comportamento | Trigger | Input | Tools | Saída | Efeito no Ambiente |
| --- | --- | --- | --- | --- | --- |
| **B1 — Responder de forma autônoma e direta à pergunta ou instrução do usuário, com informações precisas, concisas e baseadas em fatos.** | Recebimento de uma entrada textual válida (não vazia) com uma pergunta ou instrução clara. | Um texto representando a instrução ou pergunta do usuário. | Nenhuma. | Um texto contendo a resposta para a pergunta ou instrução do usuário. | Nenhum. |
| **B2 — Em caso de falha ao gerar uma resposta, interceptar o erro e retornar exatamente a mensagem de erro padrão.** | Exceção na chamada ao LLM, estouro de tempo ou resposta vazia durante a geração da resposta. | Um texto representando a instrução ou pergunta do usuário. | Nenhuma. | A string exata: `Error: Unable to get response from agent.` | Nenhum. |
| **B3 — Admitir explicitamente a falta de conhecimento quando não souber a resposta ou não puder confirmar que a informação é atual.** | Entrada textual válida cuja resposta o agente não conhece ou não consegue confirmar como atual. | Um texto representando a instrução ou pergunta do usuário. | Nenhuma. | Um texto que declara explicitamente que o agente não sabe a resposta (ou não pode confirmá-la como atual), sem apresentar informação inventada como fato. | Nenhum. |
| **B4 — Tratar entrada inválida retornando a mensagem de erro padrão.** | Entrada vazia, composta apenas por espaços em branco ou que não seja um texto. | Uma entrada vazia, apenas com espaços ou não textual. | Nenhuma. | A string exata: `Error: Unable to get response from agent.` | Nenhum. |
 
| Campo | Critério de corretude |
| --- | --- |
| Comportamento | Cada comportamento está especificado como um requisito individual e identificável, separado do Goal? Cada requisito é atômico o suficiente para ser testado isoladamente (dá pra escrever um caso de teste "dado X, o agente deve fazer Y")? A especificação cobre explicitamente o comportamento esperado diante de: entrada inválida, falha de ferramenta/tool call, ambiguidade, ausência de informação necessária? Os requisitos, somados, são suficientes para realizar o Goal declarado (nada essencial ficou implícito)? |
| Trigger | A condição ou evento que inicia o comportamento está explicitamente definido e é observável? |
| Input | A especificação define qual é a entrada esperada? |
| Tools | As ferramentas necessárias para o comportamento estão identificadas e seu propósito de uso está definido? |
| Expected Outcome | A especificação define qual alteração deve ocorrer no ambiente? A alteração esperada é observável e verificável após a execução do comportamento? |
| Environment Effect | A especificação define qual ambiente o agente vai alterar?  A especificação define como será feita a alteração no ambiente? |