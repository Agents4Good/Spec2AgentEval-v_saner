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
| Goal | Objetivo e comportamento principal do agente na voz ativa | O agente atua analisando consultas de suporte ao cliente recebidas em linguagem natural. Em uma única chamada de LLM, ele categoriza a consulta em Technical, Billing ou General e analisa o sentimento como Positive, Neutral ou Negative, retornando a resposta estruturada. | O comportamento principal está declarado definindo quem faz, o que faz e o resultado? |
| Persona (Opcional) | A personalidade que o agente assume ao realizar as tarefas | O agente assume a persona de um classificador de consultas de suporte ao cliente, focado unicamente em analisar e categorizar o texto de entrada. | A especificação define uma persona e sua área de atuação? A persona é adequada/coerente com o Goal? |
| Inputs | Informação enviada para o agente | Uma string (query: str) contendo a mensagem do cliente em linguagem natural. | A especificação define qual é a entrada esperada? |
| Tools | Lista com as ferramentas disponíveis e para que elas servem | Não se aplica. | A especificação define a ferramenta, seus parâmetros, o retorno e o propósito de uso dela estão definidos? |
| Constraints | Uma afirmação que restringe as ações que o agente (ou seus usuários/operadores) têm permissão de realizar. Descreve o que o agente deve, não deve, ou não pode fazer, ou define que apenas determinados papéis/condições autorizam uma certa ação. | - O agente não deve realizar múltiplos passos ou chamadas separadas para extrair Categoria e Sentimento, ambas as informações devem ser identificadas no mesmo passo analítico.<br><br>- O agente não deve responder à consulta do cliente ou tentar resolver o problema diretamente, sua ação se restringe unicamente a classificar e analisar a entrada. | A especificação declara explicitamente uma ação que o agente deve, não deve, ou só pode sob certas condições? A restrição está expressa em termos de ação/comportamento permitido ou proibido — não em termos de tecnologia, ferramenta ou biblioteca usada para implementar o agente? Quando a restrição envolve papel/autorização ("só X pode fazer Y"), a especificação define quem ou o que está autorizado? A restrição é verificável: é possível observar, no comportamento do agente, se ela foi violada ou respeitada? |
| Policies/Invariants | Afirmações que descrevem o que é verdadeiro sobre o domínio ou sobre o estado do agente em um dado momento, e que devem permanecer válidas durante toda a execução. Descrevem relações, propriedades ou condições do mundo/domínio que servem de base para o comportamento do agente. | - Todo resultado considerado válido deve obrigatoriamente mapear a Categoria como sendo exatamente um dentre: "Technical", "Billing" ou "General".<br><br>- Todo resultado considerado válido deve obrigatoriamente mapear o Sentimento como sendo exatamente um dentre: "Positive", "Neutral" ou "Negative". | A especificação declara uma relação, propriedade ou condição do domínio que é (ou deve ser) sempre verdadeira, e não uma ação permitida/proibida? O invariante está conectado a uma entrada, saída, evento ou dado que o agente efetivamente manipula (não é conhecimento de domínio irrelevante ao escopo do agente)? É possível verificar, a qualquer momento durante a execução, se o invariante ainda é válido? |
| Quality attributes (non-functional requirements) | Características observáveis do comportamento do agente durante a execução, que não descrevem o que ele faz (isso é Goal/Behavior Requirements) nem o que ele nunca deve fazer (Constraints), mas quão bem ele faz — de forma consistente, segura, eficiente e confiável. Moldam diretamente a confiança e a experiência de quem depende do agente. | Não se aplica. | O atributo está descrito de forma observável/mensurável no comportamento (não é uma característica de implementação, ex: "usar cache" não é quality attribute, é constraint de implementação)? Existe um critério de aceitação verificável (limiar, taxa, ou condição de teste), e não apenas um adjetivo vago ("deve ser rápido" sem definir o quê é rápido)? Havendo trade-off entre atributos (ex: robustez vs. performance), a especificação deixa explícita a prioridade? |

**Comportamentos:**

| Comportamento | Input | Tools | Saída | Efeito no Ambiente | Condição de Satisfação |
| --- | --- | --- | --- | --- | --- |
| **B1 — Classificação e análise** | Entrada válida: Uma string (`query: str`) contendo a mensagem do cliente em linguagem natural. | Não se aplica. | Classificação da categoria e análise do sentimento. | Não se aplica. | Dada uma entrada válida o agente deve classificar a categoria e analisar o sentimento em uma única chamada ao LLM. |
| **B2 — Formatação de saída válida** | Entrada válida. | Não se aplica. | String neste formato:<br><br>- Category: one of "Technical", "Billing", or "General"<br><br>- Sentiment: one of "Positive", "Neutral", or "Negative" | Não se aplica. | Dada uma entrada válida o agente deve retornar uma string contendo os campos Category e Sentiment. |
| **B3 — Formatação de saída inválida** | Entrada inválida (entradas vazias, nulas ou não correlacionadas ao suporte ao cliente). | Não se aplica. | String com a mensagem de erro: "Error: could not classify the query" | Não se aplica. | Dada uma entrada inválida (entradas vazias, nulas ou não correlacionadas ao suporte ao cliente) o agente deve retornar "Error: could not classify the query". |