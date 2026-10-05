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
| Goal | Objetivo e comportamento principal do agente na voz ativa | O agente deve avaliar redações do Exame Nacional do Ensino Médio (ENEM) seguindo a rubrica oficial de avaliação do INEP e produzir uma avaliação estruturada da redação. | O comportamento principal está declarado definindo quem faz, o que faz e o resultado? |
| Persona (Opcional) | A personalidade que o agente assume ao realizar as tarefas | Avaliador especialista de redações do ENEM, com domínio das diretrizes, rubricas de correção e critérios do INEP. | A especificação define uma persona e sua área de atuação? A persona é adequada/coerente com o Goal? |
| Inputs | Informação enviada para o agente | O agente recebe o tema da redação (`essay_topic: str`) e uma redação argumentativa escrita pelo estudante (`essay: str`). | A especificação define qual é a entrada esperada? |
| Tools | Lista com as ferramentas disponíveis e para que elas servem | Não se aplica. | A especificação define a ferramenta, seus parâmetros, o retorno e o **propósito de uso** dela estão definidos? |
| Constraints | Uma afirmação que restringe as ações que o agente (ou seus usuários/operadores) têm permissão de realizar. | C1 — O agente deve concluir a execução e retornar uma resposta válida dentro do limite máximo 30 segundos. Exceder o tempo limite resulta em falha funcional automática. <br><br> C2 — O agente não deve entrar em loops infinitos nem depender de tentativas indefinidas. | A especificação declara explicitamente uma ação que o agente deve, não deve, ou só pode sob certas condições? A restrição está expressa em termos de ação/comportamento permitido ou proibido — não em termos de tecnologia, ferramenta ou biblioteca usada para implementar o agente? Quando a restrição envolve papel/autorização ("só X pode fazer Y"), a especificação define quem ou o que está autorizado? A restrição é verificável: é possível observar, no comportamento do agente, se ela foi violada ou respeitada? |
| Policies/Invariants | Afirmações que descrevem o que é verdadeiro sobre o domínio ou sobre o estado do agente em um dado momento | A nota total da redação é sempre estritamente igual à soma das notas individuais atribuídas às 5 competências do INEP. Redações desqualificadas resultam sempre em nota final 0 de 1000 com nota 0 em todas as 5 competências. | A especificação declara uma relação, propriedade ou condição do domínio que é (ou deve ser) sempre verdadeira, e não uma ação permitida/proibida? O invariante está conectado a uma entrada, saída, evento ou dado que o agente efetivamente manipula (não é conhecimento de domínio irrelevante ao escopo do agente)? É possível verificar, a qualquer momento durante a execução, se o invariante ainda é válido? |
| Quality attributes (non-functional requirements) | Características observáveis do comportamento do agente durante a execução, que não descrevem o que ele faz nem o que ele nunca deve fazer, mas quão bem ele faz. | O agente deve concluir a execução em no máximo 30 segundos e retornar uma resposta válida dentro desse limite, verificado pelo tempo decorrido entre a chamada e o retorno. | O atributo está descrito de forma observável/mensurável no comportamento? Existe um critério de aceitação verificável? Havendo trade-off entre atributos, a especificação deixa explícita a prioridade? |

**Comportamentos:**

| Comportamento | Input | Tools | Saída | Efeito no Ambiente | Condição de Satisfação |
| --- | --- | --- | --- | --- | --- |
| **B1 — Validação de desqualificação** | Tema da redação (`essay_topic: str`) e redação argumentativa (`essay: str`). | Não se aplica. | Resposta em `str` atribuindo nota 0 para as 5 competências e nota final 0 de 1000 caso a redação possua condições de desqualificação do INEP. | Não se aplica. | O agente valida se a redação possui condições de desqualificação do INEP, e caso desqualificada atribui nota 0 para as 5 competências. |
| **B2 — Avaliação por competência** | Tema da redação (`essay_topic: str`) e redação argumentativa (`essay: str`). | Não se aplica. | Resposta em `str` contendo a avaliação do texto individualmente nas 5 competências do INEP (Competência 1 a Competência 5). | Não se aplica. | O agente avalia o texto individualmente nas 5 competências do INEP. |
| **B3 — Pontuação** | Avaliação das competências da redação. | Não se aplica. | Pontuação válida (entre 0 e 200) atribuída a cada competência e a soma total calculada. | Não se aplica. | O agente atribui uma pontuação válida (entre 0 e 200) a cada competência e calcula a soma total. |
| **B4 — Feedback por competência** | Avaliação das competências da redação. | Não se aplica. | Feedback detalhado por competência contendo pontos fortes, pontos fracos e sugestões de melhoria. | Não se aplica. | O agente retorna feedback detalhado para cada competência, incluindo pontos fortes, pontos fracos e sugestões de melhoria. |
| **B5 — Feedback geral** | Avaliação geral da redação. | Não se aplica. | Feedback geral resumindo a qualidade geral da redação e as principais observações. | Não se aplica. | O agente retorna um feedback geral da redação, resumindo a qualidade geral e as principais observações. |
| **B6 — Formatação da saída** | Resultado da avaliação do agente. | Não se aplica. | Resposta em `str` no formato estrito:<br><br>`Competency 1 (140/200): Feedback here.`<br><br>`Competency 2 (120/200): Feedback here.`<br><br>`Competency 3 (160/200): Feedback here.`<br><br>`Competency 4 (140/200): Feedback here.`<br><br>`Competency 5 (120/200): Feedback here.`<br><br>`Total: 680/1000. Overall feedback here.`| Não se aplica. | A saída deve seguir estritamente o formato especificado. |