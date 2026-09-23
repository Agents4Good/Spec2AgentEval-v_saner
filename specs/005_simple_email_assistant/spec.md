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
| Goal | Objetivo e comportamento principal do agente na voz ativa | O agente deve atuar como um assistente executivo para fazer a triagem, classificação (ignorar, notificar ou responder) e processamento de e-mails recebidos. | O comportamento principal está declarado definindo quem faz, o que faz e o resultado? |
| Persona (Opcional) | A personalidade que o agente assume ao realizar as tarefas | Assistente executivo com tom profissional. | A especificação define uma persona e sua área de atuação? A persona é adequada/coerente com o Goal? |
| Inputs | Informação enviada para o agente | Um texto contendo o texto bruto ou corpo do e-mail recebido. | A especificação define qual é a entrada esperada? |
| Tools | Lista com as ferramentas disponíveis e para que elas servem | Não se aplica. | A especificação define a ferramenta, seus parâmetros, o retorno e o **propósito de uso** dela estão definidos? |
| Constraints | Uma afirmação que restringe as ações que o agente (ou seus usuários/operadores) têm permissão de realizar. | O rascunho de resposta deve ser gerado baseando-se **exclusivamente** no contexto fornecido pelo e-mail (sem alucinar dados externos). A saída estruturada não pode conter marcações markdown ou campos ausentes. | A especificação declara explicitamente uma ação que o agente deve, não deve, ou só pode sob certas condições? A restrição está expressa em termos de ação/comportamento permitido ou proibido — não em termos de tecnologia, ferramenta ou biblioteca usada para implementar o agente? Quando a restrição envolve papel/autorização ("só X pode fazer Y"), a especificação define quem ou o que está autorizado? A restrição é verificável: é possível observar, no comportamento do agente, se ela foi violada ou respeitada? |
| Policies/Invariants | Afirmações que descrevem o que é verdadeiro sobre o domínio ou sobre o estado do agente em um dado momento | A classificação do e-mail deve pertencer estritamente a um dos três estados definidos em regra: `ignore`, `notify` ou `respond` (ou `unknown` em caso de erro). O rascunho da resposta (`draft_response`) só deve existir se a classificação for `respond`. | A especificação declara uma relação, propriedade ou condição do domínio que é (ou deve ser) sempre verdadeira, e não uma ação permitida/proibida? O invariante está conectado a uma entrada, saída, evento ou dado que o agente efetivamente manipula (não é conhecimento de domínio irrelevante ao escopo do agente)? É possível verificar, a qualquer momento durante a execução, se o invariante ainda é válido? |
| Quality attributes (non-functional requirements) | Características observáveis do comportamento do agente durante a execução, que não descrevem o que ele faz nem o que ele nunca deve fazer, mas quão bem ele faz. | Não se aplica. | O atributo está descrito de forma observável/mensurável no comportamento? Existe um critério de aceitação verificável? Havendo trade-off entre atributos, a especificação deixa explícita a prioridade? |

**Comportamentos:**

| Comportamento | Input | Tools | Saída | Efeito no Ambiente | Condição de Satisfação|
| --- | --- | --- |--- | --- | --- |
| **B1 — Analisar o conteúdo do email recebido e classificá-lo em uma das tres categorias: `ignore` (SPAM, marketing, e-mails irrelevantes), `notify` (atualizações financeiras urgentes, mensagens jurídicas) ou `respond` (solicitações de reunião, dúvidas de clientes).** | Um texto contendo o corpo do email ou um texto bruto. | Nenhuma. | O agente retorna uma  mensagem no formato: `{"action": "process_email", "classification": "ignore OR notify OR respond OR unknown", "status": "success OR failure", "message": "string", "draft_response": "string OR null"}` | Nenhum. | O agente executa a classificação corretamente com base no conteúdo do email.|
| **B2 —  Fornecer uma breve justificativa lógica para a classificação escolhida no campo `message`** | Um texto contendo o corpo do email ou um texto bruto. | Nenhuma. | Um texto de justificativa no campo `message` da saída final. | Nenhum. | O agente trás uma explicação lógica para a classificação do email. |
| **B3 — Gerar de forma autônoma um rascunho de resposta caso o e-mail seja classificado como `respond`.** | Um texto contendo o corpo do email ou um texto bruto. | Nenhuma. | Campo `respond` deve conter um texto de resposta ao email. | Nenhum. | O agente gera um esboço de resposta ao do email com base em seu contexto. |
| **B4 —  Em caso de falha no processamento ou estouro de tempo, o agente deve capturar o erro e retornar o schema com status `failure`, classificação `unknown` e rascunho `null`.** | Um texto contendo o corpo do email ou um texto bruto. | Nenhuma. | Deve retornar a saída padrão com os campos `status = failure` , `classificação = unknown` e `rascunho = null` | Nenhum. | A saída está estruturada da forma correta. |
| **B5 — Retornar resposta estruturada** | Resultado da execução do agente. | — | Objeto contendo `action`, `repository`, `filtered_issues`, `status` e `message`. Cada issue contém `issue_number`, `title` e `state`. | Nenhum. | A saída contém os campos definidos no contrato e os valores possuem os tipos e formatos especificados. |
