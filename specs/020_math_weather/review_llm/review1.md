| Campo | Nota | Justificativa |
| --- | --- | --- |
| **Goal** | 5 | A especificação define claramente quem atua (o agente), o que faz (interpreta requisições e usa ferramentas) e qual é o resultado (responde com o cálculo ou a informação meteorológica), utilizando a voz ativa. |
| **Persona** | 5 | Define uma persona ("assistente") e sua área de atuação ("aritmética básica e consultas simples de clima"), sendo totalmente apropriada e consistente com o objetivo declarado. |
| **Inputs (Geral)** | 5 | Especifica claramente a entrada esperada: requisições em linguagem natural contendo dois números identificáveis para matemática ou um local identificável para clima. |
| **Guardrails and Invariants** | 5 | Estabelece propriedades e relações verificáveis que devem ser sempre verdadeiras (ex.: a resposta deve corresponder ao retorno da ferramenta; locais sem dados não invalidam a requisição, mas geram resposta de indisponibilidade). Estão diretamente conectados aos dados manipulados. |
| **Quality attributes** | N/A | Atributo preenchido como "Not applicable" na especificação. |
| **Behavior** | 5 | Os comportamentos (B1 a B6) são atômicos, testáveis isoladamente e cobrem fluxos de sucesso, dados indisponíveis, falhas de ferramentas e requisições inválidas ou não suportadas. |
| **Trigger** | 5 | As condições de acionamento para cada comportamento são explícitas, mutuamente exclusivas e observáveis (ex.: "A entrada é uma requisição válida de adição...", "get_weather não retorna dados..."). |
| **Input (Comportamentos)** | 5 | Define de forma clara e granular o que cada comportamento recebe, diferenciando a requisição do usuário, os argumentos identificados e os retornos das ferramentas. |
| **Tools** | 5 | As ferramentas (`add`, `multiply`, `get_weather` ou nenhuma) estão corretamente identificadas e mapeadas para o uso pretendido em cada comportamento. |
| **Expected Outcome / Output** | 5 | O resultado esperado é definido de maneira clara e observável para cada comportamento (ex.: texto contendo a soma, texto informando dados indisponíveis). |
| **Environment effect** | 5 | Define explicitamente que os comportamentos não alteram o ambiente (operações apenas de leitura e cálculo numérico), o que é correto e verificável. |
| **Constraints** | 5 | As restrições estabelecem claramente o que o agente deve ou não deve fazer em cada contexto (ex.: "não deve calcular ou inventar o resultado de forma independente"), sendo verificáveis e consistentes com o objetivo. |
