**Task:** você é um gerador de agentes de IA e abaixo está uma especificação de um agente. Gere um agente conforme a especificação. Considere que as ferramentas do agente devem ser geradas junto ao código do agente.  
**Contexto:** Um agente de IA é uma entidade que usa LLM e ferramentas para executar tarefas específicas de forma autônoma.  
**Output:** agent.py, requirements.txt, tests.py 

**Contrato de interface com o benchmark:**

| Requisito de interface | Exemplo |
| --- | --- |
| Modelo | GPT-4o |
| Framework | LangGraph |
| GenAI Tool | ChatOpenAI |
| Tools | Toda ferramenta deve ser uma ferramenta do LangChain (@tool, StructuredTool ou obtida de servidor MCP via langchain-mcp-adapters) e deve ser executada exclusivamente por sua interface (invoke/ainvoke, diretamente ou via ToolNode), nunca pela chamada direta da função que a implementa. |
| Entrypoint | agent(...) |
| Output | state: dict |

**Detalhes de implementação:**

| Detalhe de implementação | Especificação |
| --- | --- |
|  |  |


**Requisitos:**

| Requisito | Significado | Especificação do agente |
| --- | --- | --- |
| Goal | Objetivo e comportamento principal do agente na voz ativa |  |
| Persona (Opcional) | A personalidade que o agente assume ao realizar as tarefas |  |
| Inputs | Informação enviada para o agente |  |
| Constraints | Uma afirmação que restringe as ações que o agente (ou seus usuários/operadores) têm permissão de realizar. Descreve o que o agente deve, não deve, ou não pode fazer, ou define que apenas determinados papéis/condições autorizam uma certa ação. |  |
| Guardrails and Invariants | Regras externas ou condições que devem ser sempre respeitadas pelo agente durante sua execução. Podem vir de leis, regulamentações, políticas da empresa ou regras do domínio. Quando uma condição é um invariante, ela deve ser satisfeita antes e depois da execução de qualquer comportamento do agente. |  |
| Quality attributes (non-functional requirements) | Características observáveis do comportamento do agente durante a execução, que não descrevem o que ele faz (isso é Goal/Behavior Requirements) nem o que ele nunca deve fazer (Constraints), mas quão bem ele faz — de forma consistente, segura, eficiente e confiável. Moldam diretamente a confiança e a experiência de quem depende do agente. |  |

**Tools:**

| Nome | Propósito | Entradas | Retorno | Falhas |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

**Comportamentos:**

| Comportamento | Trigger | Input | Tools | Saída | Efeito no ambiente | Constraints |
| --- | --- | --- | --- | --- | --- | --- |
| **B1 — |  |  |  |  |  |  |

**Grafo comportamental:**

```mermaid
flowchart TD
```

| Campo | Critério de corretude |
| --- | --- |
| Comportamento | Cada comportamento está especificado como um requisito individual e identificável, separado do Goal? Cada requisito é atômico o suficiente para ser testado isoladamente (dá pra escrever um caso de teste "dado X, o agente deve fazer Y")? A especificação cobre explicitamente o comportamento esperado diante de: entrada inválida, falha de ferramenta/tool call, ambiguidade, ausência de informação necessária? Os requisitos, somados, são suficientes para realizar o Goal declarado (nada essencial ficou implícito)? |
| Trigger | A condição ou evento que inicia o comportamento está explicitamente definido e é observável? |
| Input | A especificação define qual é a entrada esperada? |
| Tools | As ferramentas necessárias para o comportamento estão identificadas e seu propósito de uso está definido? |
| Expected Outcome | A especificação define qual alteração deve ocorrer no ambiente? A alteração esperada é observável e verificável após a execução do comportamento? |
| Efeito no ambiente | A especificação define qual ambiente o agente vai alterar?  A especificação define como será feita a alteração no ambiente? |
| Constraints | As restrições do comportamento estão explícitas, são verificáveis e coerentes com as restrições gerais? |
