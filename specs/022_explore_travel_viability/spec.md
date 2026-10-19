**Task:** You are an AI agent generator. Below is a design for an agent; please generate the agent accordingly. Note that the agent's tools should be generated alongside the agent's code.

**Contexto:** An AI agent is an entity that utilizes an LLM and tools to perform specific tasks autonomously.

**Output:** agent.py, requirements.txt, tests.py

**Contrato de interface com o benchmark:**

| Requisito de interface | Exemplo |
| --- | --- |
| Modelo | GPT-4o |
| Framework | LangGraph |
| GenAI Tool | ChatOpenAI |
| Tools | Every tool must be a LangChain tool (@tool, StructuredTool, or obtained from an MCP server via langchain-mcp-adapters) and must be executed exclusively through its interface (invoke/ainvoke, directly or via ToolNode), never by calling its implementation function directly. |
| Entrypoint | agent(user_request: str) -> dict |
| Output | {"status": "success \\\| failure", "response": "string"} |

**Requisitos:**

| Requisito | Significado | Especificação do agente | Critério de Corretude |
| --- | --- | --- | --- |
| Goal | Objetivo e comportamento principal do agente na voz ativa | Assess the viability of a travel destination by combining real-time weather data and recent news. | O comportamento principal está declarado definindo quem faz, o que faz e o resultado? |
| Persona (Opcional) | A personalidade que o agente assume ao realizar as tarefas | Act as a travel destination assistant that helps users evaluate a destination before traveling. | A especificação define uma persona e sua área de atuação?<br>A persona é adequada/coerente com o Goal? |
| Inputs | Informação enviada para o agente | Receive a natural-language request through user_request: str and identify the target destination. | A especificação define qual é a entrada esperada? |
| Guardrails and Invariants | Regras externas ou condições que devem ser sempre respeitadas pelo agente durante sua execução. Podem vir de leis, regulamentações, políticas da empresa ou regras do domínio. Quando uma condição é um invariante, ela deve ser satisfeita antes e depois da execução de qualquer comportamento do agente. | - Do not fabricate weather data or news.<br><br>- Return failure for empty requests, timeouts, or unhandled exceptions.<br><br>- Always comply with the required output schema. | A especificação declara uma relação, propriedade ou condição do domínio que é (ou deve ser) sempre verdadeira, e não uma ação permitida/proibida?<br>O invariante está conectado a uma entrada, saída, evento ou dado que o agente efetivamente manipula (não é conhecimento de domínio irrelevante ao escopo do agente)?<br>É possível verificar, a qualquer momento durante a execução, se o invariante ainda é válido? |
| Quality attributes (non-functional requirements) | Características observáveis do comportamento do agente durante a execução, que não descrevem o que ele faz (isso é Goal/Behavior Requirements) nem o que ele nunca deve fazer (Constraints), mas quão bem ele faz — de forma consistente, segura, eficiente e confiável. Moldam diretamente a confiança e a experiência de quem depende do agente. | Not applicable | O atributo está descrito de forma observável/mensurável no comportamento (não é uma característica de implementação, ex: "usar cache" não é quality attribute, é constraint de implementação)?<br>Existe um critério de aceitação verificável (limiar, taxa, ou condição de teste), e não apenas um adjetivo vago ("deve ser rápido" sem definir o quê é rápido)?<br>Havendo trade-off entre atributos (ex: robustez vs. performance), a especificação deixa explícita a prioridade? |

**Tools:**

| Nome | Propósito | Entradas | Retorno | Falhas |
| --- | --- | --- | --- | --- |
| get_weather | Retrieve real-time weather conditions for a destination. | City name | Current weather data from wttr.in. | API request failure, timeout, or invalid city. |
| search_destination_news | Search for recent news, safety advisories, and current events about a destination. | Destination and search query | Recent results from the Tavily Search API. | Missing or invalid API key, API request failure, or timeout. |

**Comportamentos:**

| Comportamento | Trigger | Input | Tools | Saída | Efeito no ambiente | Constraints |
| --- | --- | --- | --- | --- | --- | --- |
| B1 — Identify Destination | A non-empty request is received. | Natural-language request. | None | Target destination identified. | None | Identify the destination at city-level granularity when possible. |
| B2 — Retrieve Weather | The destination is identified. | Target city. | `get_weather` | Real-time weather data. | External request to wttr.in. | Must use real data; no fabricated results. |
| B3 — Search News | The destination is identified. | Destination and relevant search query. | `search_destination_news` | Recent news and safety-related information. | External request to Tavily. | Must use real sources; requires TAVILY_API_KEY. |
| B4 — Assess Travel Viability | Both tool results are available. | Weather data and news results. | None | Combined assessment of destination viability. | None | Must consider both sources and provide concrete evidence. |
| B5 — Handle Empty Request | The request is empty or contains only whitespace. | Empty or blank user_request. | None | Response with status: "failure". | None | Must not proceed with tool execution. |
| B6 — Handle Errors and Timeouts | An exception occurs or execution exceeds the time limit. | Error or timeout. | None | Response with status: "failure". | An external request may be interrupted. | Maximum execution time of 30 seconds; no propagated exceptions. |
| B7 — Return Structured Response | Processing finishes or fails. | Assessment or error. | None | Object containing status and response. | Returns the result to the caller. | Must match the exact output schema. |

| Campo | Critério de corretude |
| --- | --- |
| Comportamento | Cada comportamento está especificado como um requisito individual e identificável, separado do Goal? Cada requisito é atômico o suficiente para ser testado isoladamente (dá pra escrever um caso de teste "dado X, o agente deve fazer Y")? A especificação cobre explicitamente o comportamento esperado diante de: entrada inválida, falha de ferramenta/tool call, ambiguidade, ausência de informação necessária? Os requisitos, somados, são suficientes para realizar o Goal declarado (nada essencial ficou implícito)? |
| Trigger | A condição ou evento que inicia o comportamento está explicitamente definido e é observável? |
| Input | A especificação define qual é a entrada esperada? |
| Tools | As ferramentas necessárias para o comportamento estão identificadas e seu propósito de uso está definido? |
| Expected Outcome | A especificação define qual alteração deve ocorrer no ambiente? A alteração esperada é observável e verificável após a execução do comportamento? |
| Efeito no ambiente | A especificação define qual ambiente o agente vai alterar?  A especificação define como será feita a alteração no ambiente? |
| Constraints | As restrições deste comportamento declaram o que o agente deve, não deve ou só pode fazer sob determinadas condições? O responsável e as condições estão claros? As restrições são verificáveis e coerentes com os demais campos deste comportamento e com o Goal? |

**Grafo comportamental:**

```mermaid
flowchart TD
    A([User Request]) --> B5{"B5: Empty Request?"}
    B5 -->|Yes| B7["B7: Return Structured Response"]
    B5 -->|No| B1["B1: Identify Destination"]

    B1 --> B2["B2: Retrieve Weather"]
    B2 --> B3["B3: Search News"]
    B3 --> B4["B4: Assess Travel Viability"]
    B4 --> B7

    B2 -.-> B6["B6: Handle Errors and Timeouts"]
    B3 -.-> B6
    B4 -.-> B6
    B6 --> B7
```

**Detalhes de implementação:**

| Item | Instruções |
| --- | --- |
| Weather URL | https://wttr.in/{city}?format=j1 |
| Environment variable | TAVILY_API_KEY |
| Import |  |
| Imports | from dotenv import load_dotenv; load_dotenv() |
