**Task:** You are an AI agent generator. Below is a design for an agent; please generate the agent accordingly. Note that the agent's tools should be generated alongside the agent's code.

**Context:** An AI agent is an entity that utilizes an LLM and tools to perform specific tasks autonomously.

**Output:** agent.py, requirements.txt, tests.py

**Benchmark interface contract:**

| Interface requirement | Example |
| --- | --- |
| Model | GPT-4o |
| Framework | LangGraph |
| GenAI Tool | ChatOpenAI |
| Tools | Every tool must be a LangChain tool (@tool, StructuredTool, or obtained from an MCP server via langchain-mcp-adapters) and must be executed exclusively through its interface (invoke/ainvoke, directly or via ToolNode), never by calling its implementation function directly. |
| Entrypoint | agent(user_request: str) |
| Output | { "status": "success \| failure", "user_request": "Original user request", "response": "Natural-language response" } |

**Requirements:**

| Requirement | Meaning | Agent specification | Correctness criterion |
| --- | --- | --- | --- |
| Goal | The agent's objective and main behavior, stated in the active voice | Answer natural-language questions about travel destinations by selecting the appropriate action based on the user's intent. | Does the main behavior state who acts, what they do, and the result? |
| Persona (Optional) | The personality the agent assumes when performing its tasks | Act as a travel destination assistant that helps users explore destinations before deciding to travel. | Does the specification define a persona and its area of activity? Is the persona appropriate for and consistent with the Goal? |
| Inputs | Information sent to the agent | Receive a natural-language question about a travel destination through user_request: str. | Does the specification define the expected input? |
| Guardrails and Invariants | External rules or conditions the agent must always respect during execution. They may come from laws, regulations, company policies, or domain rules. An invariant must hold before and after any agent behavior. | - Return failure for empty or blank requests, timeouts, and unhandled exceptions.<br><br>- Do not fabricate weather data or news results.<br><br>- Always return the required output schema. | Does the specification state a relationship, property, or condition of the domain that is (or should be) always true, rather than a permitted/prohibited action? Is the invariant connected to an input, output, event, or piece of data the agent actually handles, rather than domain knowledge irrelevant to its scope? Can the invariant be checked at any point during execution? |
| Quality attributes (non-functional requirements) | Observable characteristics of the agent's behavior during execution that describe neither what it does (Goal/Behavior Requirements) nor what it must never do (Constraints), but how well it does it: consistently, safely, efficiently, and reliably. They directly shape the trust and experience of those who depend on the agent. | Not applicable | Is the attribute described in observable/measurable behavioral terms, rather than as an implementation characteristic (for example, "use a cache" is an implementation constraint, not a quality attribute)? Is there a verifiable acceptance criterion (a threshold, rate, or test condition), rather than just a vague adjective (such as "must be fast" without defining how fast)? Where attributes involve a trade-off (for example, robustness versus performance), does the specification make the priority explicit? |

**Tools:**

| Name | Purpose | Inputs | Return | Failures |
| --- | --- | --- | --- | --- |
| get_weather | Retrieve real-time weather conditions for a destination. | City name | Current weather data from the wttr.in API. | API request failure, timeout, or invalid city. |
| search_destination_news | Search for recent destination-related news, safety advisories, and current events. | Destination and search query | Recent search results from the Tavily Search API. | Missing or invalid API key, API request failure, or timeout. |

**Behaviors:**

| Behavior | Trigger | Input | Tools | Output | Environment effect | Constraints |
| --- | --- | --- | --- | --- | --- | --- |
| B1 — Interpret Request | A non-empty request is received. | Natural-language question about a travel destination. | None | Identified intent: weather, news/safety, or travel tips. | None | Select the appropriate action based on the user's intent. |
| B2 — Retrieve Weather | The user asks about current weather, temperature, or weather conditions. | User request and identified city. | `get_weather` | Answer based on real weather data. | External request to the wttr.in API. | Invoke at most one tool per execution. |
| B3 — Search News and Advisories | The user asks about news, safety, travel advisories, or current events. | User request and identified destination. | `search_destination_news` | Answer based on real search results. | External request to the Tavily Search API. | Invoke at most one tool per execution; do not fabricate results. |
| B4 — Answer Directly | The user asks about itineraries, activities, local culture, food, or general travel tips. | Natural-language request. | None | Natural-language answer. | None | Do not invoke tools for these requests. |
| B5 — Handle Empty Request | The input is empty or contains only whitespace. | Empty or blank user_request. | None | Response with status: "failure". | None | Preserve the original input in user_request. |
| B6 — Handle Errors and Timeouts | An exception occurs or the execution time limit is exceeded. | Error during processing. | None | Response with status: "failure". | An external request may be interrupted. | Enforce the 30-second execution limit; do not propagate exceptions. |
| B7 — Return Structured Response | An answer is produced or an error occurs. | Original input, result, or error. | None | Object containing status, user_request, and response. | Returns the result to the caller. | Strictly comply with the specified output schema. |

| Field | Correctness criterion |
| --- | --- |
| Behavior | Is each behavior specified as an individual, identifiable requirement, separate from the Goal? Is each requirement atomic enough to test in isolation (given X, the agent must do Y)? Does the specification explicitly cover invalid input, tool call failure, ambiguity, and missing required information? Together, are the requirements sufficient to achieve the declared Goal, with no essential behavior left implicit? |
| Trigger | Is the condition or event that starts the behavior explicitly defined and observable? |
| Input | Does the specification define the expected input? |
| Tools | Are the required tools identified and their intended use defined? |
| Expected Outcome | Does the specification define the expected change in the environment? Is that change observable and verifiable after the behavior runs? |
| Environment effect | Does the specification define which environment the agent changes and how that change is made? |
| Constraints | As restrições do comportamento estão explícitas, são verificáveis e coerentes com as restrições gerais? |

**Behavioral graph:**

```mermaid
flowchart TD
    A([User Request]) --> B5{"B5: Empty Request?"}
    B5 -->|Yes| B7["B7: Return Structured Response"]
    B5 -->|No| B1["B1: Interpret Request"]

    B1 --> D{"Identify Intent"}
    D -->|Weather| B2["B2: Retrieve Weather"]
    D -->|News or Safety| B3["B3: Search News"]
    D -->|Travel Tips| B4["B4: Answer Directly"]

    B2 --> B7
    B3 --> B7
    B4 --> B7

    B7 --> F([Final Response])
```

**Implementation details:**

| Item | Instruções |
| --- | --- |
| Weather API | Must access https://wttr.in/{city}?format=j1 to retrieve real-time weather data |
| News search | Must use the Tavily Search API Client (TavilyClient) to search for recent news, safety advisories, and current events. |
| Environment variable | Must read TAVILY_API_KEY to authenticate Tavily requests. |
| Imports | from dotenv import load_dotenv; load_dotenv() |
