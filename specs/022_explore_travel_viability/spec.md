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
| Entrypoint | agent(user_request: str) -> dict |
| Output | {"status": "success \\\| failure", "response": "string"} |

**Requirements:**

| Requirement | Meaning | Agent specification | Correctness criterion |
| --- | --- | --- | --- |
| Goal | The agent's objective and main behavior, stated in the active voice | Assess the viability of a travel destination by combining real-time weather data and recent news. | Does the main behavior state who acts, what they do, and the result? |
| Persona (Optional) | The personality the agent assumes when performing its tasks | Act as a travel destination assistant that helps users evaluate a destination before traveling. | Does the specification define a persona and its area of activity? Is the persona appropriate for and consistent with the Goal? |
| Inputs | Information sent to the agent | Receive a natural-language request through user_request: str and identify the target destination. | Does the specification define the expected input? |
| Guardrails and Invariants | External rules or conditions the agent must always respect during execution. They may come from laws, regulations, company policies, or domain rules. An invariant must hold before and after any agent behavior. | No information regarding the destination's climate or safety may be derived from the model's internal knowledge; any statement about the destination in the final response must be traceable to—and strictly limited to—the results returned by the `get_weather` and `search_destination_news` tools. | Does the specification state a relationship, property, or condition of the domain that is (or should be) always true, rather than a permitted/prohibited action? Is the invariant connected to an input, output, event, or piece of data the agent actually handles, rather than domain knowledge irrelevant to its scope? Can the invariant be checked at any point during execution? |
| Quality attributes (non-functional requirements) | Observable characteristics of the agent's behavior during execution that describe neither what it does (Goal/Behavior Requirements) nor what it must never do (Constraints), but how well it does it: consistently, safely, efficiently, and reliably. They directly shape the trust and experience of those who depend on the agent. | Not applicable | Is the attribute described in observable/measurable behavioral terms, rather than as an implementation characteristic (for example, "use a cache" is an implementation constraint, not a quality attribute)? Is there a verifiable acceptance criterion (a threshold, rate, or test condition), rather than just a vague adjective (such as "must be fast" without defining how fast)? Where attributes involve a trade-off (for example, robustness versus performance), does the specification make the priority explicit? |

**Tools:**

| Name | Purpose | Inputs | Return | Failures |
| --- | --- | --- | --- | --- |
| get_weather | Retrieve real-time weather conditions for a destination. | City name | Current weather data from wttr.in. | API request failure, timeout, or invalid city. |
| search_destination_news | Search for recent news, safety advisories, and current events about a destination. | Destination and search query | Recent results from the Tavily Search API. | Missing or invalid API key, API request failure, or timeout. |

**Behaviors:**

| Behavior | Trigger | Input | Tools | Output | Environment effect | Constraints |
| --- | --- | --- | --- | --- | --- | --- |
| B1 — Identify Destination | A non-empty request is received. | Natural-language request. | None | Target destination identified. | None | Identify the destination at city-level granularity when possible. |
| B2 — Retrieve Weather | The destination is identified. | Target city. | `get_weather` | Real-time weather data. | External request to wttr.in. | Must use real data; no fabricated results. |
| B3 — Search News | The destination is identified. | Destination and relevant search query. | `search_destination_news` | Recent news and safety-related information. | External request to Tavily. | Must use real sources; requires TAVILY_API_KEY. |
| B4 — Assess Travel Viability | Both tool results are available. | Weather data and news results. | None | Combined assessment of destination viability. | None | Must consider both sources and provide concrete evidence. |
| B5 — Handle Empty Request | The request is empty or contains only whitespace. | Empty or blank user_request. | None | Response with status: "failure". | None | Must not proceed with tool execution. |
| B6 — Handle Errors and Timeouts | An exception occurs or execution exceeds the time limit. | Error or timeout. | None | Response with status: "failure". | An external request may be interrupted. | Maximum execution time of 30 seconds; no propagated exceptions. |
| B7 — Return Structured Response | Processing finishes or fails. | Assessment or error. | None | Object containing status and response. | Returns the result to the caller. | Must match the exact output schema. |

| Field | Correctness criterion |
| --- | --- |
| Behavior | Is each behavior specified as an individual, identifiable requirement, separate from the Goal? Is each requirement atomic enough to test in isolation (given X, the agent must do Y)? Does the specification explicitly cover invalid input, tool call failure, ambiguity, and missing required information? Together, are the requirements sufficient to achieve the declared Goal, with no essential behavior left implicit? |
| Trigger | Is the condition or event that starts the behavior explicitly defined and observable? |
| Input | Does the specification define the expected input? |
| Tools | Are the required tools identified and their intended use defined? |
| Expected Outcome | Does the specification define the expected change in the environment? Is that change observable and verifiable after the behavior runs? |
| Environment effect | Does the specification define which environment the agent changes and how that change is made? |
| Constraints | Do this behavior’s constraints state what the agent must, must not, or may only do under certain conditions? Are the responsible party and conditions clear? Are the constraints verifiable and consistent with this behavior’s other fields and the Goal? |

**Behavioral graph:**

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

**Implementation details:**

| Item | Instruções |
| --- | --- |
| Weather URL | https://wttr.in/{city}?format=j1 |
| Environment variable | TAVILY_API_KEY |
| Import |  |
| Imports | from dotenv import load_dotenv; load_dotenv() |
