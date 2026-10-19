**Task:** you are an AI agent generator, and below is the specification of an agent. Generate an agent according to the specification. Consider that the agent's tools must be generated together with the agent code.

**Context:** An AI agent is an entity that uses an LLM and tools to perform specific tasks autonomously.

**Output:** agent.py, requirements.txt, tests.py

**Interface contract with the benchmark:**

| Interface requirement | Example |
| --- | --- |
| Model | GPT-4o |
| Framework | LangGraph |
| GenAI Tool | ChatOpenAI |
| Tools | Every tool must be a LangChain tool (@tool, StructuredTool, or obtained from an MCP server via langchain-mcp-adapters) and must be executed exclusively through its interface (invoke/ainvoke, either directly or via ToolNode), never by directly calling the function that implements it. |
| Entrypoint | agent(query: str) |
| Output | response: string |

**Implementation details:**

| Implementation detail | Specification |
| --- | --- |
| Environment variables | The agent must access the environment variables when necessary.<br>Available keys: NOTION\_PAGE\_ID, TAVILY\_API\_KEY, NOTION\_API\_KEY, OPENAI\_API\_KEY |

**Requirements:**

| Requirement | Meaning | Agent specification | Correctness criterion |
| --- | --- | --- | --- |
| Goal | The agent's main objective and behavior, in the active voice | The agent acts as a generator of personalized recipes, adapted to use the ingredients provided in the user's query or in their pantry and to respect the user's dietary preferences and restrictions. It returns the complete recipe with ingredients, equipment, instructions, and storage recommendations. | Is the main behavior stated, defining who does what and the result? |
| Persona (Optional) | The personality the agent assumes when performing tasks | The agent assumes the persona of a **chef**, whose role is to create personalized recipes. | Does the specification define a persona and its area of activity?<br>Is the persona appropriate/consistent with the Goal? |
| Inputs | Information sent to the agent | The agent receives a _query (in natural language)_ from the user, containing preferences and, optionally, specific ingredients and dietary restrictions. | Does the specification define the expected input? |
| Guardrails and Invariants | External rules or conditions that must always be respected by the agent during its execution. They may come from laws, regulations, company policies, or domain rules. When a condition is an invariant, it must be satisfied before and after the execution of any agent behavior. | • A dietary restriction implies all of its derived ingredients, even those not mentioned by the user.<br>• An ingredient mentioned under negation, absence, or hypothesis is not an available ingredient. | Does the specification state a relationship, property, or domain condition that is (or must be) always true, rather than a permitted/forbidden action?<br>Is the invariant connected to an input, output, event, or datum that the agent actually handles (not domain knowledge irrelevant to the agent's scope)?<br>Is it possible to verify, at any moment during execution, whether the invariant still holds? |
| Quality attributes (non-functional requirements) | Observable characteristics of the agent's behavior during execution that describe neither what it does (that is Goal/Behavior Requirements) nor what it must never do (Constraints), but how well it does it — consistently, safely, efficiently, and reliably. They directly shape the trust and experience of those who depend on the agent. | Not applicable. | Is the attribute described in an observable/measurable way in the behavior (not an implementation characteristic, e.g., "use cache" is not a quality attribute, it is an implementation constraint)?<br>Is there a verifiable acceptance criterion (threshold, rate, or test condition), and not just a vague adjective ("must be fast" without defining what is fast)?<br>If there is a trade-off between attributes (e.g., robustness vs. performance), does the specification make the priority explicit? |

**Tools:**

| Name | Purpose | Inputs | Return | Failures |
| --- | --- | --- | --- | --- |
| retrieve\_pantry | Obtain the ingredients available in the user's pantry, through a Notion page. | None. | The complete list of pantry ingredients. An empty pantry returns an empty list. | Missing credentials, an inaccessible page, or a timeout must return a message beginning with `Error:` followed by a message explaining the failure. |
| search\_recipes | Obtain recipes from the web, through **tavily**, as a preparation **reference** for the usable ingredients, derived from the pantry and the query. The recipes found are neither a source of base ingredients nor the final recipe. | identified ingredients compatible with the dietary restrictions, derived from the output of `retrieve_pantry` and from the query. | Up to 5 results, each with title, URL, and content excerpt. No results returns an empty list. | Missing credentials, an inaccessible page, or a timeout must return a message beginning with `Error:` followed by a message explaining the failure. |

**Behaviors:**

| Behavior | Trigger | Input | Tools | Output | Environment effect | Constraints |
| --- | --- | --- | --- | --- | --- | --- |
| **B1 — Query validation**<br>The agent classifies the query as invalid (empty, offensive, or unrelated to food/cooking) or valid (any query that meets none of the invalidity criteria), and stops the execution if it is invalid. | Receipt of the query. | User query. | None. | Boolean validity indicator. If invalid, a message beginning with `Error:` with the explanation. | None. | The indicator matches the expected classification. If invalid, the message begins with `Error:`, no tool is called, and no recipe is generated. |
| **B2 — Pantry lookup**<br>The agent retrieves the ingredients available in the user's pantry. | Query classified as valid in B1. | None. | `retrieve_pantry` | List of pantry ingredients, possibly empty. | None. | `retrieve_pantry` is the first tool called and is called exactly once. |
| **B3 — Ingredient identification**<br>The agent identifies the ingredients usable in the recipe, which are those mentioned in the query and all those returned by the pantry, and retains only those compatible with the declared dietary restrictions. | B2 completed. | User query and the return of B2. | None. | List of identified ingredients compatible with the dietary restrictions; if no ingredient is identified or compatible, an empty list. | None. | The list equals the expected set (query and pantry ingredients compatible with the dietary restrictions), taking into account regional variants and inflected forms of the ingredients, excluding ingredients mentioned under negation, absence, or hypothesis and excluding ingredients incompatible with the declared dietary restrictions. |
| **B4 — Reference recipe search**<br>The agent searches for recipes that serve as a preparation reference for the ingredients identified in B3. | List of ingredients from B3 is not empty. | List of ingredients from B3. | `search_recipes` | Up to five results, each with title, URL, and content excerpt, possibly none. | None. | `search_recipes` is called exactly once, only after `retrieve_pantry` has completed, with the list of ingredients from B3. |
| **B5 — Lack of ingredients**<br>The agent stops the execution when there is no ingredient compatible with the dietary restrictions, either in the query or in the pantry. | List of ingredients from B3 is empty. | List from B3. | None. | Message beginning with `Error:` explaining the absence of compatible ingredients. | None. | The message begins with `Error:`, `search_recipes` is not called, and no recipe is generated. |
| **B6 — Execution failure**<br>The agent handles tool or model failures without terminating with an unhandled exception. | A tool returns a message beginning with `erro:`, or an exception occurs during execution (timeout, missing credentials, or model failure). | Failure message or exception received. | None. | Message beginning with `Error:` explaining the failure. | None. | The agent does not terminate with an unhandled exception, the message begins with `Error:`, and no recipe is generated. |
| **B7 — Recipe generation**<br>The agent uses its culinary knowledge to turn the compatible ingredients into a complete recipe, taking the user's preferences into account and using the reference recipes, when available. | B4 completed without error (with or without results). | User query, ingredients from B3, and results from B4. | None. | Recipe with the fields:<br>• **Title:** recipe title;<br>• **Ingredients:** list of ingredients with their quantities;<br>• **Equipment:** list of required equipment;<br>• **Instructions:** step-by-step preparation instructions;<br>• **Storage:** storage recommendations. | None. | All fields are present and non-empty; no ingredient in the recipe violates the dietary restrictions declared in the query; the recipe contains at least one ingredient identified in B3. |

**Behavioral graph:**

```mermaid
flowchart TD
    Q(["User query"]) --> B1{"B1: valid query?"}
    B1 -->|invalid| E1["Error: invalid query"]
    B1 -->|valid| B2["B2: retrieve_pantry"]
    B2 -->|success| B3["B3: identify compatible ingredients"]
    B2 -->|error| B6["B6: Error: execution failure"]
    B3 -->|non-empty list| B4["B4: search_recipes"]
    B3 -->|empty list| B5["B5: Error: lack of ingredients"]
    B4 -->|"with or without results"| B7["B7: generate recipe"]
    B4 -->|error| B6
    B7 --> R(["Recipe"])
```

| Field | Correctness criterion |
| --- | --- |
| Behavior | Is each behavior specified as an individual, identifiable requirement, separate from the Goal? Is each requirement atomic enough to be tested in isolation (can you write a test case "given X, the agent must do Y")? Does the specification explicitly cover the expected behavior in the face of: invalid input, tool/tool call failure, ambiguity, absence of necessary information? Are the requirements, taken together, sufficient to achieve the stated Goal (nothing essential left implicit)? |
| Trigger | Is the condition or event that starts the behavior explicitly defined and observable? |
| Input | Does the specification define the expected input? |
| Tools | Are the tools needed for the behavior identified, and is their purpose of use defined? |
| Expected Outcome | Does the specification define what change must occur in the environment? Is the expected change observable and verifiable after the behavior is executed? |
| Environment effect | Does the specification define which environment the agent will change? Does the specification define how the change to the environment will be made? |
| Constraints | Are the behavior's constraints explicit, verifiable, and consistent with the general constraints? |
