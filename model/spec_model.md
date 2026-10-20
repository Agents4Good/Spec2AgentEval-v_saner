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
| Entrypoint | agent(...) |
| Output | state: dict |

**Implementation details**

| Environment variables | **…** |
| --------------------- | ----- |
|                       |       |


**Requirements**

| Requirement | Meaning | Correctness criterion |
| --- | --- | --- |
| Goal | The agent's main objective and behavior, in the active voice | Is the main behavior stated, defining who does what and the result? |
| Persona (Optional) | The personality the agent assumes when performing tasks | Does the specification define a persona and its area of activity?<br>Is the persona appropriate/consistent with the Goal? |
| Inputs | Information sent to the agent | Does the specification define the expected input? |
| Tools | List of the available tools and what they are for | Does the specification define the tool, and are its parameters, return value, and purpose of use defined? |
| Guardrails and Invariants | External rules or conditions that must always be respected by the agent during its execution. They may come from laws, regulations, company policies, or domain rules. When a condition is an invariant, it must be satisfied before and after the execution of any agent behavior. | Does the specification state a relationship, property, or domain condition that is (or must be) always true, rather than a permitted/forbidden action?<br>Is the invariant connected to an input, output, event, or datum that the agent actually handles (not domain knowledge irrelevant to the agent's scope)?<br>Is it possible to verify, at any moment during execution, whether the invariant still holds? |
| Quality attributes (non-functional requirements) | Observable characteristics of the agent's behavior during execution that describe neither what it does (that is Goal/Behavior Requirements) nor what it must never do (Constraints), but how well it does it — consistently, safely, efficiently, and reliably. They directly shape the trust and experience of those who depend on the agent. | Is the attribute described in an observable/measurable way in the behavior (not an implementation characteristic, e.g., "use cache" is not a quality attribute, it is an implementation constraint)?<br>Is there a verifiable acceptance criterion (threshold, rate, or test condition), and not just a vague adjective ("must be fast" without defining what is fast)?<br>If there is a trade-off between attributes (e.g., robustness vs. performance), does the specification make the priority explicit? |

Behaviors

| Behavioral Requirement | Trigger | Input | Tools | Expected Outcome | Environment Effect | Constraints |
| ------ | ------- | ----- | ----- | ---------------- | ------------------ | ----------- |
| Specific, testable capability that the agent must implement to achieve the declared Goal. Each requirement is atomic, uniquely identifiable, and covers not only the happy path, but also how the agent must behave in the face of invalid inputs, tool failures, ambiguity in the request, or expected error conditions | The condition or event that starts the execution of this behavior | Information sent to the agent for the execution of this behavior | List of the tools needed to execute this behavior | Agent response or artifact produced, whether final or intermediate, in format X | Behavior that changes the environment, e.g., creation of a file or an entry in a database | A statement that restricts the actions that the agent (or its users/operators) is allowed to perform in this behavior. It describes what the agent must, must not, or cannot do, or defines that only certain roles/conditions authorize a given action |

| Field | Correctness criterion |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| Behavior | Is each behavior specified as an individual, identifiable requirement, separate from the Goal? Is each requirement atomic enough to be tested in isolation (can you write a test case "given X, the agent must do Y")? Does the specification explicitly cover the expected behavior in the face of: invalid input, tool/tool call failure, ambiguity, absence of necessary information? Are the requirements, taken together, sufficient to achieve the stated Goal (nothing essential left implicit)? |
| Trigger | Is the condition or event that starts the behavior explicitly defined and observable? |
| Input | Does the specification define the expected input? |
| Tools | Are the tools needed for the behavior identified, and is their purpose of use defined? |
| Expected Outcome | Does the specification define what change must occur in the environment? Is the expected change observable and verifiable after the behavior is executed? |
| Environment Effect | Does the specification define which environment the agent will change? Does the specification define how the change to the environment will be made? |
| Constraints | Do this behavior’s constraints state what the agent must, must not, or may only do under certain conditions? Are the responsible party and conditions clear? Are the constraints verifiable and consistent with this behavior’s other fields and the Goal? |

**Expected behavioral graph:**

```mermaid
flowchart TD
```
