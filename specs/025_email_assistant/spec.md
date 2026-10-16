**\*\*Task:\*\*** You are an AI agent generator, and below is an agent specification. Generate an agent according to the specification. Consider that the agent's tools must be generated together with the agent's code. &#x20;

**\*\*Context:\*\*** An AI agent is an entity that uses an LLM and tools to perform specific tasks autonomously. &#x20;

**\*\*Output:\*\*** agent.py, requirements.txt, tests.py

**\*\*Benchmark Interface Contract:\*\***

| Interface Requirement | Example |
| --- | --- |
| Model | GPT-4o |
| Framework | LangGraph |
| GenAI Tool | ChatOpenAI |
| Entrypoint | `agent(email_input: str)` |
| Output | `Tuple[CompiledGraph, dict]` |
| Tools | Every tool must be a LangChain tool (`@tool`, `StructuredTool`, or obtained from an MCP server via langchain-mcp-adapters) and must be executed exclusively through its interface (invoke/ainvoke, directly or via ToolNode), never by directly calling the function that implements it. |

**\*\*Implementation Details\*\***

| Environment Variable | Description |
| --- | --- |
| OPENAI_API_KEY | OpenAI API access key. |
| NOTION_TOKEN | Notion integration token used by the official `notion_client` SDK. |
| NOTION_DRAFTS_DB_ID | ID of the Drafts database in Notion. |

**\*\*General Requirements:\*\***

| Requirement | Meaning | Agent Specification | Correctness Criterion |
| --- | --- | --- | --- |
| Goal | Main objective and behavior of the agent in active voice | The agent must act as an executive assistant, triaging incoming emails, classifying each one as `ignore`, `notify`, or `respond`, autonomously executing the tool corresponding to the classification when the classification is `ignore` or `respond`, executing no tool when the classification is `notify`, and returning the processing result. | Is the main behavior defined, specifying who performs the task, what they do, and the result? |
| Persona (Optional) | The personality the agent assumes when performing tasks | Executive email assistant, with a professional, objective, and direct tone. | Does the specification define a persona and its area of expertise? Is the persona appropriate and consistent with the Goal? |
| Inputs | Information provided to the agent | Text containing the raw body of the received email. | Does the specification define the expected input? |
| Tools | List of available tools and what they are used for | (T1) `archive_email`: silently archives the email; returns a confirmation string; used when the classification is `ignore`. (T2) `save_email_draft`: saves a generated reply draft to the "Drafts" database in Notion; returns a confirmation string; used when the classification is `respond`. When the classification is `notify`, no tool is executed. | Are the tool, its parameters, its return value, and its **\*\*purpose of use\*\*** defined? |
| Constraints | A statement that restricts the actions the agent (or its users/operators) are permitted to perform. | (1) The agent must execute at most one tool per processed email. (2) When the classification is `notify`, it must not execute any tool. (3) The executed tool must correspond exactly to the classification assigned to the email (`ignore` → T1, `respond` → T2). (4) The agent must not execute a tool other than the one mapped to the chosen classification. | Does the specification explicitly state an action that the agent must, must not, or may only perform under certain conditions? Is the constraint expressed in terms of permitted or prohibited actions/behavior—not in terms of the technology, tool, or library used to implement the agent? When the constraint involves a role/authorization ("only X can do Y"), does the specification define who or what is authorized? Is the constraint verifiable: is it possible to observe, in the agent's behavior, whether it was violated or respected? |
| Policies/Invariants | Statements describing what is true about the domain or the agent's state at a given moment | (1) The `action` field is always `"process_email"`. (2) `classification` always belongs to `ignore`, `notify`, `respond`, or `unknown`. (3) `classification` is `unknown` if `status` is `failure`. (4) `draft_response` is a non-empty string if `classification` is `respond` and `status` is `success`; in any other case, it is `null`. (5) `actions_taken` contains exactly one element (the name of the executed tool) when `status` is `success` and `classification` is `ignore` or `respond`, and no elements when `status` is `failure` or when `classification` is `notify`. (6) The element in `actions_taken`, when present, corresponds to the tool mapped to `classification` according to Constraint (3). | Does the invariant declare a relationship, property, or condition of the domain that must always be true, rather than an allowed/prohibited action? Is the invariant connected to an input, output, event, or data that the agent actually manipulates (not domain knowledge irrelevant to the agent's scope)? Is it possible to verify at any point during execution whether the invariant still holds? |
| Quality Attributes (non-functional requirements) | Observable characteristics of the agent's behavior during execution that do not describe what it does or what it must never do, but how well it does it. | Conciseness of the `message` field: no more than 2 sentences, justifying the classification and the action taken. | Is the attribute described in an observable/measurable way? Is there a verifiable acceptance criterion? If there is a trade-off between attributes, does the specification explicitly state the priority? |

**\*\*Behaviors:\*\***

| Behavior | Trigger | Input | Tools | Output | Effect on the Environment |
| --- | --- | --- | --- | --- | --- |
| **\*\*B1 — Classify the email as `ignore` (SPAM, marketing, irrelevant cold emails) and execute `archive_email`.\*\*** | Receipt of a valid (non-empty) email whose content is identified as irrelevant. | Text containing the email body. | T1. | `{"action": "process_email", "classification": "ignore", "actions_taken": ["archive_email"], "status": "success", "message": "string (≤ 2 sentences)", "draft_response": null}` | The email is archived (real call or mock of `archive_email`). |
| **\*\*B2 — Classify the email as `notify` (urgent financial updates, critical HR messages) and execute no tool.\*\*** | Receipt of a valid email whose content is identified as urgent/critical but does not require a direct response. | Text containing the email body. | None. | `{"action": "process_email", "classification": "notify", "actions_taken": [], "status": "success", "message": "string (≤ 2 sentences)", "draft_response": null}` | None. |
| **\*\*B3 — Classify the email as `respond` (meeting requests, customer inquiries, introductions), generate a reply draft, and execute `save_email_draft`.\*\*** | Receipt of a valid email whose content requires a direct response. | Text containing the email body. | T2. | `{"action": "process_email", "classification": "respond", "actions_taken": ["save_email_draft"], "status": "success", "message": "string (≤ 2 sentences)", "draft_response": "string"}` | The generated draft is saved to the "Drafts" database in Notion (real call or mock of `save_email_draft`). |
| **\*\*B4 — In the event of a timeout or failure to parse the LLM output, return the standard failure schema.\*\*** | LLM API timeout or failure to parse the LLM output into a structured response. | Text containing the email body. | None. | `{"action": "process_email", "classification": "unknown", "actions_taken": [], "status": "failure", "message": "string", "draft_response": null}` | None. |
| **\*\*B5 — Handle invalid input by returning the standard failure schema.\*\*** | Empty input, input consisting only of whitespace, or input that is not text. | Empty input, whitespace-only input, or non-text input. | None. | `{"action": "process_email", "classification": "unknown", "actions_taken": [], "status": "failure", "message": "string indicating that the input is invalid", "draft_response": null}` | None. |

| Field | Correctness Criterion |
| --- | --- |
| Behavior | Is each behavior specified as an individual, identifiable requirement, separate from the Goal? Is each requirement sufficiently atomic to be tested in isolation (can a test case be written as "given X, the agent must do Y")? Does the specification explicitly cover the expected behavior in the event of invalid input, tool failure, ambiguity, or missing necessary information? Are the requirements, taken together, sufficient to achieve the stated Goal (has anything essential been left implicit)? |
| Trigger | Is the condition or event that initiates the behavior explicitly defined and observable? |
| Input | Does the specification define the expected input? |
| Tools | Are the tools required for the behavior identified, and is their purpose of use defined? |
| Expected Outcome | Does the specification define what change should occur in the environment? Is the expected change observable and verifiable after execution? |
| Environment Effect | Does the specification define which environment the agent will modify? Does the specification define how the change will be made? |