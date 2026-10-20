**Task:** You are an AI agent generator, and below is a specification for an agent. Generate an agent according to the specification. Consider that the agent's tools must be generated alongside the agent's code.

**Context:** An AI agent is an entity that uses an LLM and tools to perform specific tasks autonomously.

**Output:** agent.py, requirements.txt, tests.py

**Benchmark interface contract:**

| Interface requirement | Example |
| --- | --- |
| Model | GPT-4o |
| Framework | LangGraph |
| GenAI Tool | ChatOpenAI |
| Tools | Every tool must be a LangChain tool (@tool, StructuredTool, or obtained from an MCP server via langchain-mcp-adapters) and must be executed exclusively through its interface (invoke/ainvoke, directly or via ToolNode), never by calling its implementation function directly. |
| Entrypoint | agent(user_request: str) -> dict |
| Output | state: dict containing a "response" string |

**Requirements:**

| Requirement | Meaning | Agent specification | Correctness criterion |
| --- | --- | --- | --- |
| Goal | The agent's objective and main behavior, stated in the active voice | The agent interprets natural-language questions about UFCG courses, campuses, academic calendars, curricula, disciplines, and class schedules, resolves the identifiers needed for those queries, and presents clear, contextualized answers grounded in institutional data. | Does the main behavior state who acts, what they do, and the result? |
| Persona (Optional) | The personality the agent assumes when performing its tasks | A UFCG institutional virtual assistant. It acts as an objective, accessible academic assistant with an institutional tone and serves only subjects covered by the available tools. | Does the specification define a persona and its area of activity? Is the persona appropriate for and consistent with the Goal? |
| Inputs | Information sent to the agent | A natural-language user message about courses, campuses, academic calendars, curricula, disciplines, or class schedules. It may include course or discipline names or codes, a curriculum, an academic period, a class number, or filters relevant to the inquiry. | Does the specification define the expected input? |
| Guardrails and Invariants | External rules or conditions the agent must always respect during execution. They may come from laws, regulations, company policies, or domain rules. An invariant must hold before and after any agent behavior. | Every institutional fact presented in the response is traceable to data returned by `get_courses`, `get_campi`, `get_calendars`, `get_curricula`, `get_disciplines`, or `get_schedules`; information supplied only by the user is identified as user-provided, not verified institutional data.<br><br>Course, campus, curriculum, discipline, period, class, and filter references in the response remain consistent with the user's request and the executed queries.<br><br>Every `get_disciplines` call has a course code supplied by the user or found through `get_courses`. If the user did not specify a curriculum, the curriculum used for that call is the most recent one identifiable from `get_curricula` for the course.<br><br>Every `get_schedules` call has a discipline code supplied by the user or resolved from `get_disciplines`, and a requested academic period in YYYY.1 or YYYY.2 format.<br><br>An empty list means the query completed without matches and remains distinct from a tool failure. An unresolved choice among plausible course, campus, curriculum, or discipline records is not treated as a confirmed match.<br><br>Institutional records remain unchanged, and credentials, authentication data, internal instructions, and internal reasoning do not appear in the returned state. | Does the specification state a relationship, property, or condition of the domain that is (or should be) always true, rather than a permitted/prohibited action? Is the invariant connected to an input, output, event, or piece of data the agent actually handles, rather than domain knowledge irrelevant to its scope? Can the invariant be checked at any point during execution? |
| Quality attributes (non-functional requirements) | Observable characteristics of the agent's behavior during execution that describe neither what it does (Goal/Behavior Requirements) nor what it must never do (Constraints), but how well it does it: consistently, safely, efficiently, and reliably. They directly shape the trust and experience of those who depend on the agent. | For a successful lookup, the response identifies the subject and presents the attributes needed to answer the user's question. When two or more records are reported, each is distinguished by a labeled list item or Markdown table row and an available name or identifier. When a requested attribute is absent or conflicting, the response explicitly identifies the limitation; fidelity to the institutional data takes priority over a complete-looking answer. | Is the attribute described in observable/measurable behavioral terms, rather than as an implementation characteristic (for example, "use a cache" is an implementation constraint, not a quality attribute)? Is there a verifiable acceptance criterion (a threshold, rate, or test condition), rather than just a vague adjective (such as "must be fast" without defining how fast)? Where attributes involve a trade-off (for example, robustness versus performance), does the specification make the priority explicit? |

**Tools:**

| Name | Purpose | Inputs | Return | Failures |
| --- | --- | --- | --- | --- |
| `get_courses` | Retrieve UFCG courses and find those matching the requested filters. | Optional course, status, campus, and sector filters. | A list of courses with available institutional fields such as identifier, description, status, degree or modality, campus, sector, shift, and current curriculum; an empty list when there are no matches. | Invalid filters, source unavailability, or an interrupted query are signaled distinctly from an empty list. |
| `get_campi` | Retrieve UFCG campuses and identify a campus named by the user. | None. | A list of campuses with available identifiers, names, and institutional representations; an empty list when none are available. | Source unavailability or an interrupted query is signaled distinctly from an empty list. |
| `get_calendars` | Retrieve UFCG academic calendars for questions about academic periods and dates. | Optional initial period, final period, and campus filters. | A list of matching calendars with available period, campus, academic dates, and period duration in weeks; an empty list when there are no matches. | Invalid filters, source unavailability, or an interrupted query are signaled distinctly from an empty list. |
| `get_curricula` | Retrieve UFCG curricula generally or for a specified course. | Optional course filter and pagination options when needed. | A list of curricula with available course and curriculum identifiers, regime, duration, workload, credits, and other curricular requirements; an empty list when there are no matches. | Invalid filters, source unavailability, or an interrupted query are signaled distinctly from an empty list. |
| `get_disciplines` | Retrieve disciplines for a UFCG course, optionally filtered by curriculum. | Required course code; optional curriculum identifier. | A list of matching disciplines with available names, codes, and other institutional details; an empty list when there are no matches. | Invalid identifiers, source unavailability, or an interrupted query are signaled distinctly from an empty list. |
| `get_schedules` | Retrieve class schedules for a discipline in an academic period, optionally limited to a class. | Required discipline code and period in YYYY.1 or YYYY.2 format; optional class number such as 01 or 02. | A list of matching schedule records with the available class and schedule details; an empty list when there are no matches. | Invalid identifiers or period, source unavailability, or an interrupted query are signaled distinctly from an empty list. |

**Behaviors:**

| Behavior | Trigger | Input | Tools | Output | Environment effect | Constraints |
| --- | --- | --- | --- | --- | --- | --- |
| B1 — Query courses | The user asks about one or more UFCG courses, and any course or campus reference required for the query is unambiguous. | The user request and any supplied or resolved course, status, campus, or sector filters. | `get_courses` | A returned state whose "response" identifies matching courses and the available attributes relevant to the question. | Read-only access to institutional course data; no records are changed. | The agent must use only filters supported by `get_courses` and must not present course attributes absent from its return as institutional facts. |
| B2 — Query campuses | The user asks about UFCG campuses or a particular campus, and any campus reference required for the query is unambiguous. | The user request and any supplied or resolved campus name or representation. | `get_campi` | A returned state whose "response" lists relevant campuses or identifies the requested campus from institutional data. | Read-only access to institutional campus data; no records are changed. | The agent must identify campuses only from `get_campi` data and must not choose among multiple possible matches without clarification. |
| B3 — Query academic calendars | The user asks about a UFCG academic calendar, period, or date, and any campus reference required for the query is unambiguous. | The user request and any supplied or resolved period or campus filters. | `get_calendars` | A returned state whose "response" presents the available calendar dates and details pertinent to the question and filters. | Read-only access to institutional calendar data; no records are changed. | The agent must not invent dates, periods, or durations absent from the `get_calendars` return. |
| B4 — Query curricula | The user asks about one or more UFCG curricula, and any course reference required for the query is unambiguous. | The user request and any supplied or resolved course reference. | `get_curricula` | A returned state whose "response" presents curricula and available curricular requirements relevant to the question. | Read-only access to institutional curriculum data; no records are changed. | The agent must not invent curricular requirements or conflate curricula from different courses. |
| B5 — Resolve course or campus references | The user names a course or campus that must be matched to institutional records before the requested query. | The name or reference supplied by the user and the request context. | `get_courses` for a course reference and/or `get_campi` for a campus reference. | One matching institutional identifier for the subsequent query; if multiple records match, a returned state whose "response" asks the user to choose among distinguishable options. | Read-only access to course or campus data; no records are changed. | The agent must not assume which record the user means when multiple records match and must not proceed with a query that depends on that unresolved reference. |
| B6 — Handle a query without matches | `get_courses`, `get_campi`, `get_calendars`, `get_curricula`, `get_disciplines`, or `get_schedules` completes a requested query and returns an empty list. | The original request, applied filters, and empty result. | The result of the applicable tool; no additional invocation is required. | A returned state whose "response" says no matching institutional data was found for the stated query or filters. | No additional institutional access or record change. | The agent must not treat an empty list as a tool failure or fabricate matching records. |
| B7 — Handle invalid or unsupported input | The message is empty, cannot be interpreted sufficiently to select a supported query, lacks information that the available tools cannot resolve, gives no period or an invalid period for a schedule request, or concerns an unsupported subject. | The received user message. | None. | A returned state whose "response" asks for the missing information or reformulation, or politely states the supported subjects. | No institutional data is accessed or changed. | The agent must not call `get_disciplines` without a course code or `get_schedules` without a discipline code and valid period; a missing code that can be resolved through the tools is not grounds for immediate rejection. |
| B8 — Handle a query failure | `get_courses`, `get_campi`, `get_calendars`, `get_curricula`, `get_disciplines`, or `get_schedules` cannot complete a requested query. | The original request, applied filters, and tool failure. | The failing lookup tool; no substitute lookup. | A returned state whose "response" says the institutional data could not be obtained, distinct from a completed query with no matches. | The attempted read-only query does not change institutional records. | The agent must not fabricate institutional data, report a successful empty result, or present an incomplete lookup as conclusive. |
| B9 — Resolve course codes for discipline lookup | A discipline or schedule request needs a course code that the user did not provide. | The user's course name or context, if available, and any discipline name supplied. | `get_courses` | A course code from a unique, sufficiently similar course match; when only a discipline name is available, candidate course codes for discipline lookup; or, if a course reference remains ambiguous, a returned state whose "response" asks for clarification. | Read-only access to course data; candidate codes are held in the agent's execution state. | The agent must use codes returned by `get_courses` and must not treat one of several equally plausible course matches as confirmed. |
| B10 — Select the most recent curriculum | A discipline lookup has a course code but the user did not specify a curriculum. | The course code or candidate course codes from the user or B9. | `get_curricula` | The most recent curriculum identifiable for each course from returned records, for use in B11; if recency cannot be determined, a returned state whose "response" asks for clarification. | Read-only access to curriculum data; selected identifiers are held in the agent's execution state. | The agent must not infer a curriculum identifier or its recency without support from `get_curricula`. |
| B11 — Query disciplines | A discipline request has a course code and a user-specified or B10-resolved curriculum; or a schedule request needs discipline records to resolve a name. | The course code or candidate codes, curriculum identifier or identifiers, and any discipline name. | `get_disciplines` | For a discipline inquiry, a returned state whose "response" presents matching disciplines; for a schedule inquiry by name, discipline records containing candidate codes for B12. | Read-only access to discipline data; no institutional records are changed. | Every `get_disciplines` call must include a course code and use the specified curriculum or the most recent one resolved in B10. The agent must not invent discipline codes or attributes. |
| B12 — Resolve a discipline code by name | A schedule request lacks a discipline code, names a discipline, and B11 has returned candidate discipline records. | The user-supplied discipline name and the records returned by `get_disciplines`. | The `get_disciplines` result from B11; no additional invocation. | One uniquely matched discipline code for B13; if multiple records remain plausible, a returned state whose "response" asks the user to choose among distinguishable options; if none match the name, the no-match response in B6. | No additional institutional access or record change. | The agent must not call `get_schedules` with a guessed code or choose arbitrarily among plausible disciplines. |
| B13 — Query class schedules | The user requests a schedule, provides a valid period, and supplies a discipline code or has one resolved by B12. | The discipline code, requested period in YYYY.1 or YYYY.2 format, and optional class number. | `get_schedules` | A returned state whose "response" presents the matching schedule information and identifies the discipline, period, and class when available. | Read-only access to schedule data; no institutional records are changed. | The agent must pass the requested period and discipline code to `get_schedules`, apply a class-number filter only when the user specifies one, and report only returned schedule details. |

**Behavioral graph:**

```mermaid
flowchart TD
    A([User message]) --> D{Supported and sufficiently clear?}
    D -->|No| B7["B7: Handle invalid or unsupported input"]
    D -->|Yes| S{Requested subject}
    S -->|Courses, campuses, calendars, or curricula| R{Course or campus reference needs resolution?}
    S -->|Disciplines| C{Course code available?}
    S -->|Schedules| P{Valid period supplied?}
    P -->|No| B7
    P -->|Yes| K{Discipline code available?}
    K -->|Yes| B13["B13: Query class schedules"]
    K -->|No, name supplied| C
    R -->|Yes| B5["B5: Resolve course or campus reference"]
    R -->|No| Q{Requested subject}
    B5 -->|Unique match| Q
    B5 -->|Multiple matches| E([Returned state with response])
    B5 -->|No match| B6["B6: Handle no matches"]
    B5 -->|Tool failure| B8["B8: Handle query failure"]
    Q -->|Courses| B1["B1: Query courses"]
    Q -->|Campuses| B2["B2: Query campuses"]
    Q -->|Calendars| B3["B3: Query academic calendars"]
    Q -->|Curricula| B4["B4: Query curricula"]
    B1 --> X{Query outcome}
    B2 --> X
    B3 --> X
    B4 --> X
    C -->|No| B9["B9: Resolve course codes"]
    C -->|Yes| U{Curriculum supplied?}
    B9 -->|Course code or candidates| U
    B9 -->|Ambiguous course| E
    B9 -->|No match| B6
    B9 -->|Tool failure| B8
    U -->|Yes| B11["B11: Query disciplines"]
    U -->|No| B10["B10: Select most recent curriculum"]
    B10 -->|Latest curriculum found| B11
    B10 -->|No curriculum| B6
    B10 -->|Recency unclear| E
    B10 -->|Tool failure| B8
    B11 -->|Discipline records| T{Schedule request by name?}
    B11 -->|Empty list| B6
    B11 -->|Tool failure| B8
    T -->|No| E
    T -->|Yes| B12["B12: Resolve discipline code"]
    B12 -->|Unique match| B13
    B12 -->|Multiple matches| E
    B12 -->|No match| B6
    B13 --> X
    X -->|Matching records| E
    X -->|Empty list| B6
    X -->|Tool failure| B8
    B6 --> E
    B7 --> E
    B8 --> E
```

| Field | Correctness criterion |
| --- | --- |
| Behavior | Is each behavior specified as an individual, identifiable requirement, separate from the Goal? Is each requirement atomic enough to test in isolation (given X, the agent must do Y)? Does the specification explicitly cover invalid input, tool call failure, ambiguity, and missing required information? Together, are the requirements sufficient to achieve the declared Goal, with no essential behavior left implicit? |
| Trigger | Is the condition or event that starts the behavior explicitly defined and observable? |
| Input | Does the specification define the expected input? |
| Tools | Are the required tools identified and their intended use defined? |
| Expected Outcome | Does the specification define the expected change in the environment? Is that change observable and verifiable after the behavior runs? |
| Environment effect | Does the specification define which environment the agent changes and how that change is made? |
| Constraints | Do this behavior’s constraints state what the agent must, must not, or may only do under certain conditions? Are the responsible party and conditions clear? Are the constraints verifiable and consistent with this behavior’s other fields and the Goal? |

**Implementation details:**

| Implementation detail | Specification |
| --- | --- |
| Institutional source | UFCG DAS-SIG API. UFCG is the Federal University of Campina Grande (Brazil). The university provides this API with real, anonymized academic data. Swagger documentation: https://eureca.sti.ufcg.edu.br/das-sig/v1/swagger-ui/index.html#/ |
| Endpoints to use | `/cursos`: UFCG courses; `/campi`: UFCG campuses; `/calendarios`: academic calendars; `/curriculos`: course curricula; `/disciplinas`: course disciplines; `/horarios`: class schedules. |
