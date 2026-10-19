| Field | Score (1–5) | Justification |
| :--- | :---: | :--- |
| **Goal** | 5 | The goal clearly identifies the agent as the actor, details the main multi-stage investigation process (entity discovery, academic search, and synthesis), and defines the expected outcome using active voice. |
| **Persona** | 5 | Explicitly defines the persona ("Research investigative assistant") along with its professional tone and methodical approach, fully aligning and maintaining consistency with the Goal. |
| **Inputs** | 4 | Clearly defines the expected input type ("Complex natural language text that requires multi-stage investigation"), though providing concrete format examples would have made it fully detailed. |
| **Guardrails and Invariants** | 5 | Presents a verifiable, persistent domain condition (I1: strict domain-to-tool routing with ArXiv as fallback) connected directly to the data and tools manipulated by the agent during execution. |
| **Quality Attributes** | 5 | The attribute (QA1) is described in an observable, measurable manner (exact case-insensitive string match with trimmed whitespace) with an unambiguous acceptance criterion for data chaining. |
| **Behavior** | 5 | Behaviors (B1 to B5) cover the end-to-end lifecycle, including primary discovery, T1 tool failure recovery via LLM inference, domain routing, and standard handling for invalid input or execution exceptions. |
| **Trigger** | 5 | Every behavior explicitly details a clear, observable initiating event (e.g., receipt of valid request, failure/success of T1, completion of Stage 2). |
| **Input (Behaviors)** | 5 | Precisely states the required input data for each isolated behavior, effectively mapping context transfer across stage transitions. |
| **Tools** | 5 | Tools (T1, T2, T3) are exhaustively defined with explicit purpose, inputs, return formats, and clear failure-handling constraints. |
| **Expected Outcome / Output** | 5 | Defines strict, observable outputs across all behaviors, detailing specific JSON structures (`topic`, `discovered_entity`, `papers`) or exact standardized error strings in B5. |
| **Environment Effect** | 5 | Explicitly declares the absence of environmental side effects ("None") for all behaviors, ensuring strictly read-only operations. |
| **Constraints** | 5 | Sets unambiguous, testable constraints for each flow (e.g., mandatory T1 invocation at start, prohibition of conversational formatting in final output, non-abort policy on T1 failure). |

Status: Approved