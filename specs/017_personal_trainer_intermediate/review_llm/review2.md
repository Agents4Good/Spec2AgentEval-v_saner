| Field | Score | Justification |
| --- | --- | --- |
| **Goal** | 5 | The goal clearly defines who acts (the agent), what it does (interprets the profile and retrieves data), and the result (returns a personalized routine). |
| **Persona** | 5 | The persona (personal trainer) and tone (clear and practical) are well defined and consistent with the main goal. |
| **Inputs** | 5 | The expected inputs, mandatory information (goal + fitness level or age), optional information, and strict validity conditions are fully documented. |
| **Guardrails and Invariants** | 5 | The constraints have a clear scope and are verifiable before and after execution (e.g., validation priority, plain-text formatting without Markdown, exclusive selection of one tool). |
| **Quality attributes** | 5 | The attributes (schema consistency, grounding, reliability) are observable and measurable, with objective acceptance criteria (e.g., zero tool calls for invalid inputs, a limit of 3 attempts). |
| **Behavior** | 5 | The behaviors (B1 to B7) are individual, atomic, and testable in isolation. Failure cases, invalid inputs, and alternative flows are handled exhaustively and fulfill the goal. |
| **Trigger** | 5 | The event that initiates each behavior (e.g., "Call to agent", "B1 rejects") is explicitly defined and observable in the flow. |
| **Input (Behavior)** | 5 | The inputs for each behavior (B1–B7) are clearly specified. |
| **Tools (Behavior)** | 5 | The tools (ChatOpenAI, retrieve_restrictions, retrieve_available_equipment) and their absence at specific stages are mapped in detail for each behavior. |
| **Expected Outcome** | 5 | The results generated at each step (error response, retrieved content, structured routine) are observable and verifiable. |
| **Environment Effect** | 5 | The effect on the environment (e.g., reading a specific Notion page or no access) is well defined for each action. |
| **Constraints (Behavior)** | 5 | The constraints for individual actions are clear, verifiable, and fully consistent with the established global guardrails. |
