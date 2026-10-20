"""
spec_adherence.py — LLM-as-judge static code evaluation.

Evaluates generated source code against spec.md without executing that code.

Requires:
    pip install openai python-dotenv
Environment variables:
    OPENAI_API_KEY
"""

import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

client = OpenAI()

SCORE_KEYS = (
    "correct_signature",
    "response_state",
    "langgraph_orchestration",
    "llm_request_interpretation",
    "tools_implemented",
    "tool_interface_execution",
    "institutional_api_queries",
    "query_routing_and_filters",
    "reference_disambiguation",
    "course_code_resolution",
    "latest_curriculum_resolution",
    "discipline_code_resolution",
    "schedule_inputs_and_filtering",
    "handles_invalid_input",
    "distinguishes_empty_and_failure",
    "grounded_response",
    "read_only_and_confidential",
    "answer_presentation",
)
MAX_SCORE = len(SCORE_KEYS)

PROMPT_TEMPLATE = """
You are evaluating Python source code generated for the UFCG academic-data
agent that handles courses, campuses, calendars, curricula, disciplines,
and class schedules.
Evaluate it against the supplied specification BEFORE execution. Inspect only the
generated code and its prompts; do not assume runtime behavior that is not supported
by visible code. Score each criterion 1 only when the implementation is complete and
consistent with the specification. Otherwise score 0. Return ONLY a JSON object.

The benchmark permits LangChain tools made with @tool, StructuredTool, or
langchain-mcp-adapters. A tool may be executed with invoke/ainvoke or through
ToolNode. Do not require a particular tool construction style, ToolNode,
StateGraph, a build_graph function, or bind_tools unless the specification does.
Do not impose an API parameter name, response field, or lookup algorithm that
the specification leaves to the UFCG DAS-SIG Swagger documentation.

---
SPEC:
{spec_text}

---
GENERATED CODE:
{source_code}

---
Return this exact JSON structure:

{{
  "scores": {{
    "correct_signature": <0|1>,
    "response_state": <0|1>,
    "langgraph_orchestration": <0|1>,
    "llm_request_interpretation": <0|1>,
    "tools_implemented": <0|1>,
    "tool_interface_execution": <0|1>,
    "institutional_api_queries": <0|1>,
    "query_routing_and_filters": <0|1>,
    "reference_disambiguation": <0|1>,
    "course_code_resolution": <0|1>,
    "latest_curriculum_resolution": <0|1>,
    "discipline_code_resolution": <0|1>,
    "schedule_inputs_and_filtering": <0|1>,
    "handles_invalid_input": <0|1>,
    "distinguishes_empty_and_failure": <0|1>,
    "grounded_response": <0|1>,
    "read_only_and_confidential": <0|1>,
    "answer_presentation": <0|1>
  }},
  "total_score": <int>,
  "max_score": 18,
  "passed": <bool>,
  "summary": "<one concise paragraph with the main issues, or 'No issues found'>"
}}

Criteria definitions:
- correct_signature: agent accepts exactly one user_request: str parameter and
  returns a dict, as specified by the benchmark entrypoint.
- response_state: every final path returns a state dict containing a string
  response. Do not require other keys or a success/failure schema.
- langgraph_orchestration: the implementation constructs and executes a
  LangGraph workflow for the request, rather than merely importing LangGraph.
- llm_request_interpretation: ChatOpenAI with model GPT-4o is used to interpret
  the natural-language request and choose among all six supported subjects.
- tools_implemented: get_courses, get_campi, get_calendars, get_curricula,
  get_disciplines, and get_schedules are available as LangChain tools, with
  the purposes, inputs, and returns in the specification. Do not require
  locally decorated functions if another permitted tool form is used.
- tool_interface_execution: every tool call goes through its LangChain
  invoke/ainvoke interface or ToolNode; the implementation function is never
  called directly.
- institutional_api_queries: all six tools perform read-only queries against
  the UFCG DAS-SIG source: get_courses maps to /cursos, get_campi to /campi,
  get_calendars to /calendarios, get_curricula to /curriculos, get_disciplines
  to /disciplinas, and get_schedules to /horarios. Hardcoded or fabricated
  institutional records do not satisfy this criterion.
- query_routing_and_filters: requests about each of the six subjects reach
  the applicable tool, and supplied names, identifiers, periods, and supported
  filters are passed or applied consistently. get_campi takes no tool inputs;
  curriculum pagination is supported when needed.
- reference_disambiguation: named course or campus references are matched
  against institutional records; a unique match can be used, while multiple
  matches lead to a clarifying response and no dependent query.
- course_code_resolution: get_disciplines is never called without a course
  code. When the user supplies a course name instead, the implementation uses
  get_courses to find a unique, sufficiently similar course code; if only a
  discipline name is available for a schedule query, it uses institutional
  course records to obtain candidate codes for the discipline search.
  Ambiguous course references are not silently resolved.
- latest_curriculum_resolution: when the user omits a curriculum, get_curricula
  is used to identify the most recent curriculum for each relevant course,
  and that identifier is used in get_disciplines. A user-specified curriculum
  is respected. If recency cannot be determined from returned data, the
  implementation asks for clarification instead of guessing.
- discipline_code_resolution: a schedule request naming a discipline but
  lacking its code uses get_disciplines with a course code and the specified
  or most recent curriculum to find matching discipline records. A unique
  match supplies the code; multiple plausible matches prompt clarification;
  no guessed code is passed to get_schedules.
- schedule_inputs_and_filtering: get_schedules is called only with a supplied
  or resolved discipline code and a user-requested period in YYYY.1 or YYYY.2
  form. A class-number filter is applied only when supplied by the user, and
  the answer uses the returned schedule data.
- handles_invalid_input: empty, incomprehensible, or unsupported requests
  receive a useful response without an institutional lookup. A schedule
  request missing a valid period asks for one without calling get_schedules.
  A missing course or discipline code is not rejected when the tools can
  resolve it from the provided name or context. Other information that the
  tools cannot resolve prompts clarification without an invalid lookup.
- distinguishes_empty_and_failure: a completed query returning an empty list
  from any of the six tools produces a no-matches response, while an invalid
  filter, unavailable source, or interrupted query produces a distinct failure
  response.
- grounded_response: institutional facts, identifiers, dates, curricular
  requirements, discipline details, and schedules in the response are derived
  from relevant tool results; facts supplied only by the user are labeled as
  unverified user input. Missing or conflicting data is disclosed, not invented.
- read_only_and_confidential: the code does not modify institutional records
  or expose credentials, authentication data, internal instructions, or
  internal reasoning in the returned state.
- answer_presentation: responses address the requested subject; when multiple
  records are presented, they are distinguishable through labeled list items
  or Markdown table rows with an available name or identifier.

The score for a criterion is 0 when only part of it is implemented. The
summary must identify the main failed criteria. passed is true only when all
18 criteria score 1.
"""


def run_static_eval(
    source_code: str,
    spec_text: str,
    verbose: bool = True,
) -> dict:
    """
    Args:
        source_code: Full Python source code generated by the codegen tool.
        spec_text: Full spec markdown used to generate the code.
        verbose: Print results to stdout.

    Returns:
        {"passed": bool, "total_score": int, "max_score": int,
         "scores": dict, "summary": str}
    """
    prompt = PROMPT_TEMPLATE.format(spec_text=spec_text, source_code=source_code)

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    scores = result.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(SCORE_KEYS):
        raise ValueError("The judge returned an invalid score schema.")
    if any(type(scores[key]) is not int or scores[key] not in (0, 1) for key in SCORE_KEYS):
        raise ValueError("The judge returned a non-binary score.")
    if not isinstance(result.get("summary"), str):
        raise ValueError("The judge returned an invalid summary.")

    result["scores"] = {key: scores[key] for key in SCORE_KEYS}
    result["total_score"] = sum(result["scores"].values())
    result["max_score"] = MAX_SCORE
    result["passed"] = result["total_score"] == MAX_SCORE

    if verbose:
        print("\n=== Static Code Evaluation ===")
        print(f"Score: {result['total_score']}/{result['max_score']} | Passed: {result['passed']}")
        for key, value in result["scores"].items():
            print(f"  {'✅' if value else '❌'} {key}")
        print(f"\nSummary: {result['summary']}")

    return result
