"""Black-box behavioral evaluation of the Personal Trainer Agent (spec 027).

Only the public contract from the specification is used:
  - `agent(user_profile: str) -> str`
  - `build_graph()` returns a `CompiledStateGraph`
  - tools `retrieve_restrictions` and `search_exercises` are LangChain tools run
    through `invoke`/`ainvoke` (observed here at the LangChain BaseTool boundary,
    which makes execution order observable without touching agent internals).

Functional tests use the real external APIs (OpenAI, Notion, Tavily); they are
skipped when the required environment variables are missing.

Run with: deepeval test run test_deepeval.py
"""

import os
import re
import threading
import time

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

from langchain_core.tools import BaseTool

import agent as agent_module  # noqa: E402

REQUIRED_ENV = [
    "OPENAI_API_KEY",
    "NOTION_API_KEY",
    "NOTION_RESTRICTIONS_PAGE_ID",
    "TAVILY_API_KEY",
]

requires_env = pytest.mark.skipif(
    any(not os.getenv(k) for k in REQUIRED_ENV),
    reason="Real API credentials are required: " + ", ".join(REQUIRED_ENV),
)

SECTION_TITLES = [
    "Assessment:",
    "Weekly Plan:",
    "Intensity & Progression:",
    "Nutrition Guidance:",
    "Recovery Protocol:",
    "Important Notes:",
]

TOOL_NAMES = {"retrieve_restrictions", "search_exercises"}


# --------------------------------------------------------------------------
# Tool-call observation (framework level, implementation independent)
# --------------------------------------------------------------------------
class ToolRecorder:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def record(self, name, tool_input, start, end, ok):
        with self.lock:
            self.events.append(
                {"name": name, "input": tool_input, "start": start, "end": end, "ok": ok}
            )

    def by_name(self, name):
        return sorted(
            (e for e in self.events if e["name"] == name), key=lambda e: e["start"]
        )

    @property
    def names(self):
        return [e["name"] for e in sorted(self.events, key=lambda e: e["start"])]


@pytest.fixture
def recorder(monkeypatch):
    rec = ToolRecorder()
    orig_invoke = BaseTool.invoke
    orig_ainvoke = BaseTool.ainvoke

    def invoke(self, input, config=None, **kwargs):
        start, ok = time.time(), True
        try:
            return orig_invoke(self, input, config, **kwargs)
        except BaseException:
            ok = False
            raise
        finally:
            rec.record(self.name, input, start, time.time(), ok)

    async def ainvoke(self, input, config=None, **kwargs):
        start, ok = time.time(), True
        try:
            return await orig_ainvoke(self, input, config, **kwargs)
        except BaseException:
            ok = False
            raise
        finally:
            rec.record(self.name, input, start, time.time(), ok)

    monkeypatch.setattr(BaseTool, "invoke", invoke)
    monkeypatch.setattr(BaseTool, "ainvoke", ainvoke)
    return rec


def run_agent(profile):
    return agent_module.agent(profile)


def tool_query_text(event):
    value = event["input"]
    if isinstance(value, dict):
        # ToolCall dicts carry args; plain dict inputs are the args themselves
        value = value.get("args", value)
        if isinstance(value, dict):
            return " ".join(str(v) for v in value.values())
    return str(value)


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------
VALID_PROFILE = (
    "I am 30 years old and an intermediate lifter. My goal is to build muscle. "
    "I have a barbell, a bench and dumbbells at home."
)
VALID_PROFILE_NO_EQUIPMENT = "I am a beginner and my goal is to lose weight."
VALID_PROFILE_INJURY = (
    "I am 35, intermediate level, my goal is muscle gain. I have dumbbells and a "
    "pull-up bar. I have a herniated lumbar disc, so I must avoid spinal loading "
    "and heavy axial compression."
)

INVALID_PROFILES = {
    "empty": "",
    "whitespace": "   \n\t  ",
    "none": None,
    "meaningless": "asdf qwer zxcv ;;;; ????? 12@@##",
    "missing_goal": "I am 28 years old and I have dumbbells at home.",
    "missing_level_and_age": "My goal is to build muscle and I have a full gym.",
    "contradictory_level": (
        "My goal is to gain strength. I am a complete beginner who has never "
        "trained, and also an advanced competitive powerlifter."
    ),
    "contradictory_goal": (
        "I am 30 and intermediate. I want to lose as much weight as possible and "
        "at the same time gain 10 kg of body weight, both as the main goal."
    ),
    "ambiguous_restriction": (
        "I am 40, beginner, goal is to get fit. I have some problem in my joint, "
        "it may be the knee or maybe the shoulder, or maybe none, I am not sure "
        "whether I can or cannot do any impact."
    ),
    "injury_but_invalid": "I recently tore my ACL and I want a workout plan.",
}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
error_explanation_metric = GEval(
    name="Error Explanation Without Plan",
    criteria=(
        "The actual output must be a rejection: it begins with 'Error:' and "
        "explains why the user profile cannot be accepted (what information is "
        "missing, unclear, conflicting or meaningless and what needs clarification). "
        "It must NOT contain a workout plan, weekly schedule, exercises, nutrition "
        "or recovery guidance, and it must not ask to start a dialogue."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

clarification_metric = GEval(
    name="Blocking Issue Clarification",
    criteria=(
        "The input contains a blocking problem (contradiction, ambiguity, or missing "
        "minimum information). The actual output must start with 'Error:' and "
        "specifically identify which information is unclear, conflicting or missing "
        "and state what needs to be clarified. It must not choose an interpretation "
        "on the user's behalf and must not include a plan."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

query_metric = GEval(
    name="Search Query Completeness",
    criteria=(
        "The input is the user profile; the actual output is the search query sent "
        "to the exercise search tool. The query must explicitly include the user's "
        "goal, their equipment (if any was stated), and every restriction/limitation "
        "stated in the profile (and the equivalent safe/contraindication context). "
        "It must be a meaningful free-text search query."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

grounding_metric = GEval(
    name="Profile Grounding",
    criteria=(
        "The routine must be personalized to the user profile in the input: it fits "
        "the stated goal, fitness level or age, and uses the stated equipment where "
        "applicable. The Assessment must briefly evaluate the user's fitness profile. "
        "It must not invent contradictory facts about the user."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

format_metric = GEval(
    name="Six Section Plain Text Routine",
    criteria=(
        "The output must be plain text with exactly six sections in this order: "
        "Assessment, Weekly Plan, Intensity & Progression, Nutrition Guidance, "
        "Recovery Protocol, Important Notes. The Weekly Plan lists days with "
        "'Day N (MUSCLE GROUP): exercises (sets x reps). Rest: X min.' and marks rest "
        "days as 'Rest/Active Recovery'. Intensity & Progression has an RPE target and "
        "a weekly progression strategy. Nutrition Guidance has caloric direction, macro "
        "priorities and meal timing. Recovery Protocol has sleep target, stretching and "
        "recovery strategies. No extra sections, no code fences, no Markdown formatting, "
        "no leftover bracket placeholders."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

safety_metric = GEval(
    name="Restriction Safety",
    criteria=(
        "The user profile states an injury/limitation. The routine must respect it: "
        "no exercise included in the Weekly Plan may load or aggravate the stated "
        "limitation (e.g. contraindicated movement patterns or excluded exercise "
        "categories). Safe alternatives providing a similar stimulus should be used, "
        "and if a muscle group has no safe alternative its training must be explicitly "
        "adapted. Important Notes must list every excluded exercise category with the "
        "reason and any substitutions made. Restrictions must never be relaxed."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

error_details_metric = GEval(
    name="Tool Failure Error",
    criteria=(
        "An external tool failed during execution. The actual output must begin with "
        "'Error:' followed by details of the failure (exception details), and must NOT "
        "contain a workout plan, weekly schedule or fabricated exercise results."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def assert_no_plan(output):
    lowered = output.lower()
    assert "weekly plan:" not in lowered
    assert "nutrition guidance:" not in lowered
    assert "recovery protocol:" not in lowered


def assert_six_sections(output):
    positions = []
    for title in SECTION_TITLES:
        assert output.count(title) == 1, f"Section '{title}' must appear exactly once"
        positions.append(output.index(title))
    assert positions == sorted(positions), "Sections are out of order"
    assert "```" not in output, "Code fences are not allowed"
    assert "**" not in output, "Markdown bold is not allowed"
    assert not re.search(r"^\s{0,3}#{1,6}\s", output, re.M), "Markdown headings not allowed"


# --------------------------------------------------------------------------
# B1/B2 - invalid input: Error:, no plan, zero tool calls
# --------------------------------------------------------------------------
@requires_env
@pytest.mark.parametrize("case", list(INVALID_PROFILES.keys()))
def test_invalid_input_returns_error_with_zero_tool_calls(case, recorder):
    profile = INVALID_PROFILES[case]
    output = run_agent(profile)

    assert isinstance(output, str)
    assert output.startswith("Error:"), f"Expected Error: prefix, got: {output[:120]}"
    assert recorder.events == [], f"Tools must not be called, got {recorder.names}"
    assert_no_plan(output)

    if profile is None or not str(profile).strip():
        return  # nothing semantic to judge
    assert_test(
        LLMTestCase(input=str(profile), actual_output=output),
        [error_explanation_metric],
    )


@requires_env
@pytest.mark.parametrize(
    "case",
    [
        "contradictory_level",
        "contradictory_goal",
        "ambiguous_restriction",
        "missing_level_and_age",
        "missing_goal",
    ],
)
def test_blocking_issues_are_explained(case, recorder):
    profile = INVALID_PROFILES[case]
    output = run_agent(profile)
    assert output.startswith("Error:")
    assert recorder.events == []
    assert_test(LLMTestCase(input=profile, actual_output=output), [clarification_metric])


# --------------------------------------------------------------------------
# Valid inputs accepted: missing optional info must not cause rejection
# --------------------------------------------------------------------------
@requires_env
@pytest.mark.parametrize(
    "profile", [VALID_PROFILE, VALID_PROFILE_NO_EQUIPMENT], ids=["full", "no_equipment"]
)
def test_valid_profile_is_not_rejected(profile, recorder):
    output = run_agent(profile)
    assert isinstance(output, str)
    assert not output.startswith("Error:"), f"Valid input rejected: {output[:200]}"
    assert "retrieve_restrictions" in recorder.names


# --------------------------------------------------------------------------
# B3/B5 - dependent sequence, never reversed or parallel
# --------------------------------------------------------------------------
@requires_env
@pytest.mark.parametrize("profile", [VALID_PROFILE, VALID_PROFILE_INJURY], ids=["no_injury", "injury"])
def test_retrieval_precedes_search_sequentially(profile, recorder):
    output = run_agent(profile)
    assert not output.startswith("Error:"), output[:200]

    retrievals = recorder.by_name("retrieve_restrictions")
    searches = recorder.by_name("search_exercises")
    assert retrievals, "retrieve_restrictions must be called for every valid profile"
    assert searches, "search_exercises must be called for a valid profile"
    assert set(recorder.names) <= TOOL_NAMES, "Only the two specified tools may be used"
    assert retrievals[0]["ok"], "Retrieval must succeed before search"
    # Search starts only after retrieval has completed (no reverse, no parallel)
    assert retrievals[0]["end"] <= searches[0]["start"]
    assert recorder.names[0] == "retrieve_restrictions"
    for r in retrievals:
        for s in searches:
            overlap = r["start"] < s["end"] and s["start"] < r["end"]
            assert not overlap, "Tools must not run in parallel"


@requires_env
def test_retrieval_required_even_without_injury_mention(recorder):
    run_agent(VALID_PROFILE_NO_EQUIPMENT)
    assert len(recorder.by_name("retrieve_restrictions")) >= 1


@requires_env
def test_search_query_includes_goal_equipment_and_restrictions(recorder):
    run_agent(VALID_PROFILE_INJURY)
    searches = recorder.by_name("search_exercises")
    assert searches
    query = tool_query_text(searches[0])
    assert query.strip()
    assert_test(
        LLMTestCase(input=VALID_PROFILE_INJURY, actual_output=query), [query_metric]
    )


# --------------------------------------------------------------------------
# B6 - output format, grounding and safety
# --------------------------------------------------------------------------
@requires_env
@pytest.mark.parametrize(
    "profile", [VALID_PROFILE, VALID_PROFILE_NO_EQUIPMENT, VALID_PROFILE_INJURY],
    ids=["full", "no_equipment", "injury"],
)
def test_successful_output_has_six_sections_in_order(profile, recorder):
    output = run_agent(profile)
    assert isinstance(output, str)
    assert not output.startswith("Error:"), output[:200]
    assert_six_sections(output)
    assert_test(LLMTestCase(input=profile, actual_output=output), [format_metric])


@requires_env
def test_routine_is_grounded_in_profile(recorder):
    output = run_agent(VALID_PROFILE)
    assert not output.startswith("Error:"), output[:200]
    assert_test(LLMTestCase(input=VALID_PROFILE, actual_output=output), [grounding_metric])


@requires_env
def test_routine_respects_user_restriction(recorder):
    output = run_agent(VALID_PROFILE_INJURY)
    assert not output.startswith("Error:"), output[:200]
    assert_six_sections(output)
    notes = output.split("Important Notes:")[-1].strip()
    assert notes, "Important Notes must list exclusions/substitutions or 'None'"
    assert_test(
        LLMTestCase(input=VALID_PROFILE_INJURY, actual_output=output), [safety_metric]
    )


# --------------------------------------------------------------------------
# B7 - tool failure handling
# --------------------------------------------------------------------------
@requires_env
def test_notion_failure_returns_error_and_blocks_search(recorder, monkeypatch):
    monkeypatch.setenv("NOTION_API_KEY", "invalid-notion-key")
    output = run_agent(VALID_PROFILE)

    assert isinstance(output, str)
    assert output.startswith("Error:"), output[:200]
    assert_no_plan(output)
    assert recorder.by_name("search_exercises") == [], "Search must not run after Notion failure"
    assert_test(LLMTestCase(input=VALID_PROFILE, actual_output=output), [error_details_metric])


@requires_env
def test_tavily_failure_returns_error_and_blocks_synthesis(recorder, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "invalid-tavily-key")
    output = run_agent(VALID_PROFILE)

    assert isinstance(output, str)
    assert output.startswith("Error:"), output[:200]
    assert_no_plan(output)
    assert recorder.by_name("retrieve_restrictions"), "Retrieval must have happened first"
    assert_test(LLMTestCase(input=VALID_PROFILE, actual_output=output), [error_details_metric])


# --------------------------------------------------------------------------
# Interface contract
# --------------------------------------------------------------------------
def test_build_graph_returns_compiled_state_graph():
    from langgraph.graph.state import CompiledStateGraph

    graph = agent_module.build_graph()
    assert isinstance(graph, CompiledStateGraph)


def test_agent_entrypoint_signature():
    import inspect

    params = list(inspect.signature(agent_module.agent).parameters)
    assert params == ["user_profile"]
