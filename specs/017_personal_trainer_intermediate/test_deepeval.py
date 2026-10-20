"""Black-box behavioral evaluation of the Personal Trainer Agent (spec 017).

Only the interface in the specification is used:
  - agent(user_profile) -> str
  - build_graph() -> CompiledStateGraph
Tool calls are observed through the messages returned by the compiled graph.
Requires real OPENAI_API_KEY / NOTION_* environment variables.

Run: deepeval test run test_deepeval.py
"""
import re

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agent import agent, build_graph

EQUIPMENT_TOOL = "retrieve_available_equipment"
RESTRICTIONS_TOOL = "retrieve_restrictions"

SECTION_TITLES = [
    "Assessment:",
    "Weekly Plan:",
    "Intensity & Progression:",
    "Nutrition Guidance:",
    "Recovery Protocol:",
    "Important Notes:",
    "Tool Used:",
]

# ---------------------------------------------------------------- inputs
INVALID_INPUTS = [
    pytest.param("", id="empty"),
    pytest.param("    \n\t  ", id="whitespace"),
    pytest.param("asdf qwer ;;;; ???? 12!@#", id="meaningless"),
    pytest.param("I want to build muscle.", id="goal_without_level_or_age"),
    pytest.param("I am 30 years old and intermediate.", id="level_and_age_without_goal"),
    pytest.param("My knee hurts a lot and I have a back injury.", id="pain_without_goal_level_age"),
]

EQUIPMENT_INPUTS = [
    pytest.param(
        "I am 28, intermediate level, and want to gain muscle mass training 4 days a week.",
        id="muscle_gain",
    ),
    pytest.param(
        "Beginner, 35 years old, I want to lose weight. I am vegetarian.",
        id="weight_loss_vegetarian",
    ),
    pytest.param(
        "Advanced lifter, goal is strength for powerlifting, 3 days per week.",
        id="level_only_no_age",
    ),
]

RESTRICTION_INPUTS = [
    pytest.param(
        "I am 45, beginner, want to lose weight. I have a herniated disc in my lower back.",
        id="explicit_injury",
    ),
    pytest.param(
        "Intermediate, 52 years old, want to get stronger. My right knee hurts when I squat.",
        id="explicit_pain",
    ),
    pytest.param(
        "I'm 60 and want to improve my general fitness. I have high blood pressure and type 2 diabetes.",
        id="health_condition",
    ),
    pytest.param(
        "Beginner, goal is to tone up. I just had shoulder surgery last month and my doctor said to be careful.",
        id="implied_restriction",
    ),
]

VALID_INPUTS = EQUIPMENT_INPUTS + RESTRICTION_INPUTS


# ---------------------------------------------------------------- helpers
def run_graph(user_profile: str):
    """Invoke the compiled graph and return (final_text, tool_names_called)."""
    graph = build_graph()
    state = graph.invoke({"messages": [("user", user_profile)]})
    messages = state["messages"] if isinstance(state, dict) and "messages" in state else []
    tool_names = []
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            tool_names.append(call["name"])
    return tool_names


def has_markdown(text: str) -> bool:
    if "```" in text or "**" in text or "__" in text:
        return True
    for line in text.splitlines():
        if re.match(r"^\s*#{1,6}\s", line):  # headings
            return True
        if re.match(r"^\s*([-*+]|\d+\.)\s+\S", line):  # list markup
            return True
    return False


def make_geval(name: str, criteria: str, threshold: float = 0.6) -> GEval:
    return GEval(
        name=name,
        criteria=criteria,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=threshold,
    )


# ---------------------------------------------------------------- build_graph
def test_build_graph_returns_compiled_state_graph():
    from langgraph.graph.state import CompiledStateGraph

    assert isinstance(build_graph(), CompiledStateGraph)


def test_agent_returns_string():
    out = agent("I am 28, intermediate level, and want to gain muscle mass.")
    assert isinstance(out, str)


# ---------------------------------------------------------------- B1/B2 invalid input
@pytest.mark.parametrize("profile", INVALID_INPUTS)
def test_invalid_input_returns_error_string(profile):
    out = agent(profile)
    assert isinstance(out, str)
    assert out.startswith("Error:"), out


@pytest.mark.parametrize("profile", INVALID_INPUTS)
def test_invalid_input_has_no_plan(profile):
    out = agent(profile)
    for title in SECTION_TITLES[1:]:
        assert title not in out, f"Error output contains plan section {title}"


@pytest.mark.parametrize("profile", INVALID_INPUTS)
def test_invalid_input_makes_zero_tool_calls(profile):
    assert run_graph(profile) == []


@pytest.mark.parametrize("profile", INVALID_INPUTS)
def test_invalid_input_error_is_explanatory(profile):
    out = agent(profile)
    tc = LLMTestCase(input=profile, actual_output=out)
    assert_test(
        tc,
        [
            make_geval(
                "Invalid input explanation",
                "The output must be a single plain-text explanation beginning with 'Error:' that "
                "explains the input is invalid or lacks required information (a goal together with "
                "either a fitness level or an age). It must NOT contain a workout plan or routine.",
            )
        ],
    )


# ---------------------------------------------------------------- B3 tool routing
@pytest.mark.parametrize("profile", EQUIPMENT_INPUTS)
def test_no_restrictions_uses_only_equipment_tool(profile):
    tools = run_graph(profile)
    assert tools, "expected a tool call"
    assert set(tools) == {EQUIPMENT_TOOL}, tools


@pytest.mark.parametrize("profile", RESTRICTION_INPUTS)
def test_restrictions_use_only_restrictions_tool(profile):
    tools = run_graph(profile)
    assert tools, "expected a tool call"
    assert set(tools) == {RESTRICTIONS_TOOL}, tools


@pytest.mark.parametrize("profile", VALID_INPUTS)
def test_never_both_tools_and_no_parallel_calls(profile):
    tools = run_graph(profile)
    assert not ({EQUIPMENT_TOOL, RESTRICTIONS_TOOL} <= set(tools))
    # Retries of the selected query are capped at three total attempts.
    assert len(tools) <= 3, tools


# ---------------------------------------------------------------- B6 output format
@pytest.mark.parametrize("profile", VALID_INPUTS)
def test_success_output_has_seven_sections_in_order(profile):
    out = agent(profile)
    assert not out.startswith("Error:"), out
    positions = []
    for title in SECTION_TITLES:
        assert out.count(title) == 1, f"'{title}' should appear exactly once"
        positions.append(out.index(title))
    assert positions == sorted(positions), "sections out of order"


@pytest.mark.parametrize("profile", VALID_INPUTS)
def test_success_output_is_plain_text_without_markdown(profile):
    out = agent(profile)
    assert isinstance(out, str)
    assert not has_markdown(out), out


@pytest.mark.parametrize("profile", VALID_INPUTS)
def test_weekly_plan_day_format(profile):
    out = agent(profile)
    plan = out.split("Weekly Plan:")[1].split("Intensity & Progression:")[0]
    assert re.search(r"Day 1 \(", plan), plan
    assert re.search(r"Rest:\s*\d+", plan), "training days must include rest minutes"
    assert re.search(r"\d+\s*x\s*\d+", plan), "exercises must include sets x reps"


@pytest.mark.parametrize("profile", EQUIPMENT_INPUTS)
def test_tool_used_names_equipment_tool(profile):
    out = agent(profile)
    tail = out.split("Tool Used:")[-1]
    assert EQUIPMENT_TOOL in tail
    assert RESTRICTIONS_TOOL not in tail


@pytest.mark.parametrize("profile", RESTRICTION_INPUTS)
def test_tool_used_names_restrictions_tool(profile):
    out = agent(profile)
    tail = out.split("Tool Used:")[-1]
    assert RESTRICTIONS_TOOL in tail
    assert EQUIPMENT_TOOL not in tail


# ---------------------------------------------------------------- semantic quality (LLM-judged)
@pytest.mark.parametrize("profile", VALID_INPUTS)
def test_routine_is_complete_and_personalized(profile):
    out = agent(profile)
    tc = LLMTestCase(input=profile, actual_output=out)
    assert_test(
        tc,
        [
            make_geval(
                "Routine completeness",
                "The output is a personalized routine for the user's profile. It must contain: "
                "an assessment of the user's fitness profile; a weekly plan listing muscle groups, "
                "exercises with sets x reps and rest minutes per training day, with rest days marked "
                "'Rest/Active Recovery'; RPE target and weekly progression; caloric direction, macro "
                "priorities and meal timing; sleep target, stretching and recovery strategies; "
                "important notes with safety considerations; and a Tool Used line. "
                "It must be consistent with the user's stated goal and level/age.",
            )
        ],
    )


def test_dietary_requirement_respected():
    profile = "Beginner, 35 years old, I want to lose weight. I am vegetarian."
    out = agent(profile)
    tc = LLMTestCase(input=profile, actual_output=out)
    assert_test(
        tc,
        [
            make_geval(
                "Dietary requirement",
                "The user is vegetarian. The nutrition guidance must respect this and must not "
                "recommend meat, poultry or fish.",
                threshold=0.7,
            )
        ],
    )


@pytest.mark.parametrize(
    "profile,restriction",
    [
        ("I am 45, beginner, want to lose weight. I have a herniated disc in my lower back.",
         "herniated disc / lower back"),
        ("Intermediate, 52 years old, want to get stronger. My right knee hurts when I squat.",
         "knee pain when squatting"),
    ],
)
def test_health_restrictions_respected(profile, restriction):
    out = agent(profile)
    tc = LLMTestCase(input=profile, actual_output=out)
    assert_test(
        tc,
        [
            make_geval(
                "Restriction safety",
                f"The user has this health restriction: {restriction}. The routine must take it "
                "into account: avoid or modify exercises that would aggravate it, and the "
                "Important Notes section must mention safety considerations and any constraint "
                "modifications (or 'None').",
                threshold=0.7,
            )
        ],
    )


@pytest.mark.parametrize("profile", RESTRICTION_INPUTS)
def test_notion_claims_not_fabricated(profile):
    out = agent(profile)
    tc = LLMTestCase(input=profile, actual_output=out)
    assert_test(
        tc,
        [
            make_geval(
                "Notion grounding (plausibility)",
                "Important Notes should present constraint modifications as coming from retrieved "
                "Notion restriction content only when they are concrete and relevant to the user's "
                "restriction; otherwise it should say 'None'. Penalize vague, contradictory or "
                "clearly invented claims attributed to Notion.",
                threshold=0.5,
            )
        ],
    )


@pytest.mark.parametrize("profile", RESTRICTION_INPUTS[:2])
def test_persona_clear_practical_language(profile):
    out = agent(profile)
    tc = LLMTestCase(input=profile, actual_output=out)
    assert_test(
        tc,
        [
            make_geval(
                "Persona",
                "The output reads like a personal trainer using clear, practical language.",
                threshold=0.6,
            )
        ],
    )
