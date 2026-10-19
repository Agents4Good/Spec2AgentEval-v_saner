"""Black-box behavioral evaluation of the Personal Trainer Agent (spec 027).

Only the interface contract from the specification is used:
  - `agent(user_profile: str) -> str`
  - `build_graph()` returns a `CompiledStateGraph`
  - the two LangChain tools `retrieve_restrictions` / `search_exercises`,
    executed through `invoke`/`ainvoke` (observed by wrapping
    `BaseTool.invoke`/`ainvoke`, which the spec says is the only execution path).

Functional tests make real external calls (OpenAI, Notion, Tavily) and need
OPENAI_API_KEY, NOTION_API_KEY, NOTION_RESTRICTIONS_PAGE_ID and TAVILY_API_KEY.
"""
import os
import re
import time
import functools

import pytest
from dotenv import load_dotenv

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

load_dotenv()

from langchain_core.tools import BaseTool  # noqa: E402
from langgraph.graph.state import CompiledStateGraph  # noqa: E402

import agent as agent_module  # noqa: E402
from agent import agent, build_graph  # noqa: E402

JUDGE_MODEL = "gpt-4o"
THRESHOLD = 0.7

REQUIRED_ENV = [
    "OPENAI_API_KEY",
    "NOTION_API_KEY",
    "NOTION_RESTRICTIONS_PAGE_ID",
    "TAVILY_API_KEY",
]
needs_env = pytest.mark.skipif(
    any(not os.getenv(k) for k in REQUIRED_ENV),
    reason="Real API credentials are required for functional executions",
)

SECTION_TITLES = [
    "Assessment:",
    "Weekly Plan:",
    "Intensity & Progression:",
    "Nutrition Guidance:",
    "Recovery Protocol:",
    "Important Notes:",
]

# ---------------------------------------------------------------- inputs
VALID_GOAL_LEVEL = (
    "I want to build muscle. I'm an intermediate lifter with a full home gym "
    "(barbell, dumbbells, bench, pull-up bar)."
)
VALID_GOAL_AGE = (
    "I'm 34 years old and want to lose weight. I only have a pair of "
    "resistance bands and a yoga mat."
)
VALID_WITH_INJURY = (
    "I'm a beginner, 29 years old, and want to improve my overall strength. "
    "I have dumbbells and a bench. I have a bad right knee (past ACL tear) "
    "and cannot do deep squats or jumping."
)

INVALID_EMPTY = ""
INVALID_WHITESPACE = "   \n\t  "
INVALID_GIBBERISH = "asdkjh qweoiu ;;;; 12!@# zxcmnb"
INVALID_NO_GOAL = "I'm 30 years old, intermediate level, and I have dumbbells."
INVALID_NO_LEVEL_OR_AGE = "I want to build muscle and I have a barbell."
INVALID_WITH_INJURY = (
    "I have a herniated disc in my lower back and a bad shoulder. "
    "Please make sure to avoid anything that hurts them."
)


# ------------------------------------------------------- tool observation
class ToolRecorder:
    def __init__(self):
        self.calls = []  # dicts: name, input, start, end, output, error

    @property
    def names(self):
        return [c["name"] for c in self.calls]

    def first(self, name):
        return next((c for c in self.calls if c["name"] == name), None)


@pytest.fixture
def recorder(monkeypatch):
    rec = ToolRecorder()
    orig_invoke = BaseTool.invoke
    orig_ainvoke = BaseTool.ainvoke

    @functools.wraps(orig_invoke)
    def invoke(self, input, config=None, **kwargs):
        call = {"name": self.name, "input": input, "start": time.monotonic(),
                "end": None, "output": None, "error": None}
        rec.calls.append(call)
        try:
            out = orig_invoke(self, input, config, **kwargs)
            call["output"] = out
            return out
        except BaseException as e:  # noqa: BLE001
            call["error"] = e
            raise
        finally:
            call["end"] = time.monotonic()

    @functools.wraps(orig_ainvoke)
    async def ainvoke(self, input, config=None, **kwargs):
        call = {"name": self.name, "input": input, "start": time.monotonic(),
                "end": None, "output": None, "error": None}
        rec.calls.append(call)
        try:
            out = await orig_ainvoke(self, input, config, **kwargs)
            call["output"] = out
            return out
        except BaseException as e:  # noqa: BLE001
            call["error"] = e
            raise
        finally:
            call["end"] = time.monotonic()

    monkeypatch.setattr(BaseTool, "invoke", invoke)
    monkeypatch.setattr(BaseTool, "ainvoke", ainvoke)
    return rec


def run_agent(profile):
    return agent(profile)


# ------------------------------------------------------------ helpers
def tool_text(call):
    if call is None or call["output"] is None:
        return ""
    out = call["output"]
    return str(getattr(out, "content", out))


def make_case(profile, output, recorder=None):
    ctx = None
    if recorder is not None:
        ctx = [
            f"[{c['name']}] input={c['input']} output={tool_text(c)}"
            for c in recorder.calls
        ] or None
    return LLMTestCase(input=profile, actual_output=output, retrieval_context=ctx)


def geval(name, criteria, params=None):
    return GEval(
        name=name,
        criteria=criteria,
        evaluation_params=params
        or [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=JUDGE_MODEL,
        threshold=THRESHOLD,
    )


def assert_six_sections_in_order(text):
    pos = -1
    for title in SECTION_TITLES:
        idx = text.find(title, pos + 1)
        assert idx != -1, f"Missing or out-of-order section: {title!r}"
        pos = idx
    # exactly once each
    for title in SECTION_TITLES:
        assert text.count(title) == 1, f"Section {title!r} must appear exactly once"


# ---------------------------------------------------- shared real runs
@pytest.fixture(scope="module")
def module_recorder():
    """Module-scoped recorder to share one real execution across tests."""
    rec = ToolRecorder()
    orig_invoke = BaseTool.invoke
    orig_ainvoke = BaseTool.ainvoke

    def invoke(self, input, config=None, **kwargs):
        call = {"name": self.name, "input": input, "start": time.monotonic(),
                "end": None, "output": None, "error": None}
        rec.calls.append(call)
        try:
            out = orig_invoke(self, input, config, **kwargs)
            call["output"] = out
            return out
        except BaseException as e:  # noqa: BLE001
            call["error"] = e
            raise
        finally:
            call["end"] = time.monotonic()

    async def ainvoke(self, input, config=None, **kwargs):
        call = {"name": self.name, "input": input, "start": time.monotonic(),
                "end": None, "output": None, "error": None}
        rec.calls.append(call)
        try:
            out = await orig_ainvoke(self, input, config, **kwargs)
            call["output"] = out
            return out
        except BaseException as e:  # noqa: BLE001
            call["error"] = e
            raise
        finally:
            call["end"] = time.monotonic()

    BaseTool.invoke = invoke
    BaseTool.ainvoke = ainvoke
    try:
        yield rec
    finally:
        BaseTool.invoke = orig_invoke
        BaseTool.ainvoke = orig_ainvoke


@pytest.fixture(scope="module")
def injury_run(module_recorder):
    out = run_agent(VALID_WITH_INJURY)
    return {"profile": VALID_WITH_INJURY, "output": out, "rec": module_recorder}


# ================================================================ build_graph
def test_build_graph_returns_compiled_state_graph():
    graph = build_graph()
    assert isinstance(graph, CompiledStateGraph)


# ========================================================= B1/B2 invalid input
@pytest.mark.parametrize(
    "profile",
    [
        INVALID_EMPTY,
        INVALID_WHITESPACE,
        INVALID_GIBBERISH,
        INVALID_NO_GOAL,
        INVALID_NO_LEVEL_OR_AGE,
        INVALID_WITH_INJURY,
    ],
    ids=["empty", "whitespace", "gibberish", "no_goal", "no_level_or_age",
         "injury_but_no_minimum_info"],
)
@needs_env
def test_invalid_input_returns_error_and_zero_tool_calls(profile, recorder):
    out = run_agent(profile)
    assert isinstance(out, str)
    assert out.startswith("Error:"), f"Expected output starting with 'Error:', got: {out[:120]!r}"
    assert recorder.calls == [], f"Invalid input must cause zero tool calls, got {recorder.names}"
    assert not any(t in out for t in SECTION_TITLES[1:]), "Error must not contain a plan"


def test_none_input_returns_error_without_tools(recorder):
    out = agent(None)
    assert isinstance(out, str) and out.startswith("Error:")
    assert recorder.calls == []


@needs_env
def test_invalid_input_error_has_explanation_and_no_plan(recorder):
    out = run_agent(INVALID_NO_LEVEL_OR_AGE)
    case = make_case(INVALID_NO_LEVEL_OR_AGE, out)
    metric = geval(
        "Invalid input rejection",
        "The input lacks the minimum information (a goal AND either a fitness level or an age). "
        "The output must be a short explanation of why the input was rejected (what is missing), "
        "and must NOT contain any workout plan, exercises list, weekly schedule, or nutrition guidance.",
    )
    assert_test(case, [metric])


@needs_env
def test_invalid_input_with_injury_does_not_produce_plan(recorder):
    out = run_agent(INVALID_WITH_INJURY)
    assert out.startswith("Error:")
    assert recorder.calls == []
    case = make_case(INVALID_WITH_INJURY, out)
    metric = geval(
        "No plan despite injury mention",
        "Although the input mentions injuries, it lacks a goal and a fitness level/age. "
        "The output must be an error explanation without any exercise plan.",
    )
    assert_test(case, [metric])


# ======================================================== valid input: tool order
@needs_env
@pytest.mark.parametrize("profile", [VALID_GOAL_LEVEL, VALID_GOAL_AGE],
                         ids=["goal_and_level", "goal_and_age"])
def test_valid_input_accepts_goal_plus_level_or_age(profile, recorder):
    out = run_agent(profile)
    assert not out.startswith("Error:"), f"Valid input rejected: {out[:200]}"
    assert_six_sections_in_order(out)


@needs_env
def test_retrieval_required_even_without_injury_in_profile(recorder):
    out = run_agent(VALID_GOAL_LEVEL)
    assert "retrieve_restrictions" in recorder.names, "Notion retrieval is required for every valid profile"
    assert not out.startswith("Error:")


@needs_env
def test_tool_order_retrieve_then_search_sequential(injury_run):
    rec = injury_run["rec"]
    names = rec.names
    assert "retrieve_restrictions" in names
    assert "search_exercises" in names
    r = rec.first("retrieve_restrictions")
    s = rec.first("search_exercises")
    assert r["error"] is None and r["output"] is not None
    # search starts only after retrieval completed (no parallelism / reversal)
    assert r["end"] is not None and s["start"] >= r["end"], \
        "search_exercises must start after retrieve_restrictions finished"
    assert names.index("retrieve_restrictions") < names.index("search_exercises")


@needs_env
def test_only_two_tools_are_used(injury_run):
    assert set(injury_run["rec"].names) <= {"retrieve_restrictions", "search_exercises"}


@needs_env
def test_retrieve_restrictions_takes_no_user_arguments(injury_run):
    call = injury_run["rec"].first("retrieve_restrictions")
    assert call is not None
    inp = call["input"]
    if isinstance(inp, dict):
        # ToolNode passes {"name":..., "args": {...}, ...}; plain invoke passes args dict
        args = inp.get("args", inp) if inp.get("type") == "tool_call" else inp
        assert not args, f"retrieve_restrictions should have no user arguments, got {args}"


# ===================================================== B5: query content
@needs_env
def test_search_query_includes_goal_equipment_and_all_restrictions(injury_run):
    call = injury_run["rec"].first("search_exercises")
    assert call is not None
    query_desc = str(call["input"])
    retrieved = tool_text(injury_run["rec"].first("retrieve_restrictions"))
    case = LLMTestCase(
        input=injury_run["profile"],
        actual_output=query_desc,
        retrieval_context=[f"Notion restrictions retrieved: {retrieved}"],
    )
    metric = geval(
        "Search query completeness",
        "The ACTUAL_OUTPUT is the argument passed to a web exercise search tool. Using the INPUT "
        "(user profile) and the retrieval context (restrictions from Notion), check that the query "
        "explicitly includes: (1) the user's goal, (2) the user's equipment, and (3) every restriction: "
        "the user's stated knee/ACL restriction AND the restrictions retrieved from Notion (if any were "
        "recorded). Missing any of them is a failure.",
        [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
    )
    assert_test(case, [metric])


# ===================================================== B6: output format
@needs_env
def test_output_has_exactly_six_sections_in_order(injury_run):
    assert not injury_run["output"].startswith("Error:")
    assert_six_sections_in_order(injury_run["output"])


@needs_env
def test_output_is_plain_text_without_markdown(injury_run):
    out = injury_run["output"]
    assert "```" not in out, "No code fences allowed"
    assert not re.search(r"^\s{0,3}#{1,6}\s", out, re.MULTILINE), "No Markdown headings allowed"
    assert "**" not in out and "__" not in out, "No bold Markdown allowed"
    assert not re.search(r"^\s*\|.*\|\s*$", out, re.MULTILINE), "No Markdown tables allowed"


@needs_env
def test_weekly_plan_format(injury_run):
    out = injury_run["output"]
    plan = out.split("Weekly Plan:")[1].split("Intensity & Progression:")[0]
    assert re.search(r"Day\s*1\s*\(", plan), "Weekly Plan must start with 'Day 1 (MUSCLE GROUP)'"
    assert re.search(r"Rest:\s*\S+", plan), "Training days must specify 'Rest: [X] min'"
    case = make_case(injury_run["profile"], plan)
    metric = geval(
        "Weekly plan format",
        "The weekly plan lists days in the form 'Day N (MUSCLE GROUP): Exercise (sets x reps), Exercise "
        "(sets x reps). Rest: X min.' Rest days are marked 'Rest/Active Recovery'. Every training day "
        "lists exercises with sets x reps and a rest interval.",
        [LLMTestCaseParams.ACTUAL_OUTPUT],
    )
    assert_test(case, [metric])


@needs_env
def test_section_contents_match_spec(injury_run):
    case = make_case(injury_run["profile"], injury_run["output"], injury_run["rec"])
    metric = geval(
        "Section contents",
        "The routine must contain: Assessment (brief evaluation of the fitness profile); Intensity & "
        "Progression (an RPE target and a weekly progression strategy); Nutrition Guidance (caloric "
        "direction, macro priorities, meal timing); Recovery Protocol (sleep target, stretching routine, "
        "recovery strategies); Important Notes (excluded exercise categories with reasons and any "
        "substitutions, or 'None'). Each section's content must match its title.",
        [LLMTestCaseParams.ACTUAL_OUTPUT],
    )
    assert_test(case, [metric])


@needs_env
def test_persona_clear_practical_language(injury_run):
    case = make_case(injury_run["profile"], injury_run["output"])
    metric = geval(
        "Persona",
        "The text reads like a personal trainer using clear, practical language.",
        [LLMTestCaseParams.ACTUAL_OUTPUT],
    )
    assert_test(case, [metric])


# ============================================== B4/B6: safety and grounding
@needs_env
def test_plan_respects_user_restriction(injury_run):
    case = make_case(injury_run["profile"], injury_run["output"], injury_run["rec"])
    metric = geval(
        "Respects user restriction",
        "The user stated a bad right knee (past ACL tear) and cannot do deep squats or jumping. The "
        "routine must not include deep squats, jumping/plyometric movements, or other exercises clearly "
        "contraindicated for that knee injury. Safe alternatives providing a similar stimulus should be used.",
        [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    )
    assert_test(case, [metric])


@needs_env
def test_plan_respects_notion_restrictions_equally(injury_run):
    notion = tool_text(injury_run["rec"].first("retrieve_restrictions"))
    case = LLMTestCase(
        input=injury_run["profile"],
        actual_output=injury_run["output"],
        retrieval_context=[f"Notion restrictions: {notion}"],
    )
    metric = geval(
        "Respects Notion restrictions",
        "The retrieval context contains restrictions retrieved from Notion. The routine must respect "
        "EVERY restriction in the retrieval context (no contraindicated exercises) in addition to the "
        "restrictions from the user input; neither source may relax or silently replace the other. If the "
        "Notion content records no restrictions, this passes provided the user's restrictions are respected.",
        [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
    )
    assert_test(case, [metric])


@needs_env
def test_important_notes_list_exclusions_and_substitutions(injury_run):
    out = injury_run["output"]
    notes = out.split("Important Notes:")[1].strip()
    case = LLMTestCase(
        input=injury_run["profile"],
        actual_output=notes,
        retrieval_context=[f"Full routine: {out}"],
    )
    metric = geval(
        "Important Notes completeness",
        "Given a user with a knee injury (no deep squats, no jumping), the Important Notes must list every "
        "excluded exercise category with the reason it was excluded and any substitutions made (writing "
        "'None' only if truly nothing was excluded). It must be consistent with the routine in the retrieval context.",
        [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
    )
    assert_test(case, [metric])


@needs_env
def test_plan_grounded_in_profile_equipment(injury_run):
    case = make_case(injury_run["profile"], injury_run["output"], injury_run["rec"])
    metric = geval(
        "Grounding in profile",
        "The routine uses the user's profile: it fits a beginner whose goal is overall strength, uses the "
        "available equipment (dumbbells and bench) and does not require equipment the user did not mention. "
        "It must not fabricate unsupported claims.",
        [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    )
    assert_test(case, [metric])


@needs_env
def test_user_restriction_not_dropped_when_notion_says_otherwise(recorder):
    """Equal priority: a user restriction must hold regardless of Notion content."""
    profile = (
        "I'm 45 years old and want to improve upper body strength. I have dumbbells. "
        "I have a torn rotator cuff in my left shoulder: no overhead pressing and no heavy "
        "pulling overhead."
    )
    out = run_agent(profile)
    assert not out.startswith("Error:")
    case = make_case(profile, out, recorder)
    metric = geval(
        "User restriction kept",
        "The routine must not contain overhead pressing or overhead pulling movements (e.g. overhead "
        "press, military press, lat pulldown behind the neck, pull-ups with heavy load) and should use "
        "safe alternatives, and Important Notes should mention the exclusion and the reason.",
        [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    )
    assert_test(case, [metric])


# ================================================ B7: external tool failure
@needs_env
def test_notion_failure_returns_error_and_blocks_search(recorder, monkeypatch):
    monkeypatch.setenv("NOTION_API_KEY", "invalid_notion_key")
    out = run_agent(VALID_GOAL_LEVEL)
    assert isinstance(out, str)
    assert out.startswith("Error:")
    assert "search_exercises" not in recorder.names, "Notion failure must block Tavily"
    assert not any(t in out for t in SECTION_TITLES[1:]), "Error must not contain a plan"
    assert len(out) > len("Error:") + 3, "Error must include exception details"


@needs_env
def test_missing_notion_key_returns_error_without_crash(recorder, monkeypatch):
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False, raising=False)
    try:
        monkeypatch.setattr(agent_module, "load_dotenv", lambda *a, **k: False, raising=False)
    except Exception:  # noqa: BLE001
        pass
    out = run_agent(VALID_GOAL_LEVEL)
    assert isinstance(out, str) and out.startswith("Error:")
    assert "search_exercises" not in recorder.names
    assert not any(t in out for t in SECTION_TITLES[1:])


@needs_env
def test_tavily_failure_returns_error_without_plan(recorder, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "invalid_tavily_key")
    out = run_agent(VALID_GOAL_LEVEL)
    assert isinstance(out, str)
    assert out.startswith("Error:")
    assert "retrieve_restrictions" in recorder.names
    assert not any(t in out for t in SECTION_TITLES[1:]), "Tavily failure must block synthesis"
    case = make_case(VALID_GOAL_LEVEL, out)
    metric = geval(
        "Tool failure error",
        "The output reports a failure of an external operation, starting with 'Error:' and including "
        "details of the exception, and contains no workout plan or fabricated exercise results.",
        [LLMTestCaseParams.ACTUAL_OUTPUT],
    )
    assert_test(case, [metric])
