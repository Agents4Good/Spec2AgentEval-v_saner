"""Trace-based evaluation of the recipe agent (specification 018_recipe_intermediate).

Each scenario invokes the agent once through the benchmark wrapper ``traced(agent)``.
Verdicts come only from the execution trace (``status``, ``trajectory``, ``tool_calls``).
The agent output is never evaluated. The behavioral graph (B1-B7) is the oracle for the
observed trajectory, and a requirement whose evidence the trace does not record fails.

Run with: deepeval test run test_trace.py
"""

from __future__ import annotations

import functools
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import pytest
from deepeval import assert_test
from deepeval.metrics import BaseMetric, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from trace_capture import traced
from agent import agent

SEARCH = "search_recipes"
PANTRY = "retrieve_pantry"
NO_TOOL = "none"
SUCCESS = "success"
FAILURE = "failure"

MAX_SEARCH_RESULTS = 5
# The spec writes the failure prefix as "Error:" for retrieve_pantry and "erro:" for
# search_recipes (B6 also says "erro:"). Both spellings are accepted for either tool.
FAILURE_PREFIXES = ("error:", "erro:")
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

# Ingredient concepts, matched against normalized text (lowercase, no accents).
# The patterns cover the regional variants and inflected forms that B2 requires.
CONCEPTS: Dict[str, str] = {
    "tomato": r"\bj?tomat(o|oes|e|es)\b",
    "garlic": r"\b(garlic|alho)\b",
    "basil": r"\b(basil|manjericao)\b",
    "rice": r"\b(rice|arroz)\b",
    "onion": r"\b(onions?|cebolas?)\b",
    "spinach": r"\b(spinach|espinafre)\b",
    "egg": r"\b(eggs?|ovos?)\b",
    "chicken": r"\b(chicken|frango)\b",
    "milk": r"\b(milk|buttermilk|leite)\b",
    "butter": r"\b(butter|manteiga)\b",
    "cheese": r"\b(cheese|queijo)\b",
    "honey": r"\b(honey|mel)\b",
}


@dataclass(frozen=True)
class Scenario:
    name: str
    query: str
    route: str  # tool that B2 must select: NO_TOOL, SEARCH or PANTRY
    result: str  # NO_TOOL, SUCCESS, or FAILURE when a credential is removed
    expected: Tuple[str, ...] = ()  # concepts that must reach search_recipes
    forbidden: Tuple[str, ...] = ()  # concepts that must not reach search_recipes
    unset_env: Tuple[str, ...] = ()  # credentials removed to inject a B6 failure


SCENARIOS: Dict[str, Scenario] = {
    scenario.name: scenario
    for scenario in (
        # B1: invalid queries stop before any tool call.
        Scenario("invalid_empty", "", NO_TOOL, NO_TOOL),
        Scenario("invalid_offensive", "You are a useless idiot. Shut up and answer.", NO_TOOL, NO_TOOL),
        Scenario("invalid_unrelated", "Explain how the stock market works.", NO_TOOL, NO_TOOL),
        # B2 -> B3: plain identification; negation and hypothesis; restriction with derived ingredients.
        Scenario(
            "ingredients_plain",
            "I have tomatoes, garlic and fresh basil. I want a quick dinner.",
            SEARCH,
            SUCCESS,
            expected=("tomato", "garlic", "basil"),
        ),
        Scenario(
            "ingredients_negation_hypothesis",
            "I have rice and onions, but I have no tomatoes. If I had eggs, I would add them. "
            "What can I cook for dinner?",
            SEARCH,
            SUCCESS,
            expected=("rice", "onion"),
            forbidden=("tomato", "egg"),
        ),
        Scenario(
            "ingredients_vegan_derived",
            "I am vegan. I have rice, spinach, chicken and eggs. What can I cook for lunch?",
            SEARCH,
            SUCCESS,
            expected=("rice", "spinach"),
            forbidden=("chicken", "egg", "milk", "butter", "cheese", "honey"),
        ),
        # B2 -> B4: no identified ingredient, or none compatible with the restrictions.
        Scenario("pantry_no_ingredients", "Suggest a dessert I can make tonight.", PANTRY, SUCCESS),
        Scenario(
            "pantry_only_incompatible",
            "I am vegan and I only have chicken and beef. What can I cook?",
            PANTRY,
            SUCCESS,
        ),
        # B6: a credential is removed for the call only, which makes the tool fail.
        Scenario(
            "fault_search_credentials",
            "I have tomatoes, garlic and fresh basil. I want a quick dinner.",
            SEARCH,
            FAILURE,
            unset_env=("TAVILY_API_KEY",),
        ),
        Scenario(
            "fault_pantry_credentials",
            "Suggest a dessert I can make tonight.",
            PANTRY,
            FAILURE,
            unset_env=("NOTION_API_KEY",),
        ),
        Scenario(
            "fault_model_credentials",
            "I have tomatoes, garlic and fresh basil. I want a quick dinner.",
            NO_TOOL,
            NO_TOOL,
            unset_env=("OPENAI_API_KEY",),
        ),
    )
}

SCENARIO_NAMES = list(SCENARIOS)
SEARCH_CASES = ["ingredients_plain", "ingredients_negation_hypothesis", "ingredients_vegan_derived"]
PANTRY_CASES = ["pantry_no_ingredients", "pantry_only_incompatible"]
FAULT_TOOL_CASES = ["fault_search_credentials", "fault_pantry_credentials"]
TOOL_CASES = SEARCH_CASES + PANTRY_CASES + FAULT_TOOL_CASES


@dataclass(frozen=True)
class Run:
    scenario: Scenario
    trace: dict


def _invoke(name: str) -> Run:
    scenario = SCENARIOS[name]
    with pytest.MonkeyPatch.context() as env:
        for variable in scenario.unset_env:
            # The agent is expected to read credentials when it needs them (spec: "when necessary").
            env.delenv(variable, raising=False)
        _output, trace = traced(agent)(scenario.query)  # the output is not evaluated
    return Run(scenario=scenario, trace=trace)


@pytest.fixture(scope="module")
def invalid_empty() -> Run:
    return _invoke("invalid_empty")


@pytest.fixture(scope="module")
def invalid_offensive() -> Run:
    return _invoke("invalid_offensive")


@pytest.fixture(scope="module")
def invalid_unrelated() -> Run:
    return _invoke("invalid_unrelated")


@pytest.fixture(scope="module")
def ingredients_plain() -> Run:
    return _invoke("ingredients_plain")


@pytest.fixture(scope="module")
def ingredients_negation_hypothesis() -> Run:
    return _invoke("ingredients_negation_hypothesis")


@pytest.fixture(scope="module")
def ingredients_vegan_derived() -> Run:
    return _invoke("ingredients_vegan_derived")


@pytest.fixture(scope="module")
def pantry_no_ingredients() -> Run:
    return _invoke("pantry_no_ingredients")


@pytest.fixture(scope="module")
def pantry_only_incompatible() -> Run:
    return _invoke("pantry_only_incompatible")


@pytest.fixture(scope="module")
def fault_search_credentials() -> Run:
    return _invoke("fault_search_credentials")


@pytest.fixture(scope="module")
def fault_pantry_credentials() -> Run:
    return _invoke("fault_pantry_credentials")


@pytest.fixture(scope="module")
def fault_model_credentials() -> Run:
    return _invoke("fault_model_credentials")


@pytest.fixture
def run(request: pytest.FixtureRequest) -> Run:
    return request.getfixturevalue(request.param)


# Trace accessors. They tolerate malformed values so that a broken trace produces a
# violation instead of an exception.


def _field(trace, key: str):
    return trace.get(key) if isinstance(trace, dict) else None


def _parse_time(value) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _calls(trace) -> List[dict]:
    raw = _field(trace, "tool_calls")
    calls = [call for call in raw if isinstance(call, dict)] if isinstance(raw, list) else []
    return sorted(calls, key=lambda call: _parse_time(call.get("start")) or EPOCH)


def _trajectory(trace) -> List[str]:
    raw = _field(trace, "trajectory")
    return [node for node in raw if isinstance(node, str)] if isinstance(raw, list) else []


def _first(trace, tool: str) -> Optional[dict]:
    return next((call for call in _calls(trace) if call.get("name") == tool), None)


def _has_failure_prefix(value) -> bool:
    return isinstance(value, str) and value.lstrip().lower().startswith(FAILURE_PREFIXES)


def _failed(call: dict) -> bool:
    """A tool failed when the trace records an error or the tool returned a failure message."""
    return bool(call.get("error")) or _has_failure_prefix(call.get("output"))


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char)).lower()


def _strings(value) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _strings(item)]
    return []


def _ingredients(tool_input) -> List[str]:
    items = []
    for text in _strings(tool_input):
        items.extend(part.strip() for part in re.split(r"[,;\n]", _normalize(text)) if part.strip())
    return items


def _matches(concept: str, item: str) -> bool:
    return re.search(CONCEPTS[concept], item) is not None


# Requirement checks. Each returns the violations found in the trace; an empty list means satisfied.


def _evidence_violations(run: Run) -> List[str]:
    trace = run.trace
    if not isinstance(trace, dict):
        return ["the benchmark returned no trace mapping"]
    missing = [key for key in ("query", "status", "runs_collected", "trajectory", "tool_calls") if key not in trace]
    if missing:
        return [f"the trace does not record {', '.join(missing)}"]
    violations = []
    if trace["status"] not in ("ok", "error"):
        violations.append(f"the trace status {trace['status']!r} is neither 'ok' nor 'error'")
    if run.scenario.query not in _strings(trace["query"]):
        violations.append("the trace does not record the query of this invocation")
    if not isinstance(trace["runs_collected"], int) or trace["runs_collected"] < 1:
        violations.append("no run was collected")
    if not _trajectory(trace):
        violations.append("the trajectory is missing or empty, so no behavior is recorded")
    if not isinstance(trace["tool_calls"], list):
        violations.append("tool_calls is not a list")
    for call in trace["tool_calls"] if isinstance(trace["tool_calls"], list) else []:
        name = call.get("name") if isinstance(call, dict) else None
        if not isinstance(name, str):
            violations.append(f"a tool call without a tool name is recorded: {call!r:.80}")
            continue
        if not isinstance(call.get("input"), dict):
            violations.append(f"the input of '{name}' is not recorded")
        if _parse_time(call.get("start")) is None or _parse_time(call.get("end")) is None:
            violations.append(f"the start or end time of '{name}' is not recorded")
        if "output" not in call and not call.get("error"):
            violations.append(f"'{name}' records neither a result nor a failure")
    names = [call.get("name") for call in _calls(trace)]
    recorded = [node for node in _trajectory(trace) if node in names]
    if recorded != names:
        violations.append(f"the trajectory records tools {recorded}, but tool_calls records {names}")
    return violations


def _exception_violations(run: Run) -> List[str]:
    status = _field(run.trace, "status")
    if status == "ok":
        return []
    return [f"the run did not complete: status {status!r}, error {_field(run.trace, 'error')!r}"]


def _graph_violations(run: Run) -> List[str]:
    scenario = run.scenario
    calls = _calls(run.trace)
    names = [call.get("name") for call in calls]
    if len(calls) > 1:
        # B3 and B4 both lead to B5, B6 or B7, so no second tool call is allowed.
        return [f"B3 and B4 end the execution after a single tool call, but {names} were called"]
    if scenario.route == NO_TOOL:
        return [f"no tool may be called here (B1 or B6), but {names} were called"] if calls else []
    if not calls:
        return [f"B2 routes to '{scenario.route}', but no tool call is recorded"]
    if names[0] != scenario.route:
        return [f"B2 routes to '{scenario.route}', but '{names[0]}' was called"]
    failed = _failed(calls[0])
    if scenario.result == SUCCESS and failed:
        return [f"'{names[0]}' failed although no credential was removed"]
    if scenario.result == FAILURE and not failed:
        return [f"a credential was removed for '{names[0]}', but no failure is recorded (B6 is not evidenced)"]
    return []


def _ingredient_violations(run: Run) -> List[str]:
    search = _first(run.trace, SEARCH)
    if search is None:
        return ["B3 requires a search_recipes call, but none is recorded"]
    items = _ingredients(search.get("input"))
    if not items:
        return ["search_recipes was called without ingredients, but B3 passes the B2 list"]
    expected, forbidden = run.scenario.expected, run.scenario.forbidden
    violations = []
    for concept in forbidden:
        leaked = [item for item in items if _matches(concept, item)]
        if leaked:
            violations.append(f"'{concept}' must not reach search_recipes, but it received {leaked}")
    for concept in expected:
        if not any(_matches(concept, item) for item in items):
            violations.append(f"'{concept}' is identified in the query, but search_recipes did not receive it: {items}")
    for item in items:
        if not any(_matches(concept, item) for concept in expected + forbidden):
            violations.append(f"'{item}' is not an ingredient identified in the query")
    return violations


def _search_result_violations(run: Run) -> List[str]:
    search = _first(run.trace, SEARCH)
    if search is None:
        return ["B3 requires a search_recipes call, but none is recorded"]
    if _failed(search):
        return ["search_recipes failed although no credential was removed"]
    result = search.get("output")
    if not isinstance(result, list):
        return [f"search_recipes returned {type(result).__name__}, but its result is a list of results"]
    if len(result) > MAX_SEARCH_RESULTS:
        return [f"search_recipes returned {len(result)} results, but at most {MAX_SEARCH_RESULTS} are allowed"]
    return []


def _pantry_result_violations(run: Run) -> List[str]:
    pantry = _first(run.trace, PANTRY)
    if pantry is None:
        return ["B4 requires a retrieve_pantry call, but none is recorded"]
    if _failed(pantry):
        return ["retrieve_pantry failed although no credential was removed"]
    if not isinstance(pantry.get("output"), list):
        return [f"retrieve_pantry returned {type(pantry.get('output')).__name__}, but its result is a list"]
    return []


def _transition_violations(run: Run) -> List[str]:
    tools = {call.get("name") for call in _calls(run.trace)}
    trajectory = _trajectory(run.trace)
    positions = [index for index, node in enumerate(trajectory) if node in tools]
    if not positions:
        return ["the trajectory records no tool execution, so the transition after the tool cannot be evidenced"]
    last = positions[-1]
    if any(node not in tools for node in trajectory[last + 1 :]):
        return []
    return [
        f"the trajectory records no step after '{trajectory[last]}', "
        "so the transition to B5, B6 or B7 cannot be evidenced"
    ]


def _failure_message_violations(run: Run) -> List[str]:
    tool = run.scenario.route
    call = _first(run.trace, tool)
    if call is None:
        return [f"no call to '{tool}' is recorded, so the removed credential cannot be evidenced"]
    if "output" not in call:
        return [f"'{tool}' raised an exception instead of returning an error message"]
    if not _has_failure_prefix(call["output"]):
        return [f"'{tool}' returned {call['output']!r:.120}, not a message beginning with 'Error:' or 'erro:'"]
    return []


# DeepEval adapter. Trace checks run as deterministic metrics, so no judge model is needed.


class TraceRequirement(BaseMetric):
    """Passes when the trace check reports no violation for one requirement of the specification."""

    def __init__(self, requirement: str, check: Callable[[], List[str]], threshold: float = 1.0):
        # DeepEval rebuilds metrics from their constructor arguments, so the attribute names match them.
        self.requirement = requirement
        self.check = check
        self.threshold = threshold
        self.async_mode = False
        self.verbose_mode = False
        self.include_reason = True
        self.strict_mode = False
        self.evaluation_model = None
        self.skipped = False
        self.error = None
        self.score = None
        self.reason = None
        self.success = False

    @property
    def __name__(self) -> str:
        return self.requirement

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        violations = self.check()
        self.score = 0.0 if violations else 1.0
        self.success = self.score >= self.threshold
        self.reason = "; ".join(violations) if violations else "the trace satisfies this requirement"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.success


def _test_case(run: Run, expected_tools: Optional[List[str]]) -> LLMTestCase:
    tools_called = [
        ToolCall(
            name=str(call.get("name")),
            input_parameters=call.get("input") if isinstance(call.get("input"), dict) else None,
        )
        for call in _calls(run.trace)
    ]
    return LLMTestCase(
        input=run.scenario.query,
        tools_called=tools_called,
        expected_tools=None if expected_tools is None else [ToolCall(name=name) for name in expected_tools],
        name=run.scenario.name,
    )


def _assert_trace(
    run: Run,
    requirement: str,
    check: Callable[[Run], List[str]],
    expected_tools: Optional[List[str]] = None,
) -> None:
    metrics: List[BaseMetric] = [TraceRequirement(requirement, functools.partial(check, run))]
    if expected_tools is not None:
        # Exact sequence of tool names, in trace order.
        metrics.insert(0, ToolCorrectnessMetric(threshold=1.0, should_exact_match=True, should_consider_ordering=True))
    assert_test(test_case=_test_case(run, expected_tools), metrics=metrics)


# Tests


@pytest.mark.parametrize("run", SCENARIO_NAMES, indirect=True)
def test_trace_provides_sufficient_evidence(run: Run) -> None:
    """Every verdict below depends on these trace fields. If they are missing, the test fails."""
    _assert_trace(run, "Evidence: the trace records the invocation, trajectory and tool calls", _evidence_violations)


@pytest.mark.parametrize("run", SCENARIO_NAMES, indirect=True)
def test_agent_does_not_terminate_with_unhandled_exception(run: Run) -> None:
    """B6: failures are handled, so every run completes with status 'ok'."""
    _assert_trace(run, "B6: the agent does not terminate with an unhandled exception", _exception_violations)


@pytest.mark.parametrize("run", SCENARIO_NAMES, indirect=True)
def test_trajectory_is_a_path_of_the_behavioral_graph(run: Run) -> None:
    """B1-B7: the observed tool trajectory must be a path of the behavioral graph.

    Invalid queries (B1) and model failures (B6) call no tool. Every valid query calls exactly
    one tool, search_recipes (B3) or retrieve_pantry (B4). Both tool branches end in B5, B6 or B7,
    so the graph has no path with a second tool call.
    """
    expected = [] if run.scenario.route == NO_TOOL else [run.scenario.route]
    _assert_trace(
        run,
        "B1-B7: the observed trajectory is a path of the behavioral graph",
        _graph_violations,
        expected_tools=expected,
    )


@pytest.mark.parametrize("run", SEARCH_CASES, indirect=True)
def test_search_input_is_the_compatible_ingredients_of_the_query(run: Run) -> None:
    """B2/B3: search_recipes receives exactly the identified ingredients that are compatible.

    Negated, absent, hypothetical, and restricted ingredients, including derived ones such as
    eggs or dairy for a vegan user, must not appear. Each identified ingredient must appear.
    """
    _assert_trace(run, "B2/B3: search_recipes receives the identified compatible ingredients", _ingredient_violations)


@pytest.mark.parametrize("run", SEARCH_CASES, indirect=True)
def test_search_result_holds_at_most_five_recipes(run: Run) -> None:
    """Tool contract (B3): search_recipes returns a list of at most five results."""
    _assert_trace(run, "B3 tool result: at most five search results", _search_result_violations)


@pytest.mark.parametrize("run", PANTRY_CASES, indirect=True)
def test_pantry_result_is_a_list_of_ingredients(run: Run) -> None:
    """Tool contract (B4): retrieve_pantry returns a list, which is empty for an empty pantry."""
    _assert_trace(run, "B4 tool result: retrieve_pantry returns a list of ingredients", _pantry_result_violations)


@pytest.mark.parametrize("run", TOOL_CASES, indirect=True)
def test_tool_result_is_followed_by_a_recorded_transition(run: Run) -> None:
    """B3 -> B6 or B7, and B4 -> B5, B6 or B7: the next behavior after a tool result must be recorded."""
    _assert_trace(run, "B3/B4 -> B5/B6/B7: the transition after the tool result is recorded", _transition_violations)


@pytest.mark.parametrize("run", FAULT_TOOL_CASES, indirect=True)
def test_failing_tool_returns_the_specified_error_message(run: Run) -> None:
    """Tool contract (B6): with a missing credential the tool returns a message beginning with 'Error:' or 'erro:'."""
    _assert_trace(run, "B6 tool contract: a failing tool returns an error message", _failure_message_violations)
