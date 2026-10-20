"""Trace-based evaluation tests for the UFCG academic data agent (spec 019).

The behavioral graph in the specification is the oracle. All evidence comes
from the trace returned by the benchmark wrapper `traced(agent)(...)`.
No semantic validation of the agent output is performed.
"""

import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from trace_capture import traced
from agent import agent

LOOKUP_TOOLS = {"get_courses", "get_campi", "get_calendars", "get_curricula"}
RESOLUTION_TOOLS = {"get_courses", "get_campi"}  # B5


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _run(user_request):
    output, trace = traced(agent)(user_request)
    return {"request": user_request, "output": output, "trace": trace}


def _tool_calls(trace):
    calls = trace.get("tool_calls")
    assert isinstance(calls, list), "Trace has no 'tool_calls' list; insufficient evidence."
    return calls


def _names(trace):
    return [c.get("name") for c in _tool_calls(trace)]


def _succeeded(call):
    return "error" not in call and "output" in call


def _assert_trace_ok(trace):
    assert isinstance(trace, dict), "Trace missing."
    assert trace.get("status") == "ok", f"Agent raised: {trace.get('error')}"
    assert isinstance(trace.get("trajectory"), list) and trace["trajectory"], (
        "Empty trajectory: insufficient evidence of execution."
    )


def _assert_tool_correctness(run, expected):
    """DeepEval ToolCorrectnessMetric: expected tools must appear in tools_called."""
    calls = _tool_calls(run["trace"])
    called = [
        ToolCall(name=c["name"], input_parameters=c.get("input") or {}, output=c.get("output"))
        for c in calls
    ]
    test_case = LLMTestCase(
        input=run["request"],
        actual_output=str(run["output"]),
        tools_called=called,
        expected_tools=[ToolCall(name=n) for n in expected],
    )
    metric = ToolCorrectnessMetric(threshold=1.0)
    metric.measure(test_case)
    assert metric.score >= 1.0, (
        f"Expected tools {expected} not observed in trace; called={_names(run['trace'])}. "
        f"{metric.reason}"
    )


def _first_index(names, tool):
    return names.index(tool) if tool in names else None


# --------------------------------------------------------------------------- #
# Scenarios (one agent invocation each)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def run_courses():
    return _run("Quais cursos de graduação ativos a UFCG oferece?")


@pytest.fixture(scope="module")
def run_campi():
    return _run("Quais são os campi da UFCG?")


@pytest.fixture(scope="module")
def run_calendars():
    return _run("Quais são os calendários acadêmicos da UFCG entre os períodos 2023.1 e 2023.2?")


@pytest.fixture(scope="module")
def run_curricula_by_course_name():
    return _run("Qual a carga horária do currículo do curso de Ciência da Computação da UFCG?")


@pytest.fixture(scope="module")
def run_empty_message():
    return _run("")


@pytest.fixture(scope="module")
def run_out_of_scope():
    return _run("Qual é a previsão do tempo para amanhã em Recife?")


ALL_RUNS = [
    "run_courses",
    "run_campi",
    "run_calendars",
    "run_curricula_by_course_name",
    "run_empty_message",
    "run_out_of_scope",
]


# --------------------------------------------------------------------------- #
# Global trace invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fixture_name", ALL_RUNS)
def test_execution_completed_and_trace_present(fixture_name, request):
    run = request.getfixturevalue(fixture_name)
    _assert_trace_ok(run["trace"])
    assert isinstance(run["output"], dict) and "response" in run["output"]


@pytest.mark.parametrize("fixture_name", ALL_RUNS)
def test_only_specified_tools_are_called(fixture_name, request):
    """Only get_courses, get_campi, get_calendars and get_curricula exist in the spec."""
    run = request.getfixturevalue(fixture_name)
    _assert_trace_ok(run["trace"])
    unknown = [n for n in _names(run["trace"]) if n not in LOOKUP_TOOLS]
    assert not unknown, f"Tools outside the specification were called: {unknown}"


@pytest.mark.parametrize("fixture_name", ALL_RUNS)
def test_tool_call_records_are_complete(fixture_name, request):
    """Each tool event must carry evidence: either an output or an error, and timing."""
    run = request.getfixturevalue(fixture_name)
    for c in _tool_calls(run["trace"]):
        assert c.get("name"), "Tool call without name."
        assert isinstance(c.get("input"), dict), f"Tool call {c.get('name')} without input."
        assert ("output" in c) or ("error" in c), f"Tool call {c['name']} lacks output/error."
        assert c.get("start") and c.get("end"), f"Tool call {c['name']} lacks timing."
        assert c["start"] <= c["end"], f"Tool call {c['name']} ends before it starts."


@pytest.mark.parametrize("fixture_name", ALL_RUNS)
def test_tool_calls_ordered_by_start(fixture_name, request):
    run = request.getfixturevalue(fixture_name)
    starts = [c["start"] for c in _tool_calls(run["trace"])]
    assert starts == sorted(starts), "Tool calls are not ordered by start time."


@pytest.mark.parametrize("fixture_name", ALL_RUNS)
def test_tool_calls_reflected_in_trajectory(fixture_name, request):
    run = request.getfixturevalue(fixture_name)
    trajectory = run["trace"].get("trajectory") or []
    for name in set(_names(run["trace"])):
        assert name in trajectory, f"Tool {name} in tool_calls but absent from trajectory."


@pytest.mark.parametrize("fixture_name", ALL_RUNS)
def test_b8_no_substitute_lookup_after_failure(fixture_name, request):
    """B8: after a tool failure, no substitute lookup with a different tool."""
    run = request.getfixturevalue(fixture_name)
    calls = _tool_calls(run["trace"])
    for i, c in enumerate(calls):
        if "error" in c:
            later = [x["name"] for x in calls[i + 1:] if x["name"] != c["name"]]
            assert not later, (
                f"Substitute lookups {later} after failure of {c['name']} (violates B8)."
            )


# --------------------------------------------------------------------------- #
# B1 — Query courses
# --------------------------------------------------------------------------- #
def test_b1_courses_tool_called(run_courses):
    _assert_trace_ok(run_courses["trace"])
    _assert_tool_correctness(run_courses, ["get_courses"])


# --------------------------------------------------------------------------- #
# B2 — Query campuses
# --------------------------------------------------------------------------- #
def test_b2_campi_tool_called(run_campi):
    _assert_trace_ok(run_campi["trace"])
    _assert_tool_correctness(run_campi, ["get_campi"])


def test_b2_get_campi_called_without_arguments(run_campi):
    """get_campi takes no inputs per the specification."""
    calls = [c for c in _tool_calls(run_campi["trace"]) if c["name"] == "get_campi"]
    assert calls, "get_campi not observed in trace."
    for c in calls:
        assert not (c.get("input") or {}), f"get_campi called with arguments: {c['input']}"


# --------------------------------------------------------------------------- #
# B3 — Query academic calendars
# --------------------------------------------------------------------------- #
def test_b3_calendars_tool_called(run_calendars):
    _assert_trace_ok(run_calendars["trace"])
    _assert_tool_correctness(run_calendars, ["get_calendars"])


# --------------------------------------------------------------------------- #
# B5 -> B4 — Resolve course reference, then query curricula
# --------------------------------------------------------------------------- #
def test_b5_course_reference_resolved_with_get_courses(run_curricula_by_course_name):
    _assert_trace_ok(run_curricula_by_course_name["trace"])
    _assert_tool_correctness(run_curricula_by_course_name, ["get_courses"])


def test_b5_resolution_precedes_curricula_query(run_curricula_by_course_name):
    """If get_curricula runs, a successful get_courses resolution must precede it."""
    calls = _tool_calls(run_curricula_by_course_name["trace"])
    names = [c["name"] for c in calls]
    idx_cur = _first_index(names, "get_curricula")
    idx_res = _first_index(names, "get_courses")
    assert idx_res is not None, "No course resolution (get_courses) observed in trace."
    if idx_cur is not None:
        assert idx_res < idx_cur, "get_curricula executed before course resolution (B5)."
        assert any(_succeeded(c) for c in calls[:idx_cur] if c["name"] == "get_courses"), (
            "get_curricula executed without a successful prior get_courses resolution."
        )


def test_b5_failed_resolution_stops_dependent_query(run_curricula_by_course_name):
    """B5 -> B8: if every resolution attempt failed, the dependent query must not run."""
    calls = _tool_calls(run_curricula_by_course_name["trace"])
    names = [c["name"] for c in calls]
    idx_cur = _first_index(names, "get_curricula")
    resolutions = [c for c in calls[: idx_cur if idx_cur is not None else len(calls)]
                   if c["name"] == "get_courses"]
    if resolutions and all("error" in c for c in resolutions):
        assert idx_cur is None, "get_curricula executed after failed course resolution."


def test_b5_empty_resolution_stops_dependent_query(run_curricula_by_course_name):
    """B5 -> B6: if resolution returned an empty list, the dependent query must not run."""
    calls = _tool_calls(run_curricula_by_course_name["trace"])
    names = [c["name"] for c in calls]
    idx_cur = _first_index(names, "get_curricula")
    resolutions = [c for c in calls[: idx_cur if idx_cur is not None else len(calls)]
                   if c["name"] == "get_courses" and _succeeded(c)]
    if resolutions and all(c.get("output") in ([], ()) for c in resolutions):
        assert idx_cur is None, "get_curricula executed although course resolution had no match."


# --------------------------------------------------------------------------- #
# B7 — Invalid or unsupported input: no lookup tools
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fixture_name", ["run_empty_message", "run_out_of_scope"])
def test_b7_no_lookup_tool_invoked(fixture_name, request):
    run = request.getfixturevalue(fixture_name)
    _assert_trace_ok(run["trace"])
    called = [n for n in _names(run["trace"]) if n in LOOKUP_TOOLS]
    assert not called, f"Lookup tools invoked for invalid/unsupported input: {called}"
