"""Trace-based evaluation tests for the UFCG academic data agent (spec 029).

The agent is invoked only through the benchmark wrapper `traced`, and every
assertion is made on the captured execution trace (tool_calls / status),
using the behavioral graph of the specification as the oracle. No semantic
validation of the agent output is performed.
"""
import json
import re

import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from trace_capture import traced
from agent import agent

SPEC_TOOLS = {
    "get_courses",
    "get_campi",
    "get_calendars",
    "get_curricula",
    "get_disciplines",
    "get_schedules",
}
PERIOD_RE = re.compile(r"^\d{4}\.[12]$")

_cache = {}


def run(request: str):
    if request not in _cache:
        _cache[request] = traced(agent)(request)
    return _cache[request]


# --------------------------------------------------------------- helpers
def calls(trace):
    tc = trace.get("tool_calls")
    assert isinstance(tc, list), "trace has no tool_calls list: insufficient evidence"
    return tc


def calls_named(trace, name):
    return [c for c in calls(trace) if c.get("name") == name]


def successful(trace, name):
    return [c for c in calls_named(trace, name) if "error" not in c]


def scalars(obj):
    """All scalar values (as strings) found in a nested structure."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from scalars(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from scalars(v)
    elif isinstance(obj, str) and obj.strip().startswith(("[", "{")):
        try:
            yield from scalars(json.loads(obj))
        except ValueError:
            yield obj.strip()
    elif obj is not None:
        yield str(obj).strip()


def input_values(call):
    return {v for v in scalars(call.get("input") or {}) if v}


def output_values(call):
    return {v for v in scalars(call.get("output")) if v}


def first_index(trace, name):
    for i, c in enumerate(calls(trace)):
        if c.get("name") == name:
            return i
    return None


def assert_tools_called(trace, query, expected):
    """DeepEval deterministic tool correctness: expected tools must be called."""
    tools_called = [ToolCall(name=c["name"]) for c in calls(trace)]
    assert tools_called, "no tool calls observed in trace: insufficient evidence"
    test_case = LLMTestCase(
        input=query,
        actual_output="",
        tools_called=tools_called,
        expected_tools=[ToolCall(name=n) for n in expected],
    )
    metric = ToolCorrectnessMetric(threshold=1.0, should_exact_match=False)
    metric.measure(test_case)
    assert metric.score >= 1.0, metric.reason


def assert_ok(trace):
    assert trace.get("status") == "ok", f"agent raised: {trace.get('error')}"


# --------------------------------------------------------------- scenarios
Q_COURSES = "Quais cursos de graduação a UFCG oferece no campus de Campina Grande?"
Q_CAMPI = "Quais são os campi da UFCG?"
Q_CALENDAR = "Qual é o calendário acadêmico da UFCG para o período 2023.1?"
Q_CURRICULA = "Quais são os currículos do curso de Ciência da Computação da UFCG?"
Q_DISCIPLINES = "Quais disciplinas fazem parte do curso de Ciência da Computação da UFCG?"
Q_SCHEDULE_BY_NAME = (
    "Qual é o horário da disciplina Teoria da Computação do curso de "
    "Ciência da Computação no período 2023.1?"
)
Q_SCHEDULE_CODE = "Qual é o horário da disciplina de código 1411174, turma 01, no período 2023.1?"
Q_SCHEDULE_NO_PERIOD = "Qual é o horário da disciplina de código 1411174?"
Q_SCHEDULE_BAD_PERIOD = "Qual é o horário da disciplina de código 1411174 no período 2023.7?"
Q_UNSUPPORTED = "Qual é a previsão do tempo para amanhã em Recife?"
Q_EMPTY = ""

ALL_QUERIES = [
    Q_COURSES, Q_CAMPI, Q_CALENDAR, Q_CURRICULA, Q_DISCIPLINES,
    Q_SCHEDULE_BY_NAME, Q_SCHEDULE_CODE, Q_SCHEDULE_NO_PERIOD,
    Q_SCHEDULE_BAD_PERIOD, Q_UNSUPPORTED, Q_EMPTY,
]


# --------------------------------------------------------------- global invariants
@pytest.mark.parametrize("query", ALL_QUERIES)
def test_only_specified_tools_are_called(query):
    _, trace = run(query)
    for c in calls(trace):
        assert c.get("name") in SPEC_TOOLS, f"unexpected tool {c.get('name')}"


@pytest.mark.parametrize("query", ALL_QUERIES)
def test_every_get_disciplines_call_has_grounded_course_code(query):
    """Invariant: course code supplied by user or found through get_courses;
    curriculum (when not user-specified) found through get_curricula."""
    _, trace = run(query)
    tc = calls(trace)
    for i, c in enumerate(tc):
        if c.get("name") != "get_disciplines":
            continue
        vals = input_values(c)
        assert vals, "get_disciplines called without any course code"
        prior_courses = set()
        prior_curricula = set()
        for p in tc[:i]:
            if p.get("name") == "get_courses" and "error" not in p:
                prior_courses |= output_values(p)
            if p.get("name") == "get_curricula" and "error" not in p:
                prior_curricula |= output_values(p)
        grounded = {v for v in vals if v in query or v in prior_courses}
        assert grounded, (
            f"get_disciplines input {vals} not traceable to user request or get_courses"
        )
        ungrounded = vals - {v for v in vals if v in query or v in prior_courses or v in prior_curricula}
        assert not ungrounded, (
            f"get_disciplines input values {ungrounded} not supplied by user nor "
            "returned by get_courses/get_curricula"
        )


@pytest.mark.parametrize("query", ALL_QUERIES)
def test_every_get_schedules_call_has_valid_period_and_grounded_code(query):
    _, trace = run(query)
    tc = calls(trace)
    for i, c in enumerate(tc):
        if c.get("name") != "get_schedules":
            continue
        vals = input_values(c)
        assert any(PERIOD_RE.match(v) for v in vals), (
            f"get_schedules called without a YYYY.1/YYYY.2 period: {vals}"
        )
        prior_disc = set()
        for p in tc[:i]:
            if p.get("name") == "get_disciplines" and "error" not in p:
                prior_disc |= output_values(p)
        codes = {v for v in vals if not PERIOD_RE.match(v)}
        assert any(v in query or v in prior_disc for v in codes), (
            f"get_schedules discipline code in {codes} not from user nor get_disciplines"
        )


# --------------------------------------------------------------- B1..B4
def test_b1_courses_query_calls_get_courses():
    _, trace = run(Q_COURSES)
    assert_ok(trace)
    assert_tools_called(trace, Q_COURSES, ["get_courses"])


def test_b2_campi_query_calls_get_campi():
    _, trace = run(Q_CAMPI)
    assert_ok(trace)
    assert_tools_called(trace, Q_CAMPI, ["get_campi"])


def test_b3_calendar_query_calls_get_calendars():
    _, trace = run(Q_CALENDAR)
    assert_ok(trace)
    assert_tools_called(trace, Q_CALENDAR, ["get_calendars"])


def test_b4_curricula_query_calls_get_curricula():
    _, trace = run(Q_CURRICULA)
    assert_ok(trace)
    assert_tools_called(trace, Q_CURRICULA, ["get_curricula"])


# --------------------------------------------------------------- B9 -> B10 -> B11
def test_disciplines_by_course_name_follow_b9_b10_b11_order():
    _, trace = run(Q_DISCIPLINES)
    assert_ok(trace)
    assert_tools_called(trace, Q_DISCIPLINES, ["get_courses"])
    if not successful(trace, "get_courses") or not any(
        output_values(c) for c in successful(trace, "get_courses")
    ):
        pytest.fail("get_courses produced no usable result: cannot reach B11 per graph")
    i_courses = first_index(trace, "get_courses")
    i_curr = first_index(trace, "get_curricula")
    i_disc = first_index(trace, "get_disciplines")
    if i_disc is not None:
        assert i_curr is not None, "get_disciplines called without get_curricula (B10)"
        assert i_courses < i_curr < i_disc, "expected order get_courses -> get_curricula -> get_disciplines"


# --------------------------------------------------------------- schedule flows
def test_schedule_by_name_resolves_discipline_before_get_schedules():
    _, trace = run(Q_SCHEDULE_BY_NAME)
    assert_ok(trace)
    assert calls(trace), "no tool calls: name resolution not evidenced"
    i_sched = first_index(trace, "get_schedules")
    if i_sched is not None:
        i_disc = first_index(trace, "get_disciplines")
        assert i_disc is not None and i_disc < i_sched, (
            "get_schedules called before resolving discipline code via get_disciplines"
        )
        i_courses = first_index(trace, "get_courses")
        assert i_courses is not None and i_courses < i_disc, (
            "course code for get_disciplines not resolved via get_courses first"
        )
        for c in calls_named(trace, "get_schedules"):
            assert "2023.1" in input_values(c), "requested period not passed to get_schedules"
    else:
        assert first_index(trace, "get_disciplines") is not None or first_index(
            trace, "get_courses"
        ) is not None, "no resolution attempt observed"


def test_b13_schedule_with_code_and_class():
    _, trace = run(Q_SCHEDULE_CODE)
    assert_ok(trace)
    assert_tools_called(trace, Q_SCHEDULE_CODE, ["get_schedules"])
    for c in calls_named(trace, "get_schedules"):
        vals = input_values(c)
        assert "1411174" in vals, f"discipline code not passed: {vals}"
        assert "2023.1" in vals, f"requested period not passed: {vals}"
        assert any(v in ("01", "1") for v in vals), f"class number not passed: {vals}"


def test_b13_no_class_filter_when_not_requested():
    _, trace = run(Q_SCHEDULE_BY_NAME)
    for c in calls_named(trace, "get_schedules"):
        extras = {v for v in input_values(c) if re.fullmatch(r"0?\d{1,2}", v)}
        assert not extras, f"class filter applied without user request: {extras}"


# --------------------------------------------------------------- B7
@pytest.mark.parametrize("query", [Q_SCHEDULE_NO_PERIOD, Q_SCHEDULE_BAD_PERIOD])
def test_b7_schedule_without_valid_period_does_not_call_get_schedules(query):
    _, trace = run(query)
    assert_ok(trace)
    assert not calls_named(trace, "get_schedules"), "get_schedules called without valid period"


@pytest.mark.parametrize("query", [Q_UNSUPPORTED, Q_EMPTY])
def test_b7_unsupported_or_empty_input_accesses_no_data(query):
    _, trace = run(query)
    assert_ok(trace)
    assert calls(trace) == [], "institutional data accessed for unsupported/empty input"


# --------------------------------------------------------------- B6 / B8
@pytest.mark.parametrize("query", ALL_QUERIES)
def test_b8_no_substitute_lookup_after_failure(query):
    """After a tool failure the agent must not substitute another lookup
    for the same tool (no retry with a different tool as data source)."""
    _, trace = run(query)
    tc = calls(trace)
    for i, c in enumerate(tc):
        if "error" in c:
            later = [x.get("name") for x in tc[i + 1:]]
            dependent = {
                "get_courses": {"get_curricula", "get_disciplines", "get_schedules"},
                "get_curricula": {"get_disciplines", "get_schedules"},
                "get_disciplines": {"get_schedules"},
            }.get(c.get("name"), set())
            assert not dependent.intersection(later), (
                f"agent proceeded with dependent queries after {c.get('name')} failed"
            )


@pytest.mark.parametrize("query", ALL_QUERIES)
def test_b6_no_dependent_query_after_empty_result(query):
    _, trace = run(query)
    tc = calls(trace)
    dependent = {
        "get_disciplines": {"get_schedules"},
    }
    for i, c in enumerate(tc):
        if "error" not in c and c.get("output") in ([], "[]"):
            later = [x.get("name") for x in tc[i + 1:]]
            blocked = dependent.get(c.get("name"), set())
            # only applies if every get_disciplines call returned empty
            if c.get("name") == "get_disciplines" and all(
                d.get("output") in ([], "[]") for d in calls_named(trace, "get_disciplines")
            ):
                assert not blocked.intersection(later), (
                    "get_schedules called although discipline lookup returned no records"
                )
