"""Trace-based evaluation tests for the Autonomous Research Agent (spec 024).

Oracle: the Behavioral Graph of the specification.
All evidence comes from the trace returned by the benchmark wrapper
`traced(agent)(user_request)`. No semantic validation of the output is done.
"""

import json

import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from trace_capture import traced
from agent import agent

T1 = "get_wikipedia_summary"
T2 = "search_arxiv"
T3 = "search_pubmed"
STAGE2 = {T2, T3}
ALL_TOOLS = {T1, T2, T3}

REQUESTS = {
    "life_sciences": (
        "Find out which gene is most associated with hereditary breast cancer "
        "(BRCA1) and then look for recent research papers about it."
    ),
    "exact_sciences": (
        "Identify the famous theorem in number theory proved by Andrew Wiles "
        "(Fermat's Last Theorem) and find academic papers about it."
    ),
    "humanities": (
        "Discover the Renaissance painting known as the Mona Lisa and then "
        "find academic papers related to it."
    ),
    "not_found": (
        "Investigate the entity Zqxvbrtlmnop Kwyjibo Flarnstead Theorem 9481 "
        "and then find academic papers about it."
    ),
}


# --------------------------------------------------------------------------- #
# Fixtures: each scenario invokes the agent exactly once.
# --------------------------------------------------------------------------- #
def _run(request):
    output, trace = traced(agent)(request)
    assert isinstance(trace, dict), "Trace not captured"
    return output, trace


@pytest.fixture(scope="module")
def life_run():
    return _run(REQUESTS["life_sciences"])


@pytest.fixture(scope="module")
def exact_run():
    return _run(REQUESTS["exact_sciences"])


@pytest.fixture(scope="module")
def humanities_run():
    return _run(REQUESTS["humanities"])


@pytest.fixture(scope="module")
def not_found_run():
    return _run(REQUESTS["not_found"])


@pytest.fixture(scope="module")
def empty_run():
    return _run("")


@pytest.fixture(scope="module")
def whitespace_run():
    return _run("   \n\t ")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _calls(trace):
    calls = trace.get("tool_calls")
    assert isinstance(calls, list), "Trace has no 'tool_calls' list"
    return calls


def _names(trace):
    return [c.get("name") for c in _calls(trace)]


def _require_valid_trace(trace):
    assert trace.get("status") == "ok", f"Agent raised: {trace.get('error')}"
    assert trace.get("runs_collected", 0) > 0, "No runs collected in trace"
    assert isinstance(trace.get("trajectory"), list) and trace["trajectory"], (
        "Empty trajectory: insufficient evidence"
    )


def _first_index(names, targets):
    for i, n in enumerate(names):
        if n in targets:
            return i
    return None


def _tool_trajectory(trace):
    """Ordered tool events from the trajectory (nodes are ignored)."""
    return [s for s in trace.get("trajectory", []) if s in ALL_TOOLS]


def _search_term(call):
    inp = call.get("input")
    assert isinstance(inp, dict) and inp, "Stage 2 tool call has no input"
    values = [v for v in inp.values() if isinstance(v, str) and v.strip()]
    assert values, "Stage 2 tool call has no string search term"
    return values[0]


def _t1_failed(call):
    """T1 failure observable in trace: error field or explicit no-result."""
    if "error" in call:
        return True
    out = call.get("output")
    if out is None:
        return True
    text = out if isinstance(out, str) else json.dumps(out, default=str)
    low = text.lower()
    return any(k in low for k in ("not found", "error", "no result", "404"))


def _assert_tool_correctness(trace, expected):
    """DeepEval deterministic tool-correctness check over observed tool calls."""
    test_case = LLMTestCase(
        input=json.dumps(trace.get("query"), default=str),
        actual_output="<not evaluated>",
        tools_called=[ToolCall(name=n) for n in _names(trace)],
        expected_tools=[ToolCall(name=n) for n in expected],
    )
    metric = ToolCorrectnessMetric(threshold=1.0)
    metric.measure(test_case)
    assert metric.score >= 1.0, metric.reason


# --------------------------------------------------------------------------- #
# Generic graph-path checks (valid inputs)
# --------------------------------------------------------------------------- #
VALID_RUNS = ["life_run", "exact_run", "humanities_run", "not_found_run"]


@pytest.mark.parametrize("run", VALID_RUNS)
def test_execution_completed_with_trace_evidence(run, request):
    _, trace = request.getfixturevalue(run)
    _require_valid_trace(trace)
    assert _calls(trace), "No tool calls recorded: insufficient evidence"


@pytest.mark.parametrize("run", VALID_RUNS)
def test_b1_discovery_tool_called(run, request):
    _, trace = request.getfixturevalue(run)
    _require_valid_trace(trace)
    assert T1 in _names(trace), f"{T1} (Stage 1) was not called"


@pytest.mark.parametrize("run", VALID_RUNS)
def test_b1_investigation_begins_with_discovery(run, request):
    """B1: investigation must begin with get_wikipedia_summary."""
    _, trace = request.getfixturevalue(run)
    _require_valid_trace(trace)
    names = _names(trace)
    assert names, "No tool calls recorded"
    assert names[0] == T1, f"First tool call was {names[0]!r}, expected {T1}"
    traj_tools = _tool_trajectory(trace)
    assert traj_tools and traj_tools[0] == T1, (
        f"Trajectory does not start investigation with {T1}: {traj_tools}"
    )


@pytest.mark.parametrize("run", VALID_RUNS)
def test_b3_stage2_called_after_stage1_completes(run, request):
    """B1/B3: no In-depth Research before Discovery completes."""
    _, trace = request.getfixturevalue(run)
    _require_valid_trace(trace)
    calls = _calls(trace)
    names = [c.get("name") for c in calls]
    i1 = _first_index(names, {T1})
    i2 = _first_index(names, STAGE2)
    assert i1 is not None, f"{T1} not called"
    assert i2 is not None, "No Stage 2 tool (search_arxiv/search_pubmed) called"
    assert i1 < i2
    t1_end = calls[i1].get("end")
    s2_start = calls[i2].get("start")
    assert t1_end and s2_start, "Missing timestamps: insufficient evidence"
    assert t1_end <= s2_start, "Stage 2 started before Stage 1 completed"


@pytest.mark.parametrize("run", VALID_RUNS)
def test_stage2_tools_produced_output(run, request):
    """B3 -> B4 transition requires a completed Stage 2 call."""
    _, trace = request.getfixturevalue(run)
    _require_valid_trace(trace)
    s2 = [c for c in _calls(trace) if c.get("name") in STAGE2]
    assert s2, "No Stage 2 call"
    assert any("output" in c and "error" not in c for c in s2), (
        "No Stage 2 call completed successfully"
    )


@pytest.mark.parametrize("run", VALID_RUNS)
def test_single_database_branch(run, request):
    """Graph: entity domain selects exactly one branch (T2 xor T3)."""
    _, trace = request.getfixturevalue(run)
    _require_valid_trace(trace)
    used = {n for n in _names(trace) if n in STAGE2}
    assert len(used) == 1, f"Expected exactly one database branch, got {used}"


@pytest.mark.parametrize("run", VALID_RUNS)
def test_no_unknown_tools(run, request):
    _, trace = request.getfixturevalue(run)
    _require_valid_trace(trace)
    unknown = set(_names(trace)) - ALL_TOOLS
    assert not unknown, f"Tools outside the specification: {unknown}"


# --------------------------------------------------------------------------- #
# I1 — Domain-to-database invariant
# --------------------------------------------------------------------------- #
def test_i1_life_sciences_uses_pubmed(life_run):
    _, trace = life_run
    _require_valid_trace(trace)
    names = _names(trace)
    assert T3 in names, "Life Sciences entity must be searched on PubMed"
    assert T2 not in names, "Life Sciences entity must not use ArXiv"
    _assert_tool_correctness(trace, [T1, T3])


def test_i1_exact_sciences_uses_arxiv(exact_run):
    _, trace = exact_run
    _require_valid_trace(trace)
    names = _names(trace)
    assert T2 in names, "Exact Sciences entity must be searched on ArXiv"
    assert T3 not in names, "Exact Sciences entity must not use PubMed"
    _assert_tool_correctness(trace, [T1, T2])


def test_i1_other_domain_defaults_to_arxiv(humanities_run):
    _, trace = humanities_run
    _require_valid_trace(trace)
    names = _names(trace)
    assert T2 in names, "Other-domain entity must default to ArXiv"
    assert T3 not in names, "Other-domain entity must not use PubMed"
    _assert_tool_correctness(trace, [T1, T2])


# --------------------------------------------------------------------------- #
# QA1 — Data chaining: Stage 2 term == entity discovered in Stage 1
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("run", ["life_run", "exact_run", "humanities_run"])
def test_qa1_stage2_term_chained_from_stage1(run, request):
    _, trace = request.getfixturevalue(run)
    _require_valid_trace(trace)
    calls = _calls(trace)
    t1 = next((c for c in calls if c.get("name") == T1), None)
    s2 = next((c for c in calls if c.get("name") in STAGE2), None)
    assert t1 is not None and s2 is not None, "Missing Stage 1 or Stage 2 call"
    assert not _t1_failed(t1), "Stage 1 failed; chaining cannot be evidenced"

    term = _search_term(s2).strip().lower()
    t1_out = t1.get("output")
    t1_text = (
        t1_out if isinstance(t1_out, str) else json.dumps(t1_out, default=str)
    ).lower()
    t1_inputs = [
        v.strip().lower()
        for v in (t1.get("input") or {}).values()
        if isinstance(v, str)
    ]
    assert term in t1_text or term in t1_inputs, (
        f"Stage 2 term {term!r} is not evidenced as the Stage 1 entity"
    )
    query = trace.get("query")
    original = query[0] if isinstance(query, list) and query else str(query)
    assert term != str(original).strip().lower(), (
        "Stage 2 used the original user request instead of the entity"
    )


# --------------------------------------------------------------------------- #
# B2 — Discovery failure: infer entity and proceed (no abort)
# --------------------------------------------------------------------------- #
def test_b2_discovery_failure_proceeds_to_stage2(not_found_run):
    _, trace = not_found_run
    _require_valid_trace(trace)
    calls = _calls(trace)
    names = [c.get("name") for c in calls]
    t1_calls = [c for c in calls if c.get("name") == T1]
    assert t1_calls, f"{T1} not called"
    assert any(_t1_failed(c) for c in t1_calls), (
        "Trace does not evidence a Stage 1 failure/no-result for this scenario"
    )
    i2 = _first_index(names, STAGE2)
    assert i2 is not None, "Agent aborted: no Stage 2 call after failed discovery"
    assert _first_index(names, {T1}) < i2
    term = _search_term(calls[i2]).strip()
    assert term, "Stage 2 called without an inferred entity"


# --------------------------------------------------------------------------- #
# B5 — Invalid input: return error without entering the investigation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("run", ["empty_run", "whitespace_run"])
def test_b5_invalid_input_calls_no_tools(run, request):
    _, trace = request.getfixturevalue(run)
    assert trace.get("status") == "ok", (
        f"Agent raised instead of returning error JSON: {trace.get('error')}"
    )
    assert _calls(trace) == [], (
        f"Tools called for invalid input: {_names(trace)}"
    )
    assert not _tool_trajectory(trace), "Tool events in trajectory for invalid input"
