"""Trace-based evaluation tests for spec 011_list_prs_or_issues.

The oracle is the behavioral graph of the specification:

    Request -> B7 --unsupported--> Rejection            (no listing tool)
               B7 --supported--> {B1 (list_issues) | B2 (list_pull_requests)}
               B1/B2 -> B3/B5 -> (B4) -> B6 -> Final response   (no extra tools)

Only trace evidence (tool_calls / trajectory / status) is used. The agent output
is never inspected semantically.
"""

import json
import os
from datetime import datetime

import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from trace_capture import traced
from agent import agent

LIST_ISSUES = "list_issues"
LIST_PRS = "list_pull_requests"
LISTING_TOOLS = {LIST_ISSUES, LIST_PRS}

REPO = "octocat/Hello-World"
INVALID_REPO = "octocat/this-repository-does-not-exist-0x011"

SCENARIOS = {
    # B7 (supported) -> B1 -> B3 -> B6
    "issues_open": (REPO, "List the open issues.", LIST_ISSUES),
    # B7 (supported) -> B1 -> B3 (filters) -> B4/B6
    "issues_closed_bug": (REPO, "Show me all closed bug issues.", LIST_ISSUES),
    # B7 (supported) -> B2 -> B3 -> B6
    "prs_open": (REPO, "List the open pull requests.", LIST_PRS),
    # B7 (supported) -> B2 -> B3 (filters) -> B4/B6
    "prs_closed": (REPO, "Show me all closed pull requests.", LIST_PRS),
}

UNSUPPORTED = {
    # B7: both item types simultaneously
    "both_types": (REPO, "List all issues and all pull requests of this repository."),
    # B7: unsupported operation
    "create_issue": (REPO, "Create a new issue titled 'Bug in login page'."),
}


@pytest.fixture(scope="session")
def github_token():
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not token:
        pytest.skip("GITHUB_TOKEN not set")
    return token


_cache = {}


def _run(key, repository, request, token):
    if key not in _cache:
        output, trace = traced(agent)(repository, request, token)
        _cache[key] = (output, trace)
    return _cache[key]


@pytest.fixture(scope="session", params=list(SCENARIOS))
def supported_run(request, github_token):
    repo, user_request, expected_tool = SCENARIOS[request.param]
    _, trace = _run(request.param, repo, user_request, github_token)
    return request.param, repo, expected_tool, trace


@pytest.fixture(scope="session", params=list(UNSUPPORTED))
def unsupported_run(request, github_token):
    repo, user_request = UNSUPPORTED[request.param]
    _, trace = _run(request.param, repo, user_request, github_token)
    return request.param, trace


@pytest.fixture(scope="session")
def invalid_repo_run(github_token):
    _, trace = _run("invalid_repo", INVALID_REPO, "List the open issues.", github_token)
    return trace


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _require_trace(trace):
    assert isinstance(trace, dict), "No trace captured"
    assert "tool_calls" in trace and isinstance(trace["tool_calls"], list), (
        "Trace has no tool_calls evidence"
    )
    assert "trajectory" in trace and isinstance(trace["trajectory"], list), (
        "Trace has no trajectory evidence"
    )
    assert trace.get("runs_collected", 0) > 0, "Trace collected no runs"


def _listing_calls(trace):
    return [c for c in trace["tool_calls"] if c.get("name") in LISTING_TOOLS]


def _ts(value):
    return datetime.fromisoformat(value)


def _tool_correctness(trace, expected):
    test_case = LLMTestCase(
        input=json.dumps(trace.get("query"), default=str),
        actual_output="",
        tools_called=[
            ToolCall(name=c["name"], input_parameters=c.get("input") or {})
            for c in _listing_calls(trace)
        ],
        expected_tools=[ToolCall(name=n) for n in expected],
    )
    metric = ToolCorrectnessMetric(threshold=1.0, should_exact_match=True)
    metric.measure(test_case)
    return metric


# --------------------------------------------------------------------------- #
# Supported requests: B7 -> B1|B2 -> B3 -> (B4) -> B6
# --------------------------------------------------------------------------- #
def test_supported_run_completed(supported_run):
    _, _, _, trace = supported_run
    _require_trace(trace)
    assert trace["status"] == "ok", f"Agent raised: {trace.get('error')}"


def test_selected_tool_invoked_exactly_once(supported_run):
    """B1/B2: the matching listing tool is invoked exactly once."""
    _, _, expected_tool, trace = supported_run
    _require_trace(trace)
    calls = [c for c in trace["tool_calls"] if c.get("name") == expected_tool]
    assert len(calls) == 1, f"{expected_tool} called {len(calls)} times"


def test_other_tool_not_invoked(supported_run):
    """Invariant + B1/B2/B3: the other listing tool must never be invoked."""
    _, _, expected_tool, trace = supported_run
    _require_trace(trace)
    other = (LISTING_TOOLS - {expected_tool}).pop()
    assert not any(c.get("name") == other for c in trace["tool_calls"]), (
        f"{other} must not be invoked for this request"
    )


def test_exactly_one_tool_invocation_per_run(supported_run):
    """Invariant: exactly one tool invocation per execution (B3/B4/B6 add none)."""
    _, _, _, trace = supported_run
    _require_trace(trace)
    assert len(trace["tool_calls"]) == 1, (
        f"Expected exactly one tool call, got {[c.get('name') for c in trace['tool_calls']]}"
    )


def test_tool_correctness_deepeval(supported_run):
    _, _, expected_tool, trace = supported_run
    _require_trace(trace)
    metric = _tool_correctness(trace, [expected_tool])
    assert metric.score >= 1.0, metric.reason


def test_tool_call_targets_requested_repository(supported_run):
    """B1/B2 input: repository identifier owner/repository is passed to the tool."""
    _, repo, expected_tool, trace = supported_run
    _require_trace(trace)
    calls = [c for c in trace["tool_calls"] if c.get("name") == expected_tool]
    assert calls, f"No {expected_tool} call observed in trace"
    serialized = json.dumps(calls[0].get("input") or {}, default=str).lower()
    owner, name = repo.lower().split("/")
    assert owner in serialized and name in serialized, (
        f"Tool input does not reference repository {repo}: {calls[0].get('input')}"
    )


def test_tool_call_has_execution_evidence(supported_run):
    """The tool call must show it actually executed (output or error, valid timing)."""
    _, _, expected_tool, trace = supported_run
    _require_trace(trace)
    calls = [c for c in trace["tool_calls"] if c.get("name") == expected_tool]
    assert calls, f"No {expected_tool} call observed in trace"
    call = calls[0]
    assert "output" in call or "error" in call, "Tool call has neither output nor error"
    assert call.get("start") and call.get("end"), "Tool call lacks timing evidence"
    assert _ts(call["start"]) <= _ts(call["end"]), "Tool call ends before it starts"


def test_tool_appears_in_trajectory(supported_run):
    """The listing tool execution is part of the observed trajectory."""
    _, _, expected_tool, trace = supported_run
    _require_trace(trace)
    assert expected_tool in trace["trajectory"], (
        f"{expected_tool} not present in trajectory {trace['trajectory']}"
    )
    other = (LISTING_TOOLS - {expected_tool}).pop()
    assert other not in trace["trajectory"]


# --------------------------------------------------------------------------- #
# Unsupported requests: B7 -> Rejection (no listing tool)
# --------------------------------------------------------------------------- #
def test_unsupported_run_completed(unsupported_run):
    _, trace = unsupported_run
    _require_trace(trace)
    assert trace["status"] == "ok", f"Agent raised: {trace.get('error')}"


def test_unsupported_request_invokes_no_listing_tool(unsupported_run):
    """B7: must not invoke either listing tool."""
    key, trace = unsupported_run
    _require_trace(trace)
    calls = _listing_calls(trace)
    assert not calls, f"[{key}] listing tools invoked: {[c['name'] for c in calls]}"
    for tool in LISTING_TOOLS:
        assert tool not in trace["trajectory"], f"[{key}] {tool} in trajectory"


def test_unsupported_request_never_invokes_both_tools(unsupported_run):
    """Invariant: invoking both tools in a single run is not allowed."""
    _, trace = unsupported_run
    _require_trace(trace)
    names = {c.get("name") for c in trace["tool_calls"]}
    assert not LISTING_TOOLS <= names


# --------------------------------------------------------------------------- #
# Invalid repository: B7 -> B1 -> B5 -> B6
# --------------------------------------------------------------------------- #
def test_invalid_repo_run_completed(invalid_repo_run):
    _require_trace(invalid_repo_run)
    assert invalid_repo_run["status"] == "ok", (
        f"Agent raised: {invalid_repo_run.get('error')}"
    )


def test_invalid_repo_respects_single_tool_invariant(invalid_repo_run):
    """B5: the selected tool, if invoked, is invoked once; the other is never invoked."""
    trace = invalid_repo_run
    _require_trace(trace)
    assert not any(c.get("name") == LIST_PRS for c in trace["tool_calls"])
    issue_calls = [c for c in trace["tool_calls"] if c.get("name") == LIST_ISSUES]
    assert len(issue_calls) <= 1, f"list_issues called {len(issue_calls)} times"
    assert len(trace["tool_calls"]) <= 1
