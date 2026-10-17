"""Trace-based evaluation for spec 021 - classify PRs by review status.

The behavioral graph from the spec is the oracle:

    B1 (list_pull_requests) -> B2 -> B3 (get_pull_request_reviews) -> B4 -> B7
    B1 -> B5 (empty)  -> B7
    B1 -> B6 (error)  -> B7
    B3 -> B6 (error)  -> B7

Only tool calls in the benchmark trace are used as evidence. LangGraph node
names are not part of the spec, so the tests never depend on them.

Environment variables:
    GITHUB_TOKEN             token with read permissions (required)
    TRACE_REPO_WITH_PRS      repo with open PRs (default: langchain-ai/langgraph)
    TRACE_REPO_EMPTY         repo with no open PRs (empty-result scenario is
                             skipped if this is not set)
    TRACE_REPO_INVALID       repo that does not exist
"""

import os

import pytest
from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from trace_capture import traced
from agent import agent

LIST_PRS = "list_pull_requests"
GET_REVIEWS = "get_pull_request_reviews"
LIST_ISSUES = "list_issues"
ALLOWED_TOOLS = {LIST_PRS, GET_REVIEWS}
MAX_SELECTED = 3

REQUEST = "Which open pull requests still need review and which have already been reviewed?"

REPO_WITH_PRS = os.getenv("TRACE_REPO_WITH_PRS", "langchain-ai/langgraph")
REPO_EMPTY = os.getenv("TRACE_REPO_EMPTY")
REPO_INVALID = os.getenv(
    "TRACE_REPO_INVALID", "this-owner-does-not-exist-021/this-repo-does-not-exist-021"
)


# --------------------------------------------------------------------------- #
# Helpers (read the trace only)
# --------------------------------------------------------------------------- #
def _token():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        pytest.skip("GITHUB_TOKEN not set")
    return token


def _run(repository):
    _output, trace = traced(agent)(repository, REQUEST, _token())
    assert isinstance(trace, dict), "Trace was not produced"
    assert isinstance(trace.get("tool_calls"), list), "Trace has no tool_calls list"
    return trace


def _calls(trace, name=None):
    return [c for c in trace["tool_calls"] if name is None or c.get("name") == name]


def _index(trace, name):
    return [i for i, c in enumerate(trace["tool_calls"]) if c.get("name") == name]


def _failed(call):
    return "error" in call and call["error"]


def _numbers_in(obj):
    """Collect values of 'number' keys found in a tool output structure."""
    found = set()
    if isinstance(obj, dict):
        if isinstance(obj.get("number"), int):
            found.add(obj["number"])
        for v in obj.values():
            found |= _numbers_in(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= _numbers_in(v)
    elif isinstance(obj, str):
        import json

        try:
            return _numbers_in(json.loads(obj))
        except (ValueError, TypeError):
            return found
    return found


def _input_number(call):
    for key in ("pull_number", "pullNumber", "number", "pr_number", "pull_request_number"):
        val = call.get("input", {}).get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                return None
    return None


def _input_targets_repo(call, repository):
    owner, repo = repository.split("/", 1)
    values = [str(v) for v in call.get("input", {}).values()]
    joined = " ".join(values)
    return repository in values or (owner in values and repo in values) or repository in joined


# --------------------------------------------------------------------------- #
# Fixtures - one agent invocation per scenario
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def trace_with_prs():
    return _run(REPO_WITH_PRS)


@pytest.fixture(scope="module")
def trace_empty():
    if not REPO_EMPTY:
        pytest.skip("TRACE_REPO_EMPTY not set")
    return _run(REPO_EMPTY)


@pytest.fixture(scope="module")
def trace_invalid():
    return _run(REPO_INVALID)


ALL_SCENARIOS = ["trace_with_prs", "trace_empty", "trace_invalid"]


# --------------------------------------------------------------------------- #
# Guardrails (all scenarios)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_exactly_one_listing_tool_and_it_is_list_pull_requests(scenario, request):
    trace = request.getfixturevalue(scenario)
    listing = _calls(trace, LIST_PRS) + _calls(trace, LIST_ISSUES)
    assert len(listing) == 1, f"Expected exactly one listing call, got {len(listing)}"
    assert listing[0]["name"] == LIST_PRS, "PR review intent must use list_pull_requests"


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_only_read_tools_specified_are_used(scenario, request):
    trace = request.getfixturevalue(scenario)
    used = {c.get("name") for c in trace["tool_calls"]}
    assert used <= ALLOWED_TOOLS, f"Unexpected (possibly modifying) tools: {used - ALLOWED_TOOLS}"


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_tool_calls_target_requested_repository(scenario, request):
    trace = request.getfixturevalue(scenario)
    repository = trace["query"][0] if isinstance(trace["query"], list) else trace["query"].get("repository")
    assert repository, "Trace query does not expose the repository"
    for call in trace["tool_calls"]:
        assert _input_targets_repo(call, repository), f"{call['name']} not targeting {repository}: {call['input']}"


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_list_pull_requests_is_first_behavior(scenario, request):
    trace = request.getfixturevalue(scenario)
    assert trace["tool_calls"], "No tool calls observed in trace"
    assert trace["tool_calls"][0]["name"] == LIST_PRS, "B1 must be the first tool behavior"


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_agent_does_not_raise(scenario, request):
    trace = request.getfixturevalue(scenario)
    assert trace["status"] == "ok", f"Agent raised instead of returning a response: {trace['error']}"


# --------------------------------------------------------------------------- #
# Happy path: B1 -> B2 -> B3 -> B4 -> B7
# --------------------------------------------------------------------------- #
def test_happy_path_list_succeeded_with_prs(trace_with_prs):
    lists = _calls(trace_with_prs, LIST_PRS)
    assert lists and not _failed(lists[0]), "list_pull_requests failed in happy-path scenario"
    assert "output" in lists[0], "list_pull_requests output missing from trace"
    assert _numbers_in(lists[0]["output"]), "No pull request numbers observable in list output"


def test_happy_path_reviews_follow_listing(trace_with_prs):
    list_idx = _index(trace_with_prs, LIST_PRS)
    review_idx = _index(trace_with_prs, GET_REVIEWS)
    assert review_idx, "B3 (get_pull_request_reviews) not observed after PRs were found"
    assert all(i > list_idx[0] for i in review_idx), "Reviews retrieved before listing PRs"
    list_end = _calls(trace_with_prs, LIST_PRS)[0]["end"]
    for c in _calls(trace_with_prs, GET_REVIEWS):
        assert c["start"] >= list_end, "Review call started before list_pull_requests ended"


def test_happy_path_at_most_three_prs_selected(trace_with_prs):
    numbers = [_input_number(c) for c in _calls(trace_with_prs, GET_REVIEWS)]
    assert None not in numbers, "PR number not identifiable in a review call input"
    assert 1 <= len(set(numbers)) <= MAX_SELECTED, f"Selected PRs: {set(numbers)}"


def test_happy_path_selected_prs_come_from_listing(trace_with_prs):
    listed = _numbers_in(_calls(trace_with_prs, LIST_PRS)[0].get("output"))
    assert listed, "Listed PR numbers not observable in trace"
    for c in _calls(trace_with_prs, GET_REVIEWS):
        n = _input_number(c)
        assert n in listed, f"Reviews requested for PR #{n} not returned by list_pull_requests"


def test_happy_path_reviews_requested_for_each_selected_pr(trace_with_prs):
    listed = _numbers_in(_calls(trace_with_prs, LIST_PRS)[0].get("output"))
    requested = {_input_number(c) for c in _calls(trace_with_prs, GET_REVIEWS)}
    assert listed and requested <= listed and 1 <= len(requested) <= min(MAX_SELECTED, len(listed))
    # each review call must have evidence of completion (output or error)
    for c in _calls(trace_with_prs, GET_REVIEWS):
        assert "output" in c or _failed(c), "Review call has no output/error evidence"


def test_happy_path_tool_correctness_deepeval(trace_with_prs):
    tools_called = [ToolCall(name=c["name"], input_parameters=c.get("input")) for c in trace_with_prs["tool_calls"]]
    test_case = LLMTestCase(
        input=REQUEST,
        actual_output="(trace evaluation)",
        tools_called=tools_called,
        expected_tools=[ToolCall(name=LIST_PRS), ToolCall(name=GET_REVIEWS)],
    )
    assert_test(test_case, [ToolCorrectnessMetric(threshold=1.0)], run_async=False)


# --------------------------------------------------------------------------- #
# Empty path: B1 -> B5 -> B7
# --------------------------------------------------------------------------- #
def test_empty_path_listing_succeeded_and_empty(trace_empty):
    call = _calls(trace_empty, LIST_PRS)[0]
    assert not _failed(call), "list_pull_requests failed in empty scenario"
    assert "output" in call, "No output evidence for list_pull_requests"
    assert not _numbers_in(call["output"]), "Listing was not empty"


def test_empty_path_no_review_retrieval(trace_empty):
    assert not _calls(trace_empty, GET_REVIEWS), "Reviews retrieved although no PRs were found"


# --------------------------------------------------------------------------- #
# Error path: B1 -> B6 -> B7
# --------------------------------------------------------------------------- #
def test_error_path_listing_failure_observed(trace_invalid):
    call = _calls(trace_invalid, LIST_PRS)[0]
    assert _failed(call), "Trace provides no evidence that list_pull_requests failed"


def test_error_path_no_review_retrieval(trace_invalid):
    assert not _calls(trace_invalid, GET_REVIEWS), "Reviews retrieved after listing failed"
