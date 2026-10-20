"""Black-box behavioral evaluation (DeepEval 3.3.9) for spec 011 - list PRs or issues.

Observation points (all derived from the spec's interface contract):
  * Entrypoint: agent(repository, user_request, github_token) -> dict
  * Tools are LangChain tools executed only through their interface, so
    invocations are observed by wrapping BaseTool.run / BaseTool.arun.

Requires: GITHUB_TOKEN (read permission) and OPENAI_API_KEY (agent + GEval judge).
"""

import json
import os

import pytest
from dotenv import load_dotenv
from langchain_core.tools import BaseTool

from deepeval import assert_test
from deepeval.metrics import GEval, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall

load_dotenv()

from agent import agent  # noqa: E402

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
PUBLIC_REPO = os.getenv("EVAL_REPO", "octocat/Hello-World")
MISSING_REPO = "this-owner-does-not-exist-xyz123/no-such-repo-xyz123"
LIST_TOOLS = {"list_issues", "list_pull_requests"}

pytestmark = pytest.mark.skipif(
    not (GITHUB_TOKEN and os.getenv("OPENAI_API_KEY")),
    reason="GITHUB_TOKEN and OPENAI_API_KEY are required",
)


# --------------------------------------------------------------------------- #
# Tool-call recorder (via the LangChain tool interface)
# --------------------------------------------------------------------------- #
@pytest.fixture
def tool_calls(monkeypatch):
    calls = []
    orig_run, orig_arun = BaseTool.run, BaseTool.arun

    def run(self, tool_input, *args, **kwargs):
        calls.append(ToolCall(name=self.name, input_parameters=_as_dict(tool_input)))
        return orig_run(self, tool_input, *args, **kwargs)

    async def arun(self, tool_input, *args, **kwargs):
        calls.append(ToolCall(name=self.name, input_parameters=_as_dict(tool_input)))
        return await orig_arun(self, tool_input, *args, **kwargs)

    monkeypatch.setattr(BaseTool, "run", run)
    monkeypatch.setattr(BaseTool, "arun", arun)
    return calls


def _as_dict(tool_input):
    if isinstance(tool_input, dict):
        return tool_input.get("args", tool_input) if tool_input.get("type") == "tool_call" else tool_input
    return {"input": str(tool_input)}


def listing_calls(calls):
    return [c for c in calls if c.name in LIST_TOOLS]


def run_agent(repository, user_request):
    out = agent(repository=repository, user_request=user_request, github_token=GITHUB_TOKEN)
    if isinstance(out, str):
        out = json.loads(out)
    return out


# --------------------------------------------------------------------------- #
# Schema helpers (B6)
# --------------------------------------------------------------------------- #
def assert_schema(out):
    assert isinstance(out, dict)
    assert out.get("status") in {"success", "failure"}
    assert isinstance(out.get("message"), str) and out["message"].strip()
    assert "result" in out


def assert_success_result(out, action, repository):
    assert_schema(out)
    assert out["status"] == "success", out
    result = out["result"]
    assert isinstance(result, dict)
    assert result.get("action") == action
    assert result.get("repository") == repository
    assert isinstance(result.get("items"), list)
    for item in result["items"]:
        assert isinstance(item.get("number"), int)
        assert isinstance(item.get("title"), str)
        assert item.get("state") in {"open", "closed"}
    return result["items"]


# --------------------------------------------------------------------------- #
# GEval metrics
# --------------------------------------------------------------------------- #
PARAMS = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]

empty_result_metric = GEval(
    name="Empty result handling",
    criteria=(
        "The actual output is the agent's JSON response to the input request. It must have "
        "status 'success', an empty items list, and a message that explains no matching "
        "items were found. It must not describe the situation as an error or failure."
    ),
    evaluation_params=PARAMS,
    threshold=0.7,
)

invalid_repo_metric = GEval(
    name="Invalid repository handling",
    criteria=(
        "The input refers to a repository that does not exist or is inaccessible. The actual "
        "output must have status 'failure', a message explaining the repository could not be "
        "accessed, and must not contain fabricated issues or pull requests nor claim success."
    ),
    evaluation_params=PARAMS,
    threshold=0.7,
)

rejection_metric = GEval(
    name="Unsupported request rejection",
    criteria=(
        "The input asks for an unsupported operation (e.g., modifying data or listing both "
        "issues and pull requests at once). The actual output must be a failure/rejection "
        "response whose message explains that the supported operations are listing either "
        "issues or pull requests, and must not contain listed items."
    ),
    evaluation_params=PARAMS,
    threshold=0.7,
)


def _case(repository, request, out, calls=None, expected=None):
    return LLMTestCase(
        input=f"repository={repository}\nuser_request={request}",
        actual_output=json.dumps(out, ensure_ascii=False),
        tools_called=calls,
        expected_tools=expected,
    )


def _tool_metric():
    return ToolCorrectnessMetric(threshold=1.0, should_exact_match=True)


# --------------------------------------------------------------------------- #
# B1 - List issues
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("request_text", [
    "List the issues of this repository",
    "Show me all open issues",
])
def test_b1_lists_issues_with_exactly_one_list_issues_call(tool_calls, request_text):
    out = run_agent(PUBLIC_REPO, request_text)
    assert_success_result(out, "list_issues", PUBLIC_REPO)
    calls = listing_calls(tool_calls)
    assert [c.name for c in calls] == ["list_issues"]
    assert_test(
        _case(PUBLIC_REPO, request_text, out, calls, [ToolCall(name="list_issues")]),
        [_tool_metric()],
    )


# --------------------------------------------------------------------------- #
# B2 - List pull requests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("request_text", [
    "List the pull requests of this repository",
    "Show me open PRs waiting for review",
])
def test_b2_lists_prs_with_exactly_one_list_pull_requests_call(tool_calls, request_text):
    out = run_agent(PUBLIC_REPO, request_text)
    assert_success_result(out, "list_pull_requests", PUBLIC_REPO)
    calls = listing_calls(tool_calls)
    assert [c.name for c in calls] == ["list_pull_requests"]
    assert_test(
        _case(PUBLIC_REPO, request_text, out, calls, [ToolCall(name="list_pull_requests")]),
        [_tool_metric()],
    )


# --------------------------------------------------------------------------- #
# B3 - Apply filters (state), no extra tool invocation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("request_text,action,state", [
    ("List only closed issues", "list_issues", "closed"),
    ("List only open issues", "list_issues", "open"),
    ("List only closed pull requests", "list_pull_requests", "closed"),
    ("List only open pull requests", "list_pull_requests", "open"),
])
def test_b3_state_filter_applied(tool_calls, request_text, action, state):
    out = run_agent(PUBLIC_REPO, request_text)
    items = assert_success_result(out, action, PUBLIC_REPO)
    assert all(i["state"] == state for i in items), items
    assert [c.name for c in listing_calls(tool_calls)] == [action]


# --------------------------------------------------------------------------- #
# B4 - Empty results are success
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("request_text,action", [
    ("List issues with the label 'zz-nonexistent-label-98765'", "list_issues"),
    ("List pull requests authored by user 'zz-no-such-author-98765'", "list_pull_requests"),
])
def test_b4_empty_results_reported_as_success(tool_calls, request_text, action):
    out = run_agent(PUBLIC_REPO, request_text)
    items = assert_success_result(out, action, PUBLIC_REPO)
    assert items == []
    assert [c.name for c in listing_calls(tool_calls)] == [action]
    assert_test(_case(PUBLIC_REPO, request_text, out), [empty_result_metric])


# --------------------------------------------------------------------------- #
# B5 - Invalid / inaccessible repository
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("request_text", ["List open issues", "List open pull requests"])
def test_b5_invalid_repository_returns_failure(tool_calls, request_text):
    out = run_agent(MISSING_REPO, request_text)
    assert_schema(out)
    assert out["status"] == "failure"
    result = out.get("result") or {}
    assert not (isinstance(result, dict) and result.get("items")), "fabricated items"
    assert len(listing_calls(tool_calls)) <= 1
    assert_test(_case(MISSING_REPO, request_text, out), [invalid_repo_metric])


def test_b5_invalid_token_returns_failure(tool_calls):
    out = agent(repository=PUBLIC_REPO, user_request="List open issues",
                github_token="ghp_invalidtoken000000000000000000000000")
    if isinstance(out, str):
        out = json.loads(out)
    assert_schema(out)
    assert out["status"] == "failure"
    result = out.get("result") or {}
    assert not (isinstance(result, dict) and result.get("items"))


# --------------------------------------------------------------------------- #
# B6 - Structured response preserves issue/PR distinction
# --------------------------------------------------------------------------- #
def test_b6_issue_and_pr_responses_are_distinguished(tool_calls):
    issues = run_agent(PUBLIC_REPO, "List issues")
    prs = run_agent(PUBLIC_REPO, "List pull requests")
    assert_success_result(issues, "list_issues", PUBLIC_REPO)
    assert_success_result(prs, "list_pull_requests", PUBLIC_REPO)


# --------------------------------------------------------------------------- #
# B7 - Reject unsupported requests without invoking any listing tool
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("request_text", [
    "List all issues and all pull requests",
    "Close issue #1",
    "Create a new pull request from branch dev to main",
    "Delete this repository",
])
def test_b7_unsupported_requests_rejected(tool_calls, request_text):
    out = run_agent(PUBLIC_REPO, request_text)
    assert_schema(out)
    assert out["status"] == "failure"
    assert listing_calls(tool_calls) == []
    assert_test(_case(PUBLIC_REPO, request_text, out), [rejection_metric])


# --------------------------------------------------------------------------- #
# Invariant - never both tools, at most one listing call per run
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("request_text", [
    "Show me all closed bug issues",
    "List open pull requests waiting for review",
    "Which issues are open?",
])
def test_invariant_single_tool_invocation(tool_calls, request_text):
    run_agent(PUBLIC_REPO, request_text)
    calls = listing_calls(tool_calls)
    assert len(calls) == 1, [c.name for c in calls]
