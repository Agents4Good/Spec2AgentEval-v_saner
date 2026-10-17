"""
Black-box behavioral evaluation for the "classify PRs by review status" agent.

Requirements source: spec.md (only).
Compatible with DeepEval 3.3.9.

Environment variables:
  GITHUB_TOKEN        GitHub token with read permissions (required)
  OPENAI_API_KEY      used by DeepEval GEval judge (required for GEval tests)
  TEST_REPO           owner/repo with open pull requests (default: langchain-ai/langgraph)
  EMPTY_REPO          owner/repo with NO open pull requests (optional; B5 test skipped if unset)

Ground truth for PRs/reviews is fetched independently from the GitHub REST API
(an external oracle), never from the agent's implementation.
"""

import json
import os
import time

import pytest
import requests
from dotenv import load_dotenv

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

load_dotenv()

from agent import agent  # noqa: E402  (entrypoint defined by the spec)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
TEST_REPO = os.getenv("TEST_REPO", "langchain-ai/langgraph")
EMPTY_REPO = os.getenv("EMPTY_REPO")
INVALID_REPO = "this-owner-does-not-exist-zz9/no-such-repo-zz9"
REQUEST = "Which open pull requests still need review and which have already been reviewed?"
GROUPS = {"needs_review", "reviewed"}
JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", "gpt-4o")

pytestmark = pytest.mark.skipif(not GITHUB_TOKEN, reason="GITHUB_TOKEN not set")


# --------------------------------------------------------------------------- helpers
def _gh(path, **params):
    r = requests.get(
        f"https://api.github.com{path}",
        headers={"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"},
        params=params,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _run(repository=TEST_REPO, user_request=REQUEST, token=None):
    return agent(repository=repository, user_request=user_request,
                 github_token=GITHUB_TOKEN if token is None else token)


def _items(resp):
    groups = resp["result"]["groups"]
    return [(g, it) for g in GROUPS for it in groups.get(g, [])]


@pytest.fixture(scope="module")
def success_response():
    return _run()


@pytest.fixture(scope="module")
def open_prs():
    return _gh(f"/repos/{TEST_REPO}/pulls", state="open", sort="updated", direction="desc", per_page=100)


# --------------------------------------------------------------------------- B7: output contract
def test_output_contract(success_response):
    resp = success_response
    assert isinstance(resp, dict)
    assert resp.get("status") == "success"
    assert isinstance(resp.get("message"), str) and resp["message"].strip()
    result = resp["result"]
    assert result["repository"] == TEST_REPO
    groups = result["groups"]
    assert set(groups.keys()) == GROUPS, "both classification groups must be returned"
    for g in GROUPS:
        assert isinstance(groups[g], list)
        for item in groups[g]:
            assert isinstance(item["number"], int)
            assert isinstance(item["title"], str)


def test_note_states_at_most_three_analyzed(success_response):
    note = success_response.get("note") or ""
    tc = LLMTestCase(input=REQUEST, actual_output=note)
    metric = GEval(
        name="AtMostThreeNote",
        criteria="The actual output must inform the user that at most three (3) pull requests were analyzed.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7,
        model=JUDGE_MODEL,
    )
    assert_test(tc, [metric])


# --------------------------------------------------------------------------- B2: selection
def test_at_most_three_prs_and_no_duplicates(success_response):
    nums = [it["number"] for _, it in _items(success_response)]
    assert len(nums) <= 3
    assert len(nums) == len(set(nums)), "a PR must belong to exactly one group"


def test_selection_is_deterministic(success_response):
    second = _run()
    first_nums = sorted(it["number"] for _, it in _items(success_response))
    second_nums = sorted(it["number"] for _, it in _items(second))
    assert first_nums == second_nums


# --------------------------------------------------------------------------- B1: real, open, most recently updated
def test_prs_are_real_and_open(success_response, open_prs):
    real = {p["number"]: p["title"] for p in open_prs}
    for _, it in _items(success_response):
        assert it["number"] in real, f"PR #{it['number']} is not an open PR (fabricated?)"
        assert it["title"] == real[it["number"]]


def test_prioritizes_most_recently_updated(success_response, open_prs):
    returned = {it["number"] for _, it in _items(success_response)}
    if not open_prs:
        pytest.skip("TEST_REPO has no open PRs")
    expected = {p["number"] for p in open_prs[: min(3, len(open_prs))]}
    # Allow for PRs updated between the agent run and the oracle call.
    assert len(returned & expected) >= len(expected) - 1, (
        f"selected {returned}, most recently updated are {expected}"
    )
    assert len(returned) == min(3, len(open_prs))


# --------------------------------------------------------------------------- B3/B4: classification rule
def test_classification_matches_actual_reviews(success_response):
    for group, it in _items(success_response):
        reviews = _gh(f"/repos/{TEST_REPO}/pulls/{it['number']}/reviews", per_page=100)
        states = {r.get("state") for r in reviews}
        expected = "reviewed" if states & {"APPROVED", "CHANGES_REQUESTED"} else "needs_review"
        assert group == expected, f"PR #{it['number']} states={states} classified {group}"


# --------------------------------------------------------------------------- B5: empty results
@pytest.mark.skipif(not EMPTY_REPO, reason="EMPTY_REPO not set")
def test_empty_repo_is_success_with_empty_groups():
    resp = _run(repository=EMPTY_REPO)
    assert resp["status"] == "success"
    assert resp["result"]["groups"] == {"needs_review": [], "reviewed": []}


# --------------------------------------------------------------------------- B6: errors
def test_invalid_repository_returns_failure():
    resp = _run(repository=INVALID_REPO)
    assert resp["status"] == "failure"
    assert isinstance(resp.get("message"), str) and resp["message"].strip()
    groups = (resp.get("result") or {}).get("groups") or {}
    assert all(not groups.get(g) for g in GROUPS), "no results may be fabricated after a failure"


def test_invalid_repository_message_is_explanatory():
    resp = _run(repository=INVALID_REPO)
    tc = LLMTestCase(
        input=f"repository={INVALID_REPO}; request={REQUEST}",
        actual_output=json.dumps(resp),
    )
    metric = GEval(
        name="FailureExplanation",
        criteria=(
            "The repository in the input does not exist. The actual output must report failure, "
            "include a message explaining that the repository is invalid or inaccessible, and must "
            "NOT claim success nor list any pull requests or reviews."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7,
        model=JUDGE_MODEL,
    )
    assert_test(tc, [metric])


def test_invalid_token_returns_failure():
    resp = _run(token="ghp_invalidtoken000000000000000000000000")
    assert resp["status"] == "failure"
    groups = (resp.get("result") or {}).get("groups") or {}
    assert all(not groups.get(g) for g in GROUPS)


def test_execution_respects_30_second_limit():
    start = time.monotonic()
    resp = _run()
    elapsed = time.monotonic() - start
    assert resp["status"] in {"success", "failure"}
    assert elapsed <= 35, f"execution took {elapsed:.1f}s (30s limit + margin)"


# --------------------------------------------------------------------------- Guardrail: no modification / no fabrication
def test_repository_not_modified():
    before = [(p["number"], p["state"], p["updated_at"]) for p in
              _gh(f"/repos/{TEST_REPO}/pulls", state="open", sort="updated", direction="desc", per_page=3)]
    resp = _run()
    after = [(p["number"], p["state"], p["updated_at"]) for p in
             _gh(f"/repos/{TEST_REPO}/pulls", state="open", sort="updated", direction="desc", per_page=3)]
    analyzed = {it["number"] for _, it in _items(resp)}
    changed = {b[0] for b, a in zip(before, after) if b != a}
    assert not (changed & analyzed), f"analyzed PRs changed during execution: {changed & analyzed}"


def test_summary_message_faithful_to_result(success_response):
    tc = LLMTestCase(
        input=REQUEST,
        actual_output=success_response["message"],
        context=[json.dumps(success_response["result"])],
    )
    metric = GEval(
        name="SummaryFaithfulness",
        criteria=(
            "The actual output is a summary of the pull request review classification. It must be "
            "consistent with the structured result in the context (same PRs, same needs_review/reviewed "
            "grouping) and must not mention pull requests or reviews that are absent from the context."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT],
        threshold=0.7,
        model=JUDGE_MODEL,
    )
    assert_test(tc, [metric])
