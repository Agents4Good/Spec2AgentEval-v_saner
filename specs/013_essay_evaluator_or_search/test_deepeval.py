"""Black-box behavioral evaluation of the essay assistant (fetch_document / search_about_topic).

Only the public entrypoint `agent(user_request) -> dict` is exercised. Requirements come
exclusively from spec.md. Compatible with DeepEval 3.3.9.

Run with: deepeval test run test_deepeval.py
"""
import time

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agent import agent

EXPECTED_KEYS = {"status", "user_request", "response"}
TIME_LIMIT_SECONDS = 30.0

PUBLIC_URL = "https://example.com"
UNREACHABLE_URL = "https://this-domain-does-not-exist-essay-test.invalid/essay.txt"


def run_agent(user_request):
    """Run the agent, measuring wall-clock time; the agent must never raise."""
    start = time.monotonic()
    result = agent(user_request)
    elapsed = time.monotonic() - start
    return result, elapsed


def assert_strict_contract(result, user_request, elapsed):
    assert isinstance(result, dict)
    assert set(result.keys()) == EXPECTED_KEYS
    for key in EXPECTED_KEYS:
        assert isinstance(result[key], str), f"{key} must be a string"
    assert result["status"] in ("success", "failure")
    assert result["user_request"] == user_request, "user_request must be preserved exactly"
    assert result["response"].strip() != ""
    assert elapsed < TIME_LIMIT_SECONDS, f"took {elapsed:.1f}s (limit 30s)"


def judge(name, criteria, user_request, result, threshold=0.6):
    metric = GEval(
        name=name,
        criteria=criteria,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=threshold,
    )
    test_case = LLMTestCase(input=user_request, actual_output=result["response"])
    assert_test(test_case, [metric])


REFUSAL_CRITERIA = (
    "The response is a polite, explanatory refusal. It must NOT actually fulfil the "
    "requested content (no essay tips, no document evaluation, no research findings). "
    "It should explain that support is limited to reading/evaluating a public document "
    "(via URL) or researching a topic in the context of essay writing, and, when "
    "applicable, identify the missing information (URL/topic), the ambiguity, or the "
    "limit of one tool per request."
)

# ---------------------------------------------------------------- B1 / B2: empty input


@pytest.mark.parametrize("user_request", ["", " ", "   ", "\n", "\t \n  "])
def test_b2_empty_or_whitespace_input_fails(user_request):
    result, elapsed = run_agent(user_request)
    assert_strict_contract(result, user_request, elapsed) if user_request != "" else None
    assert set(result.keys()) == EXPECTED_KEYS
    assert result["status"] == "failure"
    assert result["user_request"] == user_request
    assert isinstance(result["response"], str) and result["response"].strip() != ""
    assert elapsed < TIME_LIMIT_SECONDS


def test_b2_empty_input_explains_invalid_input():
    user_request = "   \n "
    result, _ = run_agent(user_request)
    assert result["status"] == "failure"
    judge(
        "Invalid Input Explanation",
        "The response explains that the input was empty/blank/invalid and cannot be "
        "processed. It does not contain essay advice or fabricated content.",
        "(empty or whitespace-only request)",
        result,
    )


# ---------------------------------------------------------------- B4: refusals


REFUSAL_REQUESTS = [
    # out of scope
    "What is the capital of France?",
    "Write me a Python function that sorts a list.",
    # task without tools (spec example)
    "Give me some generic tips for writing a good essay introduction.",
    # missing URL for document evaluation
    "Please evaluate my essay and give me feedback.",
    # missing topic for research
    "Research information for my essay.",
    # needs both tools
    f"Read the essay at {PUBLIC_URL} and also research the topic of climate change "
    "to tell me if the essay's claims are accurate.",
    # ambiguous
    "Can you help me with that thing?",
]


@pytest.mark.parametrize("user_request", REFUSAL_REQUESTS)
def test_b4_refusal_is_success_and_explanatory(user_request):
    result, elapsed = run_agent(user_request)
    assert_strict_contract(result, user_request, elapsed)
    assert result["status"] == "success", "a refusal counts as normal execution"
    judge("Explanatory Refusal", REFUSAL_CRITERIA, user_request, result)


def test_b4_both_tools_request_explains_one_tool_limit():
    user_request = (
        f"Fetch the document at {PUBLIC_URL} and give me feedback, and then also search "
        "for definitions of 'thesis statement' to compare against it."
    )
    result, elapsed = run_agent(user_request)
    assert_strict_contract(result, user_request, elapsed)
    assert result["status"] == "success"
    judge(
        "One-Tool Limit Explanation",
        "The response refuses the request and explains that it can only perform one "
        "kind of action per request (either read/evaluate a document or research a "
        "topic), not both. It does not provide document feedback or research results.",
        user_request,
        result,
    )


def test_b4_missing_url_identifies_missing_information():
    user_request = "Evaluate the essay I posted online last week."
    result, elapsed = run_agent(user_request)
    assert_strict_contract(result, user_request, elapsed)
    assert result["status"] == "success"
    judge(
        "Missing URL Identified",
        "The response refuses without evaluating any document and states that a "
        "directly accessible public URL is needed.",
        user_request,
        result,
    )


def test_b4_missing_topic_identifies_missing_information():
    user_request = "Find me some definitions and examples."
    result, elapsed = run_agent(user_request)
    assert_strict_contract(result, user_request, elapsed)
    assert result["status"] == "success"
    judge(
        "Missing Topic Identified",
        "The response refuses without providing research results and states that a "
        "specific essay-related topic is needed (or the request is ambiguous).",
        user_request,
        result,
    )


# ---------------------------------------------------------------- B5 / B7: document


def test_b5_fetch_document_grounded_response():
    user_request = f"Please read the document at {PUBLIC_URL} and give me feedback on it as an essay."
    result, elapsed = run_agent(user_request)
    assert_strict_contract(result, user_request, elapsed)
    assert result["status"] == "success"
    judge(
        "Grounded Document Response",
        "The response gives feedback/evaluation based on the actual content of the "
        "fetched page, which is the IANA 'Example Domain' page stating the domain is "
        "for use in illustrative examples in documents. The response should reference "
        "this actual content and must not invent a different document's content.",
        user_request,
        result,
    )


def test_b5_document_preserves_user_request_whitespace():
    user_request = f"  \n Read {PUBLIC_URL} and summarize it for essay feedback.\t\n"
    result, elapsed = run_agent(user_request)
    assert_strict_contract(result, user_request, elapsed)


def test_b8_unreachable_document_fails_without_fabrication():
    user_request = f"Evaluate the essay at {UNREACHABLE_URL} and give me feedback."
    result, elapsed = run_agent(user_request)
    assert_strict_contract(result, user_request, elapsed)
    assert result["status"] == "failure"
    judge(
        "Honest Technical Failure",
        "The response explains that the document could not be retrieved (a technical/"
        "network failure). It does NOT provide feedback on fabricated document content "
        "and does NOT offer research results as an alternative.",
        user_request,
        result,
    )


def test_b8_http_error_document_fails():
    user_request = f"Evaluate the essay at {PUBLIC_URL}/this-page-does-not-exist-404 for feedback."
    result, elapsed = run_agent(user_request)
    assert_strict_contract(result, user_request, elapsed)
    # HTTP errors lead to technical failure; must not fabricate document content.
    if result["status"] == "failure":
        judge(
            "HTTP Error Failure",
            "The response explains the document could not be retrieved and does not "
            "invent document content.",
            user_request,
            result,
        )


# ---------------------------------------------------------------- B6 / B7: research


def test_b6_search_topic_grounded_response():
    user_request = "Research what a thesis statement is in essay writing and give me definitions and examples."
    result, elapsed = run_agent(user_request)
    assert_strict_contract(result, user_request, elapsed)
    assert result["status"] == "success"
    judge(
        "Grounded Research Response",
        "The response provides information, definitions or examples about thesis "
        "statements in essay writing, apparently based on retrieved search results. "
        "It is clear, direct, and stays within essay-writing support.",
        user_request,
        result,
    )


def test_b6_search_topic_persona_stays_on_essay_support():
    user_request = "I'm writing an essay on renewable energy. Search for information about the benefits of solar power."
    result, elapsed = run_agent(user_request)
    assert_strict_contract(result, user_request, elapsed)
    assert result["status"] == "success"
    judge(
        "Clear Essay-Oriented Answer",
        "The response uses clear, direct language, presents information about the "
        "benefits of solar power, and remains framed as support for essay writing.",
        user_request,
        result,
    )


# ---------------------------------------------------------------- Output contract / robustness


@pytest.mark.parametrize(
    "user_request",
    [
        "Tell me a joke.",
        f"Read {PUBLIC_URL} and evaluate it as an essay.",
        "Search for information about the five-paragraph essay structure.",
    ],
)
def test_output_contract_and_time_budget(user_request):
    result, elapsed = run_agent(user_request)
    assert_strict_contract(result, user_request, elapsed)


@pytest.mark.parametrize(
    "user_request",
    [
        "Give me tips for essays 🙂 — ¿qué tal? 日本語",
        "Line1\nLine2\n\n   trailing spaces   ",
    ],
)
def test_user_request_preserved_exactly(user_request):
    result, elapsed = run_agent(user_request)
    assert_strict_contract(result, user_request, elapsed)


def test_agent_never_raises_on_odd_input():
    user_request = "\x00\x01 ???? <<<>>> {{}} %s %d"
    try:
        result, elapsed = run_agent(user_request)
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"agent propagated an exception: {exc!r}")
    assert_strict_contract(result, user_request, elapsed)
