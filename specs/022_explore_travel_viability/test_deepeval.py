"""Black-box behavioral evaluation (DeepEval 3.3.9) for the Travel Destination
Viability agent, derived exclusively from spec.md.

The spec leaves the entrypoint unspecified, so it is configured via env var:
    AGENT_ENTRYPOINT="package.module:function"   # callable(user_request: str) -> {"status", "response"}
Optional:
    AGENT_IS_ASYNC=1  if the callable is a coroutine function.

Run:  deepeval test run test_deepeval.py   (or: pytest test_deepeval.py)
"""

import asyncio
import importlib
import inspect
import os
import time

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

MAX_SECONDS = 30  # B6: maximum execution time of 30 seconds


# --------------------------------------------------------------------------- #
# Agent invocation helpers
# --------------------------------------------------------------------------- #
def _load_agent():
    target = os.getenv("AGENT_ENTRYPOINT")
    if not target or ":" not in target:
        pytest.skip("Set AGENT_ENTRYPOINT='module:function' to run the evaluation.")
    module_name, func_name = target.split(":", 1)
    return getattr(importlib.import_module(module_name), func_name)


@pytest.fixture(scope="session")
def agent():
    return _load_agent()


def run_agent(agent, user_request: str):
    """Invoke the agent; return (result, elapsed_seconds, raised_exception)."""
    start = time.monotonic()
    try:
        if inspect.iscoroutinefunction(agent) or os.getenv("AGENT_IS_ASYNC") == "1":
            result = asyncio.run(agent(user_request=user_request))
        else:
            result = agent(user_request=user_request)
        exc = None
    except Exception as e:  # noqa: BLE001 - B6 forbids propagated exceptions
        result, exc = None, e
    return result, time.monotonic() - start, exc


def assert_schema(result):
    """B7 / Guardrail: object containing `status` and `response`."""
    assert isinstance(result, dict), f"Output must be an object, got {type(result)!r}"
    assert "status" in result, "Output missing 'status'"
    assert "response" in result, "Output missing 'response'"


def response_text(result) -> str:
    resp = result["response"]
    return resp if isinstance(resp, str) else str(resp)


# --------------------------------------------------------------------------- #
# Metrics (GEval, LLM-as-judge)
# --------------------------------------------------------------------------- #
def destination_metric():
    return GEval(
        name="B1 Destination Identification",
        criteria=(
            "Determine whether the actual output is an assessment about the destination "
            "requested in the input, identified at city-level granularity when the input "
            "allows it. The expected output names the destination that should be assessed."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.6,
    )


def combined_assessment_metric():
    return GEval(
        name="B4 Combined Viability Assessment",
        evaluation_steps=[
            "Check that the actual output gives an assessment of whether traveling to the destination in the input is viable/advisable.",
            "Check that the actual output references current weather conditions (e.g., temperature, precipitation, sky conditions, wind).",
            "Check that the actual output references recent news, safety advisories, or current events about the destination.",
            "Check that the assessment is supported by concrete evidence (specific values, facts, or cited events) rather than generic statements.",
            "Penalize heavily if either the weather dimension or the news dimension is missing.",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.6,
    )


def no_fabrication_metric():
    return GEval(
        name="Guardrail: No Fabricated Weather/News",
        evaluation_steps=[
            "The retrieval context contains the information known to be real for this run (if any).",
            "Check whether the actual output presents weather data or news that is presented as real-time/recent "
            "but appears invented, hedged as hypothetical, or placeholder-like (e.g., 'assume it is sunny', 'example news').",
            "Check whether the actual output, when data is unavailable, honestly states so instead of inventing values.",
            "Score high only if no fabricated weather data or news is presented as factual.",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7,
    )


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
VALID_REQUESTS = [
    ("Is it a good idea to travel to Lisbon this week?", "Lisbon"),
    ("I'm planning a trip to Tokyo, Japan. Should I go?", "Tokyo"),
    ("Quero viajar para o Rio de Janeiro amanhã, é uma boa ideia?", "Rio de Janeiro"),
    ("Can you check whether Paris is a viable destination right now?", "Paris"),
]

EMPTY_REQUESTS = ["", " ", "   ", "\n", "\t", " \n\t  "]


# --------------------------------------------------------------------------- #
# B5 — Handle Empty Request
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("user_request", EMPTY_REQUESTS, ids=repr)
def test_b5_empty_request_returns_failure(agent, user_request):
    result, elapsed, exc = run_agent(agent, user_request)
    assert exc is None, f"Exception propagated for empty request: {exc!r}"
    assert_schema(result)
    assert result["status"] == "failure"
    assert elapsed <= MAX_SECONDS


# --------------------------------------------------------------------------- #
# B7 — Structured response on success path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("user_request,_dest", VALID_REQUESTS[:2])
def test_b7_valid_request_returns_structured_response(agent, user_request, _dest):
    result, _, exc = run_agent(agent, user_request)
    assert exc is None, f"Exception propagated: {exc!r}"
    assert_schema(result)
    assert isinstance(result["status"], str) and result["status"]
    assert response_text(result).strip(), "Response must not be empty"


# --------------------------------------------------------------------------- #
# B6 — Errors, timeouts, no propagated exceptions, 30s limit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "user_request",
    [
        "Should I travel to Qwxzplk Vrrnnt City?",          # invalid city for weather tool
        "Travel viability for " + "Atlantis " * 200,        # oversized/odd input
        "Is it safe to go to 🙂🙂🙂?",                      # non-identifiable destination
    ],
)
def test_b6_problematic_requests_do_not_raise_and_respect_schema(agent, user_request):
    result, elapsed, exc = run_agent(agent, user_request)
    assert exc is None, f"Exception propagated: {exc!r}"
    assert_schema(result)
    assert elapsed <= MAX_SECONDS + 2, f"Execution took {elapsed:.1f}s (> {MAX_SECONDS}s)"


@pytest.mark.parametrize("user_request,_dest", VALID_REQUESTS)
def test_b6_execution_within_time_limit(agent, user_request, _dest):
    result, elapsed, exc = run_agent(agent, user_request)
    assert exc is None, f"Exception propagated: {exc!r}"
    assert_schema(result)
    assert elapsed <= MAX_SECONDS + 2, f"Execution took {elapsed:.1f}s (> {MAX_SECONDS}s)"


def test_b6_missing_tavily_key_returns_failure_without_exception(agent, monkeypatch):
    """B3 requires TAVILY_API_KEY; a tool failure must surface as status 'failure'."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "")
    result, elapsed, exc = run_agent(agent, VALID_REQUESTS[0][0])
    assert exc is None, f"Exception propagated: {exc!r}"
    assert_schema(result)
    assert result["status"] == "failure"
    assert elapsed <= MAX_SECONDS + 2


# --------------------------------------------------------------------------- #
# B1 — Identify Destination (LLM-judged)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("user_request,destination", VALID_REQUESTS)
def test_b1_identifies_destination(agent, user_request, destination):
    result, _, exc = run_agent(agent, user_request)
    assert exc is None
    assert_schema(result)
    if result["status"] == "failure":
        pytest.fail(f"Agent failed on a valid request: {response_text(result)[:300]}")
    test_case = LLMTestCase(
        input=user_request,
        actual_output=response_text(result),
        expected_output=f"An assessment about {destination}.",
    )
    assert_test(test_case, [destination_metric()])


# --------------------------------------------------------------------------- #
# B2 + B3 + B4 — Combined assessment using weather and news
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("user_request,destination", VALID_REQUESTS)
def test_b4_assessment_combines_weather_and_news_with_evidence(agent, user_request, destination):
    result, _, exc = run_agent(agent, user_request)
    assert exc is None
    assert_schema(result)
    if result["status"] == "failure":
        pytest.fail(f"Agent failed on a valid request: {response_text(result)[:300]}")
    test_case = LLMTestCase(input=user_request, actual_output=response_text(result))
    assert_test(test_case, [combined_assessment_metric()])


# --------------------------------------------------------------------------- #
# Guardrail — Do not fabricate weather data or news
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("user_request,_dest", VALID_REQUESTS[:2])
def test_guardrail_no_fabricated_data(agent, user_request, _dest):
    result, _, exc = run_agent(agent, user_request)
    assert exc is None
    assert_schema(result)
    test_case = LLMTestCase(input=user_request, actual_output=response_text(result))
    assert_test(test_case, [no_fabrication_metric()])


def test_guardrail_no_fabrication_when_news_unavailable(agent, monkeypatch):
    """With the news source unavailable, the agent must not invent news."""
    monkeypatch.setenv("TAVILY_API_KEY", "invalid-key-for-test")
    user_request = VALID_REQUESTS[0][0]
    result, _, exc = run_agent(agent, user_request)
    assert exc is None
    assert_schema(result)
    if result["status"] == "failure":
        return  # B6: failure status is the specified outcome for tool failures
    test_case = LLMTestCase(input=user_request, actual_output=response_text(result))
    assert_test(test_case, [no_fabrication_metric()])
