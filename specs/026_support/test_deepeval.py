"""Black-box behavioral tests for the customer-support agent in spec 026.

Every check calls the public entrypoint agent(query) and inspects the returned
state. Nothing here reads, patches, or traces the agent's internals.

Run with:
    deepeval test run test_deepeval.py

agent.py must be importable from this directory or from PYTHONPATH. The GEval
judge needs OPENAI_API_KEY.

Exact requirements use plain asserts. Requirements about the meaning of a
response use DeepEval GEval metrics through assert_test.

Covered:
    - Output contract: state is a dict with a str "response" field.
    - B3: None, non-string, empty, whitespace-only, and unrelated inputs return
      the fixed error message.
    - B1, B2, Policies: successful responses use the exact format, with exactly
      one allowed Category and one allowed Sentiment. Positive and Neutral
      queries are included because they must still be forwarded.
    - Goal, B5: a support query ends with a handoff confirmation.
    - Constraint: the suggested response is for human support only.
    - Runtime dependencies, B6: an invalid or missing OPENAI_API_KEY returns the
      analysis error.

Not covered, because they cannot be observed through agent(query): tool-call
counts, arguments, and order; the no-retry rules; the single analytical LLM
call; and B7 tool failures, which need fault injection inside the tools.
"""

import json
import os
import re
import subprocess
import sys

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agent import agent

ERROR_RESPONSE = "Error: could not classify the query"
ALLOWED_CATEGORIES = {"Technical", "Billing", "General"}
ALLOWED_SENTIMENTS = {"Positive", "Neutral", "Negative"}
SUCCESS_FORMAT = re.compile(
    r"Category: (?P<category>[^\n]*)\nSentiment: (?P<sentiment>[^\n]*)"
    r"\nResponse: (?P<handoff>.*)",
    re.DOTALL,
)
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPORT_QUERY = "My invoice shows a charge I do not recognize."

# Calls agent(query) in a fresh interpreter, so the environment applies at import.
RUNNER = (
    "import json, sys\n"
    "from agent import agent\n"
    "state = agent(sys.argv[1])\n"
    "print('STATE=' + json.dumps(state, default=str))\n"
)

handoff_metric = GEval(
    name="Handoff confirmation",
    criteria=(
        "The text after 'Response:' in the actual output must confirm that the "
        "customer's request was forwarded to human support. A restatement of the "
        "classification alone does not satisfy this."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

no_customer_reply_metric = GEval(
    name="Suggested response kept for human support",
    criteria=(
        "The suggested response is for human support only. The actual output must "
        "contain only the Category and Sentiment lines and a handoff confirmation to "
        "human support. It must not answer the customer's request, for example with "
        "troubleshooting steps, a solution, or a promised refund or other resolution."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)


def run_agent(query):
    state = agent(query)
    assert isinstance(state, dict), (
        f"agent(query) must return a dict, got {type(state).__name__}"
    )
    assert isinstance(state.get("response"), str), "state['response'] must be a str"
    return state["response"]


def parse_success_response(response):
    """Return (category, sentiment, handoff) for a successful response."""
    match = SUCCESS_FORMAT.fullmatch(response)
    assert match is not None, f"Not in the success format: {response!r}"
    assert response.count("Category:") == 1, "A successful result has exactly one Category"
    assert response.count("Sentiment:") == 1, "A successful result has exactly one Sentiment"
    category, sentiment, handoff = match["category"], match["sentiment"], match["handoff"]
    assert category in ALLOWED_CATEGORIES, f"Category {category!r} is not allowed"
    assert sentiment in ALLOWED_SENTIMENTS, f"Sentiment {sentiment!r} is not allowed"
    assert handoff.strip(), "The handoff confirmation must not be empty"
    assert not handoff.startswith("Error:"), f"A tool error was returned: {handoff!r}"
    return category, sentiment, handoff


def call_agent_in_fresh_process(query, api_key):
    env = dict(os.environ)
    if api_key is None:
        env.pop("OPENAI_API_KEY", None)
    else:
        env["OPENAI_API_KEY"] = api_key
    completed = subprocess.run(
        [sys.executable, "-c", RUNNER, query],
        cwd=AGENT_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("STATE=")]
    assert lines, f"agent(query) did not return a state.\nstderr:\n{completed.stderr[-2000:]}"
    state = json.loads(lines[-1].removeprefix("STATE="))
    assert isinstance(state, dict), "agent(query) must return a dict"
    return state


def test_entrypoint_returns_state_with_response_string():
    """Output contract: agent(query) returns a dict with a str 'response' field."""
    state = agent(SUPPORT_QUERY)
    assert isinstance(state, dict)
    assert isinstance(state.get("response"), str)


INVALID_INPUTS = [
    pytest.param(None, id="none"),
    pytest.param(42, id="integer"),
    pytest.param(["My invoice is wrong."], id="list"),
    pytest.param({"query": "My invoice is wrong."}, id="dict"),
    pytest.param(b"My invoice is wrong.", id="bytes"),
    pytest.param("", id="empty"),
    pytest.param(" \n\t ", id="whitespace-only"),
]


@pytest.mark.parametrize("query", INVALID_INPUTS)
def test_invalid_input_returns_error_response(query):
    """B3: None, non-string, empty, and whitespace-only inputs return the error."""
    assert run_agent(query) == ERROR_RESPONSE


UNRELATED_QUERIES = [
    pytest.param("What is the capital of Australia?", id="geography"),
    pytest.param("Write a haiku about autumn leaves.", id="poem"),
    pytest.param("Give me a recipe for banana bread.", id="recipe"),
]


@pytest.mark.parametrize("query", UNRELATED_QUERIES)
def test_unrelated_text_returns_error_response(query):
    """B3 via B1: text unrelated to customer support returns the error."""
    assert run_agent(query) == ERROR_RESPONSE


SUPPORT_CASES = [
    pytest.param(
        "The mobile app crashes every time I try to upload a photo. "
        "This is really frustrating.",
        "Technical",
        "Negative",
        id="technical-negative",
    ),
    pytest.param(
        "I was charged twice on my invoice this month and I am angry about it. "
        "Please refund the duplicate payment.",
        "Billing",
        "Negative",
        id="billing-negative",
    ),
    pytest.param(
        "How do I change the language setting in the desktop application?",
        "Technical",
        "Neutral",
        id="technical-neutral",
    ),
    pytest.param(
        "What are your customer support hours?",
        "General",
        "Neutral",
        id="general-neutral",
    ),
    pytest.param(
        "Thanks for fixing the payment on my invoice so quickly, I really appreciate it!",
        "Billing",
        "Positive",
        id="billing-positive",
    ),
]


@pytest.mark.parametrize("query, expected_category, expected_sentiment", SUPPORT_CASES)
def test_support_query_is_classified_and_forwarded(
    query, expected_category, expected_sentiment
):
    """B1, B2, B5, Goal: exact labels, then a handoff with no customer reply."""
    response = run_agent(query)
    category, sentiment, _ = parse_success_response(response)
    assert category == expected_category
    assert sentiment == expected_sentiment
    assert_test(
        test_case=LLMTestCase(input=query, actual_output=response),
        metrics=[handoff_metric, no_customer_reply_metric],
    )


@pytest.mark.parametrize(
    "api_key",
    [
        pytest.param("invalid-key", id="invalid-key"),
        pytest.param(None, id="missing-key"),
    ],
)
def test_invalid_llm_configuration_returns_analysis_error(api_key):
    """Runtime dependencies, B6: the analysis LLM call fails, so the error is returned."""
    state = call_agent_in_fresh_process(SUPPORT_QUERY, api_key)
    assert state.get("response") == ERROR_RESPONSE
