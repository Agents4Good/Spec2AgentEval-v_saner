"""Black-box behavioral evaluation of the customer-support agent.

Requirements come only from specs/016_support_medium/spec.md. The agent is called
through its benchmark entrypoint, agent(query), and judged on the state it returns.

Not covered, because the behavior is not observable from the returned state: the B7
tool-failure messages, the single-call and no-retry rules, and the unchanged query
passed to the selected tool.

Run with: deepeval test run test_deepeval.py
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

from agent import agent  # Benchmark entrypoint: agent(query) -> state dict.


CLASSIFY_ERROR = "Error: could not classify the query"  # B3, B6

# B2 success format. The tool result is free text, so it is captured as-is.
SUCCESS_FORMAT = re.compile(
    r"Category: (?P<category>Technical|Billing|General)\n"
    r"Sentiment: (?P<sentiment>Positive|Neutral|Negative)\n"
    r"Response: (?P<tool_result>.+)",
    re.DOTALL,
)

SUPPORT_QUERY = "How do I export my monthly report to PDF in the app?"

ESCALATION_ROUTING = GEval(
    name="Escalation Routing",
    criteria=(
        "The Response section confirms that the customer's query was forwarded to "
        "human support, as a handoff confirmation. It must not answer the customer's "
        "question directly."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

SUPPORT_RESPONSE_ROUTING = GEval(
    name="Support Response Routing",
    criteria=(
        "The Response section is a customer-facing support answer that addresses the "
        "customer's query and matches the Category in the output (Technical: product "
        "use or technical problems; Billing: charges, payments or invoices; General: "
        "other support requests). It must not say that the query was forwarded to "
        "human support."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

# Runs the entrypoint in a fresh interpreter so the given environment applies at import.
AGENT_CALL_SCRIPT = (
    "import json, sys\n"
    "from agent import agent\n"
    "print(json.dumps(agent(sys.argv[1]), default=str))\n"
)


def check_contract(state):
    assert isinstance(state, dict), "agent must return a dict"
    assert isinstance(state.get("response"), str), "state['response'] must be a str"
    return state["response"]


def run_agent(query):
    return check_contract(agent(query))


def run_agent_with_env(query, env):
    result = subprocess.run(
        [sys.executable, "-c", AGENT_CALL_SCRIPT, query],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"agent raised instead of returning a state:\n{result.stderr}"
    )
    lines = result.stdout.strip().splitlines()
    assert lines, f"agent returned no state:\n{result.stderr}"
    return check_contract(json.loads(lines[-1]))


@pytest.mark.parametrize(
    "query",
    [pytest.param(SUPPORT_QUERY, id="valid"), pytest.param(None, id="none")],
)
def test_entrypoint_returns_state_with_string_response(query):
    # Benchmark interface contract: state is a dict with a required response: str.
    check_contract(agent(query))


@pytest.mark.parametrize(
    ("query", "category"),
    [
        pytest.param(
            "The app crashes every time I open the reports page. This is ridiculous "
            "and I am furious that it still is not fixed.",
            "Technical",
            id="technical",
        ),
        pytest.param(
            "I was charged twice for my invoice this month and I am really angry "
            "about it. Fix this now.",
            "Billing",
            id="billing",
        ),
        pytest.param(
            "Nobody answered my support email for a whole week and I am very annoyed "
            "with your customer service.",
            "General",
            id="general",
        ),
    ],
)
def test_negative_sentiment_is_escalated_regardless_of_category(query, category):
    # B4 and Constraints: escalate when Sentiment is Negative, whatever the Category.
    response = run_agent(query)
    match = SUCCESS_FORMAT.fullmatch(response)
    assert match, f"response does not follow the success format:\n{response!r}"
    assert match["category"] == category
    assert match["sentiment"] == "Negative"
    assert match["tool_result"].strip(), "escalate result must be non-empty"
    assert_test(
        test_case=LLMTestCase(input=query, actual_output=response),
        metrics=[ESCALATION_ROUTING],
    )


@pytest.mark.parametrize(
    ("query", "category", "sentiment"),
    [
        pytest.param(
            "Thanks for the quick fix last week, the export works great now! Could you "
            "tell me how to schedule a weekly report email?",
            "Technical",
            "Positive",
            id="technical-positive",
        ),
        pytest.param(
            "How do I export my monthly report to PDF in the app?",
            "Technical",
            "Neutral",
            id="technical-neutral",
        ),
        pytest.param(
            "Thank you so much for refunding my last payment so quickly, I really "
            "appreciate it! Can you send me a copy of my latest invoice?",
            "Billing",
            "Positive",
            id="billing-positive",
        ),
        pytest.param(
            "When will my next invoice be issued?",
            "Billing",
            "Neutral",
            id="billing-neutral",
        ),
        pytest.param(
            "Thank you for the excellent service! Could you tell me what your support "
            "hours are on weekends?",
            "General",
            "Positive",
            id="general-positive",
        ),
        pytest.param(
            "What are your customer support opening hours?",
            "General",
            "Neutral",
            id="general-neutral",
        ),
    ],
)
def test_non_negative_sentiment_gets_support_response(query, category, sentiment):
    # B5 and Constraints: handle_support for Positive or Neutral sentiment, using the
    # Category from B1.
    response = run_agent(query)
    match = SUCCESS_FORMAT.fullmatch(response)
    assert match, f"response does not follow the success format:\n{response!r}"
    assert match["category"] == category
    assert match["sentiment"] == sentiment
    assert match["tool_result"].strip(), "handle_support result must be non-empty"
    # handle_support must not repeat the Category or Sentiment analysis.
    assert "Category:" not in match["tool_result"]
    assert "Sentiment:" not in match["tool_result"]
    assert_test(
        test_case=LLMTestCase(input=query, actual_output=response),
        metrics=[SUPPORT_RESPONSE_ROUTING],
    )


def test_query_with_surrounding_whitespace_is_classified():
    # Inputs and B1: a string with at least one non-whitespace character is analyzed.
    query = f"   {SUPPORT_QUERY}   "
    response = run_agent(query)
    match = SUCCESS_FORMAT.fullmatch(response)
    assert match, f"response does not follow the success format:\n{response!r}"
    assert match["category"] == "Technical"
    assert match["sentiment"] == "Neutral"
    assert_test(
        test_case=LLMTestCase(input=query, actual_output=response),
        metrics=[SUPPORT_RESPONSE_ROUTING],
    )


@pytest.mark.parametrize(
    "query",
    [
        pytest.param(None, id="none"),
        pytest.param(42, id="integer"),
        pytest.param(["How do I export my monthly report?"], id="list"),
        pytest.param(b"I was charged twice", id="bytes"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="spaces-only"),
        pytest.param("\n\t \r\n", id="line-breaks-and-tabs-only"),
    ],
)
def test_invalid_input_returns_classification_error(query):
    # B3: None, non-string values, and empty or whitespace-only strings.
    assert run_agent(query) == CLASSIFY_ERROR


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("What is the capital of France?", id="geography"),
        pytest.param("Write a short poem about the ocean.", id="creative-writing"),
        pytest.param("Give me a recipe for banana bread.", id="cooking"),
    ],
)
def test_text_unrelated_to_support_returns_classification_error(query):
    # B1 -> B3: text unrelated to customer support.
    assert run_agent(query) == CLASSIFY_ERROR


@pytest.mark.parametrize(
    "api_key",
    [
        pytest.param("sk-invalid-test-key", id="invalid-key"),
        pytest.param(None, id="missing-key"),
    ],
)
def test_invalid_llm_configuration_returns_classification_error(api_key):
    # Runtime dependencies -> B6: missing or invalid LLM configuration.
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    if api_key is not None:
        env["OPENAI_API_KEY"] = api_key
    assert run_agent_with_env(SUPPORT_QUERY, env) == CLASSIFY_ERROR
