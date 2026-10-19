"""Black-box behavioral evaluation of the Email Assistant agent (spec 015).

Derived exclusively from spec.md. Only the public entrypoint
`agent(email_input: str) -> Tuple[CompiledGraph, dict]` is exercised.

Run with:
    deepeval test run test_deepeval.py
"""

import time
from importlib import import_module

import pytest

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

agent = import_module("agent").agent

# ---------------------------------------------------------------------------
# Spec constants
# ---------------------------------------------------------------------------
EXPECTED_KEYS = {"action", "classification", "actions_taken", "status", "message", "draft_response"}
VALID_CLASSIFICATIONS = {"ignore", "notify", "respond", "unknown"}
RESPOND_ACTIONS = ["get_calendar_events", "save_email_draft"]
MAX_LATENCY_SECONDS = 30  # QA3

# ---------------------------------------------------------------------------
# Inputs (QA2 categories)
# ---------------------------------------------------------------------------
IGNORE_EMAILS = [
    "CONGRATULATIONS!!! You have been selected to win a FREE iPhone 15. Click here now to claim your prize before it expires: http://win-prize.example.biz",
    "Black Friday mega sale! 70% off all shoes this weekend only. Unsubscribe at any time.",
    "Hi, I'm reaching out from GrowthHackers SEO Agency. We can get your website to page one of Google. Interested in a 15-minute chat about our packages?",
]
NOTIFY_EMAILS = [
    "URGENT: The quarterly financial report shows a $2M cash shortfall. The CFO has flagged that payroll funding for next month is at risk. Please make the executive aware immediately.",
    "From HR: A formal harassment complaint has been filed against a senior manager in the sales department. This is a critical, confidential matter that requires leadership attention.",
]
RESPOND_EMAILS = [
    "Hi, I'd like to schedule a 30-minute meeting next Tuesday afternoon to discuss the Q3 partnership proposal. Please let me know what time works. Best, Maria Lopez",
    "Hello, I'm a customer of your enterprise plan and I have a question about the renewal terms for our contract. Could we set up a short call this week? Thanks, John Park",
    "Hi! I'd like to introduce you to Dr. Alan Smith, CTO of DataCorp, who is interested in collaborating with your team. Would you be available for an intro call tomorrow? Regards, Sarah",
]
EMPTY_INPUTS = ["", "   ", "\n\t  \n"]

# ---------------------------------------------------------------------------
# Metrics (LLM-as-judge, GEval)
# ---------------------------------------------------------------------------
classification_metric = GEval(
    name="Classification Accuracy (QA2)",
    criteria=(
        "The input is an email received by an executive assistant. The actual output is the agent's "
        "result dictionary. Determine whether the 'classification' field logically corresponds to the "
        "email content: spam, marketing and irrelevant cold emails must be 'ignore'; urgent financial "
        "updates and critical HR messages must be 'notify'; meeting requests, customer inquiries and "
        "introductions must be 'respond'."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

draft_quality_metric = GEval(
    name="Draft Persona and Tone",
    criteria=(
        "The input is a received email and the actual output is the reply draft written by an executive "
        "assistant. Evaluate whether the draft: (1) is a pertinent reply to the email, (2) is written in a "
        "professional, cordial and objective tone, and (3) is signed as 'Executive Assistant'."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

failure_message_metric = GEval(
    name="Failure Message Describes Failure (B0)",
    criteria=(
        "The input was empty or invalid for an email-processing agent. Determine whether the actual output "
        "message describes a failure (e.g. empty/invalid input) rather than claiming successful processing."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_agent(email_input):
    start = time.monotonic()
    result = agent(email_input)
    elapsed = time.monotonic() - start
    assert isinstance(result, tuple) and len(result) == 2, "Entrypoint must return a 2-tuple"
    graph, output = result
    assert graph is not None, "First tuple element must be the compiled graph"
    return output, elapsed


def assert_invariants(output):
    # QA1 / I4
    assert isinstance(output, dict)
    assert set(output.keys()) == EXPECTED_KEYS, f"Schema mismatch: {sorted(output.keys())}"
    assert output["action"] == "process_email"
    # I1
    assert output["classification"] in VALID_CLASSIFICATIONS
    assert output["status"] in {"success", "failure"}
    assert isinstance(output["message"], str)
    assert isinstance(output["actions_taken"], list)
    assert all(isinstance(a, str) for a in output["actions_taken"])
    assert output["draft_response"] is None or isinstance(output["draft_response"], str)
    # I2 / QA4
    if output["classification"] == "respond":
        assert output["actions_taken"] == RESPOND_ACTIONS
    else:
        assert output["actions_taken"] == []
    # I3
    if output["status"] == "failure":
        assert output["classification"] == "unknown"
        assert output["draft_response"] is None
        assert output["actions_taken"] == []


def assert_failure_dict(output):
    assert_invariants(output)
    assert output["classification"] == "unknown"
    assert output["actions_taken"] == []
    assert output["status"] == "failure"
    assert output["draft_response"] is None
    assert output["message"].strip()


# ---------------------------------------------------------------------------
# B0 — Empty input / failure handling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("email_input", EMPTY_INPUTS, ids=["empty", "spaces", "whitespace_mixed"])
def test_b0_empty_or_whitespace_input_returns_failure(email_input):
    output, _ = run_agent(email_input)  # C1: must not raise
    assert_failure_dict(output)
    assert_test(
        test_case=LLMTestCase(input=repr(email_input), actual_output=output["message"]),
        metrics=[failure_message_metric],
    )


@pytest.mark.parametrize("bad_input", [None, 123], ids=["none", "int"])
def test_b0_invalid_input_does_not_raise(bad_input):
    output, _ = run_agent(bad_input)  # C1: exceptions must not propagate
    assert_failure_dict(output)


# ---------------------------------------------------------------------------
# B1 + B2 — Negative classification (ignore / notify), no tools
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("email_input", IGNORE_EMAILS)
def test_b2_ignore_emails(email_input):
    output, elapsed = run_agent(email_input)
    assert_invariants(output)
    assert elapsed <= MAX_LATENCY_SECONDS
    assert output["classification"] == "ignore"
    assert output["status"] == "success"
    assert output["actions_taken"] == []  # C2
    assert output["draft_response"] is None
    assert output["message"].strip()
    assert_test(
        test_case=LLMTestCase(input=email_input, actual_output=str(output)),
        metrics=[classification_metric],
    )


@pytest.mark.parametrize("email_input", NOTIFY_EMAILS)
def test_b2_notify_emails(email_input):
    output, elapsed = run_agent(email_input)
    assert_invariants(output)
    assert elapsed <= MAX_LATENCY_SECONDS
    assert output["classification"] == "notify"
    assert output["status"] == "success"
    assert output["actions_taken"] == []  # C2
    assert output["draft_response"] is None
    assert output["message"].strip()
    assert_test(
        test_case=LLMTestCase(input=email_input, actual_output=str(output)),
        metrics=[classification_metric],
    )


# ---------------------------------------------------------------------------
# B1 + B3 + B4 — Positive classification (respond), calendar + draft
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("email_input", RESPOND_EMAILS)
def test_b4_respond_emails_produce_saved_draft(email_input):
    output, elapsed = run_agent(email_input)
    assert_invariants(output)
    assert elapsed <= MAX_LATENCY_SECONDS
    if output["status"] == "failure":
        # Tool failures (T1/T2) are allowed to degrade to the B0 dict.
        assert_failure_dict(output)
        pytest.fail(f"Respond flow failed (tool/environment issue?): {output['message']}")
    assert output["classification"] == "respond"
    assert output["status"] == "success"
    assert output["actions_taken"] == RESPOND_ACTIONS
    assert isinstance(output["draft_response"], str) and output["draft_response"].strip()
    assert output["message"].strip()
    assert_test(
        test_case=LLMTestCase(input=email_input, actual_output=str(output)),
        metrics=[classification_metric],
    )
    assert_test(
        test_case=LLMTestCase(input=email_input, actual_output=output["draft_response"]),
        metrics=[draft_quality_metric],
    )


def test_b4_draft_signed_as_executive_assistant():
    output, _ = run_agent(RESPOND_EMAILS[0])
    assert_invariants(output)
    assert output["status"] == "success", output["message"]
    assert "executive assistant" in output["draft_response"].lower()


# ---------------------------------------------------------------------------
# QA1 / B5 — schema compliance across mixed inputs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "email_input",
    IGNORE_EMAILS[:1] + NOTIFY_EMAILS[:1] + RESPOND_EMAILS[:1] + EMPTY_INPUTS[:1] + ["asdf qwerty 12345 ???"],
)
def test_b5_schema_always_compliant(email_input):
    output, elapsed = run_agent(email_input)
    assert_invariants(output)
    assert elapsed <= MAX_LATENCY_SECONDS + 5  # small tolerance for wall-clock overhead
