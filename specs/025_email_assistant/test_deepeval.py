"""Black-box behavioral evaluation of the email assistant agent (spec 025).

Derived exclusively from spec.md. Entrypoint: agent(email_input: str) -> (CompiledGraph, dict).

Run with:
    deepeval test run test_deepeval.py
"""

import json
import re

import pytest

from deepeval import assert_test
from deepeval.metrics import GEval, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall

from agent import agent


VALID_CLASSIFICATIONS = {"ignore", "notify", "respond", "unknown"}
TOOL_FOR_CLASSIFICATION = {"ignore": "archive_email", "respond": "save_email_draft"}
EXPECTED_KEYS = {"action", "classification", "actions_taken", "status", "message", "draft_response"}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

IGNORE_EMAILS = [
    "CONGRATULATIONS!!! You have been selected to win a FREE iPhone 15. Click "
    "http://win-prizes.example to claim now. Limited time offer, act fast!",
    "Hi there, I'm reaching out from GrowthHackz. We help companies 10x their "
    "SEO with our AI-powered backlink packages. Starting at only $99/month. "
    "Reply STOP to unsubscribe.",
    "BLACK FRIDAY MEGA SALE: 70% off all shoes this weekend only! Shop now at "
    "shoes-outlet.example. You are receiving this because you subscribed to our newsletter.",
]

NOTIFY_EMAILS = [
    "From: CFO\nSubject: Q3 financial results\n\nFor your information: Q3 revenue "
    "came in 12% below forecast and the board has been informed. No action is "
    "needed from you; the full report will be shared at next month's review.",
    "From: HR Department\nSubject: Critical - mandatory policy change effective today\n\n"
    "Effective immediately, all employees must use the new VPN for remote access. "
    "This is an informational notice; no reply is required.",
]

RESPOND_EMAILS = [
    "Hi, I'm Maria from Acme Corp. Could we schedule a 30-minute meeting next "
    "Tuesday or Wednesday to discuss the partnership proposal? Please let me "
    "know which time works best for you.",
    "Hello, I'm a customer and I was charged twice for my last order #48213. "
    "Could you please look into this and tell me how I can get a refund?",
    "Hi! John suggested I reach out to you. I'm the founder of a climate-tech "
    "startup and would love to introduce myself and hear your thoughts on our "
    "product. Would you be open to a short call?",
]

INVALID_INPUTS = ["", "   ", "\n\t  \n", None, 12345, ["not", "text"], {"body": "x"}]


# ---------------------------------------------------------------------------
# Metrics (GEval judged on observable output only)
# ---------------------------------------------------------------------------

def message_quality_metric() -> GEval:
    return GEval(
        name="Message Justification",
        criteria=(
            "The actual output is a JSON processing result of an executive email "
            "assistant. Its 'message' field must contain no more than two sentences "
            "and must justify the classification and the action taken (or the "
            "absence of an action), using a professional, objective and direct tone."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
    )


def classification_metric() -> GEval:
    return GEval(
        name="Classification Correctness",
        criteria=(
            "Given the input email and the agent's JSON result, judge whether the "
            "'classification' matches the expected classification. Classification "
            "rules: spam, marketing or irrelevant cold emails -> 'ignore'; urgent or "
            "critical emails (e.g. financial updates, critical HR messages) that do "
            "not require a direct response -> 'notify'; emails requiring a direct "
            "response (e.g. meeting request, customer inquiry, introduction) -> 'respond'."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
    )


def draft_quality_metric() -> GEval:
    return GEval(
        name="Draft Response Relevance",
        criteria=(
            "The 'draft_response' field of the actual output must be a reply draft "
            "that directly addresses the request in the input email, written in a "
            "professional, objective and direct tone."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
    )


def failure_message_metric(reason: str) -> GEval:
    return GEval(
        name="Failure Message",
        criteria=(
            "The 'message' field of the actual output must clearly describe a "
            f"failure ({reason}) and must NOT claim that the email was classified, "
            "archived, or that a draft was saved successfully."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_agent(email_input):
    out = agent(email_input)
    assert isinstance(out, tuple) and len(out) == 2, "agent() must return (CompiledGraph, dict)"
    graph, result = out
    assert graph is not None
    assert isinstance(result, dict), "second element of the output must be a dict"
    return result


def count_sentences(text: str) -> int:
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    return len(parts)


def assert_invariants(result: dict):
    assert EXPECTED_KEYS.issubset(result.keys()), f"missing keys: {EXPECTED_KEYS - set(result)}"
    # Invariant 1
    assert result["action"] == "process_email"
    # Invariant 2
    assert result["classification"] in VALID_CLASSIFICATIONS
    assert result["status"] in {"success", "failure"}
    assert isinstance(result["actions_taken"], list)
    assert isinstance(result["message"], str) and result["message"].strip()
    # Constraint 1: at most one tool per email
    assert len(result["actions_taken"]) <= 1
    # Invariant 3
    if result["status"] == "failure":
        assert result["classification"] == "unknown"
    cls, status = result["classification"], result["status"]
    # Invariant 4
    if cls == "respond" and status == "success":
        assert isinstance(result["draft_response"], str) and result["draft_response"].strip()
    else:
        assert result["draft_response"] is None
    # Invariant 5 / 6
    if status == "success" and cls in TOOL_FOR_CLASSIFICATION:
        assert result["actions_taken"] == [TOOL_FOR_CLASSIFICATION[cls]]
    else:
        assert result["actions_taken"] == []
    for name in result["actions_taken"]:
        assert name == TOOL_FOR_CLASSIFICATION.get(cls)


def make_test_case(email_input, result, expected_output=None, expected_tools=None) -> LLMTestCase:
    return LLMTestCase(
        input=str(email_input),
        actual_output=json.dumps(result, ensure_ascii=False),
        expected_output=expected_output,
        tools_called=[ToolCall(name=n) for n in result.get("actions_taken", [])],
        expected_tools=expected_tools,
    )


# ---------------------------------------------------------------------------
# B1 — ignore -> archive_email
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("email", IGNORE_EMAILS)
def test_b1_ignore_and_archive(email):
    result = run_agent(email)
    assert_invariants(result)
    assert result["classification"] == "ignore"
    assert result["status"] == "success"
    assert result["actions_taken"] == ["archive_email"]
    assert result["draft_response"] is None
    assert count_sentences(result["message"]) <= 2

    assert_test(
        make_test_case(email, result, expected_output='classification: "ignore"',
                       expected_tools=[ToolCall(name="archive_email")]),
        [classification_metric(), message_quality_metric(), ToolCorrectnessMetric()],
    )


# ---------------------------------------------------------------------------
# B2 — notify -> no tool
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("email", NOTIFY_EMAILS)
def test_b2_notify_without_tool(email):
    result = run_agent(email)
    assert_invariants(result)
    assert result["classification"] == "notify"
    assert result["status"] == "success"
    assert result["actions_taken"] == []
    assert result["draft_response"] is None
    assert count_sentences(result["message"]) <= 2

    assert_test(
        make_test_case(email, result, expected_output='classification: "notify"'),
        [classification_metric(), message_quality_metric()],
    )


# ---------------------------------------------------------------------------
# B3 — respond -> draft + save_email_draft
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("email", RESPOND_EMAILS)
def test_b3_respond_with_draft(email):
    result = run_agent(email)
    assert_invariants(result)
    assert result["classification"] == "respond"
    assert result["status"] == "success"
    assert result["actions_taken"] == ["save_email_draft"]
    assert isinstance(result["draft_response"], str) and result["draft_response"].strip()
    assert count_sentences(result["message"]) <= 2

    assert_test(
        make_test_case(email, result, expected_output='classification: "respond"',
                       expected_tools=[ToolCall(name="save_email_draft")]),
        [classification_metric(), message_quality_metric(), draft_quality_metric(),
         ToolCorrectnessMetric()],
    )


# ---------------------------------------------------------------------------
# B4 — LLM failure (timeout / unparseable output)
# Black-box trigger: an invalid OpenAI key makes the LLM request fail.
# ---------------------------------------------------------------------------

def test_b4_llm_failure(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-invalid-key-for-testing")
    email = RESPOND_EMAILS[0]
    try:
        result = run_agent(email)
    finally:
        monkeypatch.undo()
    assert_invariants(result)
    assert result == {
        **result,
        "action": "process_email",
        "classification": "unknown",
        "actions_taken": [],
        "status": "failure",
        "draft_response": None,
    }
    assert result["message"].strip()

    assert_test(
        make_test_case(email, result),
        [failure_message_metric("LLM processing failed")],
    )


# ---------------------------------------------------------------------------
# B5 — invalid input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_input", INVALID_INPUTS, ids=repr)
def test_b5_invalid_input(bad_input):
    result = run_agent(bad_input)
    assert_invariants(result)
    assert result["classification"] == "unknown"
    assert result["status"] == "failure"
    assert result["actions_taken"] == []
    assert result["draft_response"] is None
    assert result["message"].strip()


def test_b5_invalid_input_message_quality():
    result = run_agent("   ")
    assert_test(
        make_test_case("   ", result),
        [failure_message_metric("the input is invalid (empty, whitespace-only, or not text)")],
    )


# ---------------------------------------------------------------------------
# B6 — action tool failure
# Black-box trigger: missing/invalid Notion configuration makes save_email_draft fail.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "env_change",
    [
        {"NOTION_TOKEN": None},
        {"NOTION_DRAFTS_DB_ID": None},
        {"NOTION_TOKEN": "invalid-notion-token", "NOTION_DRAFTS_DB_ID": "00000000000000000000000000000000"},
    ],
    ids=["missing_token", "missing_db_id", "invalid_credentials"],
)
def test_b6_respond_tool_failure(monkeypatch, env_change):
    for key, value in env_change.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    email = RESPOND_EMAILS[0]
    result = run_agent(email)
    assert_invariants(result)
    assert result["classification"] == "unknown"
    assert result["status"] == "failure"
    assert result["actions_taken"] == []
    assert result["draft_response"] is None

    assert_test(
        make_test_case(email, result),
        [failure_message_metric("the save_email_draft tool failed; the draft must not be claimed as saved")],
    )


# ---------------------------------------------------------------------------
# Cross-cutting guardrails over a mixed set of emails
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("email", IGNORE_EMAILS[:1] + NOTIFY_EMAILS[:1] + RESPOND_EMAILS[:1])
def test_guardrails_hold_for_any_email(email):
    result = run_agent(email)
    assert_invariants(result)
    if result["status"] == "success":
        assert count_sentences(result["message"]) <= 2
