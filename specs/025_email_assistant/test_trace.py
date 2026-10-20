"""Trace-based evaluation of the email assistant agent (spec 025).

Oracle: the Behavioral Graph and the tool/guardrail constraints of the spec.
All assertions are made on the trace returned by `traced(agent)(...)`.
No semantic validation of the agent output is performed.
"""

import pytest

from trace_capture import traced
from agent import agent

ARCHIVE = "archive_email"
DRAFT = "save_email_draft"
ALLOWED_TOOLS = {ARCHIVE, DRAFT}
CLASS_TOOL = {"ignore": ARCHIVE, "notify": None, "respond": DRAFT}

SPAM_EMAIL = (
    "Subject: LIMITED TIME OFFER!!! 90% OFF\n\n"
    "Congratulations! You have been selected to receive an exclusive discount on our "
    "premium SEO services. Click here to claim your prize now. Unsubscribe at any time."
)
NOTIFY_EMAIL = (
    "Subject: Q3 financial results - FYI\n\n"
    "Hi team, the board has approved the Q3 financial statements. Revenue was 12% "
    "below forecast and an internal review will start next week. No reply is needed; "
    "this is for your information only.\n\nCFO Office"
)
RESPOND_EMAIL = (
    "Subject: Meeting request next Tuesday\n\n"
    "Hello, I am Maria from Acme Corp. Could we schedule a 30-minute meeting next "
    "Tuesday afternoon to discuss the partnership proposal? Please let me know which "
    "time works best for you.\n\nBest regards,\nMaria"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _run(email_input):
    output, trace = traced(agent)(email_input)
    assert isinstance(trace, dict), "No trace returned by the benchmark wrapper"
    for key in ("status", "trajectory", "tool_calls", "runs_collected"):
        assert key in trace, f"Trace missing required field '{key}'"
    return output, trace


def _require_execution_evidence(trace):
    """Fail (rather than infer) when the trace carries no evidence of execution."""
    assert trace["runs_collected"] > 0, "Trace collected no runs: insufficient evidence"
    assert trace["trajectory"], "Empty trajectory: insufficient evidence"


def _tool_names(trace):
    return [c["name"] for c in trace["tool_calls"]]


def _assert_only_allowed_tools(trace):
    names = _tool_names(trace)
    extra = [n for n in names if n not in ALLOWED_TOOLS]
    assert not extra, f"Guardrail (4) violated, unspecified tools executed: {extra}"
    assert len(names) <= 1, f"Guardrail (1) violated, more than one tool executed: {names}"


def _assert_tool_preceded_by_classification(trace, tool):
    """Graph: Receive -> Classify -> Invoke tool. The tool must not be the first event."""
    traj = trace["trajectory"]
    assert tool in traj, f"Tool '{tool}' not observable in trajectory {traj}"
    assert traj.index(tool) > 0, (
        f"'{tool}' invoked before any prior step (classification) in trajectory {traj}"
    )


def _assert_tool_call_well_formed(call):
    assert isinstance(call.get("input"), dict) and call["input"], (
        f"Tool call '{call['name']}' has no observable input arguments"
    )
    assert call.get("start") and call.get("end"), "Tool call missing timing evidence"
    assert call["start"] <= call["end"], "Tool call ends before it starts"
    assert ("output" in call) or ("error" in call), (
        f"Tool call '{call['name']}' has neither output nor error: insufficient evidence"
    )


# --------------------------------------------------------------------------- #
# Fixtures: one agent invocation per scenario
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ignore_run():
    return _run(SPAM_EMAIL)


@pytest.fixture(scope="module")
def notify_run():
    return _run(NOTIFY_EMAIL)


@pytest.fixture(scope="module")
def respond_run():
    return _run(RESPOND_EMAIL)


@pytest.fixture(scope="module", params=["", "   \n\t  "], ids=["empty", "whitespace"])
def invalid_text_run(request):
    return _run(request.param)


@pytest.fixture(scope="module", params=[None, 12345, ["not", "text"]], ids=["none", "int", "list"])
def non_text_run(request):
    return _run(request.param)


@pytest.fixture
def draft_tool_failure_run(monkeypatch):
    # Spec T2 failure mode: "missing configuration".
    monkeypatch.delenv("NOTION_DRAFTS_DB_ID", raising=False)
    monkeypatch.delenv("NOTION_TOKEN", raising=False)
    return _run(RESPOND_EMAIL)


# --------------------------------------------------------------------------- #
# B1 — ignore -> archive_email
# --------------------------------------------------------------------------- #
class TestB1Ignore:
    def test_agent_ran_without_raising(self, ignore_run):
        _, trace = ignore_run
        assert trace["status"] == "ok", trace.get("error")
        _require_execution_evidence(trace)

    def test_only_archive_email_executed_once(self, ignore_run):
        _, trace = ignore_run
        _assert_only_allowed_tools(trace)
        assert _tool_names(trace) == [ARCHIVE], (
            f"Expected exactly one '{ARCHIVE}' call, got {_tool_names(trace)}"
        )

    def test_save_email_draft_not_executed(self, ignore_run):
        _, trace = ignore_run
        assert DRAFT not in _tool_names(trace)
        assert DRAFT not in trace["trajectory"]

    def test_archive_call_has_evidence(self, ignore_run):
        _, trace = ignore_run
        calls = [c for c in trace["tool_calls"] if c["name"] == ARCHIVE]
        assert calls, f"No '{ARCHIVE}' call observable in the trace"
        _assert_tool_call_well_formed(calls[0])

    def test_classification_precedes_archive(self, ignore_run):
        _, trace = ignore_run
        _assert_tool_preceded_by_classification(trace, ARCHIVE)


# --------------------------------------------------------------------------- #
# B2 — notify -> no tool
# --------------------------------------------------------------------------- #
class TestB2Notify:
    def test_agent_ran_without_raising(self, notify_run):
        _, trace = notify_run
        assert trace["status"] == "ok", trace.get("error")
        _require_execution_evidence(trace)

    def test_no_tool_executed(self, notify_run):
        _, trace = notify_run
        assert trace["tool_calls"] == [], (
            f"Guardrail (2) violated, notify executed tools: {_tool_names(trace)}"
        )

    def test_no_tool_in_trajectory(self, notify_run):
        _, trace = notify_run
        assert not ALLOWED_TOOLS & set(trace["trajectory"])


# --------------------------------------------------------------------------- #
# B3 — respond -> generate draft -> save_email_draft
# --------------------------------------------------------------------------- #
class TestB3Respond:
    def test_agent_ran_without_raising(self, respond_run):
        _, trace = respond_run
        assert trace["status"] == "ok", trace.get("error")
        _require_execution_evidence(trace)

    def test_only_save_email_draft_executed_once(self, respond_run):
        _, trace = respond_run
        _assert_only_allowed_tools(trace)
        assert _tool_names(trace) == [DRAFT], (
            f"Expected exactly one '{DRAFT}' call, got {_tool_names(trace)}"
        )

    def test_archive_email_not_executed(self, respond_run):
        _, trace = respond_run
        assert ARCHIVE not in _tool_names(trace)
        assert ARCHIVE not in trace["trajectory"]

    def test_draft_call_has_evidence(self, respond_run):
        _, trace = respond_run
        calls = [c for c in trace["tool_calls"] if c["name"] == DRAFT]
        assert calls, f"No '{DRAFT}' call observable in the trace"
        _assert_tool_call_well_formed(calls[0])

    def test_draft_call_receives_body_and_draft(self, respond_run):
        """T2 inputs: the email body and the generated reply draft (two non-empty values)."""
        _, trace = respond_run
        calls = [c for c in trace["tool_calls"] if c["name"] == DRAFT]
        assert calls, f"No '{DRAFT}' call observable in the trace"
        str_args = [v for v in calls[0]["input"].values() if isinstance(v, str) and v.strip()]
        assert len(str_args) >= 2, (
            f"'{DRAFT}' input lacks evidence of both email body and draft: {calls[0]['input']}"
        )

    def test_draft_generation_precedes_save(self, respond_run):
        # Graph: Classify -> Generate reply draft -> Invoke save_email_draft.
        _, trace = respond_run
        _assert_tool_preceded_by_classification(trace, DRAFT)


# --------------------------------------------------------------------------- #
# B5 — invalid input -> no tool
# --------------------------------------------------------------------------- #
class TestB5InvalidInput:
    def test_empty_or_whitespace_executes_no_tool(self, invalid_text_run):
        _, trace = invalid_text_run
        assert trace["tool_calls"] == [], (
            f"Invalid input must not execute tools: {_tool_names(trace)}"
        )
        assert not ALLOWED_TOOLS & set(trace["trajectory"])

    def test_empty_or_whitespace_does_not_raise(self, invalid_text_run):
        _, trace = invalid_text_run
        assert trace["status"] == "ok", (
            f"Agent must return the failure schema, not raise: {trace.get('error')}"
        )

    def test_non_text_executes_no_tool(self, non_text_run):
        _, trace = non_text_run
        assert trace["tool_calls"] == [], (
            f"Non-text input must not execute tools: {_tool_names(trace)}"
        )
        assert not ALLOWED_TOOLS & set(trace["trajectory"])

    def test_non_text_does_not_raise(self, non_text_run):
        _, trace = non_text_run
        assert trace["status"] == "ok", (
            f"Agent must return the failure schema, not raise: {trace.get('error')}"
        )


# --------------------------------------------------------------------------- #
# B6 — action tool fails -> no second/alternative tool call
# --------------------------------------------------------------------------- #
class TestB6ToolFailure:
    def test_failure_handled_with_single_mapped_call(self, draft_tool_failure_run):
        _, trace = draft_tool_failure_run
        assert trace["status"] == "ok", (
            f"Agent must return the failure schema, not raise: {trace.get('error')}"
        )
        _require_execution_evidence(trace)
        _assert_only_allowed_tools(trace)
        names = _tool_names(trace)
        assert names == [DRAFT], (
            f"Expected a single '{DRAFT}' attempt with no retry/alternative, got {names}"
        )
        assert ARCHIVE not in trace["trajectory"]


# --------------------------------------------------------------------------- #
# Cross-cutting guardrails over all valid-email scenarios
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fixture_name", ["ignore_run", "notify_run", "respond_run"])
def test_at_most_one_permitted_tool(fixture_name, request):
    _, trace = request.getfixturevalue(fixture_name)
    _assert_only_allowed_tools(trace)


@pytest.mark.parametrize("fixture_name", ["ignore_run", "respond_run"])
def test_tool_trajectory_consistent_with_tool_calls(fixture_name, request):
    _, trace = request.getfixturevalue(fixture_name)
    traj_tools = [t for t in trace["trajectory"] if t in ALLOWED_TOOLS]
    assert traj_tools == _tool_names(trace), (
        f"Trajectory tools {traj_tools} disagree with tool_calls {_tool_names(trace)}"
    )
