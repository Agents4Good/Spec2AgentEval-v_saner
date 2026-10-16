"""Black-box behavioral evals for the academic triage agent.

Requirements come only from specs/014_academic_triage_agent/spec.md: behaviors
B1-B7, policies (1)-(4), and the interface contract. The agent is treated as a
black box. The suite uses only its entrypoint, the string it returns, the tool
runs it dispatches (matched by the spec's tool names), and OPENAI_API_KEY.

Run from the directory that contains agent.py:

    deepeval test run test_deepeval.py

Needs OPENAI_API_KEY (the agent's LLM and DeepEval's default judge model) and
network access to arXiv, PubMed and Wikipedia for the B1-B3 tool runs.

Not covered, and why:
- B7 tool-failure and no-results paths: no black-box trigger without faulting the external services.
- Paper requests in humanities or social sciences: the spec defines no tool for them.
- Goal (LLM as reasoning engine), Persona and Quality Attributes: not observable or not applicable.
"""

import asyncio
import inspect
import json
import os
import subprocess
import sys
from contextvars import ContextVar

import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook

from agent import agent

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Exact standard messages (B4-B7).
EMPTY_REQUEST_ERROR = "Error: The user request cannot be empty."
MULTI_DOMAIN_ERROR = (
    "Error: The query spans multiple distinct domains. "
    "Please make one specific research request at a time."
)
OUT_OF_SCOPE_ERROR = "Error: Query outside of academic scope."
AGENT_FAILURE_ERROR = "Error: Unable to get response from agent."
STANDARD_ERRORS = {
    EMPTY_REQUEST_ERROR,
    MULTI_DOMAIN_ERROR,
    OUT_OF_SCOPE_ERROR,
    AGENT_FAILURE_ERROR,
}

# Tool names from the spec's Tools section.
SEARCH_ARXIV = "search_arxiv"  # T1: Exact Sciences/Technology papers
SEARCH_PUBMED = "search_pubmed"  # T2: Life Sciences/Health papers
GET_WIKIPEDIA_SUMMARY = "get_wikipedia_summary"  # T3: concept explanations

EXACT_SCIENCES_PAPER_REQUESTS = [
    "Find recent academic papers on artificial intelligence.",
    "I need research papers about quantum computing.",
]
LIFE_SCIENCES_PAPER_REQUESTS = [
    "Find recent research papers on melanoma.",
    "Are there academic papers on type 2 diabetes treatment?",
]
CONCEPT_REQUESTS = [
    # Life Sciences concept, no paper request
    "What is photosynthesis?",
    # Exact Sciences concept, no paper request
    "Explain what a black hole is.",
    # Humanities and social science concepts are in scope (policy 4)
    "What is existentialism?",
    "Explain the concept of supply and demand in economics.",
]
# Same topic, paper request vs. concept request: policy 3 maps them to different tools.
PAPER_VS_CONCEPT_CASES = [
    ("Find research papers on machine learning.", SEARCH_ARXIV),
    ("What is machine learning?", GET_WIKIPEDIA_SUMMARY),
    ("Find research papers on melanoma.", SEARCH_PUBMED),
    ("What is melanoma?", GET_WIKIPEDIA_SUMMARY),
]
EMPTY_REQUESTS = ["", "   ", "\n\t \n"]
MULTI_DOMAIN_REQUESTS = [
    "Find papers about artificial intelligence and melanoma.",
    "What is photosynthesis, and are there research papers on quantum computing?",
]
OUT_OF_SCOPE_REQUESTS = [
    # personal advice
    "Should I break up with my partner? I need personal advice.",
    # transaction
    "Book a table for two at an Italian restaurant tonight.",
    # creative generation
    "Write a short poem about the ocean.",
    # other non-academic request
    "What will the weather be like in Recife tomorrow?",
]


_active_recorder = ContextVar("active_tool_recorder", default=None)


def _as_text(value):
    content = getattr(value, "content", value)  # unwrap ToolMessage
    if content is None:
        return ""
    return content if isinstance(content, str) else json.dumps(content, default=str)


class ToolCallRecorder(BaseCallbackHandler):
    """Records every tool run the agent dispatches through LangChain, with the data it returned.

    The spec requires tools to run through invoke/ainvoke, so every tool run is observed here.
    """

    def __init__(self):
        self.calls = []
        self._calls_by_run_id = {}

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        call = {"name": (serialized or {}).get("name"), "output": None}
        self._calls_by_run_id[run_id] = call
        self.calls.append(call)

    def on_tool_end(self, output, *, run_id, **kwargs):
        if run_id in self._calls_by_run_id:
            self._calls_by_run_id[run_id]["output"] = _as_text(output)


# Lets the recorder see every LangChain run started while it is set, without touching the agent.
register_configure_hook(_active_recorder, inheritable=True)


def run_agent(user_request):
    """Calls the entrypoint once. Returns its output and the tool runs it triggered."""
    recorder = ToolCallRecorder()
    token = _active_recorder.set(recorder)
    try:
        output = agent(user_request)
        if inspect.isawaitable(output):
            output = asyncio.run(output)
    finally:
        _active_recorder.reset(token)
    assert isinstance(output, str), f"agent must return str, got {type(output).__name__}"
    return output, recorder.calls


def observed_tool_names(calls):
    return [call["name"] for call in calls]


def assert_only_tool_ran_once(calls, expected_tool):
    observed = observed_tool_names(calls)
    assert observed == [expected_tool], (
        f"expected exactly one run of {expected_tool!r}, observed {observed}"
    )


def assert_no_tool_ran(calls):
    observed = observed_tool_names(calls)
    assert observed == [], f"no tool may run for this request, observed {observed}"


def assert_synthesized_response(output):
    assert output.strip() and output not in STANDARD_ERRORS, (
        f"expected a synthesized response, got {output!r}"
    )


PAPER_LIST_CRITERIA = (
    "The actual output must contain a brief introduction followed by a list of academic papers. "
    "Every listed paper must include a title and a short summary."
)
CONCEPT_CRITERIA = (
    "The actual output must be a coherent explanatory paragraph about the concept named in the "
    "input. It must explain the concept itself rather than list research papers."
)


def paper_list_metric():
    return GEval(
        name="Paper List Format",
        criteria=PAPER_LIST_CRITERIA,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7,
    )


def concept_explanation_metric():
    return GEval(
        name="Concept Explanation Format",
        criteria=CONCEPT_CRITERIA,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7,
    )


def grounding_metric():
    # "Based exclusively on the data returned by the tool": every claim must be supported by it.
    return FaithfulnessMetric(threshold=0.8)


def build_test_case(user_request, output, calls):
    retrieval_context = [call["output"] for call in calls if call["output"]]
    assert retrieval_context, "the tool returned no data to ground the response in"
    return LLMTestCase(
        input=user_request,
        actual_output=output,
        retrieval_context=retrieval_context,
    )


@pytest.mark.parametrize("user_request", EXACT_SCIENCES_PAPER_REQUESTS)
def test_b1_exact_sciences_paper_request_runs_search_arxiv_once(user_request):
    """B1: one run of search_arxiv; the response is a paper list grounded in its data."""
    output, calls = run_agent(user_request)
    assert_only_tool_ran_once(calls, SEARCH_ARXIV)
    assert_synthesized_response(output)
    assert_test(
        test_case=build_test_case(user_request, output, calls),
        metrics=[paper_list_metric(), grounding_metric()],
    )


@pytest.mark.parametrize("user_request", LIFE_SCIENCES_PAPER_REQUESTS)
def test_b2_life_sciences_paper_request_runs_search_pubmed_once(user_request):
    """B2: one run of search_pubmed; the response is a paper list grounded in its data."""
    output, calls = run_agent(user_request)
    assert_only_tool_ran_once(calls, SEARCH_PUBMED)
    assert_synthesized_response(output)
    assert_test(
        test_case=build_test_case(user_request, output, calls),
        metrics=[paper_list_metric(), grounding_metric()],
    )


@pytest.mark.parametrize("user_request", CONCEPT_REQUESTS)
def test_b3_concept_request_runs_get_wikipedia_summary_once(user_request):
    """B3: one run of get_wikipedia_summary; the response explains the concept from its data."""
    output, calls = run_agent(user_request)
    assert_only_tool_ran_once(calls, GET_WIKIPEDIA_SUMMARY)
    assert_synthesized_response(output)
    assert_test(
        test_case=build_test_case(user_request, output, calls),
        metrics=[concept_explanation_metric(), grounding_metric()],
    )


@pytest.mark.parametrize(("user_request", "expected_tool"), PAPER_VS_CONCEPT_CASES)
def test_policy3_same_topic_paper_and_concept_requests_use_different_tools(
    user_request, expected_tool
):
    """Policy 3: explicit paper requests and concept requests map to different tools."""
    output, calls = run_agent(user_request)
    assert_only_tool_ran_once(calls, expected_tool)
    assert_synthesized_response(output)


@pytest.mark.parametrize(
    "user_request", EMPTY_REQUESTS, ids=["empty", "spaces", "tabs-and-newlines"]
)
def test_b4_empty_request_returns_standard_error_without_tools(user_request):
    """B4: empty or whitespace-only input gets the exact error and runs no tool."""
    output, calls = run_agent(user_request)
    assert output == EMPTY_REQUEST_ERROR
    assert_no_tool_ran(calls)


@pytest.mark.parametrize("user_request", MULTI_DOMAIN_REQUESTS)
def test_b5_multi_domain_request_returns_standard_error_without_tools(user_request):
    """B5: a request spanning several domains gets the exact error and runs no tool."""
    output, calls = run_agent(user_request)
    assert output == MULTI_DOMAIN_ERROR
    assert_no_tool_ran(calls)


@pytest.mark.parametrize("user_request", OUT_OF_SCOPE_REQUESTS)
def test_b6_out_of_scope_request_returns_standard_error_without_tools(user_request):
    """B6: a non-academic request gets the exact scope error and runs no tool."""
    output, calls = run_agent(user_request)
    assert output == OUT_OF_SCOPE_ERROR
    assert_no_tool_ran(calls)


# Runs the agent in a fresh interpreter so OPENAI_API_KEY is set before the agent loads.
_AGENT_PROBE = """
import asyncio
import inspect
import json
import sys

from agent import agent

output = agent(sys.argv[1])
if inspect.isawaitable(output):
    output = asyncio.run(output)
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(output, handle)
"""


def run_agent_with_api_key(user_request, api_key, tmp_path):
    result_path = tmp_path / "agent_output.json"
    completed = subprocess.run(
        [sys.executable, "-c", _AGENT_PROBE, user_request, str(result_path)],
        cwd=AGENT_DIR,
        env={**os.environ, "OPENAI_API_KEY": api_key},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, (
        f"agent raised instead of returning a response:\n{completed.stderr}"
    )
    return json.loads(result_path.read_text(encoding="utf-8"))


def test_b7_agent_failure_returns_standard_error(tmp_path):
    """B7: an exception during execution returns the exact failure message.

    An invalid OPENAI_API_KEY (spec: Implementation Details) makes the LLM call raise.
    """
    output = run_agent_with_api_key("Explain what photosynthesis is.", "sk-invalid-key", tmp_path)
    assert output == AGENT_FAILURE_ERROR


def test_b7_control_same_request_succeeds_with_valid_key(tmp_path):
    """Control for B7: with a valid key the same request is answered, so the failure above
    comes from the injected key rather than from a broken agent.
    """
    valid_key = os.environ.get("OPENAI_API_KEY")
    if not valid_key:
        pytest.skip("OPENAI_API_KEY is required for the control run")
    output = run_agent_with_api_key("Explain what photosynthesis is.", valid_key, tmp_path)
    assert_synthesized_response(output)
