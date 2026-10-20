"""Black-box behavioral evaluation for the Essay Writer Agent (spec v1.0).

Run with: deepeval test run test_deepeval.py

Environment:
    ESSAY_DOC_URL  Public Google Doc URL containing an ENEM topic and supporting
                   texts. Tests that need a successful run are skipped without it.
    OPENAI_API_KEY Required by the agent (gpt-4o) and by the DeepEval judge.
"""
import os
import re
import time

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agent import agent, build_graph

TIME_LIMIT_SECONDS = 30
EXPECTED_KEYS = {"status", "user_request", "response"}
DOC_URL = os.getenv("ESSAY_DOC_URL", "").strip()

requires_doc = pytest.mark.skipif(
    not DOC_URL, reason="ESSAY_DOC_URL not set (public Google Doc with a topic)"
)

SUCCESS_REQUESTS = [
    f"Write an essay which the theme is available at: {DOC_URL}",
    f"Please write an ENEM argumentative essay. The topic is in this document: {DOC_URL}",
]


def _run(request):
    start = time.monotonic()
    result = agent(request)
    elapsed = time.monotonic() - start
    return result, elapsed


def _assert_contract(result, request):
    assert isinstance(result, dict)
    assert set(result.keys()) == EXPECTED_KEYS
    assert result["user_request"] == request
    assert result["status"] in ("success", "failure")
    assert isinstance(result["response"], str)
    assert result["response"].strip()


def _assert_failure(result, request):
    _assert_contract(result, request)
    assert result["status"] == "failure"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
enem_structure_metric = GEval(
    name="ENEM Essay Structure",
    criteria=(
        "Evaluate whether the actual output contains a complete ENEM-style "
        "argumentative essay: (1) an introduction that presents the topic and a "
        "thesis; (2) a development with AT LEAST 2 paragraphs containing arguments "
        "and evidence; (3) a conclusion that restates the thesis and proposes an "
        "intervention."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.6,
)

grounding_metric = GEval(
    name="Grounding in Retrieved Topic",
    criteria=(
        "The actual output must include the topic retrieved from the document and an "
        "essay grounded in that topic (not generic or off-topic). It should be "
        "coherent with a topic and supporting texts that a Google Doc would provide, "
        "and must not contain fabricated placeholder content."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.6,
)

formal_persona_metric = GEval(
    name="Formal Professional Tone",
    criteria=(
        "The essay must use clear, formal, objective and respectful language "
        "appropriate for an ENEM essay writing assistant."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.6,
)

useful_error_metric = GEval(
    name="Useful Error Message",
    criteria=(
        "The actual output is an error message explaining why the essay could not be "
        "produced (e.g. empty request, missing/invalid URL, retrieval problem). It "
        "must be understandable and must NOT contain a fabricated essay or claim "
        "that a file was saved."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5,
)


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------
def test_build_graph_returns_compiled_graph():
    graph = build_graph()
    assert graph is not None
    assert hasattr(graph, "invoke")
    assert hasattr(graph, "get_graph")


# ---------------------------------------------------------------------------
# B1: request validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("request_text", ["", " ", "   ", "\n\t  \n"])
def test_empty_or_blank_input_fails(request_text):
    result, elapsed = _run(request_text)
    _assert_failure(result, request_text)
    assert elapsed <= TIME_LIMIT_SECONDS


@pytest.mark.parametrize(
    "request_text",
    [
        "Write an essay about climate change.",
        "Escreva uma redação sobre educação no Brasil.",
    ],
)
def test_missing_url_fails_with_useful_message(request_text):
    result, elapsed = _run(request_text)
    _assert_failure(result, request_text)
    assert elapsed <= TIME_LIMIT_SECONDS
    test_case = LLMTestCase(input=request_text, actual_output=result["response"])
    assert_test(test_case, [useful_error_metric])


def test_original_request_whitespace_preserved_on_failure():
    request_text = "  \n  Write an essay without any link.  \n"
    result, _ = _run(request_text)
    _assert_contract(result, request_text)
    assert result["status"] == "failure"


# ---------------------------------------------------------------------------
# B2/B6: retrieval failure
# ---------------------------------------------------------------------------
def test_unretrievable_document_returns_failure():
    request_text = (
        "Write an essay which the theme is available at: "
        "https://docs.google.com/document/d/invalid_nonexistent_id_000000/edit"
    )
    result, elapsed = _run(request_text)
    _assert_failure(result, request_text)
    assert elapsed <= TIME_LIMIT_SECONDS
    assert "saved" not in result["response"].lower() or "not" in result["response"].lower()
    test_case = LLMTestCase(input=request_text, actual_output=result["response"])
    assert_test(test_case, [useful_error_metric])


# ---------------------------------------------------------------------------
# B3-B5: successful flow
# ---------------------------------------------------------------------------
@requires_doc
@pytest.mark.parametrize("request_text", SUCCESS_REQUESTS)
def test_success_contract_and_time_limit(request_text):
    result, elapsed = _run(request_text)
    _assert_contract(result, request_text)
    assert elapsed <= TIME_LIMIT_SECONDS
    assert result["status"] == "success", result["response"]


@requires_doc
def test_success_response_contains_valid_saved_path():
    request_text = SUCCESS_REQUESTS[0]
    result, _ = _run(request_text)
    assert result["status"] == "success", result["response"]
    candidates = re.findall(r"(/[^\s\"'`<>|*?]+)", result["response"])
    existing = [
        p.rstrip(".,;:)") for p in candidates
        if os.path.isabs(p.rstrip(".,;:)")) and os.path.isfile(p.rstrip(".,;:)"))
    ]
    assert existing, f"No existing absolute file path found in response: {result['response'][:500]}"
    assert os.path.getsize(existing[-1]) > 0


@requires_doc
def test_success_response_contains_full_essay_not_just_path():
    result, _ = _run(SUCCESS_REQUESTS[0])
    assert result["status"] == "success", result["response"]
    assert len(result["response"].split()) >= 200


@requires_doc
@pytest.mark.parametrize("request_text", SUCCESS_REQUESTS)
def test_essay_enem_structure_and_grounding(request_text):
    result, _ = _run(request_text)
    assert result["status"] == "success", result["response"]
    test_case = LLMTestCase(input=request_text, actual_output=result["response"])
    assert_test(
        test_case,
        [enem_structure_metric, grounding_metric, formal_persona_metric],
    )


@requires_doc
def test_repeated_requests_each_return_valid_contract():
    for request_text in SUCCESS_REQUESTS:
        result, elapsed = _run(request_text)
        _assert_contract(result, request_text)
        assert elapsed <= TIME_LIMIT_SECONDS


# ---------------------------------------------------------------------------
# Never propagate exceptions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "request_text",
    [
        "https://",
        "Write an essay: not-a-url://docs.google.com/document/d/",
        "Write an essay from https://example.com/not-a-google-doc",
    ],
)
def test_malformed_input_never_raises(request_text):
    result, elapsed = _run(request_text)
    _assert_failure(result, request_text)
    assert elapsed <= TIME_LIMIT_SECONDS
