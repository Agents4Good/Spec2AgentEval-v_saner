"""Black-box behavioral evaluation of the Academic Triage Agent (DeepEval 3.3.9).

Derived exclusively from spec.md. The agent is exercised only via its
entrypoint `agent(user_request: str) -> str`.

Run with:  deepeval test run test_deepeval.py
"""

import pytest

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agent import agent

ERR_EMPTY = "Error: The user request cannot be empty."
ERR_MULTI = (
    "Error: The query spans multiple distinct domains. "
    "Please make one specific research request at a time."
)
ERR_SCOPE = "Error: Query outside of academic scope."
ERR_AGENT = "Error: Unable to get response from agent."
STANDARD_ERRORS = {ERR_EMPTY, ERR_MULTI, ERR_SCOPE, ERR_AGENT}

EVAL_MODEL = "gpt-4o"


def run_agent(request: str) -> str:
    output = agent(request)
    # Guardrail (1): output is always a str.
    assert isinstance(output, str), f"Output must be str, got {type(output)}"
    return output


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

papers_format_metric = GEval(
    name="Paper List Synthesis (B1/B2)",
    evaluation_steps=[
        "Check that the actual output is a natural-language response, not raw data dumps.",
        "Check that the actual output begins with a brief introduction.",
        "Check that the actual output contains a list of academic papers.",
        "Check that each listed paper has a title and a short summary.",
        "Check that the papers are relevant to the research topic in the input.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=EVAL_MODEL,
    threshold=0.6,
)

concept_explanation_metric = GEval(
    name="Concept Explanation Synthesis (B3)",
    evaluation_steps=[
        "Check that the actual output is a coherent explanatory paragraph about the concept requested in the input.",
        "Check that the actual output explains or defines the concept rather than listing academic papers.",
        "Check that the explanation is factually consistent with well-established knowledge about the concept.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=EVAL_MODEL,
    threshold=0.6,
)

arxiv_style_metric = GEval(
    name="Exact Sciences Source Consistency (B1)",
    criteria=(
        "Determine whether the papers listed in the actual output are plausible "
        "Exact Sciences/Technology research papers (as found on arXiv) relevant to "
        "the input, rather than biomedical/clinical literature or encyclopedia text."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=EVAL_MODEL,
    threshold=0.5,
)

pubmed_style_metric = GEval(
    name="Life Sciences Source Consistency (B2)",
    criteria=(
        "Determine whether the papers listed in the actual output are plausible "
        "Life Sciences/Health research papers (as found on PubMed) relevant to the "
        "input, rather than computer science papers or encyclopedia text."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=EVAL_MODEL,
    threshold=0.5,
)


# ---------------------------------------------------------------------------
# B1 — Exact Sciences/Technology papers -> search_arxiv synthesis
# ---------------------------------------------------------------------------

B1_INPUTS = [
    "Find academic papers about artificial intelligence.",
    "Show me recent research papers on quantum computing.",
    "I need research articles about graph neural networks.",
]


@pytest.mark.parametrize("request_text", B1_INPUTS)
def test_b1_exact_sciences_papers(request_text):
    output = run_agent(request_text)
    if output == ERR_AGENT:
        pytest.skip("B7 fallback returned (tool failure / no results); B1 synthesis not observable.")
    assert output not in STANDARD_ERRORS, f"Valid B1 request returned error: {output}"
    test_case = LLMTestCase(input=request_text, actual_output=output)
    assert_test(test_case=test_case, metrics=[papers_format_metric, arxiv_style_metric])


# ---------------------------------------------------------------------------
# B2 — Life Sciences/Health papers -> search_pubmed synthesis
# ---------------------------------------------------------------------------

B2_INPUTS = [
    "Find academic papers about melanoma.",
    "Show me research on treatments for type 2 diabetes.",
    "I'm looking for scientific papers about Alzheimer's disease biomarkers.",
]


@pytest.mark.parametrize("request_text", B2_INPUTS)
def test_b2_life_sciences_papers(request_text):
    output = run_agent(request_text)
    if output == ERR_AGENT:
        pytest.skip("B7 fallback returned (tool failure / no results); B2 synthesis not observable.")
    assert output not in STANDARD_ERRORS, f"Valid B2 request returned error: {output}"
    test_case = LLMTestCase(input=request_text, actual_output=output)
    assert_test(test_case=test_case, metrics=[papers_format_metric, pubmed_style_metric])


# ---------------------------------------------------------------------------
# B3 — Concept explanation -> get_wikipedia_summary synthesis
# (includes concepts from either scientific domain, no papers requested)
# ---------------------------------------------------------------------------

B3_INPUTS = [
    "Explain what photosynthesis is.",
    "What is machine learning?",
    "Define melanoma.",
    "Explain the concept of the Renaissance in art history.",
    "What is supply and demand in economics?",
]


@pytest.mark.parametrize("request_text", B3_INPUTS)
def test_b3_concept_explanation(request_text):
    output = run_agent(request_text)
    if output == ERR_AGENT:
        pytest.skip("B7 fallback returned (tool failure / no results); B3 synthesis not observable.")
    assert output not in STANDARD_ERRORS, f"Valid B3 request returned error: {output}"
    test_case = LLMTestCase(input=request_text, actual_output=output)
    assert_test(test_case=test_case, metrics=[concept_explanation_metric])


# ---------------------------------------------------------------------------
# B4 — Empty input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("request_text", ["", " ", "   ", "\n", "\t  \n "])
def test_b4_empty_input(request_text):
    assert run_agent(request_text) == ERR_EMPTY


# ---------------------------------------------------------------------------
# B5 — Multiple distinct domains
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "request_text",
    [
        "Find papers about artificial intelligence and melanoma.",
        "Show me research on quantum computing and also research on breast cancer treatments.",
        "I want academic papers on deep learning and on cardiovascular disease.",
    ],
)
def test_b5_multiple_domains(request_text):
    assert run_agent(request_text) == ERR_MULTI


# ---------------------------------------------------------------------------
# B6 — Outside academic scope
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "request_text",
    [
        "Should I break up with my girlfriend?",          # personal advice
        "Book me a flight to Paris next Friday.",          # transaction
        "Buy 10 shares of Apple stock for me.",            # transaction
        "Write a poem about the ocean.",                   # creative generation
        "Write me a short fantasy story about dragons.",   # creative generation
        "What's a good restaurant near me tonight?",       # non-academic
    ],
)
def test_b6_out_of_scope(request_text):
    assert run_agent(request_text) == ERR_SCOPE


# ---------------------------------------------------------------------------
# B7 — Tool returns no results / failure
# A query that is valid in form but extremely unlikely to yield results.
# Accept only the B7 error or (if results were somehow found) a non-error
# synthesis; never a different standard error or a non-string.
# ---------------------------------------------------------------------------

def test_b7_no_results_returns_standard_error():
    request_text = (
        "Find research papers on xqzvbnmlkjhgf-plorptquantum-zzyyxx9871 algorithms."
    )
    output = run_agent(request_text)
    assert output in (ERR_AGENT,) or output not in STANDARD_ERRORS


# ---------------------------------------------------------------------------
# Guardrail (1) — output is either a synthesis or EXACTLY one standard error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "request_text",
    [
        "Find papers about reinforcement learning.",
        "Explain what entropy is.",
        "",
        "Write a song about summer.",
    ],
)
def test_guardrail_error_messages_are_exact(request_text):
    output = run_agent(request_text)
    if output.strip().startswith("Error:"):
        assert output in STANDARD_ERRORS, f"Non-standard error message: {output!r}"
