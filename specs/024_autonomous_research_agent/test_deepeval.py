"""Black-box behavioral evaluation suite for the Autonomous Research Agent (spec 024).

Derived exclusively from spec.md. Only the observable output of the entrypoint
`agent(user_request: str) -> str` is evaluated.

Run with:
    deepeval test run test_deepeval.py
"""

import json
from importlib import import_module

import pytest

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

agent = import_module("agent").agent

# B5: exact standard error string
ERROR_JSON = '{"status": "error", "message": "Execution failed or invalid request."}'
REQUIRED_KEYS = {"topic", "discovered_entity", "papers"}


# ---------------------------------------------------------------------------
# Metrics (LLM-as-judge) for requirements that need semantic judgment
# ---------------------------------------------------------------------------

entity_discovery_metric = GEval(
    name="Entity Discovery (B1/B2, Goal)",
    criteria=(
        "The user request requires a multi-stage investigation in which a specific entity "
        "(e.g., a gene, scientist, theorem) must first be discovered. Judge whether the "
        "'discovered_entity' field of the JSON output is a specific, concrete entity that "
        "correctly answers the discovery part of the request, rather than a restatement of "
        "the generic topic of the request."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.6,
)

data_chaining_metric = GEval(
    name="Data Chaining Accuracy (QA1, B3)",
    criteria=(
        "The research papers listed in 'papers' must have been retrieved by searching for the "
        "'discovered_entity' itself, not for the user's original generic topic. Judge whether "
        "the papers are focused on the discovered entity (rather than only the generic topic "
        "of the request)."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5,
)

domain_database_metric = GEval(
    name="Domain-Appropriate Database (I1)",
    criteria=(
        "Life Sciences/Health entities (e.g., genes, diseases) must be researched in PubMed; "
        "Exact Sciences/Technology entities, and entities from any other domain (e.g., "
        "Humanities, Arts), must be researched in ArXiv. Using the titles, summaries and any "
        "source information in 'papers', judge whether the retrieved literature is consistent "
        "with the correct database for the domain of 'discovered_entity' (biomedical "
        "literature for Life Sciences/Health; physics/math/CS-style preprints otherwise)."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5,
)

persona_metric = GEval(
    name="Objective Structured Result (Persona, B4)",
    criteria=(
        "The output must report only the final structured result: no comments, opinions, "
        "conversational text, or Markdown. Judge whether the output is purely a structured "
        "JSON result with a professional, objective content."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

SYNTHESIS_METRICS = [
    entity_discovery_metric,
    data_chaining_metric,
    domain_database_metric,
    persona_metric,
]


# ---------------------------------------------------------------------------
# Inputs: complex natural language requests requiring multi-stage investigation
# ---------------------------------------------------------------------------

LIFE_SCIENCE_REQUESTS = [
    "Find the gene most commonly associated with hereditary breast cancer and research recent papers about it.",
    "Identify the scientist who discovered penicillin and then look up academic research related to that discovery's subject.",
    "What tumor suppressor gene is known as the 'guardian of the genome'? Find research papers about it.",
]

EXACT_SCIENCE_REQUESTS = [
    "Which theorem did Andrew Wiles famously prove in the 1990s? Find academic papers about that theorem.",
    "Identify the neural network architecture introduced in the paper 'Attention Is All You Need' and search for research on it.",
    "Which physicist proposed the uncertainty principle? Find research related to that principle.",
]

OTHER_DOMAIN_REQUESTS = [
    "Identify the art movement founded by André Breton and find academic papers about it.",
]

# B2: discovery is likely to fail on Wikipedia for this obscure phrasing; the agent must not abort.
DISCOVERY_FALLBACK_REQUESTS = [
    "Find the qxzzv-unknown-topic-1234 related to the protein that carries oxygen in red blood cells and research it.",
]

ALL_VALID_REQUESTS = (
    LIFE_SCIENCE_REQUESTS
    + EXACT_SCIENCE_REQUESTS
    + OTHER_DOMAIN_REQUESTS
    + DISCOVERY_FALLBACK_REQUESTS
)

_cache: dict = {}


def run_agent(request: str) -> str:
    if request not in _cache:
        _cache[request] = agent(request)
    return _cache[request]


# ---------------------------------------------------------------------------
# Deterministic checks: output contract (Output, B4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("request_text", ALL_VALID_REQUESTS)
def test_output_is_exclusively_valid_json_with_required_structure(request_text):
    output = run_agent(request_text)

    assert isinstance(output, str)
    stripped = output.strip()
    assert not stripped.startswith("```"), "Output must not contain Markdown formatting"
    data = json.loads(stripped)  # must be exclusively a valid JSON object
    assert isinstance(data, dict)

    assert REQUIRED_KEYS.issubset(data.keys()), f"Missing keys: {REQUIRED_KEYS - set(data)}"
    assert isinstance(data["discovered_entity"], str) and data["discovered_entity"].strip()
    assert isinstance(data["papers"], list)
    for paper in data["papers"]:
        assert isinstance(paper, dict)
        assert "title" in paper
        assert any(k in paper for k in ("source", "summary")), "Paper needs source/summary"


@pytest.mark.parametrize("request_text", DISCOVERY_FALLBACK_REQUESTS)
def test_discovery_failure_does_not_abort(request_text):
    """B2: on Wikipedia failure the agent infers the entity and proceeds (no error JSON)."""
    output = run_agent(request_text)
    assert output.strip() != ERROR_JSON
    data = json.loads(output)
    assert data.get("discovered_entity", "").strip()
    assert "papers" in data


# ---------------------------------------------------------------------------
# Deterministic checks: standard error JSON (B5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_input", ["", "   ", "\n\t  \n"])
def test_empty_or_whitespace_input_returns_exact_error_json(bad_input):
    assert agent(bad_input) == ERROR_JSON


# ---------------------------------------------------------------------------
# LLM-as-judge checks: discovery, chaining, domain routing, persona
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("request_text", ALL_VALID_REQUESTS)
def test_research_synthesis_behavior(request_text):
    output = run_agent(request_text)
    test_case = LLMTestCase(input=request_text, actual_output=output)
    assert_test(test_case=test_case, metrics=SYNTHESIS_METRICS)
