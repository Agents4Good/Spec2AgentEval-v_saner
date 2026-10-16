"""Offline unit tests (LLM and HTTP mocked). Run: pytest tests.py"""

import json
from unittest.mock import MagicMock, patch

import pytest

import agent as A


def _fake_llm(category, query="topic", synth="Final answer."):
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = A.Triage(
        category=category, query=query)
    llm.invoke.return_value = MagicMock(content=synth)
    return llm


@pytest.fixture(autouse=True)
def reset_graph():
    A._GRAPH = None
    yield


def _run(category, tool_outputs=None, synth="Final answer."):
    tool_outputs = tool_outputs or {}
    calls = []
    fakes = {}
    for name in A.TOOLS:
        m = MagicMock()
        m.invoke.side_effect = lambda args, n=name: (
            calls.append(n) or tool_outputs.get(n, "[]"))
        fakes[name] = m
    with patch.object(A, "_llm", return_value=_fake_llm(category, synth=synth)), \
         patch.dict(A.TOOLS, fakes):
        return A.agent("some request"), calls


PAPERS = json.dumps([{"title": "T", "summary": "S"}])


@pytest.mark.parametrize("value", ["", "   ", "\n\t"])
def test_b4_empty(value):
    assert A.agent(value) == A.ERR_EMPTY


def test_b1_arxiv():
    out, calls = _run("exact_tech", {"search_arxiv": PAPERS})
    assert out == "Final answer." and calls == ["search_arxiv"]


def test_b2_pubmed():
    out, calls = _run("life_health", {"search_pubmed": PAPERS})
    assert out == "Final answer." and calls == ["search_pubmed"]


def test_b3_wikipedia():
    data = json.dumps({"title": "X", "extract": "X is..."})
    out, calls = _run("concept", {"get_wikipedia_summary": data})
    assert out == "Final answer." and calls == ["get_wikipedia_summary"]


def test_b5_multi_domain():
    out, calls = _run("multi_domain")
    assert out == A.ERR_MULTI and calls == []


def test_b6_out_of_scope():
    out, calls = _run("out_of_scope")
    assert out == A.ERR_SCOPE and calls == []


def test_b7_no_results():
    out, calls = _run("exact_tech", {"search_arxiv": "[]"})
    assert out == A.ERR_FAIL and calls == ["search_arxiv"]


def test_b7_exception():
    with patch.object(A, "_llm", side_effect=RuntimeError("boom")):
        assert A.agent("papers on AI") == A.ERR_FAIL


def test_b7_empty_llm_response():
    out, _ = _run("exact_tech", {"search_arxiv": PAPERS}, synth="")
    assert out == A.ERR_FAIL


def test_tools_are_langchain_tools():
    from langchain_core.tools import BaseTool
    assert all(isinstance(t, BaseTool) for t in A.TOOLS.values())
