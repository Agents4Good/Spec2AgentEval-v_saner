"""Offline tests for agent.py (LLM and HTTP are mocked)."""

import json
from unittest.mock import MagicMock, patch

import pytest

import agent as A


class FakeLLM:
    def __init__(self, triage, synth="Intro.\n- Title: summary"):
        self.triage, self.synth = triage, synth

    def with_structured_output(self, _schema):
        m = MagicMock()
        m.invoke.return_value = self.triage
        return m

    def invoke(self, _msgs):
        return MagicMock(content=self.synth)


@pytest.fixture(autouse=True)
def reset_graph():
    A._GRAPH = None
    yield


def run(category, query="q", tool_out='[{"title": "T", "summary": "S"}]', synth="Intro.\n- T: S"):
    llm = FakeLLM(A.Triage(category=category, query=query), synth)
    calls = []

    def fake_invoke(name):
        def _inv(args):
            calls.append(name)
            if isinstance(tool_out, Exception):
                raise tool_out
            return tool_out
        return _inv

    fakes = {n: MagicMock(invoke=fake_invoke(n)) for n in A.TOOLS}
    with patch.object(A, "_llm", return_value=llm), patch.dict(A.TOOLS, fakes):
        return A.agent("some request"), calls


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty(text):
    assert A.agent(text) == A.ERR_EMPTY


def test_arxiv_route():
    out, calls = run("exact_tech")
    assert calls == ["search_arxiv"] and "T" in out


def test_pubmed_route():
    out, calls = run("life_health")
    assert calls == ["search_pubmed"]


def test_wikipedia_route():
    out, calls = run("concept", tool_out=json.dumps({"title": "X", "extract": "X is..."}), synth="X is...")
    assert calls == ["get_wikipedia_summary"] and out == "X is..."


def test_multi_domain():
    out, calls = run("multi_domain")
    assert out == A.ERR_MULTI and calls == []


def test_out_of_scope():
    out, calls = run("out_of_scope")
    assert out == A.ERR_SCOPE and calls == []


def test_no_results():
    out, calls = run("exact_tech", tool_out="[]")
    assert out == A.ERR_AGENT and calls == ["search_arxiv"]


def test_tool_failure():
    out, _ = run("life_health", tool_out=RuntimeError("down"))
    assert out == A.ERR_AGENT


def test_empty_synthesis():
    out, _ = run("exact_tech", synth="")
    assert out == A.ERR_AGENT


def test_llm_exception():
    with patch.object(A, "_llm", side_effect=RuntimeError("no key")):
        assert A.agent("papers on transformers") == A.ERR_AGENT


def test_tools_are_langchain_tools():
    from langchain_core.tools import BaseTool
    assert all(isinstance(t, BaseTool) for t in A.TOOLS.values())
