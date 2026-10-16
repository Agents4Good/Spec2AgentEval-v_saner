import pytest
from unittest.mock import patch, MagicMock
from agent import agent, search_arxiv, search_pubmed, get_wikipedia_summary


# B4 — Empty input
class TestEmptyInput:
    def test_empty_string(self):
        assert agent("") == "Error: The user request cannot be empty."

    def test_whitespace_only(self):
        assert agent("   ") == "Error: The user request cannot be empty."

    def test_newlines_only(self):
        assert agent("\n\t\n") == "Error: The user request cannot be empty."


# B1 — Exact Sciences / arXiv
class TestExactSciences:
    @patch("agent.ChatOpenAI")
    def test_arxiv_called_for_ai_papers(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        tool_response = MagicMock()
        tool_response.content = ""
        tool_response.tool_calls = [
            {"name": "search_arxiv", "args": {"query": "artificial intelligence"}, "id": "call_1"}
        ]
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke = MagicMock(side_effect=[
            tool_response,
            MagicMock(content="Here are papers about AI:\n1. Title: Deep Learning\nSummary: A survey."),
        ])

        with patch.object(search_arxiv, "invoke", return_value="Title: Deep Learning\nSummary: A survey of deep learning."):
            result = agent("Find research papers about artificial intelligence")

        assert "Deep Learning" in result or "paper" in result.lower() or "AI" in result

    @patch("agent.ChatOpenAI")
    def test_arxiv_only_tool_called(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        tool_response = MagicMock()
        tool_response.content = ""
        tool_response.tool_calls = [
            {"name": "search_arxiv", "args": {"query": "quantum computing"}, "id": "call_1"}
        ]
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke = MagicMock(side_effect=[
            tool_response,
            MagicMock(content="Papers on quantum computing found."),
        ])

        with patch.object(search_arxiv, "invoke", return_value="Title: Quantum Algorithms\nSummary: Overview.") as arxiv_mock, \
             patch.object(search_pubmed, "invoke") as pubmed_mock, \
             patch.object(get_wikipedia_summary, "invoke") as wiki_mock:
            agent("Find papers on quantum computing")

        arxiv_mock.assert_called_once()
        pubmed_mock.assert_not_called()
        wiki_mock.assert_not_called()


# B2 — Life Sciences / PubMed
class TestLifeSciences:
    @patch("agent.ChatOpenAI")
    def test_pubmed_called_for_health(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        tool_response = MagicMock()
        tool_response.content = ""
        tool_response.tool_calls = [
            {"name": "search_pubmed", "args": {"query": "melanoma treatment"}, "id": "call_1"}
        ]
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke = MagicMock(side_effect=[
            tool_response,
            MagicMock(content="Here are papers on melanoma treatment."),
        ])

        with patch.object(search_pubmed, "invoke", return_value="Title: Melanoma Immunotherapy\nSource: Nature\nDate: 2024"):
            result = agent("Find research papers about melanoma treatment")

        assert isinstance(result, str)
        assert len(result) > 0


# B3 — General Concepts / Wikipedia
class TestGeneralConcepts:
    @patch("agent.ChatOpenAI")
    def test_wikipedia_for_concept(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        tool_response = MagicMock()
        tool_response.content = ""
        tool_response.tool_calls = [
            {"name": "get_wikipedia_summary", "args": {"topic": "photosynthesis"}, "id": "call_1"}
        ]
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke = MagicMock(side_effect=[
            tool_response,
            MagicMock(content="Photosynthesis is the process by which plants convert light energy into chemical energy."),
        ])

        with patch.object(get_wikipedia_summary, "invoke", return_value="Topic: Photosynthesis\nSummary: Photosynthesis is a process..."):
            result = agent("What is photosynthesis?")

        assert "photosynthesis" in result.lower()


# B5 — Multiple domains
class TestMultipleDomains:
    @patch("agent.ChatOpenAI")
    def test_multi_domain_error(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        response = MagicMock()
        response.content = "Error: The query spans multiple distinct domains. Please make one specific research request at a time."
        response.tool_calls = []
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = response

        result = agent("Find papers about artificial intelligence and melanoma")
        assert result == "Error: The query spans multiple distinct domains. Please make one specific research request at a time."


# B6 — Outside academic scope
class TestOutsideScope:
    @patch("agent.ChatOpenAI")
    def test_non_academic_error(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        response = MagicMock()
        response.content = "Error: Query outside of academic scope."
        response.tool_calls = []
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = response

        result = agent("Book me a flight to Paris")
        assert result == "Error: Query outside of academic scope."


# B7 — Error handling
class TestErrorHandling:
    @patch("agent.ChatOpenAI")
    def test_exception_returns_error(self, mock_llm_cls):
        mock_llm_cls.side_effect = Exception("API failure")
        result = agent("Find papers about physics")
        assert result == "Error: Unable to get response from agent."

    @patch("agent.ChatOpenAI")
    def test_tool_returns_no_results(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        tool_response = MagicMock()
        tool_response.content = ""
        tool_response.tool_calls = [
            {"name": "search_arxiv", "args": {"query": "xyznonexistent"}, "id": "call_1"}
        ]
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = tool_response

        with patch.object(search_arxiv, "invoke", return_value="No results found."):
            result = agent("Find papers about xyznonexistent")

        assert result == "Error: Unable to get response from agent."


# Output type
class TestOutputType:
    def test_output_is_string(self):
        result = agent("")
        assert isinstance(result, str)
