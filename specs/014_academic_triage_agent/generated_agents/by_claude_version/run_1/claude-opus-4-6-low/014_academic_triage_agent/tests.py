import pytest
from unittest.mock import patch, MagicMock
from agent import agent, search_arxiv, search_pubmed, get_wikipedia_summary


class TestEmptyInput:
    def test_empty_string(self):
        assert agent("") == "Error: The user request cannot be empty."

    def test_whitespace(self):
        assert agent("   ") == "Error: The user request cannot be empty."

    def test_none(self):
        assert agent(None) == "Error: The user request cannot be empty."


class TestToolDefinitions:
    def test_search_arxiv_is_tool(self):
        assert hasattr(search_arxiv, "invoke")
        assert search_arxiv.name == "search_arxiv"

    def test_search_pubmed_is_tool(self):
        assert hasattr(search_pubmed, "invoke")
        assert search_pubmed.name == "search_pubmed"

    def test_get_wikipedia_summary_is_tool(self):
        assert hasattr(get_wikipedia_summary, "invoke")
        assert get_wikipedia_summary.name == "get_wikipedia_summary"


class TestExactSciences:
    @patch("agent.ChatOpenAI")
    def test_arxiv_called_for_ai_papers(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        mock_bound = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound

        first_response = MagicMock()
        first_response.tool_calls = [
            {"name": "search_arxiv", "args": {"query": "artificial intelligence"}, "id": "call_1"}
        ]
        first_response.content = ""
        mock_bound.invoke.return_value = first_response

        synth_response = MagicMock()
        synth_response.content = "Here are papers about AI:\n1. Paper Title - Summary"
        mock_llm.invoke.return_value = synth_response

        with patch("agent.TOOL_MAP", {"search_arxiv": MagicMock(invoke=MagicMock(return_value="Title: AI Paper\nSummary: About AI")), "search_pubmed": search_pubmed, "get_wikipedia_summary": get_wikipedia_summary}):
            result = agent("Find research papers about artificial intelligence")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Error" not in result


class TestLifeSciences:
    @patch("agent.ChatOpenAI")
    def test_pubmed_called_for_health(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        mock_bound = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound

        first_response = MagicMock()
        first_response.tool_calls = [
            {"name": "search_pubmed", "args": {"query": "melanoma"}, "id": "call_1"}
        ]
        first_response.content = ""
        mock_bound.invoke.return_value = first_response

        synth_response = MagicMock()
        synth_response.content = "Here are papers about melanoma:\n1. Study - Summary"
        mock_llm.invoke.return_value = synth_response

        with patch("agent.TOOL_MAP", {"search_arxiv": search_arxiv, "search_pubmed": MagicMock(invoke=MagicMock(return_value="Title: Melanoma Study\nSource: Journal\nDate: 2024")), "get_wikipedia_summary": get_wikipedia_summary}):
            result = agent("Find research papers about melanoma treatment")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Error" not in result


class TestGeneralConcepts:
    @patch("agent.ChatOpenAI")
    def test_wikipedia_for_concept(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        mock_bound = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound

        first_response = MagicMock()
        first_response.tool_calls = [
            {"name": "get_wikipedia_summary", "args": {"topic": "photosynthesis"}, "id": "call_1"}
        ]
        first_response.content = ""
        mock_bound.invoke.return_value = first_response

        synth_response = MagicMock()
        synth_response.content = "Photosynthesis is the process by which plants convert light energy..."
        mock_llm.invoke.return_value = synth_response

        with patch("agent.TOOL_MAP", {"search_arxiv": search_arxiv, "search_pubmed": search_pubmed, "get_wikipedia_summary": MagicMock(invoke=MagicMock(return_value="Title: Photosynthesis\nSummary: Process of converting light..."))}):
            result = agent("Explain photosynthesis")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Error" not in result


class TestOutOfScope:
    @patch("agent.ChatOpenAI")
    def test_non_academic_request(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        mock_bound = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound

        first_response = MagicMock()
        first_response.tool_calls = []
        first_response.content = "Error: Query outside of academic scope."
        mock_bound.invoke.return_value = first_response

        result = agent("Book me a flight to Paris")
        assert result == "Error: Query outside of academic scope."


class TestMultipleDomains:
    @patch("agent.ChatOpenAI")
    def test_multi_domain(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        mock_bound = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound

        first_response = MagicMock()
        first_response.tool_calls = []
        first_response.content = "Error: The query spans multiple distinct domains. Please make one specific research request at a time."
        mock_bound.invoke.return_value = first_response

        result = agent("Find papers about artificial intelligence and melanoma")
        assert result == "Error: The query spans multiple distinct domains. Please make one specific research request at a time."


class TestErrorHandling:
    @patch("agent.ChatOpenAI")
    def test_exception_returns_error(self, mock_llm_cls):
        mock_llm_cls.side_effect = Exception("API Error")
        result = agent("Find papers about physics")
        assert result == "Error: Unable to get response from agent."
