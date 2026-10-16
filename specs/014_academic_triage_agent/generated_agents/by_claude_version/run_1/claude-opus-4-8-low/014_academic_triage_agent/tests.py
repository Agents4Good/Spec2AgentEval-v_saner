"""Tests for the Academic Triage Agent.

These tests cover the deterministic behaviors (empty input) and the agent's
routing / synthesis behaviors. Routing tests require a valid OPENAI_API_KEY and
network access; they are skipped automatically when the key is absent.
"""

import os
import unittest

import agent as agent_module
from agent import (
    ERR_EMPTY,
    ERR_GENERIC,
    ERR_MULTI,
    ERR_SCOPE,
    agent,
    get_wikipedia_summary,
    search_arxiv,
    search_pubmed,
)

HAS_KEY = bool(os.environ.get("OPENAI_API_KEY"))
skip_live = unittest.skipUnless(HAS_KEY, "requires OPENAI_API_KEY and network")


class TestEmptyInput(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(agent(""), ERR_EMPTY)

    def test_whitespace_only(self):
        self.assertEqual(agent("   \n\t "), ERR_EMPTY)

    def test_none(self):
        self.assertEqual(agent(None), ERR_EMPTY)


class TestToolsAreLangChainTools(unittest.TestCase):
    def test_tools_have_invoke(self):
        for t in (search_arxiv, search_pubmed, get_wikipedia_summary):
            self.assertTrue(hasattr(t, "invoke"))
            self.assertTrue(hasattr(t, "name"))

    def test_tool_invoke_returns_str(self):
        # Invoked through the tool interface, not the raw function.
        out = get_wikipedia_summary.invoke({"topic": "Photosynthesis"})
        self.assertIsInstance(out, str)


@skip_live
class TestRouting(unittest.TestCase):
    def test_b1_exact_sciences(self):
        out = agent("Find me papers on artificial intelligence")
        self.assertIsInstance(out, str)
        self.assertNotIn("Error:", out)

    def test_b2_life_sciences(self):
        out = agent("Find research papers about melanoma")
        self.assertIsInstance(out, str)
        self.assertNotIn("Error:", out)

    def test_b3_concept(self):
        out = agent("Explain what photosynthesis is")
        self.assertIsInstance(out, str)
        self.assertNotIn("Error:", out)

    def test_b5_multiple_domains(self):
        out = agent(
            "Find papers about artificial intelligence and papers about melanoma"
        )
        self.assertEqual(out, ERR_MULTI)

    def test_b6_out_of_scope(self):
        out = agent("Should I break up with my partner?")
        self.assertEqual(out, ERR_SCOPE)


class TestOutputIsString(unittest.TestCase):
    def test_always_string(self):
        self.assertIsInstance(agent(""), str)


if __name__ == "__main__":
    unittest.main()
