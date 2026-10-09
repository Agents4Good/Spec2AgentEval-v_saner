"""Avaliação GEval do agente de receitas (spec 018), um caso por fluxo.

Executar a partir da raiz do repositório:

    AGENT_DIR=claude/sonnet deepeval test run specs/018_recipe_intermediate/tests/evals/test_recipe_agent.py

AGENT_DIR é relativo a specs/018_recipe_intermediate/ (ex.: claude/sonnet,
claude/haiku, codex). As tools do agente são substituídas pelos mocks de cada
golden, que registram as chamadas para o GEval; nada vai à rede além do LLM do
agente e do modelo juiz.
"""

import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from dotenv import load_dotenv

from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase, ToolCall

from metrics import flow_metrics

SPEC_DIR = Path(__file__).resolve().parents[2]
for _dir in Path(__file__).resolve().parents:
    if (_dir / ".env").is_file():
        load_dotenv(_dir / ".env")
        break

AGENT_DIR = SPEC_DIR / os.getenv("AGENT_DIR", "claude/sonnet")
TOOL_NAMES = ("search_recipes", "retrieve_pantry")


def _load_agent_module():
    agent_path = AGENT_DIR / "agent.py"
    sys.path.insert(0, str(AGENT_DIR))
    spec = importlib.util.spec_from_file_location("recipe_agent_under_test", agent_path)
    module = importlib.util.module_from_spec(spec)
    # LangGraph resolve as anotações do state via sys.modules[__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ai_app = _load_agent_module()


dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path=str(Path(__file__).parent / ".dataset.json"))


def _recording_tool(name, original, mock_output, calls):
    """Substitui a tool pelo mock do golden, preservando a assinatura original."""
    signature = inspect.signature(original)
    # Alguns agentes serializam o retorno da tool em JSON (anotação -> str).
    returns_str = signature.return_annotation in (str, "str")
    output = json.dumps(mock_output, ensure_ascii=False) if returns_str and not isinstance(mock_output, str) else mock_output

    def fake(*args, **kwargs):
        try:
            params = dict(signature.bind(*args, **kwargs).arguments)
        except TypeError:
            params = {"args": list(args), **kwargs}
        calls.append(ToolCall(name=name, input_parameters=params, output=mock_output))
        return output

    return fake


def _final_output(state) -> str:
    """Resposta final do agente: mensagem de erro ou receita extraída do state."""
    if not isinstance(state, dict):
        return f"STATE INVÁLIDO ({type(state).__name__}): {state!r}"
    for key in ("error", "error_message", "message"):
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for key in ("recipe", "final_recipe"):
        value = state.get(key)
        if value:
            return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    return "STATE SEM RECEITA NEM ERRO: " + json.dumps(state, ensure_ascii=False, default=str)


@pytest.mark.parametrize("golden", dataset.goldens, ids=[g.name for g in dataset.goldens])
def test_recipe_agent(golden: Golden):
    metadata = golden.additional_metadata
    calls: list[ToolCall] = []
    fakes = {
        name: _recording_tool(name, getattr(ai_app, name), metadata["mocks"][name], calls)
        for name in TOOL_NAMES
    }

    with patch.multiple(ai_app, **fakes):
        try:
            actual_output = _final_output(ai_app.agent(golden.input))
        except Exception as exc:  # B7: exceção não tratada é avaliada como falha
            actual_output = f"EXCEPTION {type(exc).__name__}: {exc}"

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=actual_output,
        expected_output=golden.expected_output,
        tools_called=calls,
        expected_tools=golden.expected_tools,
        name=golden.name,
        additional_metadata=metadata,
    )
    assert_test(test_case=test_case, metrics=flow_metrics(metadata["flow"]))
