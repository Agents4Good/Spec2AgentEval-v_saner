"""Avaliação GEval por fluxo da spec 018_recipe_intermediate.

Uso:
    AGENT_PATH=specs/018_recipe_intermediate/<gerador>/<modelo>/agent.py \
        deepeval test run specs/018_recipe_intermediate/evals/geval/test_geval_flows.py

Pela spec, o agente expõe `agent(query)` -> state (dict) e as tools
`retrieve_pantry` e `search_recipes` no mesmo módulo. As tools são
substituídas pelo retorno definido em cada caso (cases.py) e suas chamadas
são registradas como `tools_called`.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, ToolCall

sys.path.insert(0, str(Path(__file__).parent))
from cases import CASES, tool_returns_for  # noqa: E402
from metrics import GEVAL_METRICS  # noqa: E402

TOOL_NAMES = ("retrieve_pantry", "search_recipes")


def _load_agent_module():
    agent_path = os.environ.get("AGENT_PATH")
    if not agent_path:
        pytest.skip("Defina AGENT_PATH com o caminho do agent.py a ser avaliado.")
    path = Path(agent_path).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("agent_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AGENT_MODULE = None


def _agent_module():
    global AGENT_MODULE
    if AGENT_MODULE is None:
        AGENT_MODULE = _load_agent_module()
    return AGENT_MODULE


def _fake_tool(name, result, calls):
    def fake(*args, **kwargs):
        params = dict(kwargs)
        if args:
            params["args"] = list(args)
        calls.append(ToolCall(name=name, input_parameters=params, output=result))
        return result

    return fake


def _patch_tools(module, tool_returns, calls):
    """Substitui cada tool pelo retorno do cenário, registrando as chamadas.

    Ferramentas LangChain (@tool) têm o callable em `.func`; funções simples
    são substituídas no próprio módulo.
    """
    patchers = []
    for name in TOOL_NAMES:
        tool = getattr(module, name, None)
        if tool is None:
            pytest.fail(f"Tool '{name}' não encontrada em {module.__file__}.")
        fake = _fake_tool(name, tool_returns[name], calls)
        if hasattr(tool, "func") and tool.func is not None:
            patchers.append(patch.object(tool, "func", fake))
        else:
            patchers.append(patch.object(module, name, fake))
    return patchers


def _run(case):
    module = _agent_module()
    tool_returns = tool_returns_for(case)
    calls: list[ToolCall] = []
    patchers = _patch_tools(module, tool_returns, calls)
    for p in patchers:
        p.start()
    try:
        state = module.agent(case["query"])
        output = json.dumps(state, ensure_ascii=False, default=str)
    except Exception as exc:  # B7: exceção não tratada também é avaliada
        output = f"EXCEÇÃO NÃO TRATADA: {type(exc).__name__}: {exc}"
    finally:
        for p in patchers:
            p.stop()
    return output, calls, tool_returns


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_geval_flow(case):
    output, calls, tool_returns = _run(case)
    scenario = json.dumps(tool_returns, ensure_ascii=False)
    test_case = LLMTestCase(
        input=f"{case['query']}\n\n[Cenário — retorno configurado das tools]: {scenario}",
        actual_output=output,
        tools_called=calls,
        name=case["id"],
    )
    assert_test(test_case, [GEVAL_METRICS[case["id"]]])
