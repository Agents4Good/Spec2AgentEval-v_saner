"""Processo filho do stage 08: executa o agente e avalia o trace com pytest.

Uso::

    python trace_runner.py <agent_dir> <test_file> <json_report_file>

1. Importa ``<test_file>`` só para ler o contrato de cenários:
       SCENARIOS: dict[str, dict]
       agent_inputs(scenario) -> tuple | dict   # argumentos do entrypoint
2. Carrega ``<agent_dir>/agent.py`` (``trace_capture.load_agent``) e executa
   cada cenário sob ``run_traced``, sem gravar em disco.
3. Roda pytest em ``<test_file>`` entregando os traces em memória pela
   fixture ``traces`` (``{scenario_id: trace}``). Os traces também são
   anexados ao JSON report, sob a chave ``"traces"``.

Um processo por agente: cada pasta tem seu próprio ``agent.py`` (sempre
importado como módulo ``agent``) e o agente pode travar o processo ou poluir
o estado global.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from trace_capture import _jsonable, load_agent, run_traced


class TracePlugin:
    """Expõe os traces coletados para o arquivo de teste."""

    def __init__(self, traces: dict):
        self._traces = traces

    @pytest.fixture(scope="session")
    def traces(self) -> dict:
        return self._traces

    def pytest_json_modifyreport(self, json_report):
        json_report["traces"] = self._traces


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _unwrap_output(state):
    """Agentes que devolvem ``(output, {"events": [...]})``: fica com o output."""
    if (isinstance(state, tuple) and len(state) == 2
            and isinstance(state[1], dict) and "events" in state[1]):
        return state[0]
    return state


def collect_traces(test_module, agent_dir: Path) -> dict:
    try:
        load_agent(agent_dir)
    except Exception as e:
        error = f"could not import agent.py: {type(e).__name__}: {e}"
        return {sid: {"status": "error", "error": error}
                for sid in test_module.SCENARIOS}

    traces = {}
    for sid, scenario in test_module.SCENARIOS.items():
        try:
            inputs = test_module.agent_inputs(scenario)
        except Exception as e:
            # ex.: credencial ausente — o teste decide se pula ou falha.
            traces[sid] = {"status": "not_run", "error": str(e)}
            continue
        print(f"  scenario {sid}...", flush=True)
        try:
            state, trajectory, tool_calls = run_traced(
                inputs, out_dir=None, agent_dir=agent_dir)
        except Exception as e:
            traces[sid] = {"status": "error",
                           "error": f"{type(e).__name__}: {e}",
                           "output": None, "trajectory": [], "tool_calls": []}
            continue
        traces[sid] = {"status": "ok", "error": None,
                       "output": _jsonable(_unwrap_output(state)),
                       "trajectory": trajectory, "tool_calls": tool_calls}
    return traces


def main() -> int:
    agent_dir = Path(sys.argv[1]).resolve()
    test_path = Path(sys.argv[2]).resolve()
    report_file = sys.argv[3]

    test_module = _load_module(test_path.stem, test_path)
    traces = collect_traces(test_module, agent_dir)

    return pytest.main(
        [
            str(test_path),
            "-v",
            "--tb=long",
            "-rN",  # no summary section (we have the json report)
            "--json-report",
            f"--json-report-file={report_file}",
            "--json-report-omit=keywords,warnings,streams,collectors",
        ],
        plugins=[TracePlugin(traces)],
    )


if __name__ == "__main__":
    sys.exit(main())
