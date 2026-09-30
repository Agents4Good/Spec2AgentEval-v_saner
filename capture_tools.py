"""
Captura de chamadas de ferramentas para agentes LangGraph.

Módulo compartilhado que captura as chamadas de ferramentas durante a invocação de agentes,
substituindo o trace via OpenTelemetry.

Uso:
    from capture_tools import capture_tool_calls

    with capture_tool_calls() as trace:
        result = agent.invoke(entrada)  # ou await agent.ainvoke(entrada)

    trace.events      # list[dict], ordenada por t_start
    trace.to_dict()   # dict serializável em JSON

Limitações:
    - Chamadas de ferramentas feitas em threads criadas pelo próprio agente (fora dos
      executores do LangChain) não herdam o contexto e não são capturadas.

Importação (recomendado para módulos em specs/*/):
    import sys
    from pathlib import Path

    # Encontra o diretório raiz do projeto
    BENCH_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(BENCH_ROOT))

    from capture_tools import capture_tool_calls
"""

import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator

from langchain_core.messages import ToolMessage
from langchain_core.tracers.context import collect_runs


@dataclass
class ToolCallEvent:
    """Evento de chamada de ferramenta."""

    tool: str
    input: Dict[str, Any]
    output: Any
    status: str  # "ok" | "error"
    error: Optional[str]
    t_start: float
    t_end: float
    run_id: str
    parent_run_id: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dict serializável em JSON."""
        return {
            "tool": self.tool,
            "input": self.input,
            "output": self.output,
            "status": self.status,
            "error": self.error,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
        }


@dataclass
class ToolCallTrace:
    """Trace de chamadas de ferramentas."""

    events: List[ToolCallEvent] = field(default_factory=list)
    t_start_absolute: float = field(default_factory=time.time)

    def add_event(self, event: ToolCallEvent) -> None:
        """Adiciona um evento ao trace."""
        self.events.append(event)

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dict serializável em JSON."""
        # Coleta estatísticas de uso de tokens
        total_tokens = 0
        for run in getattr(self, "_collected_runs", []):
            if hasattr(run, "metadata") and run.metadata and "usage_metadata" in run.metadata:
                usage = run.metadata.get("usage_metadata", {})
                total_tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

        return {
            "events": [event.to_dict() for event in self.events],
            "total_tool_calls": len(self.events),
            "total_latency_seconds": sum(e.t_end - e.t_start for e in self.events),
            "total_tokens": total_tokens,
        }


def _normalize_tool_input(input_data: Any) -> Dict[str, Any]:
    """
    Normaliza o input da ferramenta para um dict.

    Args:
        input_data: Dados brutos do input (pode ser dict, str, etc.)

    Returns:
        Dict com os argumentos da ferramenta
    """
    if isinstance(input_data, dict):
        # Se o dict tem um único campo 'input' com uma string,
        # tenta desserializar a string
        if len(input_data) == 1 and "input" in input_data:
            input_str = input_data["input"]
            if isinstance(input_str, str):
                try:
                    parsed = json.loads(input_str)
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    # Se não conseguir desserializar, trata como string
                    # e tenta fazer eval (cuidado com eval em produção)
                    try:
                        import ast
                        parsed = ast.literal_eval(input_str)
                        if isinstance(parsed, dict):
                            return parsed
                    except (ValueError, SyntaxError):
                        pass
                return input_data
            return input_data
        return input_data
    if isinstance(input_data, str):
        try:
            parsed = json.loads(input_data)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return {"raw_input": input_data}
    return {"raw_input": str(input_data)}


def _normalize_tool_output(output: Any) -> Any:
    """
    Normaliza a saída da ferramenta.

    Extrai `.content` de ToolMessage, converte MCP content blocks em texto,
    e desserializa JSON quando a saída for uma string JSON válida.

    Args:
        output: Saída bruta da ferramenta

    Returns:
        Saída normalizada
    """
    # Extrai .content de ToolMessage
    if isinstance(output, ToolMessage):
        output = output.content

    # Se for string, tenta desserializar como JSON
    if isinstance(output, str):
        try:
            return json.loads(output)
        except (json.JSONDecodeError, ValueError):
            return output

    # Se o output for um dict com um único campo 'output', extrai o valor
    if isinstance(output, dict) and len(output) == 1 and "output" in output:
        output = output["output"]

    # Converte MCP content blocks em texto
    if isinstance(output, dict) and "content" in output:
        content = output["content"]
        if isinstance(content, list):
            # Junta blocos de conteúdo em texto
            texts = []
            for block in content:
                if isinstance(block, dict):
                    if "text" in block:
                        texts.append(block["text"])
                    elif "type" in block and block["type"] == "text":
                        texts.append(block.get("text", ""))
                else:
                    texts.append(str(block))
            return "\n".join(texts) if texts else output
        return content

    return output


@contextmanager
def capture_tool_calls() -> Iterator[ToolCallTrace]:
    """
    Context manager que captura chamadas de ferramentas durante invocação de agente.

    Yields:
        ToolCallTrace com os events capturados

    Example:
        with capture_tool_calls() as trace:
            result = agent.invoke(input_data)

        print(trace.events)
        print(trace.to_dict())
    """
    trace = ToolCallTrace()
    start_time = time.time()

    with collect_runs() as cb:
        try:
            yield trace
        finally:
            # Processa os runs coletados
            traced_runs = cb.traced_runs if hasattr(cb, "traced_runs") else []

            # Coleta estatísticas dos runs de LLM
            for run in traced_runs:
                if hasattr(run, "metadata") and run.metadata and "usage_metadata" in run.metadata:
                    trace._collected_runs = traced_runs

            # Extrai runs com run_type == "tool"
            tool_runs = [run for run in traced_runs if getattr(run, "run_type", None) == "tool"]

            # Se não houver tool runs no nível superior, procura em child_runs
            for run in traced_runs:
                if hasattr(run, "child_runs") and run.child_runs:
                    for child_run in run.child_runs:
                        if getattr(child_run, "run_type", None) == "tool":
                            tool_runs.append(child_run)

            # Processa cada tool run
            for run in tool_runs:
                tool_name = getattr(run, "name", "unknown_tool")
                run_id = getattr(run, "id", "")
                parent_run_id = getattr(run, "parent_run_id", None)

                # Extrai tempos
                start_time_run = getattr(run, "start_time", None)
                end_time_run = getattr(run, "end_time", None)

                if start_time_run and end_time_run:
                    # Converte para segundos relativos ao início da captura
                    t_start = (
                        start_time_run.timestamp() - start_time
                        if hasattr(start_time_run, "timestamp")
                        else 0
                    )
                    t_end = (
                        end_time_run.timestamp() - start_time
                        if hasattr(end_time_run, "timestamp")
                        else 0
                    )
                else:
                    t_start = 0
                    t_end = 0

                # Extrai input e output
                inputs = getattr(run, "inputs", {}) or {}
                outputs = getattr(run, "outputs", None)

                # Normaliza input
                tool_input = _normalize_tool_input(inputs)

                # Normaliza output
                tool_output = _normalize_tool_output(outputs)

                # Determina status
                error_value = None
                status = "ok"
                if getattr(run, "error", None):
                    status = "error"
                    error_value = repr(run.error)

                # Cria evento
                event = ToolCallEvent(
                    tool=tool_name,
                    input=tool_input,
                    output=tool_output,
                    status=status,
                    error=error_value,
                    t_start=t_start,
                    t_end=t_end,
                    run_id=str(run_id),
                    parent_run_id=str(parent_run_id) if parent_run_id else None,
                )

                trace.add_event(event)

            # Ordena eventos por t_start
            trace.events.sort(key=lambda e: e.t_start)


def ensure_import_path(module_dir: Optional[Path] = None) -> None:
    """
    Garante que o módulo capture_tools pode ser importado a partir de qualquer spec.

    Uso em módulos eval_*.py:
        from pathlib import Path
        from capture_tools import ensure_import_path, capture_tool_calls

        ensure_import_path(Path(__file__))

        # Agora pode usar capture_tool_calls normalmente

    Args:
        module_dir: Diretório do módulo que está importando (usualmente __file__).
                   Se não fornecido, tenta localizar automaticamente o diretório raiz.
    """
    if module_dir is None:
        # Tenta encontrar o diretório raiz automaticamente
        module_dir = Path(__file__).parent

    if isinstance(module_dir, str):
        module_dir = Path(module_dir)

    # Localiza o diretório raiz (onde capture_tools.py está)
    if module_dir.name == "capture_tools.py":
        # Foi passado o arquivo, não o diretório
        bench_root = module_dir.parent
    else:
        # Procura capture_tools.py subindo pelos diretórios
        current = module_dir if module_dir.is_dir() else module_dir.parent
        bench_root = None

        for _ in range(10):  # Máximo de 10 níveis acima
            if (current / "capture_tools.py").exists():
                bench_root = current
                break
            current = current.parent
            if current == current.parent:  # Atingiu a raiz do filesystem
                break

        if bench_root is None:
            # Fallback: assume que o diretório raiz está 2 níveis acima
            # (para estrutura specs/<agente>/)
            bench_root = module_dir.parent.parent if module_dir.is_dir() else module_dir.parent.parent

    # Adiciona ao sys.path se ainda não estiver lá
    bench_root_str = str(bench_root)
    if bench_root_str not in sys.path:
        sys.path.insert(0, bench_root_str)
