import json, uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import ToolMessage
from langchain_core.tracers.context import collect_runs

import agent as agent_module

TRACE_DIR = Path("traces")


def _now():
    return datetime.now(timezone.utc)


def _iso(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _jsonable(obj):
    return json.loads(json.dumps(obj, ensure_ascii=False, default=repr))


def _tool_output(out):
    """Saída da tool. Via ToolNode o LangChain entrega um ToolMessage cujo
    `content` é o retorno serializado; desfazemos isso para obter o valor."""
    if isinstance(out, ToolMessage):
        if out.artifact is not None:
            return _jsonable(out.artifact)
        try:
            return json.loads(out.content)
        except (TypeError, ValueError):
            return out.content
    return _jsonable(out)


def _tool_runs(runs):
    """Percorre a árvore de runs coletada e devolve só as execuções de tools."""
    for r in runs:
        if r.run_type == "tool":
            yield r
        yield from _tool_runs(r.child_runs)


def _node_runs(runs):
    """Nós do LangGraph, como registrados pelo próprio framework
    (mesma base usada pelo LangSmith)."""
    for r in runs:
        meta = (r.extra or {}).get("metadata", {})
        if r.run_type == "chain" and meta.get("langgraph_node") == r.name:
            yield r
        yield from _node_runs(r.child_runs)


def run_traced(query: str, out_dir: str | Path | None = TRACE_DIR
               ) -> tuple[dict | None, list[str], list[dict]]:
    """Executa o agente e devolve (estado_final, trajetória, tool_calls).

    trajectory  →  nós do grafo e tools, na ordem de início
    tool_calls  →  [{name, input, output | error, start, end}], na ordem de início
    Grava <out_dir>/<prefixo>.json (mesmo se o agente lançar exceção).
    """
    run_id, started_at = uuid.uuid4().hex, _now()
    state, status, error, cb = None, "error", None, None
    try:
        with collect_runs() as cb:
            state = agent_module.agent(query)
        status = "ok"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        runs = cb.traced_runs if cb else []

        tools = sorted(_tool_runs(runs), key=lambda r: _iso(r.start_time))
        tool_calls = []
        for r in tools:
            call = {"name": r.name, "input": _jsonable(r.inputs)}
            out = (r.outputs or {}).get("output")
            if r.error:
                call["error"] = r.error.split("Traceback")[0].strip()
            elif isinstance(out, ToolMessage) and out.status == "error":
                call["error"] = out.content
            else:
                call["output"] = _tool_output(out)
            call["start"], call["end"] = _iso(r.start_time), _iso(r.end_time)
            tool_calls.append(call)

        nodes = list(_node_runs(runs))
        trajectory = [r.name for r in sorted(nodes + tools,
                                             key=lambda r: _iso(r.start_time))]

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{started_at:%Y%m%dT%H%M%S}_{run_id[:8]}.json"
            path.write_text(json.dumps({
                "run_id": run_id, "query": query, "status": status,
                "error": error,
                "runs_collected": len(runs),
                "trajectory": trajectory,
                "tool_calls": tool_calls,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
    return state, trajectory, tool_calls


def load_run(path: str | Path) -> tuple[list[str], list[dict]]:
    """Lê o arquivo de uma execução e devolve (trajetória, tool_calls)."""
    run = json.loads(Path(path).read_text(encoding="utf-8"))
    return run["trajectory"], run["tool_calls"]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "quero um bolo de cenoura com tapioca"
    state, trajectory, tool_calls = run_traced(query)
    print("\n".join(trajectory))
    for c in tool_calls:
        print(f"\n{c['name']}\n  in : {c['input']}\n  out: {c.get('output', c.get('error'))}")
    print("\n" + state["output"])