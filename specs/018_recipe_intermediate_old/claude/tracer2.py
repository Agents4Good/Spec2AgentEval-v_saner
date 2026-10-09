import functools, inspect, json, uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import ToolMessage
from langchain_core.tracers.context import collect_runs

import agent as agent_module

_trace: ContextVar[list | None] = ContextVar("_trace", default=None)
TRACE_DIR = Path("traces")


def _now():
    return datetime.now(timezone.utc)


def traced(fn):
    """Registra o nome da função na trajetória."""
    qualname = f"{fn.__module__}.{fn.__qualname__}"

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        traj = _trace.get()
        if traj is not None:
            traj.append((_now(), qualname))
        return fn(*args, **kwargs)
    return wrapper


def instrument(module):
    """Rastreia toda função definida em `module` (nós, rotas, build_graph).
    As tools não são envolvidas: são capturadas pelos callbacks do LangChain."""
    for name, obj in list(vars(module).items()):
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            setattr(module, name, traced(obj))


instrument(agent_module)


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


def run_traced(query: str, out_dir: str | Path | None = TRACE_DIR
               ) -> tuple[dict | None, list[str], list[dict]]:
    """Executa o agente e devolve (estado_final, trajetória, tool_calls).

    trajectory  →  funções do agente e tools, na ordem de início
    tool_calls  →  [{name, input, output | error}], na ordem de início
    Grava <out_dir>/<prefixo>.json (mesmo se o agente lançar exceção).
    """
    traj_raw: list = []
    run_id, started_at = uuid.uuid4().hex, _now()
    state, status = None, "error"
    tok = _trace.set(traj_raw)
    try:
        with collect_runs() as cb:
            state = agent_module.agent(query)
        status = "ok"
    finally:
        _trace.reset(tok)
        tools = sorted(_tool_runs(cb.traced_runs), key=lambda r: r.start_time)
        tool_calls = []
        for r in tools:
            call = {"name": r.name, "input": _jsonable(r.inputs)}
            if r.error:
                call["error"] = r.error.split("Traceback")[0].strip()
            else:
                call["output"] = _tool_output((r.outputs or {}).get("output"))
            tool_calls.append(call)
        events = traj_raw + [(r.start_time, r.name) for r in tools]
        trajectory = [name for _, name in sorted(events, key=lambda e: e[0])]
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{started_at:%Y%m%dT%H%M%S}_{run_id[:8]}.json"
            path.write_text(json.dumps({
                "run_id": run_id, "query": query, "status": status,
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