import importlib.util, json, os, sys, uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import ToolMessage
from langchain_core.tracers.context import register_configure_hook
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

AGENT_DIR = Path(os.environ.get("AGENT_DIR", ".")).resolve()
TRACE_DIR = Path("traces")

_agents: dict[Path, object] = {}

# O `collect_runs` do LangChain registra o coletor como NÃO herdável: ele
# chega à run raiz e às chamadas internas (LLM, tools), mas não às runs dos
# nós, que o LangGraph cria a partir do gerenciador herdado do pai. Aqui o
# mesmo coletor é registrado como herdável, para receber também os nós.
_collector_var: ContextVar[RunCollectorCallbackHandler | None] = ContextVar(
    "benchmark_run_collector", default=None)
register_configure_hook(_collector_var, inheritable=True)


@contextmanager
def collect_runs():
    cb = RunCollectorCallbackHandler()
    token = _collector_var.set(cb)
    try:
        yield cb
    finally:
        _collector_var.reset(token)


def load_agent(agent_dir: str | Path = AGENT_DIR):
    """Importa <agent_dir>/agent.py uma única vez por diretório.
    O diretório entra no sys.path para que imports locais do agente
    (ex.: `from tools import ...`) funcionem."""
    agent_dir = Path(agent_dir).resolve()
    if agent_dir not in _agents:
        path = agent_dir / "agent.py"
        if not path.is_file():
            raise FileNotFoundError(f"agent.py não encontrado em {agent_dir}")
        if str(agent_dir) not in sys.path:
            sys.path.insert(0, str(agent_dir))
        spec = importlib.util.spec_from_file_location("agent", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["agent"] = module
        spec.loader.exec_module(module)
        _agents[agent_dir] = module
    return _agents[agent_dir]


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


def _meta(r):
    return getattr(r, "metadata", None) or (r.extra or {}).get("metadata", {}) or {}


def _node_runs(runs, parent=None):
    """Nós do LangGraph, como registrados pelo próprio framework
    (mesma base usada pelo LangSmith). Um nó é a primeira run de cada
    (langgraph_node, langgraph_step); as runs internas herdam esse par."""
    for r in runs:
        m = _meta(r)
        key = (m.get("langgraph_node"), m.get("langgraph_step"))
        if r.run_type == "chain" and key[0] is not None and key != parent:
            yield key[0], r
        yield from _node_runs(r.child_runs, key)


def trace_path(name: str, out_dir: str | Path = TRACE_DIR) -> Path:
    """Caminho do JSON de uma execução; use também para localizá-lo depois."""
    return Path(out_dir) / f"{name}.json"


def _call(entrypoint, query):
    """str → agent(query); tupla/lista → agent(*query); dict → agent(**query)."""
    if isinstance(query, (tuple, list)):
        return entrypoint(*query)
    if isinstance(query, dict):
        return entrypoint(**query)
    return entrypoint(query)


def run_traced(query: str | tuple | list | dict,
               out_dir: str | Path | None = TRACE_DIR,
               name: str | None = "trace",
               agent_dir: str | Path = AGENT_DIR
               ) -> tuple[dict | None, list[str], list[dict]]:
    """Executa <agent_dir>/agent.py e devolve (estado_final, trajetória, tool_calls).

    query       →  argumento(s) do entrypoint (ver `_call`)
    trajectory  →  nós do grafo e tools, na ordem de início
    tool_calls  →  [{name, input, output | error, start, end}], na ordem de início
    Grava <out_dir>/<name>.json (padrão: trace.json), mesmo se o agente lançar
    exceção. Com name=None, usa <data>_<id>.json.
    """
    agent_module = load_agent(agent_dir)
    run_id, started_at = uuid.uuid4().hex, _now()
    state, status, error, cb = None, "error", None, None
    try:
        with collect_runs() as cb:
            state = _call(agent_module.agent, query)
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

        events = list(_node_runs(runs)) + [(r.name, r) for r in tools]
        trajectory = [n for n, r in sorted(events,
                                           key=lambda e: _iso(e[1].start_time))]

        if out_dir is not None:
            path = trace_path(name or f"{started_at:%Y%m%dT%H%M%S}_{run_id[:8]}",
                              out_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "run_id": run_id, "query": query, "status": status,
                "error": error,
                "agent_dir": str(Path(agent_dir).resolve()),
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
    query = " ".join(sys.argv[1:]) or "quero um bolo de cenoura com tapioca"
    state, trajectory, tool_calls = run_traced(query)
    print("\n".join(trajectory))
    for c in tool_calls:
        print(f"\n{c['name']}\n  in : {c['input']}\n  out: {c.get('output', c.get('error'))}")
    print("\n" + state["output"])