import functools, inspect, time
from contextvars import ContextVar

import agent as agent_module

_trace: ContextVar[list | None] = ContextVar("_trace", default=None)
_depth: ContextVar[int] = ContextVar("_depth", default=0)

def traced(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        log = _trace.get()
        if log is None:                      # fora de uma execução rastreada
            return fn(*args, **kwargs)
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
        rec = {"fn": f"{fn.__module__}.{fn.__qualname__}",
               "args": dict(bound.arguments), "depth": _depth.get()}
        tok = _depth.set(rec["depth"] + 1)
        t0 = rec["start"] = time.perf_counter()
        try:
            rec["result"] = fn(*args, **kwargs)
            rec["status"] = "ok"
            return rec["result"]
        except Exception as e:
            rec.update(status="error", error=repr(e))
            raise
        finally:
            rec["dur"] = time.perf_counter() - t0
            _depth.reset(tok)
            log.append(rec)
    return wrapper

def instrument(module):
    """Rastreia toda função definida em `module` e toda tool LangChain (@tool) dele.

    Funções são resolvidas pelo nome global do módulo no momento da chamada
    (build_graph, nós, rotas), então basta substituí-las no módulo.
    Tools são objetos StructuredTool chamados via `.invoke`; o objeto é mantido
    e só a função interna (`.func`) é envolvida.
    """
    from langchain_core.tools import BaseTool

    for name, obj in list(vars(module).items()):
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            setattr(module, name, traced(obj))
        elif isinstance(obj, BaseTool) and getattr(obj, "func", None):
            obj.func = traced(obj.func)


instrument(agent_module)


def run_traced(query: str) -> tuple[dict, list]:
    """Executa o agente e devolve (estado_final, log de chamadas na ordem de término)."""
    log: list = []
    tok = _trace.set(log)
    try:
        state = agent_module.agent(query)
    finally:
        _trace.reset(tok)
    return state, log


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "quero um bolo de cenoura com tapioca"
    state, log = run_traced(query)
    for rec in sorted(log, key=lambda r: r["start"]):
        print(f"{'  ' * rec['depth']}{rec['fn']}  [{rec['status']}] {rec['dur']:.2f}s")
    print("\n" + state["output"])
