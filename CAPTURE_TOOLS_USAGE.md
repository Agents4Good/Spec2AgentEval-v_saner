# Captura de Chamadas de Ferramentas - Guia de Uso

## Introdução

Use o módulo `capture_tools` para capturar as chamadas de ferramentas durante a invocação do agente. Este módulo substitui o trace via OpenTelemetry e fornece uma forma simples e robusta de coletar dados sobre o comportamento das ferramentas.

## API

```python
from capture_tools import capture_tool_calls

with capture_tool_calls() as trace:
    result = agent.invoke(entrada)      # ou await agent.ainvoke(entrada)

# Acessa os eventos capturados
trace.events      # list[ToolCallEvent], ordenada por t_start
trace.to_dict()   # dict serializável em JSON
```

## Estructura do Evento

Cada evento capturado (`ToolCallEvent`) contém:

- **`tool`** (str): Nome da ferramenta chamada
- **`input`** (dict): Argumentos da ferramenta (normalizado como dict)
- **`output`** (any): Resultado da ferramenta (normalizado)
- **`status`** (str): "ok" ou "error"
- **`error`** (str | None): Mensagem de erro (repr) ou None
- **`t_start`** (float): Tempo de início (segundos relativos ao início da captura)
- **`t_end`** (float): Tempo de término (segundos relativos ao início da captura)
- **`run_id`** (str): ID único do run
- **`parent_run_id`** (str | None): ID do run pai (se for um sub-run)

## Estrutura do Dict Retornado

```python
trace.to_dict()  # Retorna:
{
    "events": [
        {
            "tool": "search_recipes",
            "input": {"ingredients": ["tomate", "alho"]},
            "output": [...],  # Normalizado
            "status": "ok",
            "error": None,
            "t_start": 0.123,
            "t_end": 1.456,
            "run_id": "uuid-...",
            "parent_run_id": None
        },
        # ... mais eventos ...
    ],
    "total_tool_calls": 2,
    "total_latency_seconds": 1.333,
    "total_tokens": 1234
}
```

## Normalização Automática

O módulo normaliza automaticamente inputs e outputs:

1. **Inputs**: Convertidos para dict quando possível
2. **Outputs**:
   - Extrai `.content` de `ToolMessage`
   - Converte blocos de conteúdo MCP em texto
   - Desserializa JSON quando a saída for uma string JSON válida

## Importação

### Forma 1: Automática (recomendada)

Use a função helper `ensure_import_path`:

```python
from pathlib import Path
from capture_tools import ensure_import_path, capture_tool_calls

# Garante que o módulo pode ser importado (funciona em qualquer nível)
ensure_import_path(Path(__file__))

# Agora pode usar normalmente
with capture_tool_calls() as trace:
    result = agent.invoke(entrada)
```

A função `ensure_import_path`:
- Localiza automaticamente o arquivo `capture_tools.py` subindo pelos diretórios
- Adiciona o diretório raiz ao `sys.path` se necessário
- **Funciona em qualquer nível de profundidade** da estrutura de specs
- Funciona em diferentes máquinas e estruturas de diretórios

### Forma 2: Manual (depende da estrutura)

Se o módulo `eval_*.py` está diretamente em `specs/<agente>/`:

```python
import sys
from pathlib import Path

# Sobe 2 níveis: eval_*.py -> specs/<agente>/ -> Spec2AgentEval-v_saner/
BENCH_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BENCH_ROOT))

from capture_tools import capture_tool_calls
```

Se está em `specs/<agente>/subfolder/`:

```python
import sys
from pathlib import Path

# Sobe 3 níveis: eval_*.py -> subfolder -> specs/<agente>/ -> Spec2AgentEval-v_saner/
BENCH_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BENCH_ROOT))

from capture_tools import capture_tool_calls
```

**Recomendação**: Use `ensure_import_path` e evite calcular manualmente quantos níveis subir.

## Exemplo Completo

```python
import json
from pathlib import Path
from capture_tools import ensure_import_path, capture_tool_calls

# Garante que o módulo pode ser importado
ensure_import_path(Path(__file__))

def evaluate_agent(agent, test_input):
    """Avalia o agente capturando suas tool calls."""
    with capture_tool_calls() as trace:
        result = agent.invoke({"query": test_input})
    
    # Acessa os eventos
    print(f"Ferramentas chamadas: {len(trace.events)}")
    for event in trace.events:
        print(f"  - {event.tool}: {event.status} ({event.t_end - event.t_start:.3f}s)")
    
    # Serializa para JSON
    trace_dict = trace.to_dict()
    json_trace = json.dumps(trace_dict, indent=2)
    
    # Inclui estatísticas
    print(f"\nEstatísticas:")
    print(f"  - Total de chamadas: {trace_dict['total_tool_calls']}")
    print(f"  - Latência total: {trace_dict['total_latency_seconds']:.3f}s")
    print(f"  - Tokens: {trace_dict['total_tokens']}")
    
    return result, trace_dict
```

## Características

- ✅ Funciona com `invoke` e `ainvoke`
- ✅ Captura chamadas de ferramentas paralelas
- ✅ Não requer passar config ou callbacks na invocação
- ✅ Não grava arquivos (responsabilidade do caller)
- ✅ Ordenação automática dos eventos por tempo
- ✅ Agregação de estatísticas (total de chamadas, latência, tokens)

## Limitações

⚠️ **Chamadas feitas em threads personalizadas não são capturadas**: Se o agente cria suas próprias threads fora dos executores do LangChain, essas chamadas não herdam o contexto de rastreamento e não serão capturadas.
