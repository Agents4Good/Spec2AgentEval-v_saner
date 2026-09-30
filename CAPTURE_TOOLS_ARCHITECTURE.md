# Arquitetura do Módulo de Captura de Tool Calls

## Visão Geral

O módulo `capture_tools.py` fornece uma forma simples e robusta de capturar as chamadas de ferramentas (tool calls) durante a invocação de agentes LangGraph, substituindo o trace via OpenTelemetry.

## Localização e Importabilidade

### Escolha de Localização: Raiz do Projeto

**Decisão**: Colocar `capture_tools.py` na raiz do projeto (`Spec2AgentEval-v_saner/capture_tools.py`)

**Justificativa**:
1. **Acessibilidade**: Módulos em qualquer profundidade (`specs/<agente>/`, `specs/<agente>/subfolder/`, etc.) podem acessá-lo
2. **Simplicidade**: Evita duplicação de código e facilita manutenção centralizada
3. **Robustez**: Funciona em diferentes máquinas com estruturas de diretórios diferentes
4. **Sem dependências**: Não requer estrutura específica de pacotes ou `__init__.py`

### Estratégia de Importação

A importação é resolvida automaticamente usando a função `ensure_import_path`:

```python
from pathlib import Path
from capture_tools import ensure_import_path, capture_tool_calls

ensure_import_path(Path(__file__))
```

Esta função:
1. Localiza o arquivo `capture_tools.py` subindo pelos diretórios
2. Adiciona seu diretório pai ao `sys.path`
3. Funciona independentemente de quantos níveis de profundidade o módulo `eval_*.py` está

### Por que não usar outras abordagens?

- **Pacote Python estruturado** (`setup.py`, `pyproject.toml`): Complexo demais para um único módulo; requer instalação
- **Imports relativos**: Não funcionam quando módulos são carregados via `importlib` do arquivo
- **Paths absolutos hardcoded**: Quebram em diferentes máquinas e estruturas
- **Múltiplas cópias**: Difícil de manter sincronizado

## Design do Módulo

### API Mínima e Estável

```python
from capture_tools import capture_tool_calls

with capture_tool_calls() as trace:
    result = agent.invoke(entrada)

trace.events      # list[ToolCallEvent]
trace.to_dict()   # dict serializável em JSON
```

**Razão**: API simples que não requer mudanças nos agentes gerados ou módulos `eval_*.py` existentes.

### Rastreamento sem Callbacks

Usa `langchain_core.tracers.context.collect_runs` em vez de passar callbacks na invocação:

**Benefício**: A captura é transparente para o agente e não requer modificações no código de invocação.

### Normalização de Dados

**Inputs**: 
- Converte strings Python literais em dicts usando `ast.literal_eval`
- Handles JSON strings
- Preserva dicts como estão

**Outputs**:
- Extrai `.content` de `ToolMessage`
- Desserializa JSON strings
- Extrai de estruturas de wrapper (`{'output': valor}`)
- Converte MCP content blocks em texto

**Razão**: Torna os dados consistentes e fáceis de processar, independentemente do formato retornado pela ferramenta.

### Estrutura de Eventos

Cada evento contém:

```python
@dataclass
class ToolCallEvent:
    tool: str                    # Nome da ferramenta
    input: Dict[str, Any]       # Argumentos normalizados
    output: Any                  # Resultado normalizado
    status: str                  # "ok" ou "error"
    error: Optional[str]         # repr do erro ou None
    t_start: float              # Tempo relativo (segundos)
    t_end: float                # Tempo relativo (segundos)
    run_id: str                 # ID único do run
    parent_run_id: Optional[str] # ID do run pai
```

**Razão**: Fornece contexto completo para análise e debugging, incluindo timing e relações entre runs.

## Tratamento de Casos Especiais

### Execução Paralela de Ferramentas

Funciona automaticamente. O `collect_runs` rastreia todos os runs, independentemente da ordem de execução.

### Erros em Ferramentas

Capturados automaticamente:
- `status` é configurado para "error"
- `error` contém o `repr` da exceção com traceback

### Múltiplos Níveis de Runs

Percorre `child_runs` para encontrar ferramentas em diferentes profundidades da árvore de execução.

### Chamadas de Ferramentas em Threads Personalizadas

**Limitação documentada**: Não são capturadas porque não herdam o contexto de rastreamento do LangChain.

## Agregação de Estatísticas

O método `to_dict()` retorna:

```python
{
    "events": [...],                          # Eventos ordenados por t_start
    "total_tool_calls": int,                  # Quantidade de tool calls
    "total_latency_seconds": float,           # Soma das durações
    "total_tokens": int                       # Tokens agregados dos LLM runs
}
```

**Razão**: Fornece métricas de alto nível úteis para análise de performance e comparação entre agentes.

## Persistência

O módulo não grava arquivos. Isso é responsabilidade das etapas do pipeline:

```python
with capture_tool_calls() as trace:
    result = agent.invoke(entrada)

# Caller é responsável por persistência
json.dump(trace.to_dict(), output_file)
```

**Razão**: Flexibilidade - diferentes etapas podem ter diferentes estratégias de armazenamento.

## Compatibilidade

- ✅ LangChain 0.1.44+ (versão fixada em requirements.txt)
- ✅ LangGraph 0.0.65+ (versão fixada)
- ✅ Python 3.8+
- ✅ Sync e async (`invoke` e `ainvoke`)

## Testes

Veja os testes em `/tmp/claude-1000/.../scratchpad/`:
- `test_capture_simple.py`: Teste básico com ferramentas simples
- `test_complete.py`: Teste com múltiplas ferramentas e agregação
- `test_errors.py`: Teste com tratamento de erros
- `test_import_from_spec.py`: Teste de importação desde subdiretórios
