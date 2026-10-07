# Spec2Agent SANER Version

CLI para avaliar e comparar ferramentas de geração de código (Claude, Codex, Copilot, Gemini, OpenCode) a partir de uma especificação em Markdown. Para cada especificação o pipeline gera um agente, roda uma bateria de checagens (estática, lint, testes, LLM-as-judge, grafo, comportamental e trace) e consolida os resultados.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # preencha BENCH_DIR, GENERATED_DIR, EXPERIMENTS_RESULTS e as chaves de API
```

Variáveis obrigatórias em `.env`:

- `BENCH_DIR` — pasta com as especificações (padrão: `./specs`)
- `GENERATED_DIR` — onde cada agente gerado será escrito (padrão: `./generated_agents`)
- `EXPERIMENTS_RESULTS` — raiz dos resultados por etapa (padrão: `./experiments_results`)
- Chaves de API conforme as etapas 05–08 (`OPENAI_API_KEY`, `GEMINI_API_KEY`, etc.)

## Rodando o pipeline

```bash
./pipe.sh
```

O script lê duas listas no topo do arquivo:

```bash
llms=("claude" "codex" "opencode" "copilot" "gemini")
agents=("001_list_issues_agent")
```

Edite-as para escolher quais LLMs e quais pastas de especificação executar. Para cada combinação `LLM × agente`, as etapas rodam **sequencialmente**; combinações diferentes passam pelo mesmo loop uma após a outra. A geração (etapa 01) deve ser usada quando quiser gerar os agentes antes das checagens.

## Como o pipeline funciona

Cada etapa é um script independente em `pipeline/` invocado como `python3 pipeline/NN_*.py --llm <llm> --agent <agent>`:

| Etapa | Script | O que faz |
|-------|--------|-----------|
| 01 | `01_generate.py` | Chama o adapter do LLM e gera o agente em `GENERATED_DIR/<llm>/<agent>/` |
| 02 | `02_static_check.py` | Verifica se o agente é executável (sintaxe, imports, boot) |
| 03 | `03_lints_check.py` | Lint com `ruff`/`pylint` e análise com SonarQube |
| 04 | `04_tests_check.py` | Roda a suíte de testes do próprio agente |
| 05 | `05_spec_llm_judge_check.py` | LLM-as-judge: aplica `spec_adherence.py` da especificação |
| 06 | `06_graph_eval_check.py` | Avalia o grafo (LangGraph) do agente |
| 07 | `07_behavioral_check.py` | Roda `test_deepeval.py` (métricas DeepEval) |
| 08 | `08_trace_check.py` | Roda `test_trace.py` (verificação de trajetória/trace) |

Os artefatos de cada etapa ficam em `EXPERIMENTS_RESULTS/<NN_etapa>/<llm>/<agent>/`. O site em `website/` agrega esses JSONs em dashboards.

## Geração dos testes

Os arquivos `test_deepeval.py` e `test_trace.py` são gerados a partir do `spec.md` de cada especificação, usando os scripts em `test_generator/`:

```bash
# teste caixa-preta (DeepEval) — para a etapa 07
python3 test_generator/blackbox_generator.py specs/001_list_issues_agent/spec.md

# teste caixa-branca (trace) — para a etapa 08
python3 test_generator/whitebox_generator.py specs/001_list_issues_agent/spec.md
```

Cada gerador escreve o arquivo correspondente dentro da própria pasta da especificação. Rodar manualmente é possível com `pytest`:

```bash
pytest specs/001_list_issues_agent/test_deepeval.py -vv
pytest specs/001_list_issues_agent/test_trace.py -vv
```

## Estrutura esperada da pasta de especificação

Antes de executar o pipeline, cada pasta em `specs/<NNN_nome_do_agente>/` precisa conter:

```
specs/001_list_issues_agent/
├── spec.md             # especificação do agente (entrada humana)
├── spec_adherence.py   # critérios para o LLM-as-judge (etapa 05)
├── test_deepeval.py    # gerado por blackbox_generator.py (etapa 07)
└── test_trace.py       # gerado por whitebox_generator.py (etapa 08)
```

Sem esses quatro arquivos as etapas 05, 07 e 08 falham ou são puladas.
