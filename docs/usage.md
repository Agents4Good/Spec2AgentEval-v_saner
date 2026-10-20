# Benchmark Usage

## Prerequisites

### Code generation tools

The pipeline currently supports five code generation tools. Each one must be installed and reachable from your PATH before running stage 01.

| LLM | Install |
|-----|---------|
| **Claude Code** | `npm install -g @anthropic-ai/claude-code` |
| **Codex** | Follow the official OpenAI Codex CLI installation |
| **Copilot CLI** | `curl -fsSL https://gh.io/copilot-install \| bash` |
| **Gemini CLI** | `npm install -g @google/gemini-cli` (requires Node ≥ 20) |
| **OpenCode** | Follow the official OpenCode CLI installation |

The LLM identifier used throughout the pipeline is the lower-case name: `claude`, `codex`, `copilot`, `gemini`, `opencode`.

### Project dependencies

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Environment configuration

Copy `.env.example` to `.env` and fill in:

- `BENCH_DIR` — path to the specs folder (default: `./specs`)
- `GENERATED_DIR` — where generated agents are written (default: `./generated_agents`)
- `EXPERIMENTS_RESULTS` — root of per-stage results (default: `./experiments_results`)
- `OPENAI_API_KEY` / `GEMINI_API_KEY` — required by stages 05–08
- `SONAR_*` — only if you plan to run SonarQube in stage 03
- MCP / tool-specific keys as needed by the agent being evaluated

## Pipeline stages

Each stage is a standalone script in `pipeline/`, called as `python3 pipeline/NN_*.py --llm <llm> --agent <agent>`. Stages read inputs from `GENERATED_DIR` / `BENCH_DIR` and write their artifacts under `EXPERIMENTS_RESULTS/<stage>/<llm>/<agent>/`.

| Stage | Script                        | What it does                                                     | Prerequisites |
|-------|-------------------------------|------------------------------------------------------------------|---------------|
| 01    | `01_generate.py`              | Calls the LLM adapter and writes the agent to `GENERATED_DIR`    | `specs/<agent>/spec.md` exists |
| 02    | `02_static_check.py`          | Static / executability check on the generated agent              | Agent generated |
| 03    | `03_lints_check.py`           | Lint via `ruff` / `pylint` plus SonarQube analysis               | Agent generated |
| 04    | `04_tests_check.py`           | Runs the agent's own test suite                                  | Agent generated |
| 05    | `05_spec_llm_judge_check.py`  | LLM-as-judge against `spec_adherence.py`                         | Agent generated + `spec_adherence.py` present |
| 06    | `06_graph_eval_check.py`      | Evaluates the agent's LangGraph                                  | Agent generated |
| 07    | `07_behavioral_check.py`      | Runs `test_deepeval.py` (DeepEval metrics)                       | Agent generated + `test_deepeval.py` present |
| 08    | `08_trace_check.py`           | Runs `test_trace.py` (trajectory / trace verification)           | Agent generated + `test_trace.py` present |

Every stage emits a JSON result file following this envelope:

```json
{
  "run_id": "<unique_run_date_time>",
  "stage": "<pipeline_stage>",
  "model": "<used_llm>",
  "bench_dir": "<specs_directory>",
  "generated_dir": "<generated_agents_directory>",
  "total_agents": "<total_number_of_processed_agents>",
  "results": [
    {
      "agent": "<agent_name>",
      "llm": "<used_llm>",
      "spec_file": "<path_to_spec.md>",
      "output_dir": "<agent_output_directory>",
      "generated_file": "<path_to_generated_file>",
      "status": "<execution_status>"
    }
  ]
}
```

> Any stage can be executed standalone as long as its prerequisites are met.

## Running the full pipeline with `pipe.sh`

```bash
./pipe.sh
```

Two arrays at the top of the script control what runs:

```bash
llms=("claude" "codex" "opencode" "copilot" "gemini")
agents=("001_list_issues_agent")
```

For each `llm × agent` combination the script runs stages 02 → 08 sequentially. The generation step (01) is kept in a commented block near the top — uncomment it when you want to generate agents in parallel before the checks.

Before running `pipe.sh`:

- All dependencies installed (`requirements.txt`).
- Each `specs/<agent>/` folder contains `spec.md`, `spec_adherence.py`, `test_deepeval.py` and `test_trace.py` (see `extension.md`).
- `.env` is configured.
