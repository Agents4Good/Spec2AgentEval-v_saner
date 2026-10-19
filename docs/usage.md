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
uv venv
uv pip install -r requirements.txt
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

Each stage is a standalone script in `pipeline/`, called as `uv run python pipeline/NN_*.py --llm <llm> --agent <agent>`. Stages read inputs from `GENERATED_DIR` / `BENCH_DIR` and write reports under `EXPERIMENTS_RESULTS/<experiment>/run_<N>/` (`--experiment by_tool`, `--run 1` by default).

| Stage | Script                        | What it does                                                     | Prerequisites |
|-------|-------------------------------|------------------------------------------------------------------|---------------|
| 01    | `01_generate.py`              | Calls the LLM adapter and writes the agent to `GENERATED_DIR`    | `specs/<agent>/spec.md` exists |
| 02    | `02_static_check.py`          | Static / executability check on the generated agent              | Agent generated |
| 03    | `03_lints_check.py`           | Lint via `ruff` / `pylint` plus SonarQube analysis               | Agent generated |
| 04    | `04_tests_check.py`           | Inventories generated Python test files; does not execute them   | Agent generated |
| 05    | `05_spec_llm_judge_check.py`  | LLM-as-judge against `spec_adherence.py`                         | Agent generated + `spec_adherence.py` present |
| 06    | `06_graph_eval_check.py`      | Evaluates the agent's LangGraph                                  | Agent generated |
| 07    | `07_behavioral_check.py`      | Runs `test_deepeval.py` (DeepEval metrics)                       | Agent generated + `test_deepeval.py` present |
| 08    | `08_trace_check.py`           | Runs `test_trace.py` (trajectory / trace verification)           | Agent generated + `test_trace.py` present |

Stage 01 emits a JSON result file following this envelope; evaluation stages retain their own legacy fields and add the operational status contract below:

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

### Evaluation status and controls (phase 5, stages 02–08)

`status: ok|error|skipped` describes execution, not metric approval. Negative
metrics retain their legacy verdicts (`final_status`, `passed`, checks or opaque
`judge`) with `status: ok` and exit 0. Operational errors have a stable
`reason_code`, retain string/null `error`, make the report `error` and exit 1
after saving it. Explicitly disabled components may be `skipped`; missing
required inputs are errors, never skips. CLI argument errors retain exit 2;
cancellation uses 128+signal. Each requested `--agent` gets an entry even if
its folder/file is missing; unfiltered empty discovery is an error.

- 02 syntax, dependency and entrypoint check failures remain metric `FAILED`.
- 04 valid agents without tests return `tests.exists=false`, not an error.
- 05/06 preserve the judge payload without inferring `passed`; evaluator
  import/call failures and missing graph builders are operational errors.
- 03 runs Ruff/Pylint/Radon before Sonar setup. `SONAR_MODE=local` (default)
  starts a job-owned server; `external` uses the configured server without
  managing it; `disabled` skips Sonar only, not linters. `SONAR_REQUIRED=1`
  is default; `0` permits only missing configuration to skip. Errors from a
  configured Sonar remain errors. Tool/scanner timeout is 300s, HTTP 10s,
  boot/analysis polling 300s. No new Radon threshold is applied.
- 07/08 use the current Python interpreter and a private test/config/JSON
  workspace, preserving preexisting agent files. `PYTEST_TIMEOUT_SECONDS`
  must be positive and finite (default 300). Call-only failures are valid
  metric failures (`ok`, `passed=false`); setup/teardown errors, collection,
  timeout/spawn or invalid/missing JSON yield `error`, `passed=null`.
  Valid all-skipped tests are `ok`, `passed=null`, not implicit approval.
  Plugin autoload is disabled; required plugins are explicit and DeepEval
  telemetry is opted out before import. This is not a sandbox for local code.

```sh
SONAR_MODE=disabled uv run python pipeline/03_lints_check.py --llm opencode --agent 011_list_issues
PYTEST_TIMEOUT_SECONDS=300 uv run python pipeline/07_behavioral_check.py --llm opencode --agent 011_list_issues
```

Docker forwards these controls only to their respective stages; it does not
forward arbitrary host environment. Offline synthetic tests do not establish
compatibility with real providers, Sonar services, Docker or SELinux.

### Stage 08 managed tracing (phase 6)

In synthetic or trusted synchronous tests, use the canonical helper:

```python
from trace_capture import traced

def test_capture():
    output, trace = traced(lambda: 42)()
    assert output == 42
```

Only08 enables the internal `trace_enabled` runner option;07 is unchanged.
The runner controls a private `TRACE_DIR`, managed-export mode and `PYTHONPATH`
pointing only to the helper. Do not supply host output paths. Async functions
and awaitables are rejected. `traced` returns `(output, trace)` even on agent
failure; `run_traced` exports before rethrowing the agent exception. Capture or
export failures propagate rather than masquerading as agent failures.
LangSmith/LangChain tracing is forced off and DeepEval telemetry opted out.

Schema1 trace JSON carries `run_id` (hex32), query, execution `status`, safe
error, runs_collected, trajectory, tool_calls and `test_nodeid`. Each evaluated
pytest call needs an exactly associated trace. Missing/corrupt traces are
operational errors (`passed=null`), preserving pytest records. Valid all-skip
needs no trace and is not approval. **Trace status is not a pytest verdict**:
an expected agent error may be captured while the test passes.

`results[].trace_artifacts` links `path`, `run_id`, `test_nodeid`, `capture_status`.
Paths are relative to the report directory:
`08_traces/<llm>/<agent>/<exec_uuid>/<filename>.json`. Runner and Docker staging
validate identities, schema, regular files and symlink-free ancestors; staging
publishes only manifest references for the exact job, traces before the report,
without overwrite. No arbitrary tree or trace-supplied absolute path is copied.

Exports are0600, temporary workspaces private; published results persist until
explicit operator deletion. Set retention/access/backup policy accordingly.
Known-secret redaction is limited: arbitrary tool data, legacy streams and judge
payloads are **not guaranteed fully sanitized**. Local trusted code is not a
sandbox. See [offline synthetic smoke](docker.md#smoke-reproduzível-offline-e-sintético)
and [Docker publication details](../docker/README_MVP.md#fase-6--captura-e-publicação08).

### Stage 01 report contract (phase 2)

Stage 01 reserves one fresh output directory per agent
(`GENERATED_DIR/<experiment>/run_<N>/<llm>/<agent>`) before any CLI starts.
A pre-existing, symlinked or otherwise unsafe target is rejected; reruns
never overwrite prior artifacts or logs — pick a new `--run` for another
execution. `agent.py` must afterwards be a fresh regular symlink-free file
contained in the output directory, even when the CLI itself failed (the
artifact and parsed usage are then retained with `status: "error"`).

Every entry (success and error alike) carries the same keys: `agent`, `llm`,
`spec_file`, `output_dir`, `generated_file`, `status`, `error` (null on
success, otherwise stable safe tokens such as `cli_exit_9`, `parse_error`,
`persistence_failed`, `spawn_failed: <prog>`, `artifact_missing`,
`output_exists`/`output_unsafe`/`output_replaced` — never raw exceptions, prompts
or CLI output), `exit_code`, `parse_status` (`ok`/`partial`/`error`; a
canonical `error` fails the run while `partial` only warns, and legacy
Copilot/Gemini adapters report null here), `warnings`, `usage` (numeric
usage with harness-measured `process_duration_s`/`wall_duration_s`;
`cli_duration_s`/`cli_version` stay null unless a caller provides them) and
`log_paths` (raw stdout/stderr live only in these log files, never in the
report).

Reports are published under a UUID identity
(`01_generation_<llm>_<report_id>.json`) with atomic no-clobber semantics,
so parallel same-second jobs and reruns never overwrite each other:
temporaries are exclusively created unique files and a real UUID collision
retries with a fresh identity. The
report is always written first; any operational error then exits `1`
(`2` for missing env config or unsafe `--experiment`/`--agent`/`--run`
selectors, including symlinked run roots).

`SPEC2AGENT_SKIP_DOTENV=1` bypasses `.env` loading (isolated tests only).

### Docker backend (phase 3)

`python3 pipeline/01_generate.py --backend docker` (default `local`) runs
the same canonical adapter command inside a container via a small
transport; `--model` overrides the adapter model for that job only
(Claude alias effort preserved). Only `claude`, `codex`, `opencode`
(including `claude-opus-*-low` aliases) are supported — local-only
`copilot`/`gemini` are rejected explicitly before execution.

`docker/docker_run_gen.sh` is a thin compatibility wrapper (same
`--llm/--agent/--run/--experiment`, optional `--model`): it applies the
`BENCH_DIR`/`GENERATED_DIR`/`EXPERIMENTS_RESULTS` path defaults plus
`GEN_IMAGE` (default `spec2agent-gen:infra`) and delegates to stage 01.
The obsolete standalone `docker_gen_report.json` is gone; Docker runs emit
the canonical `01_generation_*` report above, with extra `docker_*` error
tokens on transport failures. Reservation, artifact validation and
no-clobber publication are unchanged and still owned by stage 01.
Successful setup writes metadata/status notes to `docker_setup.log`;
inspect/build stdout/stderr are discarded, and failed setup creates no
setup log. Temporary cleanup failures report `docker_cleanup_failed`
without discarding completed CLI streams, usage or exit status (CLI/parse
errors retain priority).

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
