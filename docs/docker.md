# Running with Docker

The repository ships a `Dockerfile` and `docker-compose.yml` that build a container with every CLI the pipeline needs (Python 3.11, Node 20, Gemini CLI, Copilot CLI) and run `pipe.sh` as the default command.

## Build and run

```bash
docker compose build
docker compose run --rm spec2agenteval
```

The default `CMD` is `bash ./pipe.sh`, so running the service executes the full pipeline for the LLMs and agents configured at the top of `pipe.sh`.

## Environment

Compose reads `.env` from the repo root via `env_file`. Make sure the variables documented in [`usage.md`](./usage.md) are set — the container needs them to resolve `BENCH_DIR`, `GENERATED_DIR`, `EXPERIMENTS_RESULTS` and the API keys for stages 05–08.

## Persisting results on the host

By default the container writes `generated_agents/` and `experiments_results/` inside its own filesystem. To keep them on the host, add volume mounts to `docker-compose.yml`:

```yaml
services:
  spec2agenteval:
    build: .
    container_name: spec2agenteval-pipeline
    env_file:
      - .env
    working_dir: /app
    tty: true
    volumes:
      - ./specs:/app/specs
      - ./generated_agents:/app/generated_agents
      - ./experiments_results:/app/experiments_results
```

## Running a single stage

Override the default command to run one script instead of the whole pipeline:

```bash
docker compose run --rm spec2agenteval \
  python3 pipeline/02_static_check.py --llm claude --agent 001_list_issues_agent
```

## Adding new CLIs

If you integrate a new code generation tool (see [`extension.md`](./extension.md)), install its CLI in the `Dockerfile` so the container can invoke it in stage 01.
