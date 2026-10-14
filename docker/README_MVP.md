# Docker MVP — isolamento mínimo (gen + eval)

## Build (uma vez)
```bash
docker build -f docker/Dockerfile.gen -t spec2agent-gen:infra docker/
docker build -f docker/Dockerfile.eval -t spec2agent-eval:infra .
```
`Dockerfile.eval` dá COPY só em `requirements.txt`; `pipeline/` e
`specs/` entram via mount `:ro` no run (imagem nunca carrega spec alheia).

## Login humano inicial (no HOST, uma vez)
```bash
codex login        # gera ~/.codex/auth.json
claude login       # gera ~/.claude/.credentials.json
# opencode usa modelo free: sem auth
```
Os scripts copiam o auth p/ um HOME temporário por job (0600) com
`CODEX_HOME`/`CLAUDE_CONFIG_DIR` apontando p/ ele.

## Run exemplo
```bash
USE_DOCKER=1 RUNS=1 ./pipe.sh                       # pipeline todo isolado
bash docker/docker_run_gen.sh --llm opencode --agent 011_list_issues --run 1
bash docker/docker_run_eval.sh --step 02 --llm opencode --agent 011_list_issues --run 1
```

## Como isola
- gen: monta `spec.md` em `/in:ro` + output em `/out:rw`; **repo nunca montado**.
- eval: output `:rw` (`/data`, `/results`), repo e specs `:ro`; etapas
  02/04/06 com `--network none`, 03/05/07/08 com rede + env de judge.
- `USE_DOCKER=0` (default) mantém o fluxo atual sem Docker.

## Limitações assumidas
- Sem Sonar por job, lock por provedor, métricas nativas ou proxy egress.
- Tokens/duration vão em `docker_gen_report.json` ao lado do `agent.py`;
  o relatório `01_generation_*` não é gerado no modo Docker.
- `Dockerfile.eval` instala o `requirements.txt` cheio (build lento, uma vez).
- Auth depende de login humano prévio no host; sem credencial, só opencode.
- Versões das CLIs pinadas no Dockerfile.gen (opencode fica na linha 1.x via
  pacote `opencode-ai`; revisar periodicamente).
