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
Por job, o transporte provisiona exatamente UM modo da CLI selecionada
(nunca ambos, nunca cruzado entre CLIs):
- **env**: encaminha só o NOME da variável (`-e NOME`, sem valor no argv):
  Claude `ANTHROPIC_AUTH_TOKEN` → `ANTHROPIC_API_KEY` →
  `CLAUDE_CODE_OAUTH_TOKEN`; Codex `CODEX_API_KEY` → `CODEX_ACCESS_TOKEN`.
- **arquivo**: copia só o arquivo selecionado
  (`CLAUDE_CONFIG_DIR/.credentials.json` ou `CODEX_HOME/auth.json`,
  default `~/.claude` / `~/.codex`) para um HOME temporário privado por job
  (dirs 0700, arquivo 0600); o original nunca é modificado e o temporário é
  removido no cleanup (credenciais renovadas não persistem de volta).
  Falha de remoção gera `docker_cleanup_failed`, sem descartar streams,
  uso ou código de saída já coletados; erros da CLI/parser têm prioridade.
- **OpenCode**: sem credenciais por default (ainda mais nos modelos free
  `opencode/*`); opt-in estreito via `GEN_OPENCODE_API_KEY_ENV` nomeando
  exatamente `OPENAI_API_KEY` (`openai/*`), `ANTHROPIC_API_KEY`
  (`anthropic/*`) ou `GEMINI_API_KEY` (`google/*`).
Sem `--env-file`, sem HOME/config inteiros, sem binds duplicados `/auth`.

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
- `docker_setup.log` contém apenas metadados/status após setup bem-sucedido.
  stdout/stderr de inspect/build são descartados; falha no setup não cria
  esse log.
- Sem Sonar por job, lock por provedor, métricas nativas ou proxy egress.
- O `docker_gen_report.json` plano e independente foi removido de propósito:
  a geração Docker publica o mesmo relatório canônico
  `EXPERIMENTS_RESULTS/<experiment>/run_<N>/01_generation_<llm>_<id>.json`
  do modo local (mesmo envelope, mesmos tokens de erro + `docker_*`).
- `Dockerfile.eval` instala o `requirements.txt` cheio (build lento, uma vez).
- Auth depende de login humano prévio no host; sem credencial, só opencode.
- Versões das CLIs pinadas no Dockerfile.gen (opencode fica na linha 1.x via
  pacote `opencode-ai`; revisar periodicamente).
- Evidência aqui é de orquestração offline (Docker fake): sem claims de
  compatibilidade com provedores reais, login em produção ou billing.
- `Dockerfile.eval` instala o `requirements.txt` cheio (build lento, uma vez).
- Auth depende de login humano prévio no host; sem credencial, só opencode.
- Versões das CLIs pinadas no Dockerfile.gen (opencode fica na linha 1.x via
  pacote `opencode-ai`; revisar periodicamente).
