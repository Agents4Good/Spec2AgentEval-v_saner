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
- eval: staging privado por allowlist; `/data` e `/results` `:rw`, código e
  bench selecionados `:ro`. 02/04/06 sem rede; demais com rede configurável
  e ambiente mínimo por etapa/provider (detalhes abaixo).
- `USE_DOCKER=0` (default) mantém o fluxo atual sem Docker.

## Fase 4 — staging privado da avaliação

`docker_run_eval.sh` chama `eval_staging.py` (stdlib). Os quatro mounts
`/repo:ro`, `/bench:ro`, `/data:rw`, `/results:rw` vêm exclusivamente de
temporários privados (raiz 0700), nunca dos diretórios originais. A seleção
ocorre **antes** da cópia; não se copia árvore para depois filtrar.

Allowlist por job:
- `/repo`: somente o script da etapa; 03 acrescenta
  `sonar_scripts/sonar_pipeline.py`; 07/08 acrescentam `utils/pytest_runner.py`.
- `/bench/<agent>`: 05 recebe `spec.md` e `spec_adherence.py`; 06 recebe
  `eval_2_graph.py`; 07 recebe `test_deepeval.py`; 08 recebe `test_trace.py`.
  Arquivos de bench ausentes continuam ausentes, preservando o diagnóstico
  da etapa. Rubricas alheias ao alvo, `.env`, outros agentes e arquivos
  auxiliares não entram.
- `/data/<experiment>/run_N/<llm>/<agent>`: pasta privada sempre criada;
  `agent.py` é copiado se presente (ausência obrigatória é relatada pela etapa);
  somente 04 recebe também testes Python diretos (`test*.py` ou `*test.py`),
  necessários ao inventário dessa etapa. Não há cópia recursiva nem resolução
  automática de imports de agentes. A recon dos runners encontrou apenas
  os dois imports locais acima; não há agentes gerados ou testes de bench
  no checkout para justificar dependências adicionais. Agentes multifile
  precisam de futura allowlist explícita, não de expansão automática.
- `/results` começa vazio. Resultados anteriores nunca são montados.

Publicação em `EXPERIMENTS_RESULTS/<experiment>/run_N`: somente
`02_validation`, `03_lint`, `04_tests_check`, `05_spec_judge`, `06_graph_eval`,
`07_test_deepeval`, `08_trace_eval`, seguidos de `_<llm>_YYYYMMDD_HHMMSS.json`.
07/08 também preservam os arquivos opcionais exatos `trace.json`,
`trace.jsonl`, `agent.log`, `.pytest_report.json` da pasta do agente em
`eval_artifacts/<step>_<llm>_<agent>_<job-uuid>/`. Essa é a allowlist de
diagnósticos, não uma promessa sobre nomes de traces externos desconhecidos.
Pytest incorpora stdout/stderr e seu relatório estruturado no relatório da
etapa; seu JSON próprio fica em workspace temporário, sem apagar arquivos preexistentes.
Nada é publicado em generated, e `agent.py`/outros produtos arbitrários são
ignorados. Publicação também ocorre após falha/interrupção; nunca sobrescreve
destinos existentes, e falha operacional de publicação retorna não zero.
Symlinks nos arquivos selecionados ou seus ancestrais, inclusive destinos de
publicação, são rejeitados. Componentes de job aceitam somente letras,
números, `_` e `-`; run é inteiro positivo canônico. UID/GID são os do host.

Políticas explícitas (sem fallback):
- SELinux: default `EVAL_SELINUX=Z` aplica `:Z` somente aos temporários.
  `none` omite relabel; `disable` emite aviso e usa
  `--security-opt label=disable`. Outros valores falham.
- Rede: 02/04/06 sempre `--network none`. Nas demais,
  `EVAL_NETWORK=bridge` por default; `none` é opt-in offline e nomes de redes
  customizadas são aceitos. `host` e namespaces `container:...` são proibidos.
- 03 recebe somente `SONAR_HOST_URL`, `SONAR_LOGIN`, `SONAR_PASSWORD`,
  `USER_TOKEN`, `SONAR_COMMAND`. 05/07/08 usam
  `EVAL_PROVIDER=openai|gemini|none` (default `openai`) e recebem somente a
  chave selecionada (`OPENAI_API_KEY` ou `GEMINI_API_KEY`). 02/04/06 não
  recebem credenciais. HOME, proxies e credenciais de geração não são
  encaminhados. Paths são internos e dotenv é desabilitado no container.
  Credenciais são encaminhadas somente como `-e NOME`, resolvidas pelo
  ambiente do cliente Docker; seus valores não entram no argv.

stdout/stderr permanecem nos streams do chamador. Inspect, build e run são
rastreados e canceláveis: polling a cada 0,1 s, término do grupo do cliente
com até 1 s para SIGTERM e mais 1 s para SIGKILL. Cancelamento nunca inicia
build/run subsequente. O run não tem prazo total de execução normal.
Após qualquer saída do cliente run (inclusive 125/desconexão/exceção),
teardown força `docker rm -f` e confirma ausência por listagem nomeada
bem-sucedida; cada operação tem timeout de 3 s. Falha de rm só é tolerada
se a ausência for confirmada, como no caso de auto-remoção por `--rm`.
Publicação e remoção dos temporários só ocorrem depois dessa confirmação.
Se não for possível confirmar (ou se o cliente não puder ser terminado e
recolhido dentro do prazo), retorna erro operacional, não publica e
preserva staging privado, imprimindo localização e nome do container para
recuperação explícita. Antes de remover esse staging manualmente, confirme
que o container nomeado não existe mais. Interrupção confirmada retorna
128+sinal; erros de teardown têm prioridade e conservam o contexto da exceção.

A falha de image inspect mantém o build existente como fallback de setup.
Nenhuma mudança de imagem é necessária para staging. **Limitação preexistente:**
o contexto desse build fallback continua sendo a raiz do repositório, sem
garantia de allowlist do contexto disponibilizado ao daemon. A allowlist
descrita nesta seção restringe staging/mounts de runtime, não o build.

Limites: bridge **não restringe endpoints**; código avaliado com rede pode
usar/exfiltrar a chave selecionada. Isto não é sandbox perfeito nem defesa
contra host concorrente malicioso com o mesmo UID. Não há evidência aqui de
execução Docker real, compatibilidade de modelos ou serviços autenticados.

## Fase 5 — status e configuração da avaliação

02–08 distinguem status operacional (`ok`, `error`, `skipped`) do veredicto
métrico legado. Reprovação métrica concluída retorna 0; erro operacional
retorna 1 após salvar o relatório possível. Qualquer entrada com erro torna
o relatório `error`; requisitos ausentes nunca são skips. Um alvo solicitado
sem pasta/`agent.py` chega à etapa e produz uma entrada `missing_agent`, sem
criar árvore no generated do host. Symlinks, inclusive dangling e ancestrais,
continuam proibidos. O payload `judge` permanece opaco, sem `passed` inferido.

Controles adicionais não sensíveis, enviados como `-e NOME=valor` somente
à etapa indicada (credenciais continuam somente `-e NOME`):
- 03: `SONAR_MODE=local|external|disabled` (default local),
  `SONAR_REQUIRED=1|0` (default 1). Linters rodam antes do Sonar; disabled
  pula só Sonar. Required=0 permite skip apenas por configuração ausente;
  erro configurado continua error. External não gerencia servidor; local
  encerra somente o servidor criado pelo job. Comandos/scanner: 300s,
  HTTP: 10s, boot/análise: 300s.
- 07/08: `PYTEST_TIMEOUT_SECONDS` positivo/finito (default 300). Usam Python
  corrente, workspace exclusivo e JSON fresco; call-only failure é
  `ok/passed=false`, setup/teardown ou falha operacional é `error/passed=null`.
  Todos skipped válidos não significam aprovação. Runner controla plugins e
  aplica `DEEPEVAL_TELEMETRY_OPT_OUT=1`; não se encaminha telemetry do host.

```bash
SONAR_MODE=disabled bash docker/docker_run_eval.sh --step 03 --llm opencode --agent 011_list_issues
PYTEST_TIMEOUT_SECONDS=300 bash docker/docker_run_eval.sh --step 07 --llm opencode --agent 011_list_issues
```

04 somente inventaria testes; não os executa. Não há novos mounts, cópia
recursiva, forwarding genérico de ambiente ou mudança de rede/SELinux nesta
fase. Evidência com Docker fake e etapas estagiadas locais não prova Docker,
SELinux, Sonar ou providers reais. O código local confiado não é sandbox.

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
