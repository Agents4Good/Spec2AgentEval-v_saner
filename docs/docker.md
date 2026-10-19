# Docker: geração e avaliação isoladas

O fluxo atual usa **duas imagens**, não o Compose legado da raiz:
`docker/Dockerfile.gen` para geração e `docker/Dockerfile.eval` para avaliação.
Build, login e execução real são decisões explícitas do operador; nenhum smoke
abaixo instala dependências, constrói imagens ou chama Docker real.

O [guia MVP](../docker/README_MVP.md) contém comandos de build/run e políticas
completas. A geração monta somente spec e saída, nunca o repositório. A avaliação
usa staging privado por job: script/helpers selecionados e bench `:ro`, dados e
resultados temporários `:rw`. Não monta árvores originais nem resultados antigos.
Credenciais são mínimas por etapa/provider, enviadas como `-e NOME`, sem valores
no argv; não há forwarding genérico de HOME, proxies ou `.env`.

02/04/06 usam rede `none`; demais etapas usam `EVAL_NETWORK=bridge` por default
(`none` ou rede nomeada como opt-in, nunca `host`). Bridge não restringe endpoints.
SELinux usa `EVAL_SELINUX=Z` somente nos temporários; `none` omite relabel e
`disable` desabilita explicitamente labeling com aviso. Publicação só começa após
confirmar teardown do container. Falha de confirmação preserva staging para
recuperação manual. Código local confiado não é sandbox.

08 acrescenta somente `trace_capture.py` ao staging. Seus traces canônicos são
publicados antes do relatório que os referencia, por manifesto e sem overwrite,
em `08_traces/<llm>/<agent>/<exec_uuid>/`. Veja
[captura e retenção](usage.md#stage-08-managed-tracing-phase-6).

## Smoke reproduzível, offline e sintético

Com `uv` disponível no PATH e `.venv` já preparada, a partir da raiz:

```sh
uv --no-config --offline run --no-project --no-env-file --python "$PWD/.venv/bin/python" \
  "$PWD/.venv/bin/python" scripts/smoke_offline.py
```

O entrypoint stdlib executa somente `test_trace_capture.py`, `test_trace_runner.py`,
`test_docker_eval_isolation.py` e `test_mvp_contract.py`. Usa Python absoluto,
HOME/cache/tmp/config próprios e ambiente construído do zero; PATH contém apenas
helpers shell permitidos. Dotenv, autoload pytest e telemetria DeepEval/LangSmith
ficam desabilitados. Fixtures reais de captura/pytest instalam guards de
rede/DNS/processos antes de plugins/callbacks; Docker e CLIs são fakes.
Timeout240s/cancelamento encerram o grupo antes de limpar temporários próprios.
Não executa `pipe.sh` como benchmark (apenas seu contrato com fakes), agentes de
bench, modelos, providers, autenticação, Docker/build/install reais.

Esses testes não comprovam Docker/SELinux nem serviços autenticados reais.
Smokes reais não são automáticos e exigem autorização e ambiente específicos.
O port foi seletivo de `origin/infra` (`9b69772`); `infra` local (`4096345`) já era
ancestral. Não houve merge amplo, alteração de aliases ou download de skills.
