# Contrato atual da pipeline

Resumo do comportamento implementado na árvore de trabalho atual; não é garantia de correção
semântica dos agentes nem de isolamento absoluto.

## Execução e seleção

- A etapa 01 gera agentes; os pipes executam avaliações 02–05 e 07–08.
  A etapa 06 está desabilitada nos dois pipes por solicitação do usuário, sem renumeração;
  `pipeline/06_graph_eval_check.py` e Docker `--step 06` continuam disponíveis manualmente.
- Há execução local e duas imagens Docker reutilizáveis: geração e avaliação.
- `pipe.sh` e `pipe_claude_versions.sh` mantêm execução local por padrão;
  `USE_DOCKER=1` seleciona os wrappers Docker existentes.
- `RUN=1` seleciona somente a run 1, para cada gerador/spec habilitado;
  não significa executar apenas um job. Os controles de intervalo continuam válidos.
- Cada job de geração ou etapa de avaliação usa um container novo;
  não há um container mutável compartilhado entre etapas.
- Imagem ausente pode ser construída automaticamente. Imagem existente não é
  reconstruída automaticamente após mudanças no código ou nas dependências.

## Geração e métricas

- O artefato aceito é um `agent.py` novo, regular e contido no diretório fresco
  reservado ao job. Saídas anteriores, symlinks e escapes de caminho são rejeitados.
- Parsers canônicos por CLI distinguem uso conhecido, zero e informação ausente.
  Streams e erros operacionais são preservados; custo de requisições só é
  informado quando conhecido, sem estimativa para preencher valores ausentes.
- Backends e wrappers de geração reutilizam esse contrato de métricas.
- A publicação exclusiva do relatório canônico evita sobrescrever resultado existente.
- Validar relatório e artefato não demonstra que o agente atende à especificação.

## Status operacional e resultado da avaliação

- `ok`, `error` e `skipped` descrevem operação, não o veredito da métrica.
  Uma avaliação concluída com reprovação pode ter `status=ok` e saída zero.
- Erro operacional produz `error` e saída não zero. Quando possível, o relatório
  é salvo antes desse retorno. Ausência de entrada obrigatória não vira skip ou aprovação.
- Sonar é independente dos linters: `SONAR_MODE=local|external|disabled`.
  `SONAR_REQUIRED=0` permite skip por configuração ausente, não por falha de serviço configurado.
- Análise Sonar concluída não significa quality gate verificado.
  Um servidor Sonar em Docker local tem rede e ciclo de vida separados.

## Limites do Docker e credenciais

- A avaliação usa staging privado por allowlist, não o checkout, HOME,
  benchmark completo ou árvore completa de resultados. Symlinks selecionados
  e seus ancestrais são rejeitados. Não se monta o socket Docker.
- O padrão SELinux usa temporários com `:Z`; desabilitar labels é opção explícita
  acompanhada de aviso. As etapas 02 e 04 ficam sem rede, assim como a 06 se chamada manualmente.
- Credenciais são selecionadas por etapa/provider, não herdadas genericamente
  do host. Claude/Codex recebem somente variáveis/arquivos selecionados.
- OpenCode não recebe cópia de autenticação/configuração local; uma chave
  compatível e restrita exige opt-in. Credenciais de avaliação são distintas.
- Se o teardown da avaliação não for confirmado, a operação falha e mantém
  o staging privado para diagnóstico, em vez de declarar limpeza bem-sucedida.
- Isso não é sandbox absoluto: há kernel/daemon compartilhados e etapas com
  rede para providers/serviços. Uma chave pode ser exfiltrada numa etapa com rede.

## Pytest e traces

- Pytest usa `sys.executable`, workspace/configuração/JSON privados e plugins
  explícitos, sem sobrescrever os testes ou relatórios preexistentes do agente.
  Essa proteção não se estende genericamente aos relatórios finais das etapas.
- Assertion na fase call é reprovação da métrica; falhas de setup/teardown são
  operacionais. Todos os testes skipped não equivalem a aprovação.
- Timeout/cancelamento do pytest atua sobre o grupo de subprocessos que ele possui.
- Somente a etapa 08 habilita `trace_enabled`; a 07 não habilita traces.
  A importação canônica é `from trace_capture import traced`.
- Instrumentação aceita funções síncronas; async é rejeitado. `traced` retorna
  `(output, trace)` mesmo com erro do agente; `run_traced` exporta antes de relançar o erro.
- Em execução gerenciada, `trace_path` respeita `TRACE_DIR`. Status do trace não substitui o veredito do pytest.
  A associação usa `test_nodeid` exato e `trace_artifacts` no manifesto do relatório;
  não há serviço separado de manifestos.
- São validados schemas, IDs, UUIDs, arquivos e associações. A publicação não
  sobrescreve arquivos e publica traces antes do relatório.
- Redação cobre chaves/valores conhecidos, não conteúdo arbitrário, respostas
  de judges ou streams legados.

## Caminhos e ambiente

- Artefato: `<GENERATED_DIR>/<experiment>/run_N/<cli>/<agent>/agent.py`.
- Relatórios: `<EXPERIMENTS_RESULTS>/<experiment>/run_N/`; traces abaixo desse diretório em
  `08_traces/<cli>/<agent>/<exec_uuid>/<file>.json`.
- As raízes são parametrizadas por `GENERATED_DIR`, `EXPERIMENTS_RESULTS`
  e `BENCH_DIR`, respectivamente para agentes, resultados e benchmark.
- O shell não carrega `.env` automaticamente. A etapa 01 usa
  `load_dotenv(override=True)`; a avaliação Docker recebe a allowlist exportada
  e desabilita dotenv no container.

## Limitações e evidência disponível

- Geração não tem deadline nem cancelamento próprio do grupo e acumula streams
  até o fim; cancelamento já deixou container órfão observado.
- Avaliação não tem deadline global, embora pytest tenha timeout.
- OpenCode autenticado no host não implica autenticação no Docker. Aliases não
  garantem disponibilidade real de modelos ou serviços.
- O baseline histórico registra 574 testes sintéticos e smoke de 199 testes;
  esses conjuntos não foram reexecutados para esta documentação.
- Nesta sessão foram observados geração Claude real e relatório Sonar real
  na etapa 03: houve reprovação de lint com operação concluída (`ok`).
- O pipe Claude com Docker tem 10 testes sintéticos aprovados e revisão independente;
  não foi estabelecida execução real completa de 01–08. Runtimes não mencionados
  não devem ser considerados validados por essas evidências.

## Fontes principais

`pipe.sh`, `pipe_claude_versions.sh`, `pipeline/01_generate.py`,
`pipeline/02_static_check.py` a `pipeline/08_trace_check.py`,
`pipeline/utils/trace_capture.py`, `docker/eval_staging.py`,
`docker/docker_run_gen.sh`, `docker/docker_run_eval.sh` e `docker/README_MVP.md`.
