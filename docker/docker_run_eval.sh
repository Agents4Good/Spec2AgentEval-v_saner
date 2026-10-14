#!/bin/bash
# MVP: roda etapas 02-08 num container isolado.
# 02/04/06 → --network none (offline); 03/05/07/08 → rede default + env de judge.
# Monta output :rw (/data, /results) e repo+BENCH :ro. Sem proxy, sem executor por etapa.
# Uso: docker_run_eval.sh --step <02|..|08> --llm <llm> --agent <a> --run <N> --experiment <slug>
set -euo pipefail

STEP=""; LLM=""; AGENT=""; RUN=1; EXPERIMENT="by_tool"
while [ $# -gt 0 ]; do
    case "$1" in
        --step) STEP="$2"; shift 2 ;;
        --llm) LLM="$2"; shift 2 ;;
        --agent) AGENT="$2"; shift 2 ;;
        --run) RUN="$2"; shift 2 ;;
        --experiment) EXPERIMENT="$2"; shift 2 ;;
        *) echo "arg desconhecido: $1" >&2; exit 2 ;;
    esac
done
[ -n "$STEP" ] && [ -n "$LLM" ] && [ -n "$AGENT" ] || { echo "--step/--llm/--agent obrigatórios" >&2; exit 2; }

case "$STEP" in
    02) STEP_FILE="02_static_check.py" ;;
    03) STEP_FILE="03_lints_check.py" ;;
    04) STEP_FILE="04_tests_check.py" ;;
    05) STEP_FILE="05_spec_llm_judge_check.py" ;;
    06) STEP_FILE="06_graph_eval_check.py" ;;
    07) STEP_FILE="07_behavioral_check.py" ;;
    08) STEP_FILE="08_trace_check.py" ;;
    *) echo "step inválido: $STEP (02..08)" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH="${BENCH_DIR:-$ROOT/specs}"
GEND="${GENERATED_DIR:-$ROOT/generated_agents}"
RESULTS="${EXPERIMENTS_RESULTS:-$ROOT/experiments_results}"
mkdir -p "$GEND" "$RESULTS"
IMAGE="${EVAL_IMAGE:-spec2agent-eval:infra}"

docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || docker build -f "$SCRIPT_DIR/Dockerfile.eval" -t "$IMAGE" "$ROOT"

NET_ARGS=()
JUDGE_ENV=()
case "$STEP" in
    02|04|06) NET_ARGS=(--network none) ;;           # offline: sem rede
    *) for v in OPENAI_API_KEY GEMINI_API_KEY SONAR_HOST_URL SONAR_LOGIN SONAR_PASSWORD USER_TOKEN SONAR_COMMAND; do
           [ -n "${!v:-}" ] && JUDGE_ENV+=(-e "$v")
       done ;;
esac

docker run --rm \
    --user "$(id -u):$(id -g)" \
    "${NET_ARGS[@]}" \
    -v "$GEND:/data:rw" \
    -v "$RESULTS:/results:rw" \
    -v "$ROOT:/repo:ro" \
    -v "$BENCH:/bench:ro" \
    -e GENERATED_DIR=/data -e EXPERIMENTS_RESULTS=/results -e BENCH_DIR=/bench \
    "${JUDGE_ENV[@]}" \
    "$IMAGE" python3 "/repo/pipeline/$STEP_FILE" \
        --llm "$LLM" --agent "$AGENT" --run "$RUN" --experiment "$EXPERIMENT"
echo "[docker-eval] step $STEP $LLM/$AGENT run $RUN ok (rede: ${NET_ARGS[*]:-default})"
