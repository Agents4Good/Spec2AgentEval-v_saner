#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="$SCRIPT_DIR/pipeline/"
USE_DOCKER=${USE_DOCKER:-0} # Opt-in: USE_DOCKER=1 ./pipe_claude_versions.sh

# Experimento "By Claude Version": compara 3 versões do Opus com effort low.
llms=("claude-opus-5-5-low") #"claude-opus-4-6-low" "claude-opus-4-8-low" 

agents=("001_list_issues_agent")

RUNS=${RUNS:-3}
EXPERIMENT=${EXPERIMENT:-by_claude_version}
# Optional overrides for partial reruns:
#   RUN=3 ./pipe_claude_versions.sh            → runs only run 3
#   RUNS_FROM=2 RUNS_TO=3 ...                  → runs 2 and 3
RUNS_FROM=${RUNS_FROM:-${RUN:-1}}
RUNS_TO=${RUNS_TO:-${RUN:-$RUNS}}

spinner() {
    local pid=$1
    local msg="$2"
    local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠏'
    local i=0

    tput civis 2>/dev/null || true

    while kill -0 "$pid" 2>/dev/null; do
        printf "\r%s %s" "${spin:i:1}" "$msg"
        i=$(( (i + 1) % ${#spin} ))
        sleep 0.1
    done

    tput cnorm 2>/dev/null || true
}

run_step() {
    local msg="$1"
    shift

    "$@" &
    local pid=$!

    spinner "$pid" "$msg"

    if wait "$pid"; then
        printf "\r✔ %s\n" "$msg"
    else
        local status=$?
        printf "\r✘ %s (código de saída: %s)\n" "$msg" "$status"
        return "$status"
    fi
}

for run in $(seq "$RUNS_FROM" "$RUNS_TO"); do

printf "\n############################################################\n"
printf "EXPERIMENT %s · RUN %s (selected %s..%s)\n" "$EXPERIMENT" "$run" "$RUNS_FROM" "$RUNS_TO"
printf "############################################################\n"

# ============================================================
# 01 — Geração dos agentes
#      Paralelo por variante × agente (dentro da run)
# ============================================================

pids=()
labels=()

for llm in "${llms[@]}"; do
    for agent in "${agents[@]}"; do

        gen_cmd=(python3 "$PIPELINE_DIR/01_generate.py")
        if [ "$USE_DOCKER" = "1" ]; then
            gen_cmd=(bash "$SCRIPT_DIR/docker/docker_run_gen.sh")
        fi
        "${gen_cmd[@]}" \
            --llm "$llm" \
            --agent "$agent" \
            --run "$run" \
            --experiment "$EXPERIMENT" &

        pids+=($!)
        labels+=("$llm / $agent (run $run)")

        printf "▶ Iniciando geração para %s / %s (run %s)...\n" "$llm" "$agent" "$run"
    done
done

# Aguarda todas as gerações desta run
failed=0

for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    label="${labels[$i]}"

    if wait "$pid"; then
        printf "✔ Geração concluída para %s\n" "$label"
    else
        status=$?
        printf "✘ Geração falhou para %s (código de saída: %s)\n" \
            "$label" "$status"
        failed=1
    fi
done

if [ "$failed" -ne 0 ]; then
    printf "\n✘ A geração falhou para pelo menos uma variante na run %s.\n" "$run"
    exit 1
fi

# ============================================================
# 02–08 — Executados depois que TODAS as gerações da run terminarem
# ============================================================

for llm in "${llms[@]}"; do
    for agent in "${agents[@]}"; do

        printf "\n============================================================\n"
        printf "RUN %s | VARIANTE: %s | AGENTE: %s\n" "$run" "$llm" "$agent"
        printf "============================================================\n"

        for step in 02_static_check 03_lints_check 04_tests_check \
            05_spec_llm_judge_check 06_graph_eval_check 07_behavioral_check 08_trace_check; do
            eval_cmd=(python3 "$PIPELINE_DIR/$step.py")
            if [ "$USE_DOCKER" = "1" ]; then
                eval_cmd=(bash "$SCRIPT_DIR/docker/docker_run_eval.sh" --step "${step%%_*}")
            fi
            run_step "Executando $step.py para $llm / $agent (run $run)..." \
                "${eval_cmd[@]}" --llm "$llm" --agent "$agent" \
                --run "$run" --experiment "$EXPERIMENT"
        done

    done
done

done

printf "\nPipeline finalizado (experimento %s, runs %s..%s)!\n" "$EXPERIMENT" "$RUNS_FROM" "$RUNS_TO"
