#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="$SCRIPT_DIR/pipeline/"

# Experimento "By Claude Version": compara 3 versões do Opus com effort low.
llms=("claude-opus-5-5-low" "claude-opus-4-6-low" "claude-opus-4-8-low")

agents=("018_recipe_intermediate")

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
    local cmd="$1"
    local msg="$2"

    bash -c "$cmd" &
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
printf "EXPERIMENT %s · RUN %s / %s\n" "$EXPERIMENT" "$run" "$RUNS"
printf "############################################################\n"

# ============================================================
# 01 — Geração dos agentes
#      Paralelo por variante × agente (dentro da run)
# ============================================================

pids=()
labels=()

for llm in "${llms[@]}"; do
    for agent in "${agents[@]}"; do

        python3 "$PIPELINE_DIR/01_generate.py" \
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

        run_step \
            "python3 \"$PIPELINE_DIR/02_static_check.py\" --llm \"$llm\" --agent \"$agent\" --run \"$run\" --experiment \"$EXPERIMENT\"" \
            "Executando 02_static_check.py para $llm / $agent (run $run)..."

        run_step \
            "python3 \"$PIPELINE_DIR/03_lints_check.py\" --llm \"$llm\" --agent \"$agent\" --run \"$run\" --experiment \"$EXPERIMENT\"" \
            "Executando 03_lints_check.py para $llm / $agent (run $run)..."

        run_step \
            "python3 \"$PIPELINE_DIR/04_tests_check.py\" --llm \"$llm\" --agent \"$agent\" --run \"$run\" --experiment \"$EXPERIMENT\"" \
            "Executando 04_tests_check.py para $llm / $agent (run $run)..."

        run_step \
            "python3 \"$PIPELINE_DIR/05_spec_llm_judge_check.py\" --llm \"$llm\" --agent \"$agent\" --run \"$run\" --experiment \"$EXPERIMENT\"" \
            "Executando 05_spec_llm_judge_check.py para $llm / $agent (run $run)..."

        run_step \
            "python3 \"$PIPELINE_DIR/06_graph_eval_check.py\" --llm \"$llm\" --agent \"$agent\" --run \"$run\" --experiment \"$EXPERIMENT\"" \
            "Executando 06_graph_eval_check.py para $llm / $agent (run $run)..."

        run_step \
            "python3 \"$PIPELINE_DIR/07_behavioral_check.py\" --llm \"$llm\" --agent \"$agent\" --run \"$run\" --experiment \"$EXPERIMENT\"" \
            "Executando 07_behavioral_check.py para $llm / $agent (run $run)..."

        run_step \
            "python3 \"$PIPELINE_DIR/08_trace_check.py\" --llm \"$llm\" --agent \"$agent\" --run \"$run\" --experiment \"$EXPERIMENT\"" \
            "Executando 08_trace_check.py para $llm / $agent (run $run)..."

    done
done

done

printf "\nPipeline finalizado (experimento %s, runs %s..%s)!\n" "$EXPERIMENT" "$RUNS_FROM" "$RUNS_TO"
