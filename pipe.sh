#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="$SCRIPT_DIR/pipeline/"

#llms=("codex")
llms=("claude" "codex" "opencode" "copilot" "gemini") # #("claude" "codex" "opencode" "copilot" "gemini")

agents=("001_list_issues_agent")

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

# ============================================================
# 01 — Geração dos agentes
#      Paralelo por LLM × agente
# ============================================================

# pids=()
# labels=()

# for llm in "${llms[@]}"; do
#     for agent in "${agents[@]}"; do

#         python3 "$PIPELINE_DIR/01_generate.py" \
#             --llm "$llm" \
#             --agent "$agent" &

#         pids+=($!)
#         labels+=("$llm / $agent")

#         printf "▶ Iniciando geração para %s / %s...\n" "$llm" "$agent"
#     done
# done

# # Aguarda todas as gerações
# failed=0

# for i in "${!pids[@]}"; do
#     pid="${pids[$i]}"
#     label="${labels[$i]}"

#     if wait "$pid"; then
#         printf "✔ Geração concluída para %s\n" "$label"
#     else
#         status=$?
#         printf "✘ Geração falhou para %s (código de saída: %s)\n" \
#             "$label" "$status"
#         failed=1
#     fi
# done

# if [ "$failed" -ne 0 ]; then
#     printf "\n✘ A geração falhou para pelo menos um agente.\n"
#     exit 1
# fi

# ============================================================
# 02–07 — Executados depois que TODAS as gerações terminarem
# ============================================================

for llm in "${llms[@]}"; do
    for agent in "${agents[@]}"; do

        printf "\n============================================================\n"
        printf "LLM: %s | AGENTE: %s\n" "$llm" "$agent"
        printf "============================================================\n"

        run_step \
            "python3 \"$PIPELINE_DIR/02_static_check.py\" --llm \"$llm\" --agent \"$agent\"" \
            "Executando 02_static_check.py para $llm / $agent..."

        run_step \
            "python3 \"$PIPELINE_DIR/03_lints_check.py\" --llm \"$llm\" --agent \"$agent\"" \
            "Executando 03_lints_check.py para $llm / $agent..."

        run_step \
            "python3 \"$PIPELINE_DIR/04_tests_check.py\" --llm \"$llm\" --agent \"$agent\"" \
            "Executando 04_tests_check.py para $llm / $agent..."

        run_step \
            "python3 \"$PIPELINE_DIR/05_spec_llm_judge_check.py\" --llm \"$llm\" --agent \"$agent\"" \
            "Executando 05_spec_llm_judge_check.py para $llm / $agent..."

        run_step \
            "python3 \"$PIPELINE_DIR/06_graph_eval_check.py\" --llm \"$llm\" --agent \"$agent\"" \
            "Executando 06_graph_eval_check.py para $llm / $agent..."
        
        run_step \
            "python3 \"$PIPELINE_DIR/07_behavioral_check.py\" --llm \"$llm\" --agent \"$agent\"" \
            "Executando 07_behavioral_check.py para $llm / $agent..."

        run_step \
            "python3 \"$PIPELINE_DIR/08_trace_check.py\" --llm \"$llm\" --agent \"$agent\"" \
            "Executando 08_trace_check.py para $llm / $agent..."

    done
done

printf "\nPipeline finalizado com sucesso!\n"