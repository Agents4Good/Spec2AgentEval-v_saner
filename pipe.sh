#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="$SCRIPT_DIR/pipeline/"

llms=("codex")
#llms=("claude" "codex" "opencode" "copilot" "gemini")

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

for llm in "${llms[@]}"; do
    run_step \
        "python3 \"$PIPELINE_DIR/01_generate.py\" --llm \"$llm\" --agent 001_list_issues_agent" \
        "Executando 01_generate.py para $llm..."

    # run_step "python3 \"$PIPELINE_DIR/02_static_check.py\" --llm \"$llm\" --agent 001_list_issues_agent" "Executando 02_static_check.py para $llm..."
    # run_step "python3 \"$PIPELINE_DIR/03_lints_check.py\" --llm \"$llm\" --agent 001_list_issues_agent" "Executando 03_lints_check.py para $llm..."
    # run_step "python3 \"$PIPELINE_DIR/04_tests_check.py\" --llm \"$llm\" --agent 001_list_issues_agent" "Executando 04_tests_check.py para $llm..."
    # run_step "python3 \"$PIPELINE_DIR/05_spec_llm_judge_check.py\" --llm \"$llm\" --agent 001_list_issues_agent" "Executando 05_spec_llm_judge_check.py para $llm..."
    # run_step "python3 \"$PIPELINE_DIR/06_graph_eval_check.py\" --llm \"$llm\" --agent 001_list_issues_agent" "Executando 06_graph_eval_check.py para $llm..."
    # run_step "python3 \"$PIPELINE_DIR/07_behavioral_check.py\" --llm \"$llm\" --agent 001_list_issues_agent" "Executando 07_behavioral_check.py para $llm..."
    # run_step "python3 \"$PIPELINE_DIR/08_deterministic_check.py\" --llm \"$llm\" --agent 001_list_issues_agent" "Executando 08_deterministic_check.py para $llm..."
done

printf "\nPipeline finalizado com sucesso!\n"