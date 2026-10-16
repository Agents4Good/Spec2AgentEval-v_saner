#!/bin/bash
# Thin compatibility wrapper: delegates generation to stage 01 with the
# Docker backend. All orchestration, parsing and reporting live in
# pipeline/01_generate.py; this script only applies path defaults and
# forwards selectors. No independent report contract.
# Uso: docker_run_gen.sh --llm <llm> --agent <agent> --run <N> --experiment <slug> [--model <m>]
set -euo pipefail

LLM=""; AGENT=""; RUN=1; EXPERIMENT="by_tool"; MODEL=""
while [ $# -gt 0 ]; do
    case "$1" in
        --llm) LLM="$2"; shift 2 ;;
        --agent) AGENT="$2"; shift 2 ;;
        --run) RUN="$2"; shift 2 ;;
        --experiment) EXPERIMENT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        *) echo "arg desconhecido: $1" >&2; exit 2 ;;
    esac
done
[ -n "$LLM" ] && [ -n "$AGENT" ] || { echo "--llm e --agent obrigatórios" >&2; exit 2; }
case "$LLM" in
    claude|codex|opencode|claude-opus-*-*-low) ;;
    copilot|gemini) echo "llm local-only, sem backend docker: $LLM" >&2; exit 2 ;;
    *) echo "llm desconhecido: $LLM" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export BENCH_DIR="${BENCH_DIR:-$ROOT/specs}"
export GENERATED_DIR="${GENERATED_DIR:-$ROOT/generated_agents}"
export EXPERIMENTS_RESULTS="${EXPERIMENTS_RESULTS:-$ROOT/experiments_results}"
export GEN_IMAGE="${GEN_IMAGE:-spec2agent-gen:infra}"

ARGS=(--backend docker --llm "$LLM" --agent "$AGENT" --run "$RUN" --experiment "$EXPERIMENT")
[ -n "$MODEL" ] && ARGS+=(--model "$MODEL")
exec python3 "$ROOT/pipeline/01_generate.py" "${ARGS[@]}"
