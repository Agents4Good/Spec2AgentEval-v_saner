#!/bin/bash
# MVP: gera generated_agents/<exp>/run_<N>/<llm>/<agent>/agent.py num container
# isolado. O container NÃO vê o repo: só /in (spec+prompt, :ro) e /out (:rw).
# Mantém o layout de pipeline/01_generate.py (não cria canonical novo).
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
case "$LLM" in claude|codex|opencode) ;; *) echo "llm fora do MVP: $LLM (só claude|codex|opencode)" >&2; exit 2 ;; esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH="${BENCH_DIR:-$ROOT/specs}"
GEND="${GENERATED_DIR:-$ROOT/generated_agents}"
SPEC_FILE="$BENCH/$AGENT/spec.md"
[ -f "$SPEC_FILE" ] || { echo "spec não encontrada: $SPEC_FILE" >&2; exit 1; }

OUT_DIR="$GEND/$EXPERIMENT/run_$RUN/$LLM/$AGENT"
mkdir -p "$OUT_DIR"
IMAGE="${GEN_IMAGE:-spec2agent-gen:infra}"

docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || docker build -f "$SCRIPT_DIR/Dockerfile.gen" -t "$IMAGE" "$SCRIPT_DIR"

# Prompt construído no HOST (mesmo texto dos adapters: "Implement ...").
IN_DIR="$(mktemp -d)"; chmod 755 "$IN_DIR"
{ echo "Implement the following instructions in the current directory:"; echo; cat "$SPEC_FILE"; } > "$IN_DIR/prompt.txt"
cp "$SPEC_FILE" "$IN_DIR/spec.md"

# HOME temporário por job: copia auth com 0600; CODEX_HOME/CLAUDE_CONFIG_DIR apontam p/ ele.
TMPHOME="$(mktemp -d)"; chmod 755 "$TMPHOME"
trap 'rm -rf "$TMPHOME" "$IN_DIR"' EXIT
mkdir -p "$TMPHOME/.codex" "$TMPHOME/.claude"
AUTH_MOUNTS=()
copy_auth() { # src dst_rel mount_dst
    if [ -f "$1" ]; then
        mkdir -p "$TMPHOME/$(dirname "$2")"
        cp "$1" "$TMPHOME/$2"; chmod 0600 "$TMPHOME/$2"
        AUTH_MOUNTS+=(-v "$1:$3:ro")
    elif [ "$LLM" != "opencode" ]; then
        echo "aviso: auth ausente $1 (seguindo; opencode não precisa)" >&2
    fi
}
copy_auth "$HOME/.codex/auth.json" ".codex/auth.json" "/auth/codex/auth.json"
copy_auth "$HOME/.claude/.credentials.json" ".claude/.credentials.json" "/auth/claude/.credentials.json"

# Comando interno por LLM (espelha adapters/*.py; lê prompt do arquivo p/ evitar quoting hell).
case "$LLM" in
    claude)   MODEL="${MODEL:-claude-sonnet-4-5}"
        INNER='content=$(cat /in/prompt.txt); claude --dangerously-skip-permissions --model "$MODEL" -p "$content" --output-format json' ;;
    codex)    MODEL="${MODEL:-gpt-6-luna}"
        INNER='content=$(cat /in/prompt.txt); codex exec --json --yolo --model "$MODEL" "$content"' ;;
    opencode) MODEL="${MODEL:-opencode/longcat-2.5-preview-free}"
        INNER='content=$(cat /in/prompt.txt); opencode run "$content" --model "$MODEL" --format json --auto /out' ;;
esac

START=$(date +%s)
STATUS=0
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$IN_DIR:/in:ro" \
    -v "$OUT_DIR:/out:rw" \
    -v "$TMPHOME:/home/agent:rw" \
    "${AUTH_MOUNTS[@]}" \
    -e HOME=/home/agent -e CODEX_HOME=/home/agent/.codex -e CLAUDE_CONFIG_DIR=/home/agent/.claude \
    -e MODEL="$MODEL" \
    "$IMAGE" bash -c "cd /out && $INNER" > "$OUT_DIR/docker_gen_stdout.log" 2>&1 || STATUS=$?
DURATION=$(( $(date +%s) - START ))

[ -f "$OUT_DIR/agent.py" ] || { echo "falha: agent.py não criado (exit=$STATUS, ver $OUT_DIR/docker_gen_stdout.log)" >&2; exit 1; }

# Tokens best-effort no stdout --json (ausente → null). Não altera o formato de 01_generation_*.
TOKENS=$(python3 - "$OUT_DIR/docker_gen_stdout.log" <<'EOF'
import json, re, sys
t = open(sys.argv[1], encoding="utf-8", errors="replace").read()
def num(*keys):
    for k in keys:
        m = re.search(r'"' + k + r'"\s*:\s*(\d+)', t)
        if m: return int(m.group(1))
    return None
i, o, tot = num("input_tokens", "inputTokens"), num("output_tokens", "outputTokens"), num("total_tokens", "totalTokens")
print(json.dumps({"input": i, "output": o, "total": tot if tot is not None else ((i or 0) + (o or 0) if (i, o) != (None, None) else None)}))
EOF
)
printf '{"llm":"%s","agent":"%s","run":%s,"experiment":"%s","model":"%s","status":"ok","exit_code":%s,"duration_s":%s,"tokens":%s}\n' \
    "$LLM" "$AGENT" "$RUN" "$EXPERIMENT" "$MODEL" "$STATUS" "$DURATION" "$TOKENS" > "$OUT_DIR/docker_gen_report.json"
echo "[docker-gen] $LLM/$AGENT run $RUN em ${DURATION}s → $OUT_DIR/agent.py"
