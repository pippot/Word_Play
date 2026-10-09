#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# No chat, one planted note: what do the models decide?
#
# Nobody posts. The channel holds only the handover note, pinned for every
# instance; the rest is each instance's own reports. Placebo (an empty note)
# against factual (the false note), on the same seeds. One log per model, then
# one report side by side (python -m examples.contagion.decisions).
#
# MODELS is a list of arms, each one of:
#   NAME@URL   a decision model served by SGLang at URL (letter readout)
#   llm        the generative model at SGLANG_BASE_URL, writing JSON as usual
#
#   bash tools/run_sglang_server.sh --model-path alibiserikbay/JevK5-9B --port 30001
#   bash tools/run_sglang_server.sh --model-path alibiserikbay/JevK5 --port 30002
#   bash tools/run_sglang_server.sh --model-path Qwen/Qwen3.6-27B --port 30000
#   MODELS="alibiserikbay/JevK5-9B@http://localhost:30001/v1 alibiserikbay/JevK5@http://localhost:30002/v1
#           Qwen/Qwen3.6-27B@http://localhost:30000/v1 llm" bash examples/contagion/run_no_chat.sh
#
# The Qwen3.6-27B@ arm is the control: the same weights as `llm`, deciding by
# letter readout instead of writing JSON. Knobs: SEEDS, PARALLEL, DOMAINS,
# MODELS, SGLANG_BASE_URL, PYTHON, OUT_DIR. Resumable with the same OUT_DIR.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-uv run python}"
SEEDS="${SEEDS:-10}"
PARALLEL="${PARALLEL:-16}"
DOMAINS="${DOMAINS:-bakery hiring medical oversight}"
MODELS="${MODELS:-alibiserikbay/JevK5-9B@http://localhost:30001/v1}"
OUT_DIR="${OUT_DIR:-examples/contagion/logs/nochat_$(date +%Y%m%d_%H%M%S)}"
export SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://localhost:30000/v1}"
export PYTHONUNBUFFERED=1
mkdir -p "$OUT_DIR"

logs=""
for arm in $MODELS; do
    if [ "$arm" = llm ]; then
        url="$SGLANG_BASE_URL"; flags=""; tag=llm
    else
        name="${arm%@*}"; url="${arm#*@}"
        flags="--decider-model $name --decider-base-url $url"
        tag="$(echo "$name" | tr '/' '_')"
    fi
    if ! curl -sf "$url/models" -o /dev/null; then
        echo "No model server answers at $url (arm $arm). Start it first:"
        echo "  bash tools/run_sglang_server.sh --model-path <model> --port <port>"
        exit 1
    fi
    printf '\n=== %s ===\n' "$arm"
    # shellcheck disable=SC2086
    $PY -m examples.contagion --no-chat --episodes "$SEEDS" --domains $DOMAINS --conditions placebo factual \
        $flags --parallel "$PARALLEL" --out "$OUT_DIR/$tag.jsonl" > "$OUT_DIR/$tag.run.log" 2>&1 &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        sleep ${POLL:-30}
        printf '  %s episodes done\n' "$(grep -c ' done (' "$OUT_DIR/$tag.run.log" || true)"
    done
    wait "$pid" || { echo "  $arm failed: see $OUT_DIR/$tag.run.log"; continue; }
    logs="$logs $OUT_DIR/$tag.jsonl"
done

# shellcheck disable=SC2086
$PY -m examples.contagion.decisions $logs | tee "$OUT_DIR/report.txt"
echo
echo "Everything is in $OUT_DIR. One episode, day by day:"
echo "  $PY -m examples.contagion.show $OUT_DIR/<model>.jsonl --domain medical --condition factual --seed 0"
