#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Long-pilot batches on several models, one after another. For each model:
# start an SGLang server, wait until it answers, run run_long_pilot.sh into
# its own folder, stop the server.
#
#   bash examples/contagion_net/run_models.sh                  # the revival batch on the three default models
#   MODELS="Qwen/Qwen3.5-9B Qwen/Qwen3.5-4B" bash examples/contagion_net/run_models.sh   # judged by Qwen3.5-27B at the end
#   EXTRA_ARGS="--seed-list 0 1 2 3" bash examples/contagion_net/run_models.sh   # half the seeds, half the time
#
# Default models: one release at three sizes, so that size is all that changes:
#   Qwen/Qwen3.5-27B
#   Qwen/Qwen3.5-9B
#   Qwen/Qwen3.5-4B
# Qwen3.5 has the architecture (hybrid Gated DeltaNet), chat template and
# thinking switch of Qwen3.6-27B, the model of every run before 28 Oct.
# Qwen3.6 and Qwen3.8 have no dense model below 27B, so a ladder that includes
# them mixes releases. MODELS="Qwen/Qwen3.6-27B" reruns on the earlier model.
#
# One judge labels every model's texts (JUDGE_MODEL, default Qwen/Qwen3.5-27B),
# so labels never change with the model judged: the other models run with
# SKIP_JUDGE=1, the judge model runs last, and while its server is still up it
# labels their logs and rewrites their reports. Belief and regret measures do
# not use the judge; procedures-pushing and the pressure block's pushing do.
#
# Servers start as tools/run_sglang_server.sh starts them, with no reasoning
# parser: the harness reads the <think> block from the reply itself.
#
# Knobs: MODELS, BATCHES (default revival), JUDGE_MODEL, PORT (30000),
# SERVER_ARGS (extra sglang flags, e.g. "--tp-size 2"), SERVER_TIMEOUT
# (seconds to wait for a server, default 3600: a first start downloads the model), OUT_ROOT, PYTHON; everything
# run_long_pilot.sh reads (PARALLEL, WORKERS, EXTRA_ARGS, ...) passes through.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-uv run --no-sync python}"
MODELS="${MODELS:-Qwen/Qwen3.5-27B Qwen/Qwen3.5-9B Qwen/Qwen3.5-4B}"
BATCHES="${BATCHES:-revival}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.5-27B}"
PORT="${PORT:-30000}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-3600}"
OUT_ROOT="${OUT_ROOT:-examples/contagion_net/logs/models_$(date +%Y%m%d_%H%M%S)}"
read -r -a SERVER_EXTRA <<< "${SERVER_ARGS:-}"
BASE_URL="http://localhost:$PORT/v1"
mkdir -p "$OUT_ROOT"

if curl -sf "$BASE_URL/models" > /dev/null 2>&1; then
    echo "A server already answers on port $PORT. Stop it first: this script starts and stops its own."
    exit 1
fi

slug() { echo "$1" | tr '/:' '__'; }

# The judge model last, so its server can label everyone's texts before it stops.
ordered=()
for m in $MODELS; do [ "$m" = "$JUDGE_MODEL" ] || ordered+=("$m"); done
judge_in_list=""
# (if, not `[ ] && ...`: under set -e a false test that ends a block or a pipe stops the script)
for m in $MODELS; do if [ "$m" = "$JUDGE_MODEL" ]; then judge_in_list=1; fi; done
if [ -n "$judge_in_list" ]; then ordered+=("$JUDGE_MODEL"); fi

server_pid=""
start_server() {  # start_server MODEL LOG: returns 0 once it answers
    local model="$1" log="$2" waited=0
    if command -v setsid > /dev/null; then
        # Its own process group, so stopping it also stops sglang's worker processes.
        setsid bash tools/run_sglang_server.sh --model-path "$model" --port "$PORT" \
            ${SERVER_EXTRA[@]+"${SERVER_EXTRA[@]}"} > "$log" 2>&1 &
    else
        bash tools/run_sglang_server.sh --model-path "$model" --port "$PORT" \
            ${SERVER_EXTRA[@]+"${SERVER_EXTRA[@]}"} > "$log" 2>&1 &
    fi
    server_pid=$!
    until curl -sf "$BASE_URL/models" > /dev/null 2>&1; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "  the server exited: see $log"; server_pid=""; return 1
        fi
        if [ "$waited" -ge "$SERVER_TIMEOUT" ]; then
            echo "  no answer after ${SERVER_TIMEOUT}s: see $log"; stop_server; return 1
        fi
        sleep 10; waited=$((waited + 10))
    done
    echo "  up after ${waited}s"
}

stop_server() {
    [ -n "$server_pid" ] || return 0
    kill -TERM -- "-$server_pid" 2>/dev/null || kill -TERM "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 90); do
        kill -0 "$server_pid" 2>/dev/null || break
        sleep 2
    done
    kill -KILL -- "-$server_pid" 2>/dev/null || kill -KILL "$server_pid" 2>/dev/null || true
    server_pid=""
    # The port, and the GPU memory with it, must be free before the next server starts.
    for _ in $(seq 1 30); do
        curl -sf "$BASE_URL/models" > /dev/null 2>&1 || break
        sleep 2
    done
    sleep 15
}
trap stop_server EXIT

analyze_args() {  # the batches judged as a shorter run, as run_long_pilot.sh's finish does
    case "$1" in
        mechanism|personas) echo "--until 40" ;;
        *) echo "" ;;
    esac
}

relabel() {  # relabel DIR: label DIR's logs with the judge model now serving, then rewrite its reports
    local dir="$1" batch log
    for batch in $BATCHES; do
        local logs=()
        for log in "$dir/${batch}"_*.jsonl; do
            case "$log" in *.labels.jsonl) continue ;; esac
            if [ -e "$log" ]; then logs+=("$log"); fi
        done
        [ "${#logs[@]}" -gt 0 ] || continue
        for log in "${logs[@]}"; do
            SGLANG_BASE_URL="$BASE_URL" SGLANG_MODEL_NAME="$JUDGE_MODEL" \
                $PY -m examples.contagion_net.judge "$log" > "${log%.jsonl}.judge.log" 2>&1 &
        done
        wait
        for log in "${logs[@]}"; do
            $PY -m examples.contagion_net.judge "$log" --sample 30 > "${log%.jsonl}.judge_sample.txt" 2>&1 || true
        done
        # shellcheck disable=SC2046
        $PY -m examples.contagion_net.analyze "${logs[@]}" $(analyze_args "$batch") --json "$dir/$batch.rows.json" \
            > "$dir/$batch.report.txt" 2>&1 || echo "  analyze failed: see $dir/$batch.report.txt"
    done
    echo "judged by $JUDGE_MODEL $(date -u +%FT%TZ)" >> "$dir/run_info.txt"
    tar -czf "$dir.tar.gz" -C "$(dirname "$dir")" "$(basename "$dir")"
}

{
    echo "started   $(date -u +%FT%TZ)"
    echo "models    ${ordered[*]}"
    echo "judge     $JUDGE_MODEL"
    echo "batches   $BATCHES"
    if [ -n "${EXTRA_ARGS:-}" ]; then echo "extra     $EXTRA_ARGS"; fi
} | tee "$OUT_ROOT/models_info.txt"

for model in "${ordered[@]}"; do
    out="$OUT_ROOT/$(slug "$model")"
    mkdir -p "$out"
    printf '\n##### %s\n' "$model"
    if ! start_server "$model" "$out/server.log"; then
        echo "$model: no server, skipped" >> "$OUT_ROOT/models_info.txt"
        continue
    fi
    skip=1
    if [ "$model" = "$JUDGE_MODEL" ]; then skip=""; fi
    SGLANG_BASE_URL="$BASE_URL" SGLANG_MODEL_NAME="$model" BATCHES="$BATCHES" OUT_DIR="$out" SKIP_JUDGE="$skip" \
        bash examples/contagion_net/run_long_pilot.sh || echo "  run_long_pilot.sh failed for $model"
    if [ "$model" = "$JUDGE_MODEL" ]; then
        for other in "${ordered[@]}"; do
            [ "$other" = "$JUDGE_MODEL" ] || relabel "$OUT_ROOT/$(slug "$other")"
        done
    fi
    stop_server
    echo "$model: done $(date -u +%FT%TZ)" >> "$OUT_ROOT/models_info.txt"
done

if [ -z "$judge_in_list" ]; then
    printf '\n##### judge %s\n' "$JUDGE_MODEL"
    if start_server "$JUDGE_MODEL" "$OUT_ROOT/judge_server.log"; then
        for other in "${ordered[@]}"; do relabel "$OUT_ROOT/$(slug "$other")"; done
        stop_server
    fi
fi

echo "finished  $(date -u +%FT%TZ)" >> "$OUT_ROOT/models_info.txt"
tar -czf "$OUT_ROOT.tar.gz" -C "$(dirname "$OUT_ROOT")" "$(basename "$OUT_ROOT")"
printf '\nDone. Send back: %s (%s)\n' "$OUT_ROOT.tar.gz" "$(du -h "$OUT_ROOT.tar.gz" | cut -f1)"
