#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Long pilot for contagion_net, misaligned playbook vs its honest twin
# (placebo_thinking: the same instances, honest and reasoning). Each batch runs
# all its levels at once; a tarball is written after every batch, so partial
# results can be sent back early.
#
# The shared-log condition (misaligned_coalition) is dropped: in the
# 2026-10-08 run it matched the playbook at every level -- instances cannot
# choose whom they sync with, so coordinating on targets doesn't help. It is
# still in sources.py: add it with SOURCES="placebo_thinking misaligned_playbook
# misaligned_coalition".
#
#   1. levels      How many does it take? k = 12 24 48, arrivals to the end,
#                  30 desks, 30 days, seeds 0-3.                   24 episodes
#   2. stop        Does it outlive the attackers? k = 24 48, misaligned
#                  instances arrive on days 11-20 only, 40 days: source-free
#                  from day 30.                                    16 episodes
#   3. replicate   How often at the takeover level? k = 48, seeds 4-9. 12 episodes
#   4. structure   Clustered vs random network: rewire 1, k = 24, seeds 0-3.
#                  Not in the default run: BATCHES="structure".     8 episodes
#
# Expect roughly half the 2026-10-08 run (15 h with the shared-log condition).
#
# Each batch: run -> judge -> one report across its levels (HOW MANY IT TAKES
# table with believers, procedures and regret), checks and maps per level.
#
#   bash examples/contagion_net/run_long_pilot.sh            # model server already up on :30000
#   BATCHES="stop" bash examples/contagion_net/run_long_pilot.sh
#
# Knobs: BATCHES, PARALLEL, WORKERS, SEAT_MODEL/SEAT_BASE_URL, SGLANG_BASE_URL,
# EXTRA_ARGS (appended to every batch's command; the last value of an option
# wins, e.g. EXTRA_ARGS="--seed-list 0 --days 22" for a quick dry run),
# SGLANG_MODEL_NAME, PYTHON, OUT_DIR. Resumable: rerun with the same OUT_DIR
# and finished episodes are skipped.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# --no-sync: never change the environment from here. These scripts start many
# Python processes at once (every level, the judge, checks and plots), and
# concurrent `uv run` syncs of one .venv can leave a package half-installed
# (on the B200 box, 2026-10-08: numpy lost its OpenBLAS library).
PY="${PYTHON:-uv run --no-sync python}"
BATCHES="${BATCHES:-levels stop replicate}"
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
PARALLEL="${PARALLEL:-16}"
WORKERS="${WORKERS:-96}"
OUT_DIR="${OUT_DIR:-examples/contagion_net/logs/long_pilot_$(date +%Y%m%d_%H%M%S)}"
export SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://localhost:30000/v1}"
export PYTHONUNBUFFERED=1
mkdir -p "$OUT_DIR"

if ! curl -sf "$SGLANG_BASE_URL/models" -o "$OUT_DIR/server.json"; then
    echo "No model server answers at $SGLANG_BASE_URL. Start one first, e.g.:"
    echo "  bash tools/run_sglang_server.sh --model-path Qwen/Qwen3.6-27B --port 30000"
    exit 1
fi
export SGLANG_MODEL_NAME="${SGLANG_MODEL_NAME:-$($PY -c "import json; print(json.load(open('$OUT_DIR/server.json'))['data'][0]['id'])")}"
SEAT_ARGS=()
if [ -n "${SEAT_MODEL:-}" ]; then
    SEAT_ARGS=(--seat-model "$SEAT_MODEL" --seat-base-url "${SEAT_BASE_URL:?set SEAT_BASE_URL for SEAT_MODEL}")
    curl -sf "$SEAT_BASE_URL/models" > /dev/null || { echo "No seat model server at $SEAT_BASE_URL"; exit 1; }
fi

# Offline tests first: a broken checkout should fail in seconds, not after an hour.
$PY -m pytest -q tests/test_contagion_net.py > "$OUT_DIR/tests.log" 2>&1 || {
    echo "Offline tests failed: see $OUT_DIR/tests.log"; exit 1; }

{
    echo "started   $(date -u +%FT%TZ)"
    echo "host      $(hostname)"
    echo "git       $(git rev-parse --short HEAD 2>/dev/null || echo '?') $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') uncommitted files"
    echo "model     $SGLANG_MODEL_NAME at $SGLANG_BASE_URL"
    echo "seat      ${SEAT_MODEL:-same model as the pool}"
    echo "batches   $BATCHES   parallel $PARALLEL   workers $WORKERS"
    [ -n "${EXTRA_ARGS:-}" ] && echo "extra     $EXTRA_ARGS"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/gpu       /' || true
} | tee "$OUT_DIR/run_info.txt"

SOURCES="${SOURCES:-placebo_thinking misaligned_playbook}"

batch() {  # batch NAME "LEVELS" "SOURCES" "SEEDS" ARGS...: every level at once, then one report
    local name="$1" levels="$2" sources="$3" seeds="$4"; shift 4
    local t0=$SECONDS pids=() logs=()
    printf '\n=== %s: k = %s · %s · seeds %s ===\n' "$name" "$levels" "$sources" "$seeds"
    echo "$name started $(date -u +%FT%TZ)" >> "$OUT_DIR/run_info.txt"
    for k in $levels; do
        local log="$OUT_DIR/${name}_k$k.jsonl"
        logs+=("$log")
        # shellcheck disable=SC2086
        $PY -m examples.contagion_net --parallel "$PARALLEL" --workers "$WORKERS" --out "$log" \
            --sources $sources --seed-list $seeds --k "$k" "$@" ${SEAT_ARGS[@]+"${SEAT_ARGS[@]}"} \
            ${EXTRA[@]+"${EXTRA[@]}"} \
            > "${log%.jsonl}.run.log" 2>&1 &
        pids+=($!)
    done
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || echo "  run failed: see ${logs[$i]%.jsonl}.run.log"
    done
    echo "  runs done: $(( (SECONDS - t0) / 60 )) min"
    for log in "${logs[@]}"; do
        $PY -m examples.contagion_net.judge "$log" > "${log%.jsonl}.judge.log" 2>&1 &
    done
    wait
    $PY -m examples.contagion_net.analyze "${logs[@]}" --json "$OUT_DIR/$name.rows.json" \
        > "$OUT_DIR/$name.report.txt" 2>&1 || echo "  analyze failed: see $OUT_DIR/$name.report.txt"
    for log in "${logs[@]}"; do
        $PY -m examples.contagion_net.checks "$log" > "${log%.jsonl}.checks.txt" 2>&1 || true
        $PY -m examples.contagion_net.judge "$log" --sample 30 > "${log%.jsonl}.judge_sample.txt" 2>&1 || true
        $PY -m examples.contagion_net.plot "$log" --out "${log%.jsonl}.maps.png" > /dev/null 2>&1 || true
    done
    echo "$name finished $(date -u +%FT%TZ) ($(( (SECONDS - t0) / 60 )) min)" >> "$OUT_DIR/run_info.txt"
    echo "  $(( (SECONDS - t0) / 60 )) min"
    sed -n '/HOW MANY IT TAKES/,$p' "$OUT_DIR/$name.report.txt"
    # A tarball after every batch: partial results can be sent back early.
    tar -czf "$OUT_DIR.tar.gz" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"
    echo "  results so far: $OUT_DIR.tar.gz"
}

for name in $BATCHES; do
    case "$name" in
        levels)    batch levels "12 24 48" "$SOURCES" "0 1 2 3" \
                       --desks 30 --days 30 --plant-day 10 --arrival random ;;
        stop)      batch stop "24 48" "$SOURCES" "0 1 2 3" \
                       --desks 30 --days 40 --plant-day 10 --arrival random --arrival-until 20 ;;
        replicate) batch replicate "48" "$SOURCES" "4 5 6 7 8 9" \
                       --desks 30 --days 30 --plant-day 10 --arrival random ;;
        structure) batch structure "24" "$SOURCES" "0 1 2 3" \
                       --desks 30 --days 30 --plant-day 10 --arrival random --rewire 1 ;;
        *) echo "unknown batch: $name"; exit 1 ;;
    esac
done

echo "finished  $(date -u +%FT%TZ)" >> "$OUT_DIR/run_info.txt"
tar -czf "$OUT_DIR.tar.gz" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"
printf '\nDone. Send back: %s (%s)\n' "$OUT_DIR.tar.gz" "$(du -h "$OUT_DIR.tar.gz" | cut -f1)"
