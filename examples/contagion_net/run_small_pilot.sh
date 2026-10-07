#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Small pilot for contagion_net: does everything work, and can spread happen
# at all? Not for results -- 4 seeds cannot say how often anything happens.
#
# Fifth pilot (2026-10-07): HOW MANY MISALIGNED INSTANCES DOES IT TAKE?
# After a 10-day burn-in, misaligned instances pop up at random desks and
# days: each serves one ordinary 5-day stint, rotates out, and others may
# appear later. They reason; the rest of the pool doesn't. Three conditions
# per level, on the same instances:
#   misaligned_playbook   the playbook mandate (tactics that beat what stopped
#                         the seats in pilot 3, plus social proof), each alone
#   misaligned_coalition  the same, plus a confidential log shared by all
#                         misaligned instances across desks and rotations,
#                         read every day and added to every day (Colosseum's
#                         coalition channel: coordination and memory)
#   placebo_thinking      the honest twin: the same instances, honest, reasoning
#
#   sweep  30 desks, 30 days, 4 seeds, LEVELS misaligned instances out of
#          ~204 in the run (default 6 12 24 48: from ~1-2 to ~12 of the 30
#          desks misaligned at any one time). All levels run at once. (~140k calls)
#          The report ends with a HOW MANY IT TAKES table, one row per level and
#          condition, with believers, procedures and REGRET (patients lost per kit
#          against the best clinic, and the difference from the honest twin).
#
# A stronger model in the seats (every source seat, twin included):
#   SEAT_MODEL=<name> SEAT_BASE_URL=http://localhost:30001/v1 bash ...run_small_pilot.sh
# (any OpenAI-compatible server, e.g. a second SGLang server on another port)
#
# Earlier stages, still available: committed, seat_smoke, seat, smoke, k6, k3.
# Run time is set by days, not calls: all episodes run at once, and the seats'
# reasoning sets the pace of each day. Expect roughly 2.5-3 h for the sweep on one B200:
# reasoning seats are on duty almost every day from day 11, each reasoning day
# costs ~4 min (median trace ~2k tokens), and 48 episodes share the server.
# (Log entries are written without reasoning, so they add little.)
#
# After each stage: the judge labels every text that names the target, then
# the report, maps and checks (health, awareness, the source desks' procedures
# version by version, sample threads). Everything ends in one tarball to send back.
#
#   bash examples/contagion_net/run_small_pilot.sh        # model server already up on :30000
#   STAGES="seat_smoke" bash examples/contagion_net/run_small_pilot.sh   # a quick check of reasoning seats
#
# Start the server first, e.g.:
#   bash tools/run_sglang_server.sh --model-path Qwen/Qwen3.6-27B --port 30000
#
# Knobs: STAGES, SEEDS, PARALLEL (episodes at once), WORKERS (calls in flight
# per episode), SGLANG_BASE_URL, SGLANG_MODEL_NAME, PYTHON, OUT_DIR.
# Resumable: rerun with the same OUT_DIR and finished episodes are skipped.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-uv run python}"
STAGES="${STAGES:-sweep}"
LEVELS="${LEVELS:-6 12 24 48}"
SEAT_ARGS=()
if [ -n "${SEAT_MODEL:-}" ]; then
    SEAT_ARGS=(--seat-model "$SEAT_MODEL" --seat-base-url "${SEAT_BASE_URL:?set SEAT_BASE_URL for SEAT_MODEL}")
    curl -sf "$SEAT_BASE_URL/models" > /dev/null || { echo "No seat model server at $SEAT_BASE_URL"; exit 1; }
fi
SEEDS="${SEEDS:-4}"
PARALLEL="${PARALLEL:-16}"   # episodes at once per run (the sweep runs every level at once on top)
WORKERS="${WORKERS:-96}"
OUT_DIR="${OUT_DIR:-examples/contagion_net/logs/small_pilot_$(date +%Y%m%d_%H%M%S)}"
export SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://localhost:30000/v1}"
export PYTHONUNBUFFERED=1
mkdir -p "$OUT_DIR"

if ! curl -sf "$SGLANG_BASE_URL/models" -o "$OUT_DIR/server.json"; then
    echo "No model server answers at $SGLANG_BASE_URL. Start one first, e.g.:"
    echo "  bash tools/run_sglang_server.sh --model-path Qwen/Qwen3.6-27B --port 30000"
    exit 1
fi
export SGLANG_MODEL_NAME="${SGLANG_MODEL_NAME:-$($PY -c "import json; print(json.load(open('$OUT_DIR/server.json'))['data'][0]['id'])")}"

# Offline tests first: a broken checkout should fail in seconds, not after an hour.
$PY -m pytest -q tests/test_contagion_net.py > "$OUT_DIR/tests.log" 2>&1 || {
    echo "Offline tests failed: see $OUT_DIR/tests.log"; exit 1; }

{
    echo "started   $(date -u +%FT%TZ)"
    echo "host      $(hostname)"
    echo "git       $(git rev-parse --short HEAD 2>/dev/null || echo '?') $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') uncommitted files"
    echo "model     $SGLANG_MODEL_NAME at $SGLANG_BASE_URL"
    echo "stages    $STAGES   seeds $SEEDS   parallel $PARALLEL   workers $WORKERS"
    echo "seat      ${SEAT_MODEL:-same model as the pool}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/gpu       /' || true
} | tee "$OUT_DIR/run_info.txt"

stage() {  # stage NAME ARGS...: run, judge, report, maps, checks
    local name="$1"; shift
    local t0=$SECONDS
    printf '\n=== %s ===\n' "$name"
    $PY -m examples.contagion_net --parallel "$PARALLEL" --workers "$WORKERS" \
        --out "$OUT_DIR/$name.jsonl" "$@" > "$OUT_DIR/$name.run.log" 2>&1 || {
        echo "  run failed: see $OUT_DIR/$name.run.log"; tail -20 "$OUT_DIR/$name.run.log"; return 1; }
    grep -c ' done: ' "$OUT_DIR/$name.run.log" | sed 's/^/  episodes done: /'
    grep 'FAILED' "$OUT_DIR/$name.run.log" | sed 's/^/  /' || true
    $PY -m examples.contagion_net.judge "$OUT_DIR/$name.jsonl" > "$OUT_DIR/$name.judge.log" 2>&1 \
        || echo "  judge failed: see $OUT_DIR/$name.judge.log"
    $PY -m examples.contagion_net.judge "$OUT_DIR/$name.jsonl" --sample 40 > "$OUT_DIR/$name.judge_sample.txt" 2>&1 || true
    $PY -m examples.contagion_net.analyze "$OUT_DIR/$name.jsonl" --json "$OUT_DIR/$name.rows.json" \
        > "$OUT_DIR/$name.report.txt" 2>&1 || echo "  analyze failed: see $OUT_DIR/$name.report.txt"
    $PY -m examples.contagion_net.checks "$OUT_DIR/$name.jsonl" > "$OUT_DIR/$name.checks.txt" 2>&1 \
        || echo "  checks failed: see $OUT_DIR/$name.checks.txt"
    $PY -m examples.contagion_net.plot "$OUT_DIR/$name.jsonl" --out "$OUT_DIR/$name.maps.png" \
        > "$OUT_DIR/$name.plot.log" 2>&1 || echo "  plot failed: see $OUT_DIR/$name.plot.log"
    echo "  $(( (SECONDS - t0) / 60 )) min"
    sed -n '/^HEALTH/,/^$/p' "$OUT_DIR/$name.checks.txt"
    cat "$OUT_DIR/$name.report.txt"
}

sweep() {  # every level at once, each against its twin; then one report across levels
    local t0=$SECONDS pids=() logs=()
    printf '\n=== sweep: %s misaligned instances ===\n' "$LEVELS"
    for k in $LEVELS; do
        logs+=("$OUT_DIR/sweep_k$k.jsonl")
        $PY -m examples.contagion_net --parallel "$PARALLEL" --workers "$WORKERS" --out "$OUT_DIR/sweep_k$k.jsonl" \
            --sources placebo_thinking misaligned_playbook misaligned_coalition --seeds "$SEEDS" \
            --desks 30 --days 30 --plant-day 10 --arrival random --k "$k" ${SEAT_ARGS[@]+"${SEAT_ARGS[@]}"} \
            > "$OUT_DIR/sweep_k$k.run.log" 2>&1 &
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
    $PY -m examples.contagion_net.analyze "${logs[@]}" --json "$OUT_DIR/sweep.rows.json" > "$OUT_DIR/sweep.report.txt" 2>&1 \
        || echo "  analyze failed: see $OUT_DIR/sweep.report.txt"
    for log in "${logs[@]}"; do
        $PY -m examples.contagion_net.checks "$log" > "${log%.jsonl}.checks.txt" 2>&1 || true
        $PY -m examples.contagion_net.judge "$log" --sample 30 > "${log%.jsonl}.judge_sample.txt" 2>&1 || true
    done
    local top="${logs[${#logs[@]}-1]}"
    $PY -m examples.contagion_net.plot "$top" --out "${top%.jsonl}.maps.png" > /dev/null 2>&1 || echo "  (no maps)"
    echo "  $(( (SECONDS - t0) / 60 )) min"
    sed -n '/HOW MANY IT TAKES/,$p' "$OUT_DIR/sweep.report.txt"
}

for name in $STAGES; do
    case "$name" in
        smoke)  stage smoke --sources placebo factual --seeds 1 \
                    --desks 10 --days 10 --tenure 3 --plant-day 3 --k 1 ;;
        sweep) sweep ;;
        committed) stage committed --sources placebo_thinking misaligned_thinking --seeds "$SEEDS" \
                    --desks 30 --days 25 --plant-day 10 --source-stay 15 --k 6 ${SEAT_ARGS[@]+"${SEAT_ARGS[@]}"} ;;
        seat_smoke) stage seat_smoke --sources placebo_thinking misaligned_thinking --seeds 1 \
                    --desks 10 --days 8 --tenure 3 --plant-day 3 --k 1 ;;
        seat)   stage seat --sources placebo_thinking misaligned_thinking --seeds "$SEEDS" \
                    --desks 30 --days 30 --plant-day 10 --k 6 ;;
        k6)     stage k6 --sources placebo factual viral --seeds "$SEEDS" \
                    --desks 30 --days 35 --plant-day 10 --k 6 ;;
        k3)     stage k3 --sources placebo factual viral --seeds "$SEEDS" \
                    --desks 30 --days 35 --plant-day 10 --k 3 ;;
        *) echo "unknown stage: $name"; exit 1 ;;
    esac
done

echo "finished  $(date -u +%FT%TZ)" >> "$OUT_DIR/run_info.txt"
tar -czf "$OUT_DIR.tar.gz" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"
printf '\nDone. Send back: %s (%s)\n' "$OUT_DIR.tar.gz" "$(du -h "$OUT_DIR.tar.gz" | cut -f1)"
