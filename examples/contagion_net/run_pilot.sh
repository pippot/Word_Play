#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Contagion on a network: hunt for an episode where the planted idea takes
# over the pool. We look for the exception, not the average.
#
# Stages (STAGES, in order):
#   smoke      10 desks, 10 days (plant on day 3), 1 seed, placebo + factual (~650 calls).
#              Read the transcripts before going on.
#   calibrate  30 desks, 35 days (plant on day 10), factual with k=1 and k=3 source desks, every
#              seed with its placebo twin (~3.4k calls per episode played to the end)
#   hunt       placebo AND factual on many seeds, stopping an episode once
#              believers stay under 5% for STOP days -- dead episodes cost
#              little. The placebo hunt is not optional: searching many
#              episodes finds chance extremes, and a factual takeover only
#              means something next to a placebo hunt of the same size.
#   confirm    every seed that took over in hunt (either source), replayed
#              REPS times with its twin (is it the world or chance?)
#
#   bash examples/contagion_net/run_pilot.sh                       # smoke + calibrate
#   STAGES="hunt confirm" DESKS=100 SEEDS=40 bash examples/contagion_net/run_pilot.sh
#
# Knobs: STAGES, SEEDS, DESKS, DAYS, TENURE, DEGREE, REWIRE, K, PLANT (burn-in
# days before the source arrives), STOP, REPS,
# PARALLEL (episodes at once), WORKERS (calls in flight per episode),
# SGLANG_BASE_URL, SGLANG_MODEL_NAME, PYTHON, OUT_DIR. Resumable: rerun with
# the same OUT_DIR and finished episodes are skipped.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-uv run python}"
STAGES="${STAGES:-smoke calibrate}"
SEEDS="${SEEDS:-6}"
DESKS="${DESKS:-30}"
DAYS="${DAYS:-35}"
TENURE="${TENURE:-5}"
DEGREE="${DEGREE:-6}"
REWIRE="${REWIRE:-0}"
K="${K:-6}"
PLANT="${PLANT:-10}"
STOP="${STOP:-3}"
REPS="${REPS:-3}"
PARALLEL="${PARALLEL:-4}"
WORKERS="${WORKERS:-64}"
OUT_DIR="${OUT_DIR:-examples/contagion_net/logs/pilot_$(date +%Y%m%d_%H%M%S)}"
export SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://localhost:30000/v1}"
export PYTHONUNBUFFERED=1
mkdir -p "$OUT_DIR"

if ! curl -sf "$SGLANG_BASE_URL/models" -o "$OUT_DIR/server.json"; then
    echo "No model server answers at $SGLANG_BASE_URL. Start one first, e.g.:"
    echo "  bash tools/run_sglang_server.sh --model-path Qwen/Qwen3.6-27B --port 30000"
    exit 1
fi
export SGLANG_MODEL_NAME="${SGLANG_MODEL_NAME:-$($PY -c "import json; print(json.load(open('$OUT_DIR/server.json'))['data'][0]['id'])")}"
echo "server: $SGLANG_BASE_URL  model: $SGLANG_MODEL_NAME  stages: $STAGES  out: $OUT_DIR"

net() {  # net NAME ARGS...: one run into $OUT_DIR/NAME.jsonl, then its report and maps
    local name="$1"; shift
    printf '\n=== %s ===\n' "$name"
    $PY -m examples.contagion_net --parallel "$PARALLEL" --workers "$WORKERS" --out "$OUT_DIR/$name.jsonl" "$@" \
        > "$OUT_DIR/$name.run.log" 2>&1
    $PY -m examples.contagion_net.judge "$OUT_DIR/$name.jsonl" > "$OUT_DIR/$name.judge.log" 2>&1
    $PY -m examples.contagion_net.analyze "$OUT_DIR/$name.jsonl" --json "$OUT_DIR/$name.rows.json" \
        > "$OUT_DIR/$name.report.txt"
    $PY -m examples.contagion_net.plot "$OUT_DIR/$name.jsonl" --out "$OUT_DIR/$name.maps.png" > /dev/null 2>&1 \
        || echo "  (no maps: see matplotlib)"
    cat "$OUT_DIR/$name.report.txt"
}

shape=(--desks "$DESKS" --days "$DAYS" --tenure "$TENURE" --degree "$DEGREE" --rewire "$REWIRE" --plant-day "$PLANT")

for stage in $STAGES; do
    case "$stage" in
        smoke)
            net smoke --sources placebo factual --seeds 1 --desks 10 --days 10 --tenure 3 --plant-day 3 --k 1 ;;
        calibrate)
            for k in 1 3; do
                net "calibrate_k$k" --sources placebo factual --seeds "$SEEDS" "${shape[@]}" --k "$k"
            done ;;
        hunt)
            net hunt --sources placebo factual --seeds "$SEEDS" "${shape[@]}" --k "$K" --stop-when-extinct "$STOP" ;;
        confirm)
            seeds=$($PY -c "import json; print(' '.join(sorted({str(r['seed']) for r in json.load(open('$OUT_DIR/hunt.rows.json')) if r['takeover']}, key=int)))")
            if [ -z "$seeds" ]; then
                echo "No takeover in hunt: nothing to confirm. Try more seeds, a larger K, or a shorter burn-in."
                continue
            fi
            # shellcheck disable=SC2086
            net confirm --sources placebo factual --seed-list $seeds --reps "$REPS" "${shape[@]}" --k "$K" ;;
        *) echo "unknown stage: $stage"; exit 1 ;;
    esac
done

printf '\nEverything is in %s (per stage: .report.txt, .maps.png, .jsonl)\n' "$OUT_DIR"
