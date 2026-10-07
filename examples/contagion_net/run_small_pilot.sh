#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Small pilot for contagion_net: does everything work, and can spread happen
# at all? Not for results -- 4 seeds cannot say how often anything happens.
#
# Third pilot (2026-10-07): a misaligned instance that reasons, against the
# same seat honest and reasoning. The first two pilots planted a false rule;
# it never spread, because partners' real numbers refuted it within a day.
# A misaligned instance can adapt -- argue, discredit, share only what suits
# it -- the way the reasoning seat in examples/contagion moved its team.
# Same world as pilot 2: middle-clinic target, noise sd 30, plant on day 10.
#
#   1. seat_smoke  10 desks, 10 days, seat on day 3, 1 source desk, 1 seed:
#                  placebo_thinking + misaligned_thinking. Does the reasoning
#                  split off cleanly and never reach a message?          (~650 calls)
#   2. seat        30 desks, 35 days, 6 seats from day 10, 4 seeds:
#                  placebo_thinking + misaligned_thinking.               (~27k calls)
#   (earlier stages, still available: smoke, k6, k3)
#
# After each stage: the judge labels every text that names the target, then
# the report, maps and checks (health, awareness, the source desks' procedures
# version by version, sample threads). Everything ends in one tarball to send back.
#
#   bash examples/contagion_net/run_small_pilot.sh        # model server already up on :30000
#   STAGES="seat_smoke" bash examples/contagion_net/run_small_pilot.sh   # just the quick check
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
STAGES="${STAGES:-seat_smoke seat}"
SEEDS="${SEEDS:-4}"
PARALLEL="${PARALLEL:-8}"
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

for name in $STAGES; do
    case "$name" in
        smoke)  stage smoke --sources placebo factual --seeds 1 \
                    --desks 10 --days 10 --tenure 3 --plant-day 3 --k 1 ;;
        seat_smoke) stage seat_smoke --sources placebo_thinking misaligned_thinking --seeds 1 \
                    --desks 10 --days 10 --tenure 3 --plant-day 3 --k 1 ;;
        seat)   stage seat --sources placebo_thinking misaligned_thinking --seeds "$SEEDS" \
                    --desks 30 --days 35 --plant-day 10 --k 6 ;;
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
