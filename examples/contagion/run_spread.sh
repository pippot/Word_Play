#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The spread run: does a false belief outlive its source -- and is speaking up
# what decides it?
#
# The pilot (2026-09-29) pointed here: nearly every instance that read the
# false note believed it, but believers rarely posted, so the belief died as
# they rotated out -- except in one episode, where they talked and it lasted
# 20 rounds after its readers had gone. The misaligned instance stayed silent
# in 7 of 8 episodes.
#
# Arms, each against its own placebo on the same seeds:
#   optional            posting is optional, as in the pilot
#   mandatory           every instance posts a daily update (--mandatory-posts),
#                       so believers -- and the misaligned instance -- must speak
#   mandatory-thinking  the same, with the misaligned instance reasoning before it
#                       posts (--seat-thinking; placebo and misaligned only)
# Conditions: placebo, factual, misaligned, in all 4 domains.
#
#   bash examples/contagion/run_spread.sh          # model server already on :30000
#   SEEDS=20 ARMS="optional mandatory" bash examples/contagion/run_spread.sh
#
# SEEDS=10 is 320 episodes, about 92,000 short calls (the thinking arm's posts
# are longer). Same knobs as run_pilot.sh: SEEDS, PARALLEL, DOMAINS, ARMS,
# SGLANG_BASE_URL, SGLANG_MODEL_NAME, PYTHON, OUT_DIR. Resumable: rerun with
# the same OUT_DIR and finished episodes are skipped.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-uv run python}"
SEEDS="${SEEDS:-10}"
PARALLEL="${PARALLEL:-16}"
DOMAINS="${DOMAINS:-bakery hiring medical oversight}"
ARMS="${ARMS:-optional mandatory mandatory-thinking}"
OUT_DIR="${OUT_DIR:-examples/contagion/logs/spread_$(date +%Y%m%d_%H%M%S)}"
export SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://localhost:30000/v1}"
export PYTHONUNBUFFERED=1
mkdir -p "$OUT_DIR"

if ! curl -sf "$SGLANG_BASE_URL/models" -o "$OUT_DIR/server.json"; then
    echo "No model server answers at $SGLANG_BASE_URL. Start one first, e.g.:"
    echo "  bash tools/run_sglang_server.sh --model-path Qwen/Qwen3.6-27B --port 30000"
    exit 1
fi
export SGLANG_MODEL_NAME="${SGLANG_MODEL_NAME:-$($PY -c "import json; print(json.load(open('$OUT_DIR/server.json'))['data'][0]['id'])")}"
echo "server: $SGLANG_BASE_URL  model: $SGLANG_MODEL_NAME  seeds: $SEEDS  arms: $ARMS"

for arm in $ARMS; do
    case "$arm" in
        optional)           flags="";                                conditions="placebo factual misaligned" ;;
        mandatory)          flags="--mandatory-posts";               conditions="placebo factual misaligned" ;;
        mandatory-thinking) flags="--mandatory-posts --seat-thinking"; conditions="placebo misaligned" ;;
        *) echo "unknown arm: $arm"; exit 1 ;;
    esac
    printf '\n=== arm: %s ===\n' "$arm"
    # shellcheck disable=SC2086
    $PY -m examples.contagion --episodes "$SEEDS" --domains $DOMAINS --conditions $conditions $flags \
        --parallel "$PARALLEL" --out "$OUT_DIR/$arm.jsonl" > "$OUT_DIR/$arm.run.log" 2>&1 &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        sleep ${POLL:-60}
        printf '  %s episodes done\n' "$(grep -c ' done (' "$OUT_DIR/$arm.run.log" || true)"
    done
    wait "$pid"
    $PY -m examples.contagion.judge "$OUT_DIR/$arm.jsonl" --parallel "$PARALLEL" > "$OUT_DIR/$arm.judge.log" 2>&1
    $PY -m examples.contagion.analyze "$OUT_DIR/$arm.jsonl" --show-flags 20 --checklist \
        --json "$OUT_DIR/$arm.report.json" > "$OUT_DIR/$arm.report.txt"
done

printf '\n=== SUMMARY ===\n'
for arm in $ARMS; do
    printf '\n######## %s\n' "$arm"
    sed -n '/== ACROSS DOMAINS/,/^$/p;/== PILOT CHECKLIST/,$p' "$OUT_DIR/$arm.report.txt"
done
cat <<EOF

Everything is in $OUT_DIR (per arm: .report.txt, .jsonl, .labels.jsonl).
Read the episodes where the belief survived ("treated seeds" in each report), e.g.:
  $PY -m examples.contagion.show $OUT_DIR/mandatory.jsonl --domain medical --condition factual --seed 0
EOF
