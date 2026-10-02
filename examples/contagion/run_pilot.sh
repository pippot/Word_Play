#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Contagion pilot: 4 domains x 4 conditions on a few seeds, then post labels,
# the report, the go/no-go checklist and transcripts to read.
#
# 1. Start the model server (another terminal), as for Lifeline:
#      bash tools/run_sglang_server.sh --model-path Qwen/Qwen3.6-27B --port 30000
# 2. From the repository root:
#      bash examples/contagion/run_pilot.sh
#
# Knobs (environment variables):
#   SEEDS=2            seeds per domain x condition (2 -> 32 episodes, ~9,200 calls)
#   PARALLEL=16        episodes played at once (SGLang batches their requests)
#   DOMAINS="bakery hiring medical oversight"
#   CONDITIONS="placebo factual value misaligned"
#   SGLANG_BASE_URL=http://localhost:30000/v1
#   SGLANG_MODEL_NAME  default: whatever the server serves
#   PYTHON="uv run python"
#   OUT_DIR            default: examples/contagion/logs/pilot_<timestamp>
#
# Everything lands in OUT_DIR; the last thing printed is where to look.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-uv run python}"
SEEDS="${SEEDS:-2}"
PARALLEL="${PARALLEL:-16}"
DOMAINS="${DOMAINS:-bakery hiring medical oversight}"
CONDITIONS="${CONDITIONS:-placebo factual value misaligned}"
OUT_DIR="${OUT_DIR:-examples/contagion/logs/pilot_$(date +%Y%m%d_%H%M%S)}"
export SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://localhost:30000/v1}"
export PYTHONUNBUFFERED=1  # progress lines reach run.log as they happen
mkdir -p "$OUT_DIR/transcripts"

step() { printf '\n=== %s ===\n' "$*"; }

step "0/5 model server"
if ! curl -sf "$SGLANG_BASE_URL/models" -o "$OUT_DIR/server.json"; then
    echo "No model server answers at $SGLANG_BASE_URL. Start one first, e.g.:"
    echo "  bash tools/run_sglang_server.sh --model-path Qwen/Qwen3.6-27B --port 30000"
    exit 1
fi
SERVED="$($PY -c "import json; print(json.load(open('$OUT_DIR/server.json'))['data'][0]['id'])")"
export SGLANG_MODEL_NAME="${SGLANG_MODEL_NAME:-$SERVED}"
echo "server: $SGLANG_BASE_URL  model: $SGLANG_MODEL_NAME"

step "1/5 smoke test: one short misaligned episode per domain"
# shellcheck disable=SC2086
$PY -m examples.contagion --episodes 1 --rounds 3 --conditions misaligned --domains $DOMAINS \
    --parallel 4 --out "$OUT_DIR/smoke.jsonl" > "$OUT_DIR/smoke.log" 2>&1
$PY - "$OUT_DIR/smoke.jsonl" <<'PYEOF'
import json, sys
records = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
calls = sum(r["config"]["rounds"] * r["config"]["agents"] * 3 for r in records)
bad = [u for r in records for e in r["rounds"] for u in e.get("unusable", [])]
print(f"{len(records)} episodes, {len(bad)}/{calls} unusable replies")
for u in bad[:5]:
    print(f"  unusable {u['call']} from {u['agent']}: {str(u['raw'])[:300]!r}")
if len(records) == 0 or len(bad) > 0.05 * calls:
    sys.exit("Too many unusable replies: fix the server or model before running the pilot.")
posts = [p["text"] for r in records for e in r["rounds"] for p in e["posts"]]
for text in posts[:3]:
    print(f"  sample post: {text[:200]}")
PYEOF

step "2/5 pilot: $SEEDS seed(s) x [$DOMAINS] x [$CONDITIONS]"
# shellcheck disable=SC2086
$PY -m examples.contagion --episodes "$SEEDS" --domains $DOMAINS --conditions $CONDITIONS \
    --parallel "$PARALLEL" --out "$OUT_DIR/pilot.jsonl" > "$OUT_DIR/run.log" 2>&1 &
RUN_PID=$!
while kill -0 "$RUN_PID" 2>/dev/null; do
    sleep 30
    printf '  %s episodes done\n' "$(grep -c ' done (' "$OUT_DIR/run.log" || true)"
done
wait "$RUN_PID"

step "3/5 labelling which posts urge the target"
$PY -m examples.contagion.judge "$OUT_DIR/pilot.jsonl" --parallel "$PARALLEL" > "$OUT_DIR/judge.log" 2>&1
$PY -m examples.contagion.judge "$OUT_DIR/pilot.jsonl" --sample 40 > "$OUT_DIR/label_sample.txt"

step "4/5 report and checklist"
$PY -m examples.contagion.analyze "$OUT_DIR/pilot.jsonl" --show-flags 30 --checklist \
    --json "$OUT_DIR/report.json" > "$OUT_DIR/report.txt"
if grep -q "== ACROSS DOMAINS" "$OUT_DIR/report.txt"; then
    sed -n '/== ACROSS DOMAINS/,$p' "$OUT_DIR/report.txt"
else
    sed -n '/== PILOT CHECKLIST/,$p' "$OUT_DIR/report.txt"
fi

step "5/5 transcripts to read"
for domain in $DOMAINS; do
    for condition in $CONDITIONS; do
        $PY -m examples.contagion.show "$OUT_DIR/pilot.jsonl" --domain "$domain" --condition "$condition" \
            --seed 0 > "$OUT_DIR/transcripts/${domain}_${condition}_seed0.txt" 2>/dev/null || true
    done
done

if $PY -m examples.contagion.plot "$OUT_DIR/pilot.jsonl" --out "$OUT_DIR/plots" > "$OUT_DIR/plots.log" 2>&1; then
    figures="the research questions as figures"
else
    figures="none: see plots.log (figures need matplotlib: pip install matplotlib)"
fi

cat <<EOF

Done. Everything is in $OUT_DIR:
  report.txt          the full report (per domain, across domains, test-aware flags, checklist)
  label_sample.txt    40 judged posts: check the YES/no labels by hand
  transcripts/        one episode per domain x condition: read at least the misaligned and value ones
  plots/              $figures
  pilot.jsonl         the raw log (resumable: rerun with OUT_DIR=$OUT_DIR)
Go on to the full run only if the checklist says ok everywhere, or you know why it doesn't.
EOF
