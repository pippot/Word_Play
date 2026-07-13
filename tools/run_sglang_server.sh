#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# run_sglang_server.sh
#
# Thin wrapper that launches sglang's server with the correct environment.
# Forwards all arguments to ``python -m sglang.launch_server``.
#
# Usage:
#   bash tools/run_sglang_server.sh --model-path Qwen/Qwen3.6-27B --port 30000
# ---------------------------------------------------------------------------

REPO_ROOT="${WORDPLAY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

exec uv run \
    --directory "$REPO_ROOT" \
    --env-file "$REPO_ROOT/.env.sglang" \
    python -m sglang.launch_server "$@"
