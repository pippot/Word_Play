#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# setup_sglang_server.sh
#
# One-command setup for SGLang inference on a fresh machine.
#
# Usage:
#   bash tools/setup_sglang_server.sh
#
# Prerequisites:
#   - Python 3.13 (uv will download it if not found)
#   - Rust toolchain (for building outlines-core).  If cargo is not on PATH
#     the script will install it via rustup.
#
# This script is idempotent.
# ---------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== SGLang server setup ==="

# Rust is needed to build outlines-core (no prebuilt wheel for the
# exact version sglang pins).
if ! command -v cargo &>/dev/null; then
    echo "Installing Rust toolchain via rustup …"
    if ! curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; then
        # Some networks block or fail to resolve sh.rustup.rs, which only
        # serves the installer; the same installer is on static.rust-lang.org.
        echo "sh.rustup.rs unreachable; fetching rustup-init from static.rust-lang.org …"
        # rustup-init picks its behaviour from its file name, so keep the name.
        dir="$(mktemp -d)"
        curl --proto '=https' --tlsv1.2 -sSf -o "$dir/rustup-init" \
            "https://static.rust-lang.org/rustup/dist/$(uname -m)-unknown-linux-gnu/rustup-init"
        chmod +x "$dir/rustup-init"
        "$dir/rustup-init" -y
        rm -rf "$dir"
    fi
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi

# uv sync --extra sglang-server will:
#   1.  Create .venv (Python 3.13, constrained in pyproject.toml).
#   2.  Install sglang, sgl-kernel, torch, and all runtime deps.
#   3.  Install the CUDA 12 shared-library packages needed by sgl-kernel.
uv sync --extra sglang-server

echo ""
echo "Done.  Launch with:"
echo "  bash tools/run_sglang_server.sh --model-path Qwen/Qwen3.6-27B --port 30000"
echo ""
echo "  # or equivalently:"
echo "  uv run --env-file .env.sglang python -m sglang.launch_server ..."
