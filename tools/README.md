# tools

Standalone debugging and maintenance utilities for Word Play. These are
not part of the runtime package; they are small scripts meant to be run
directly from the repo root with `uv run python tools/<script>.py`.

| Script | Purpose |
| --- | --- |
| `dump_replay_log.py` | Reads an `ExperimentRecorder` pickle (the file written by any example that calls `record_step`, e.g. `examples/llm_among_us.py`) and writes a human-readable `.txt` transcript next to it. Prints each step's chosen actions, the full observation text each LLM saw, and every emitted event (`speech`, `kill`, `meeting`, `report`, `winner`, etc.). |
| `setup_sglang_server.sh` | One-shot environment setup for SGLang inference: creates a venv, installs sglang, sgl-kernel, and CUDA 12 runtime libraries. Idempotent. |
| `run_sglang_server.sh` | Wrapper that launches `python -m sglang.launch_server` with the correct `LD_LIBRARY_PATH` and other environment variables required by the CUDA 12 builds of sgl-kernel. Arguments are forwarded verbatim. |

Add new utility scripts here rather than under `examples/`, which is
reserved for runnable demos meant to be read top-to-bottom.
