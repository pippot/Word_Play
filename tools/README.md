# tools

Standalone debugging and maintenance utilities for Word Play. These are
not part of the runtime package; they are small scripts meant to be run
directly from the repo root with `uv run python tools/<script>.py`.

| Script | Purpose |
| --- | --- |
| `dump_replay_log.py` | Reads an `ExperimentRecorder` pickle (the file written by any example that calls `record_step`, e.g. `examples/llm_among_us.py`) and writes a human-readable `.txt` transcript next to it. Prints each step's chosen actions, the full observation text each LLM saw, and every emitted event (`speech`, `kill`, `meeting`, `report`, `winner`, etc.). |

Add new utility scripts here rather than under `examples/`, which is
reserved for runnable demos meant to be read top-to-bottom.
