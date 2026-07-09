"""
DUMP REPLAY LOG
===============

Reads a recorded experiment pickle (the file produced by
``ExperimentRecorder`` -- for example by the ``llm_among_us`` example or
the live ``rendering_demo.py``) and writes a readable text transcript
of the run to a ``.txt`` file in the same directory as the pickle.

The transcript contains:

  * The run's metadata (model, seed, etc.) and the resolved file path.
  * For every recorded frame:
      - the action each agent selected
      - the full observation text each LLM saw (system prompt + role +
        game state + nearby entities + action list)
      - every emitted ``render_state_event`` (``speech``, ``kill``,
        ``meeting``, ``report``, ``winner``, ``hit``, etc.), with any
        ``__entity_ref__`` entries resolved to entity names.
  * A footer with the final ``winner`` event (if any) and a deduplicated
    kill log.

The tool is intentionally generic: it does not know in advance which
event kinds a given experiment will produce, and works for any pickle
written by ``word_play.presets.renderers.ExperimentRecorder``.

USAGE
-----

::

    # Resolve by title slug (looks for the matching "_newest.pkl"):
    uv run python tools/dump_replay_log.py llm_among_us

    # Or pass a path directly:
    uv run python tools/dump_replay_log.py experiments/logs/llm_among_us_20260709_103154.pkl

The output file is written next to the source pickle with the same
basename and a ``.txt`` extension, e.g. ``llm_among_us_20260709_103154.txt``.
A short status line is printed to stderr so you know where the file went.

This script is read-only: it never imports the heavy LLM deps, never
touches the network, and never instantiates pygame. It just reads a
pickle and writes a text transcript.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TextIO

# Make ``src/`` importable when the script is launched directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Reuse the existing public helpers so this script stays in lock-step with
# how the pygame replay tool resolves log paths.
from word_play.presets.renderers import (  # noqa: E402
    load_recording_payload,
    replay_log_path,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

ENTITY_REF_KEY = "__entity_ref__"

# Friendly labels for the event kinds the Among Us example emits. Unknown
# kinds are still printed (uppercased) so the helper works for any recorder
# payload.
FRIENDLY_KINDS: dict[str, str] = {
    "speech": "CHAT",
    "kill": "KILL",
    "meeting": "MEETING",
    "report": "REPORT",
    "winner": "WINNER",
    "hit": "HIT",
}

# Width of the section separators.
SEPARATOR_WIDTH = 72


# ============================================================================
# HELPERS
# ============================================================================

def _resolve_log_path(spec: str) -> Path:
    """
    Resolve a CLI argument to a concrete pickle file.

    * If ``spec`` ends in ``.pkl`` or exists on disk, return it as-is.
    * Otherwise treat it as a title slug and use the same lookup the pygame
      replay tool uses (``replay_log_path``).
    """
    p = Path(spec)
    if p.suffix == ".pkl" or p.exists():
        return p
    return replay_log_path(spec)


def _entity_name_from_ref(frame: dict, ref: Any) -> str:
    """Resolve an ``__entity_ref__`` payload to the entity's name in this frame."""
    if not isinstance(ref, dict) or ENTITY_REF_KEY not in ref:
        return ""
    idx = ref[ENTITY_REF_KEY]
    entities = frame.get("entities", [])
    if isinstance(idx, int) and 0 <= idx < len(entities):
        return str(entities[idx].get("name", f"<entity#{idx}>"))
    return f"<ref#{idx}>"


def _format_event(frame: dict, event: dict) -> str:
    """Render a single ``render_state_event`` as one line."""
    kind = event.get("kind", "?")
    label = FRIENDLY_KINDS.get(kind, kind.upper())
    payload = event.get("payload", {}) or {}

    parts: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "entity":
                name = _entity_name_from_ref(frame, value)
                if name:
                    parts.append(f"entity={name}")
            elif key == "step":
                parts.append(f"step={value}")
            elif key == "text":
                # Speech text. Print compactly on one line.
                text = str(value).replace("\n", " ").strip()
                if len(text) > 240:
                    text = text[:240] + "..."
                parts.append(f'"{text}"')
            else:
                parts.append(f"{key}={value}")
    elif payload:
        parts.append(repr(payload))

    if not parts:
        return f"  {label}"
    return f"  {label:<8}  " + ", ".join(parts)


def _format_action(sel: dict) -> str:
    """Render one ``selected_actions`` entry."""
    actor = sel.get("actor_name", "?")
    action_type = sel.get("action_type", "?")
    label = sel.get("label", "")
    return f"[{actor:<8}] ACTION: {action_type} ({label})"


def _format_observation(obs: dict) -> str:
    """Render the observation text + action list for one agent."""
    name = obs.get("agent_name", "?")
    lines = [f"[{name}] observation text:"]
    for raw_line in (obs.get("text") or "").splitlines():
        lines.append(f"  {raw_line}")
    lines.append("")
    lines.append("  possible actions:")
    possible = obs.get("possible_actions") or []
    if not possible:
        lines.append("    (no actions available)")
    else:
        for i, pa in enumerate(possible):
            label = pa.get("label", "") if isinstance(pa, dict) else str(pa)
            lines.append(f"    [{i}] {label}")
    return "\n".join(lines)


def _format_metadata(metadata: Any) -> str:
    """Render the metadata dict as ``key: value`` lines."""
    if not isinstance(metadata, dict) or not metadata:
        return "  (none)"
    out: list[str] = []
    width = max((len(str(k)) for k in metadata), default=0)
    for key, value in metadata.items():
        out.append(f"  {str(key):<{width}} : {value}")
    return "\n".join(out)


# ============================================================================
# TRANSCRIPT BUILDER
# ============================================================================

def build_transcript(payload: dict, log_path: Path) -> str:
    """Return the full text transcript of a recorder payload."""
    title = payload.get("title", "?")
    frames = payload.get("frames", [])
    metadata = payload.get("metadata", {})

    out: list[str] = []
    out.append("=" * SEPARATOR_WIDTH)
    out.append(f"REPLAY LOG: {title}")
    out.append("=" * SEPARATOR_WIDTH)
    out.append(f"file:    {log_path}")
    out.append(f"version: {payload.get('version', '?')}")
    out.append(f"frames:  {len(frames)}")
    out.append("metadata:")
    out.append(_format_metadata(metadata))
    out.append("")

    for frame in frames:
        step = frame.get("cur_step", frame.get("tick", "?"))
        out.append("=" * SEPARATOR_WIDTH)
        out.append(f"STEP {step}")
        out.append("=" * SEPARATOR_WIDTH)
        out.append("")

        selected = frame.get("selected_actions", []) or []
        if selected:
            for sel in selected:
                out.append(_format_action(sel))
            out.append("")

        observations = frame.get("agent_observations", []) or []
        if observations:
            out.append("--- observations (what each LLM saw) ---")
            out.append("")
            for obs in observations:
                out.append(_format_observation(obs))
                out.append("")

        events = frame.get("render_state_events", []) or []
        if events:
            out.append("--- events ---")
            for event in events:
                out.append(_format_event(frame, event))
            out.append("")

    # --- footer (winner + kill log if present) ----------------------------
    #
    # NOTE: the recorder does not clear ``render_state.events`` between
    # captures, so the same event can appear in more than one frame.
    # Deduplicate by (step, killer, victim) so the kill log is accurate.
    winner: Any = None
    kill_log: list[dict] = []
    seen_kills: set[tuple] = set()
    for frame in frames:
        for event in frame.get("render_state_events", []) or []:
            if event.get("kind") == "winner":
                winner = (event.get("payload") or {}).get("winner", winner)
            elif event.get("kind") == "kill":
                pl = event.get("payload") or {}
                key = (pl.get("step"), pl.get("killer"), pl.get("victim"))
                if key in seen_kills:
                    continue
                seen_kills.add(key)
                kill_log.append(
                    {
                        "step": pl.get("step"),
                        "killer": pl.get("killer"),
                        "victim": pl.get("victim"),
                    }
                )

    out.append("=" * SEPARATOR_WIDTH)
    out.append("RESULT")
    out.append("=" * SEPARATOR_WIDTH)
    if winner is not None:
        out.append(f"Winner:        {winner}")
    else:
        out.append("Winner:        (none recorded)")
    out.append(f"Total frames:  {len(frames)}")
    out.append(f"Total kills:   {len(kill_log)}")
    for k in kill_log:
        out.append(
            f"  step {k['step']:>2}: {k['killer']:<10} -> {k['victim']}"
        )

    return "\n".join(out) + "\n"


def output_path_for(pkl_path: Path) -> Path:
    """Return the sibling .txt path for a given .pkl path."""
    return pkl_path.with_suffix(".txt")


# ============================================================================
# MAIN
# ============================================================================

def dump(spec: str, out: TextIO | None = None) -> Path:
    """
    Load the pickle at ``spec``, write the transcript to a .txt file
    next to it, and return the output path.

    If ``out`` is provided, also tee the transcript to that stream
    (used by the smoke tests). The .txt file is always written.
    """
    log_path = _resolve_log_path(spec)
    payload = load_recording_payload(log_path)
    transcript = build_transcript(payload, log_path)

    txt_path = output_path_for(log_path)
    txt_path.write_text(transcript, encoding="utf-8")

    if out is not None:
        out.write(transcript)

    return txt_path


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: dump_replay_log.py <log-title-or-path>", file=sys.stderr)
        print("Examples:", file=sys.stderr)
        print("  dump_replay_log.py llm_among_us", file=sys.stderr)
        print("  dump_replay_log.py experiments/logs/llm_among_us_20260709_103154.pkl", file=sys.stderr)
        sys.exit(1)

    try:
        written = dump(sys.argv[1])
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    # One-line confirmation to stderr so the user knows where the file went.
    print(f"wrote transcript: {written}", file=sys.stderr)
