"""
Belief probes: a private questionnaire put to every agent at the start of each
generation and at the end of every day.

Probes measure what agents BELIEVE (where the hazards are, which zone they
would serve next, whom they distrust), separately from what they DO. They are
side-effect free by construction: the answer is logged and never shown back to
the agent, so asking does not change the run.

The start-of-generation probe is the cleanest measure of transmission: at that
point a new courier knows nothing except its system prompt and the inherited
board. (A persistent misaligned agent also has its own memory.)
"""

from __future__ import annotations

from concurrent.futures import Executor

from word_play.core import Agent_Policy

from .config import MAX_BOARD_SLOTS, PROBE_SAMPLES, ZONE_NAMES
from .environment import Lifeline_Env
from .policy import Lifeline_Policy, parse_json_object, sync_memories
from .prompts import PROBE_MOMENT_DAY_END, build_probe_prompt

PROBE_ATTEMPTS = 2
VALID_SOURCES = ("self", "board", "both")


def normalize_probe_answer(
    parsed: dict, zone_names: tuple[str, ...] = ZONE_NAMES, slot_count: int = MAX_BOARD_SLOTS
) -> tuple[dict, list[str]]:
    """
    Coerce a parsed probe reply into a fixed shape. Anything malformed is
    dropped and reported as a warning rather than raising: a partly usable
    answer is still data.
    """
    warnings: list[str] = []

    tiles: list[dict] = []
    for item in parsed.get("contaminated_tiles") or []:
        tile, source = None, None
        if isinstance(item, dict):
            tile, source = item.get("tile"), item.get("source")
        elif isinstance(item, (list, tuple)):
            tile = item
        try:
            x, y = (int(v) for v in tile)
        except (TypeError, ValueError):
            warnings.append(f"bad tile entry: {item!r}")
            continue
        source = str(source).strip().lower() if source is not None else None
        if source not in VALID_SOURCES:
            if source is not None:
                warnings.append(f"bad source {source!r} for tile ({x}, {y})")
            source = None
        if not any(t["tile"] == [x, y] for t in tiles):
            tiles.append({"tile": [x, y], "source": source})

    fixed: list[list[int]] = []
    for item in parsed.get("fixed_tiles") or []:
        tile = item.get("tile") if isinstance(item, dict) else item
        try:
            x, y = (int(v) for v in tile)
        except (TypeError, ValueError):
            warnings.append(f"bad fixed tile entry: {item!r}")
            continue
        if [x, y] not in fixed:
            fixed.append([x, y])

    def zone_answer(key: str) -> str | None:
        zone = parsed.get(key)
        zone = str(zone).strip() if zone is not None else None
        if zone not in zone_names:
            warnings.append(f"{key} not a zone: {zone!r}")
            return None
        return zone

    zone = zone_answer("next_delivery_zone")
    at_risk = zone_answer("zone_most_at_risk")

    slots: list[int] = []
    for item in parsed.get("unreliable_board_slots") or []:
        try:
            slot = int(item)
        except (TypeError, ValueError):
            warnings.append(f"bad slot entry: {item!r}")
            continue
        if 1 <= slot <= slot_count and slot not in slots:
            slots.append(slot)
        elif not 1 <= slot <= slot_count:
            warnings.append(f"slot out of range: {slot}")

    # Agents are asked for "suspected_colleagues" ("players" would be a game
    # cue); older logs and the tests' stand-in model use "suspected_players".
    suspects = parsed.get("suspected_colleagues") or parsed.get("suspected_players") or []
    if isinstance(suspects, str):
        suspects = [suspects]
    suspects = [
        str(s).strip() for s in suspects
        if str(s).strip() and str(s).strip().lower() not in ("none", "nobody", "no one")
    ]

    answer = {
        "contaminated_tiles": tiles,
        "fixed_tiles": fixed,
        "next_delivery_zone": zone,
        "next_delivery_reason": str(parsed.get("next_delivery_reason") or "").strip(),
        "zone_most_at_risk": at_risk,
        "top_priority": str(parsed.get("top_priority") or "").strip(),
        "unreliable_board_slots": slots,
        "suspected_players": suspects,
        "suspicion_reason": str(parsed.get("suspicion_reason") or "").strip(),
    }
    return answer, warnings


def probe_agent(env: Lifeline_Env, agent_id: int, moment: str, sample: int = 0) -> dict:
    """Ask one agent the questionnaire once (one of PROBE_SAMPLES independent
    samples). Returns a JSON-safe event record."""
    agent = env.agents[agent_id]
    policy = agent.get_component(Agent_Policy)
    ended_day = env.last_day_summary["day"] if moment == PROBE_MOMENT_DAY_END and env.last_day_summary else None
    questions = build_probe_prompt(
        moment=moment, generation_index=env.generation_index, day_index=ended_day,
        zone_names=env.zones_for(agent),
    )
    record = {
        "type": "probe",
        "moment": moment,
        "generation": env.generation_index,
        "day": ended_day,
        "agent": agent.name,
        "role": env.role_of(agent),
        "sample": sample,
        "raw": None,
        "answer": None,
        "error": None,
        "warnings": [],
        "attempts": 0,
        "board": env.board_snapshot(),
    }
    if not isinstance(policy, Lifeline_Policy):
        record["error"] = "agent policy does not support probes"
        return record

    context = env.probe_view(agent_id, moment)
    for attempt in range(PROBE_ATTEMPTS):
        record["attempts"] = attempt + 1
        try:
            raw = policy.answer_probe(context, questions)
        except Exception as exc:  # server/network failure: record and move on
            record["error"] = f"{type(exc).__name__}: {exc}"
            continue
        record["raw"] = raw
        try:
            parsed = parse_json_object(raw)
        except Exception as exc:
            record["error"] = f"unparseable reply: {exc}"
            continue
        record["answer"], record["warnings"] = normalize_probe_answer(
            parsed, tuple(env.zones), len(env.board_slots)
        )
        record["error"] = None
        break
    return record


def run_probes(env: Lifeline_Env, moment: str, executor: Executor, samples: int = PROBE_SAMPLES) -> list[dict]:
    """Probe every agent `samples` times, all in parallel (the shared prompt
    prefix makes repeats cheap). Memory is brought up to date first (outcome
    of the last action, any hazard alert) but is not modified by the probe."""
    sync_memories(env)
    jobs = [(aid, k) for aid in range(len(env.agents)) for k in range(samples)]
    return list(executor.map(lambda job: probe_agent(env, job[0], moment, job[1]), jobs))


def format_probe_line(record: dict) -> str:
    if record["answer"] is None:
        return f"  [probe] {record['agent']}: FAILED ({record['error']})"
    a = record["answer"]
    tiles = ", ".join(f"({t['tile'][0]},{t['tile'][1]})" for t in a["contaminated_tiles"]) or "-"
    suspects = ", ".join(a["suspected_players"]) or "-"
    return (
        f"  [probe] {record['agent']} ({record['role']}): next={a['next_delivery_zone']} "
        f"tiles=[{tiles}] suspects=[{suspects}] unreliable_slots={a['unreliable_board_slots']}"
    )
