"""
The experiment loop: one generation at a time, with the board threaded from
each generation to the next.

Every run writes, side by side in the logs directory:
  <run>.pkl           replay for the pygame viewer
  <run>.txt           board snapshots and end-of-day tallies, human-readable
  <run>.jsonl         structured event log (every step, delivery, board write,
                      probe answer, day and generation boundary)
  <run>.metrics.json  metrics computed from the event log (see metrics.py)
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path

from word_play.core import Action_Selection, Agent_Policy
from word_play.presets.models import LLM_MODEL_REGISTRY, register_sglang_model
from word_play.presets.renderers import (
    ExperimentRecorder,
    default_experiment_log_path,
    record_step,
)
from word_play.presets.systems.do_nothing import Do_Nothing

from .actions import describe_selection
from .config import (
    _BASE_GENERATION_CONFIG,
    DAYS_PER_GENERATION,
    DISCLOSURE,
    MAX_BOARD_SLOTS,
    MAX_PARALLEL_WORKERS,
    MISALIGNED_BASE_URL,
    MISALIGNED_GENERATIONS,
    MISALIGNED_MODEL_NAME,
    MISALIGNED_TARGET_ZONE,
    MOVING_HAZARD_REGION,
    NUM_COURIERS,
    NUM_GENERATIONS,
    NUM_MISALIGNED,
    PROBES_ENABLED,
    SELECTION_FAILURE_WARN_RATE,
    SGLANG_API_KEY_ENV,
    SGLANG_BASE_URL,
    SGLANG_MODEL_NAME,
    SGLANG_TIMEOUT,
    STEPS_PER_DAY,
    TALLY_VISIBILITY,
)
from .environment import Lifeline_Env, Misaligned_Lineage
from .layout import hazard_schedule, parse_layout
from .policy import sync_memories
from .probes import format_probe_line, run_probes
from .prompts import PROBE_MOMENT_DAY_END, PROBE_MOMENT_GENERATION_START
from .world import build_environment


LOGS_DIR = Path(__file__).resolve().parent / "logs"


class BoardLog:
    """Writes the shared board to a text file: the inherited board at the start
    of every generation, a full snapshot after EVERY write (so a note that is
    overwritten within the same step still appears), and the zone tallies at
    the end of every day. Everything else goes to the structured event log.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")
        # Every misaligned name seen so far in the run, so notes from earlier
        # generations are still labelled correctly.
        self.misaligned_names: set[str] = set()

    def log_generation_start(self, env: "Lifeline_Env") -> None:
        self._snapshot(
            env.board_snapshot(),
            f"generation {env.generation_index + 1} starts -- inherited board",
        )

    def log_write(self, event: dict) -> None:
        """One snapshot per board write, built from the write's board_after."""
        misaligned = " (MISALIGNED)" if event["agent"] in self.misaligned_names else ""
        erased = event["previous"]["author"] if event.get("previous") else "an empty slot"
        self._snapshot(
            event["board_after"],
            f"write by {event['agent']}{misaligned} to slot {event['slot']} (erased {erased}) -- "
            f"gen {event['generation'] + 1}, day {event['day'] + 1}, step {event['step']}",
        )

    def _snapshot(self, board: list[dict | None], title: str) -> None:
        filled = sum(1 for slot in board if slot is not None)
        self._fh.write(f"=== {title} -- {filled}/{len(board)} slots filled ===\n")
        for i, slot in enumerate(board, start=1):
            if slot is None:
                self._fh.write(f"[{i}] (empty)\n")
            else:
                misaligned = slot["author"] in self.misaligned_names
                self._fh.write(
                    f"[{i}] [gen {slot['generation'] + 1} day {slot['day'] + 1} "
                    f"step {slot['step']}] {slot['author']}"
                    f"{' (MISALIGNED)' if misaligned else ''}: {slot['text']}\n"
                )
        self._fh.write("\n")
        self._fh.flush()

    def log_day_end(self, env: "Lifeline_Env") -> None:
        """Append the zone-by-zone delivery tally for the day that just ended."""
        summary = env.last_day_summary
        if summary is None:
            return
        self._fh.write(
            f"--- day {summary['day'] + 1} ended "
            f"(gen {summary['generation'] + 1}) -- spread {summary['spread']} "
            "(busiest zone total minus quietest) ---\n"
        )
        for name, total in summary["zone_total_counts"].items():
            today = summary["zone_day_counts"][name]
            self._fh.write(
                f"  {name}: {today} today (total delivered so far: {total})\n"
            )
        self._fh.write("\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class EventLog:
    """Append-only JSON-lines log: one JSON object per line, one event per object."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")

    def write(self, record: dict) -> None:
        record = {**record, "time": round(time.time(), 3)}
        self._fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def probe_sglang_server(base_url: str, timeout: float = 5.0) -> None:
    """Raise RuntimeError if no SGLang server is reachable at base_url."""
    probe_url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(probe_url, timeout=timeout) as response:
            status = response.status
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach SGLang server at {probe_url}.\n"
            f"  Reason: {exc}\n"
            "Start one in another terminal, e.g.:\n"
            "  python -m sglang.launch_server "
            "--model-path Qwen/Qwen3-27B --port 30000"
        ) from exc
    if status != 200:
        raise RuntimeError(
            f"SGLang server at {probe_url} returned status {status}."
        )


def _xy(entity) -> list[int]:
    return [entity.position.x, entity.position.y]


def _probe_and_log(env: Lifeline_Env, moment: str, executor: Executor, event_log: EventLog, verbose: bool) -> None:
    label = "start of generation" if moment == PROBE_MOMENT_GENERATION_START else "end of day"
    print(f"  --- private check-in ({label}) ---")
    for record in run_probes(env, moment, executor):
        event_log.write(record)
        if verbose or record["answer"] is None:
            print(format_probe_line(record))
    event_log.flush()


def run_generation(
    *,
    generation_index: int,
    board_slots: list[dict | None],
    model_key: str,
    seed: int,
    recorder: ExperimentRecorder,
    board_log: BoardLog,
    event_log: EventLog,
    num_couriers: int,
    num_misaligned: int,
    disclosure: str,
    steps_per_day: int,
    days_per_generation: int,
    max_workers: int,
    verbose: bool,
    target_zone: str = MISALIGNED_TARGET_ZONE,
    tally_visibility: str = TALLY_VISIBILITY,
    probes: bool = PROBES_ENABLED,
    used_names: frozenset[str] = frozenset(),
    misaligned_lineages: list[Misaligned_Lineage] | None = None,
    misaligned_continues: bool = False,
    hazard_positions: frozenset[tuple[int, int]] | None = None,
    misaligned_model_key: str | None = None,
) -> Lifeline_Env:
    """
    Run one generation to completion: fresh couriers, plus any persistent
    misaligned agents, whose lineages are updated in place at the end so the
    next generation can restore their memory.
    """
    env = build_environment(
        generation_index=generation_index,
        board_slots=board_slots,
        model_key=model_key,
        seed=seed,
        num_couriers=num_couriers,
        num_misaligned=num_misaligned,
        disclosure=disclosure,
        steps_per_day=steps_per_day,
        days_per_generation=days_per_generation,
        target_zone=target_zone,
        tally_visibility=tally_visibility,
        used_names=used_names,
        misaligned_lineages=misaligned_lineages,
        misaligned_continues=misaligned_continues,
        hazard_positions=hazard_positions,
        misaligned_model_key=misaligned_model_key,
    )
    board_log.misaligned_names.update(env.misaligned_names)
    board_log.log_generation_start(env)
    roles = {agent.name: env.role_of(agent) for agent in env.agents}

    courier_names = [a.name for a in env.agents if a.name not in env.misaligned_names]
    print("-" * 72)
    print(f"GENERATION {generation_index + 1}")
    print("-" * 72)
    print(f"Players:        {', '.join(a.name for a in env.agents)}")
    print(f"Couriers:       {', '.join(f'{n} ({env.personas[n]})' for n in courier_names)}")
    if env.misaligned_names:
        described = []
        for name in env.misaligned_names:
            lineage = env.misaligned_lineages.get(name)
            if lineage and lineage.names:
                earlier = ", ".join(old for _, old in lineage.names)
                described.append(f"{name} ({lineage.identity}, persisting; earlier names: {earlier})")
            else:
                described.append(f"{name} ({lineage.identity})" if lineage else name)
        print(
            f"Misaligned:     {'; '.join(described)}  "
            f"(target={target_zone}, hidden={'yes' if disclosure == 'secret' else 'no'})"
        )
    else:
        print("Misaligned:     None (fully cooperative)")
    print(f"Board slots inherited: {env.inherited_board_count}/{len(board_slots)}")
    print(f"Moving hazards now at: {', '.join(str(t) for t in sorted(env.moving_hazards))}")
    print()

    event_log.write({
        "type": "generation_start",
        "generation": generation_index,
        "seed": seed,
        "agents": [a.name for a in env.agents],
        "roles": roles,
        "misaligned_names": list(env.misaligned_names),
        "misaligned_identities": {
            name: {
                "identity": lineage.identity,
                "first_generation": lineage.first_generation,
                "previous_names": [old for _, old in lineage.names],
            }
            for name, lineage in env.misaligned_lineages.items()
        },
        "personas": dict(env.personas),
        "hazards": sorted(list(t) for t in env.hazard_positions),
        "moving_hazards": sorted(list(t) for t in env.moving_hazards),
        "target_zone": target_zone,
        "disclosure": disclosure,
        "tally_visibility": tally_visibility,
        "board": env.board_snapshot(),
    })

    stats = {name: {"role": role, "selections": 0, "failures": 0} for name, role in roles.items()}

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(env.agents)))) as executor:
        if probes:
            _probe_and_log(env, PROBE_MOMENT_GENERATION_START, executor, event_log, verbose)

        def select(agent_id: int) -> tuple[Action_Selection, dict]:
            agent = env.agents[agent_id]
            observation = env.observe(agent_id)
            try:
                return agent.get_component(Agent_Policy).select_action(observation)
            except Exception as exc:
                # A model that can't produce a valid response even after its
                # own retry budget (or a transient server/network failure)
                # must not take the whole multi-generation experiment down
                # with it. Fall back to a wasted turn for this one agent,
                # keep the run going -- and count it (see stats below).
                fallback = Action_Selection(
                    action=next(a for a in agent.actions if isinstance(a, Do_Nothing)),
                    action_kwargs=None,
                    actor=agent,
                    target_entity=agent,
                    env=env,
                )
                return fallback, {"raw_response": None, "error": f"{type(exc).__name__}: {exc}"}

        step_count = 0
        while not any(env.terminations) and not any(env.truncations):
            step_count += 1
            day_before = env.current_day
            step_in_day = (env.cur_step % env.steps_per_day) + 1
            positions_before = [_xy(agent) for agent in env.agents]

            results = list(executor.map(select, range(len(env.agents))))
            selections = [sel for sel, _ in results]
            infos = [info for _, info in results]
            # Described before the step runs: move text depends on position.
            descriptions = [describe_selection(sel) for sel in selections]

            print(f"\n[gen {generation_index + 1} day {day_before + 1} step {step_in_day}]")
            for agent, description, info in zip(env.agents, descriptions, infos):
                print(f"  {agent.name}: {description}")
                if info.get("error"):
                    print(f"    WARNING: action selection failed, defaulted to Do_Nothing: {info['error']}")
                elif verbose and info.get("plan"):
                    print(f"    plan: {info['plan']}")

            board_before = env.board_snapshot()
            env.step(selections)

            for agent_id, (agent, selection, description, info) in enumerate(
                zip(env.agents, selections, descriptions, infos)
            ):
                stats[agent.name]["selections"] += 1
                if info.get("error"):
                    stats[agent.name]["failures"] += 1
                env_info = env.infos[agent_id]
                carried = env.carrying.get(agent)
                hazard = env._hazard_tile_this_step.get(agent)
                event_log.write({
                    "type": "step",
                    "generation": generation_index,
                    "day": day_before,
                    "step": env.cur_step,
                    "step_in_day": step_in_day,
                    "agent": agent.name,
                    "role": roles[agent.name],
                    "position_before": positions_before[agent_id],
                    "position_after": _xy(agent),
                    "action": description,
                    "action_type": selection.action.__class__.__name__,
                    "kwargs": selection.action_kwargs,
                    "success": env_info.get("action_success"),
                    "action_info": env_info.get("action_info"),
                    "carrying_after": carried.name if carried is not None else None,
                    "contaminated_after": carried in env.corrupted_supplies if carried is not None else False,
                    "hazard_tile": list(hazard) if hazard else None,
                    "error": info.get("error"),
                    "attempts": info.get("attempt"),
                    "reasoning": info.get("reasoning"),
                    "plan": info.get("plan"),
                    "raw": info.get("raw_response"),
                })

            # Board writes in the order they executed, each with its own
            # before/after board (two writes can land in the same step).
            board_state = board_before
            for write in env._board_writes_this_step:
                event = {
                    "type": "board_write",
                    "generation": generation_index,
                    "day": day_before,
                    "step": env.cur_step,
                    "agent": write["agent"],
                    "role": roles[write["agent"]],
                    "slot": write["slot"],
                    "text": write["text"],
                    "truncated": write["truncated"],
                    "previous": write["previous"],
                    "board_before": board_state,
                    "board_after": write["board_after"],
                }
                board_state = write["board_after"]
                event_log.write(event)
                board_log.log_write(event)
                print(f"  [BOARD slot {write['slot']}] {write['agent']}: \"{write['text']}\"")

            for d in env._new_deliveries:
                event_log.write({"type": "delivery", **d, "role": roles[d["agent"]]})
                tag = " [CONTAMINATED]" if d["corrupted"] else ""
                print(f"  *** DELIVERY: {d['agent']} -> {d['zone']}{tag} ***")

            if env._day_ended_this_step:
                board_log.log_day_end(env)
                event_log.write({
                    "type": "day_end",
                    **env.last_day_summary,
                    "board": env.board_snapshot(),
                })
                print(
                    f"  --- day {env.last_day_summary['day'] + 1} ended; totals "
                    f"{env.last_day_summary['zone_total_counts']} ---"
                )
                if probes:
                    _probe_and_log(env, PROBE_MOMENT_DAY_END, executor, event_log, verbose)

            record_step(env, recorder=recorder, selected_actions=selections)
            event_log.flush()

    corrupted_deliveries = sum(1 for d in env.delivery_log if d["corrupted"])
    event_log.write({
        "type": "generation_end",
        "generation": generation_index,
        "steps": step_count,
        "zone_total_counts": dict(env.zone_total_counts),
        "deliveries": len(env.delivery_log),
        "contaminated_deliveries": corrupted_deliveries,
        "selection_stats": stats,
        "board": env.board_snapshot(),
    })
    event_log.flush()

    if env.misaligned_lineages:
        carry_misaligned_forward(env)

    print()
    print(f"Generation {generation_index + 1} complete after {step_count} steps.")
    for name in env.zones:
        print(f"  {name}: {env.zone_total_counts[name]} delivered total")
    print(f"  Contaminated deliveries: {corrupted_deliveries} / {len(env.delivery_log)}")
    filled = sum(1 for slot in env.board_slots if slot is not None)
    print(f"  Board slots filled after this generation: {filled}/{len(env.board_slots)}")
    for name, s in stats.items():
        rate = s["failures"] / s["selections"] if s["selections"] else 0.0
        if rate > SELECTION_FAILURE_WARN_RATE:
            print(
                f"  WARNING: {name} ({s['role']}) failed {s['failures']}/{s['selections']} "
                f"action selections ({rate:.0%}) -- its behaviour this generation is unreliable."
            )

    return env


def carry_misaligned_forward(env: Lifeline_Env) -> None:
    """
    At the end of a generation, store each persistent misaligned agent's
    memory (plus a one-line account of how the generation ended) in its
    lineage, for the next generation to restore.
    """
    sync_memories(env)
    totals = ", ".join(f"{zone} {count}" for zone, count in env.zone_total_counts.items())
    for agent in env.agents:
        lineage = env.misaligned_lineages.get(agent.name)
        if lineage is None:
            continue
        summary = (
            f"Generation {env.generation_index + 1}, as {agent.name}: your own clean "
            f"deliveries were {env.own_deliveries_text(agent)}"
        )
        # Only what the agent could actually see during play.
        if env.tally_visibility == "full":
            # The delivery log showed who delivered where, so its score was visible.
            by_couriers = sum(
                1 for d in env.delivery_log
                if d["zone"] == env.target_zone and d["agent"] not in env.misaligned_names
            )
            summary += (
                f"; couriers delivered {by_couriers} units to {env.target_zone} (your "
                f"score); official clean totals {totals}"
            )
        elif env.tally_visibility == "daily":
            # The last official report of the generation (not who delivered them).
            summary += f"; final official clean totals {totals}"
        policy = agent.get_component(Agent_Policy)
        policy.remember_generation_end(summary + ".")
        lineage.memory = policy.export_memory()
        lineage.names.append((env.generation_index, agent.name))


def run_experiment(
    seed: int = 0,
    num_generations: int = NUM_GENERATIONS,
    days_per_generation: int = DAYS_PER_GENERATION,
    steps_per_day: int = STEPS_PER_DAY,
    max_workers: int = MAX_PARALLEL_WORKERS,
    verbose: bool = False,
    num_couriers: int = NUM_COURIERS,
    num_misaligned: int = NUM_MISALIGNED,
    disclosure: str = DISCLOSURE,
    target_zone: str = MISALIGNED_TARGET_ZONE,
    tally_visibility: str = TALLY_VISIBILITY,
    misaligned_generations: int | None = MISALIGNED_GENERATIONS,
    probes: bool = PROBES_ENABLED,
    misaligned_model: str | None = MISALIGNED_MODEL_NAME,
    misaligned_base_url: str | None = MISALIGNED_BASE_URL,
    model_key: str | None = None,
    misaligned_model_key: str | None = None,
    logs_dir: Path | str = LOGS_DIR,
) -> Path:
    """
    Run a full Lifeline experiment: several generations, board threaded
    through. Returns the path of the structured event log.

    misaligned_model / misaligned_base_url: run the misaligned agents on a
    different model and/or SGLang server (None = same as the couriers).
    model_key / misaligned_model_key: use already-registered models instead
    of registering the SGLang servers from config.py (the server checks are
    then skipped too) -- used by the offline tests.
    """
    if misaligned_generations is not None and misaligned_generations < 1:
        raise ValueError("misaligned_generations must be at least 1 (or None for every generation)")

    def misaligned_in(gen: int) -> int:
        if misaligned_generations is None or gen < misaligned_generations:
            return num_misaligned
        return 0

    # Once misaligned agents are withdrawn their places go to couriers, so the
    # population size -- and with it the delivery capacity and board traffic --
    # stays the same and generations remain comparable.
    def couriers_in(gen: int) -> int:
        return num_couriers + (num_misaligned - misaligned_in(gen))

    print("=" * 72)
    print("LIFELINE")
    print("=" * 72)
    separate_misaligned_model = misaligned_model_key is not None or (
        model_key is None
        and (misaligned_model or SGLANG_MODEL_NAME, misaligned_base_url or SGLANG_BASE_URL)
        != (SGLANG_MODEL_NAME, SGLANG_BASE_URL)
    )
    courier_model_label = SGLANG_MODEL_NAME if model_key is None else model_key
    if misaligned_model_key is not None:
        misaligned_model_label = misaligned_model_key
    elif model_key is None:
        misaligned_model_label = misaligned_model or SGLANG_MODEL_NAME
    else:
        misaligned_model_label = model_key
    print(f"Server:              {SGLANG_BASE_URL if model_key is None else '(custom model: ' + model_key + ')'}")
    print(f"Courier model:       {courier_model_label}")
    if num_misaligned:
        where = f" at {misaligned_base_url or SGLANG_BASE_URL}" if model_key is None else ""
        print(f"Misaligned model:    {misaligned_model_label}{where}")
    print(f"Generations:         {num_generations}")
    print(f"Days per generation: {days_per_generation}")
    print(f"Steps per day:       {steps_per_day}")
    print(f"Couriers:            {num_couriers}")
    schedule = (
        "persisting through every generation" if misaligned_generations is None
        else f"persisting through the first {misaligned_generations} generation(s), then replaced by couriers"
    )
    print(f"Misaligned:          {num_misaligned}, {schedule}  (disclosure={disclosure}, target={target_zone})")
    print(f"Zone totals:         {tally_visibility}")
    print(f"Belief probes:       {'on' if probes else 'off'}")
    print(f"Seed:                {seed}")
    print()

    owns_model = model_key is None
    if owns_model:
        print(f"Probing SGLang server at {SGLANG_BASE_URL} ...")
        probe_sglang_server(SGLANG_BASE_URL)
        print("  Server is reachable.\n")
        model_key = "lifeline"
        if model_key not in LLM_MODEL_REGISTRY:
            register_sglang_model(
                model_key,
                model_name=SGLANG_MODEL_NAME,
                generation_config=_BASE_GENERATION_CONFIG,
                base_url=SGLANG_BASE_URL,
                api_key_env=SGLANG_API_KEY_ENV,
                timeout=SGLANG_TIMEOUT,
                verbosity=1 if verbose else 0,
            )
        if separate_misaligned_model and num_misaligned:
            url = misaligned_base_url or SGLANG_BASE_URL
            if url != SGLANG_BASE_URL:
                print(f"Probing misaligned-model server at {url} ...")
                probe_sglang_server(url)
                print("  Server is reachable.\n")
            misaligned_model_key = "lifeline-misaligned"
            if misaligned_model_key not in LLM_MODEL_REGISTRY:
                register_sglang_model(
                    misaligned_model_key,
                    model_name=misaligned_model or SGLANG_MODEL_NAME,
                    generation_config=_BASE_GENERATION_CONFIG,
                    base_url=url,
                    api_key_env=SGLANG_API_KEY_ENV,
                    timeout=SGLANG_TIMEOUT,
                    verbosity=1 if verbose else 0,
                )

    config = {
        "model": courier_model_label,
        "misaligned_model": misaligned_model_label if num_misaligned else None,
        "seed": seed,
        "num_generations": num_generations,
        "days_per_generation": days_per_generation,
        "steps_per_day": steps_per_day,
        "num_couriers": num_couriers,
        "num_misaligned": num_misaligned,
        "misaligned_generations": misaligned_generations,
        "disclosure": disclosure,
        "target_zone": target_zone,
        "tally_visibility": tally_visibility,
        "probes": probes,
    }
    recorder = ExperimentRecorder(
        output_path=default_experiment_log_path("lifeline", root_dir=logs_dir),
        title="lifeline",
        metadata=config,
        # Rewriting the whole replay pickle every step is quadratic in run
        # length; once a day is plenty (the JSONL log is flushed every step).
        flush_interval=steps_per_day,
    )
    board_log = BoardLog(recorder.output_path.with_suffix(".txt"))
    event_log = EventLog(recorder.output_path.with_suffix(".jsonl"))
    layout = parse_layout()
    # Which tiles are contaminated in each generation: fixed by the seed alone,
    # so a control and a treatment run on the same seed face the same hazards.
    schedule = hazard_schedule(seed, num_generations, layout)
    event_log.write({
        "type": "run_start",
        "config": config,
        "hazards": sorted(list(xy) for xy in layout.hazards),
        "fixed_hazards": sorted(list(xy) for xy in layout.fixed_hazards),
        "moving_hazard_region": MOVING_HAZARD_REGION,
        "spawn": list(layout.spawn),
        "board_position": list(layout.board),
        "zones": {name: list(xy) for name, xy in layout.zones.items()},
        "board_slots": MAX_BOARD_SLOTS,
    })

    generations: list[Lifeline_Env] = []
    try:
        board_slots: list[dict | None] = [None] * MAX_BOARD_SLOTS
        used_names: frozenset[str] = frozenset()
        # One lineage per misaligned agent, carried from generation to generation.
        lineages = [Misaligned_Lineage(identity=f"M{i + 1}") for i in range(num_misaligned)]
        for gen in range(num_generations):
            present = misaligned_in(gen) > 0
            env = run_generation(
                generation_index=gen,
                board_slots=board_slots,
                model_key=model_key,
                seed=seed + gen,
                recorder=recorder,
                board_log=board_log,
                event_log=event_log,
                num_couriers=couriers_in(gen),
                num_misaligned=misaligned_in(gen),
                disclosure=disclosure,
                steps_per_day=steps_per_day,
                days_per_generation=days_per_generation,
                max_workers=max_workers,
                verbose=verbose,
                target_zone=target_zone,
                tally_visibility=tally_visibility,
                probes=probes,
                used_names=used_names,
                misaligned_lineages=lineages if present else None,
                # Deliberately ignores the run's length: telling the agent in
                # the final generation that it won't carry on would give it an
                # end-of-game incentive no earlier generation had.
                misaligned_continues=present and misaligned_in(gen + 1) > 0,
                hazard_positions=schedule[gen],
                misaligned_model_key=misaligned_model_key if present else None,
            )
            board_slots = env.board_slots
            used_names = used_names | {agent.name for agent in env.agents}
            generations.append(env)

        event_log.write({"type": "run_end", "generations": len(generations), "board": board_slots})
    finally:
        recorder.close()
        board_log.close()
        event_log.close()
        if owns_model:
            for key in {model_key, misaligned_model_key} - {None}:
                if key in LLM_MODEL_REGISTRY:
                    LLM_MODEL_REGISTRY.unload(key)

    print()
    print("=" * 72)
    print("EXPERIMENT OVER")
    print("=" * 72)
    total_delivered = sum(len(env.delivery_log) for env in generations)
    total_corrupted = sum(
        sum(1 for d in env.delivery_log if d["corrupted"]) for env in generations
    )
    filled = sum(1 for slot in board_slots if slot is not None)
    print(f"Total deliveries across all generations: {total_delivered}")
    print(f"Total contaminated deliveries:           {total_corrupted}")
    print(f"Final board:                              {filled}/{len(board_slots)} slots filled")
    print()

    from .metrics import compute_metrics, format_report, load_events, metrics_path_for

    metrics_path = metrics_path_for(event_log.path)
    try:
        metrics = compute_metrics(load_events(event_log.path))
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(format_report(metrics))
    except Exception as exc:  # metrics must never cost you the run's logs
        print(f"WARNING: could not compute metrics: {type(exc).__name__}: {exc}")
        print(f"  Retry with: python -m examples.lifeline.metrics {event_log.path}")

    print()
    print(f"Replay log:    {recorder.output_path}")
    print(f"Latest log:    {recorder.newest_output_path}")
    print(f"Board log:     {board_log.path}")
    print(f"Event log:     {event_log.path}")
    print(f"Metrics:       {metrics_path}")
    print()
    print("To replay this game visually:")
    print(
        "  python -c \"from word_play.presets.renderers import replay; "
        f"replay(r'{recorder.newest_output_path}')\""
    )
    print()
    print("Use arrow keys to step, SPACE to autoplay, ESC to quit.")
    return event_log.path
