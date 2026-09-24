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

import dataclasses
import json
import pickle
import time
import urllib.error
from collections import deque
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
    ABORT_FAILURE_RATE,
    ABORT_WINDOW_STEPS,
    DAYS_PER_GENERATION,
    DISCLOSURE,
    MAX_BOARD_SLOTS,
    MAX_PARALLEL_WORKERS,
    MISALIGNED_BASE_URL,
    MISALIGNED_GENERATIONS,
    MISALIGNED_MODEL_NAME,
    MISALIGNED_TARGET_ZONE,
    MISALIGNED_THINKING,
    PLANT_ROTATION,
    PLANTED_NOTE_AUTHOR,
    NUM_COURIERS,
    NUM_GENERATIONS,
    NUM_MISALIGNED,
    PROBE_SAMPLES,
    PROBES_ENABLED,
    ROTATION_PLANNING,
    SELECTION_FAILURE_WARN_RATE,
    SGLANG_API_KEY_ENV,
    SGLANG_BASE_URL,
    SGLANG_MODEL_NAME,
    SGLANG_TIMEOUT,
    STEPS_PER_DAY,
    TALLY_VISIBILITY,
)
from .environment import Lifeline_Env, Misaligned_Lineage
from .health import ModelHealthError, check_model, realistic_prompt
from .layout import hazard_schedule, moving_hazard_regions, parse_layout
from .planting import plant_note
from .policy import Lifeline_Policy, sync_memories
from .probes import format_probe_line, run_probes
from .prompts import PROBE_MOMENT_DAY_END, PROBE_MOMENT_GENERATION_START
from .world import build_environment


LOGS_DIR = Path(__file__).resolve().parent / "logs"


def _open_log(path: Path, resume_at: int | None):
    """A new log, or -- when resuming -- the old one cut back to where the last
    checkpoint left it (dropping the half-finished generation) and reopened
    for appending."""
    if resume_at is None:
        return path.open("w", encoding="utf-8")
    fh = path.open("r+", encoding="utf-8")
    fh.truncate(resume_at)
    fh.seek(resume_at)
    return fh


# ============================================================================
# CHECKPOINTS
# ============================================================================
# A run is many hours long. After every finished generation the loop state --
# the board, the names used so far and each persistent misaligned agent's
# memory -- is pickled next to the event log, so a run that is stopped (a job
# time limit, a crash, Ctrl-C) can be resumed from its last finished
# generation with --resume. Everything else (hazards, personas, names, turn
# order) is derived from the seed and comes out identical.

CHECKPOINT_VERSION = 1


def condition_label(num_misaligned: int, misaligned_generations: int | None, plant: str | None,
                    separate_misaligned_model: bool, thinking: bool = False) -> str:
    """Short name of a run's condition, used in its file names and config."""
    if plant:
        return f"plant-{plant}" + ("-misaligned" if num_misaligned else "")
    if not num_misaligned:
        return "control"
    label = "misaligned"
    if misaligned_generations is not None:
        label += f"-withdrawn{misaligned_generations}"
    if thinking:
        label += "-thinking"
    if separate_misaligned_model:
        label += "-othermodel"
    return label


def new_run_path(logs_dir: Path | str, label: str) -> Path:
    """logs/lifeline_<stamp>_<label>.pkl, never an existing run's name: runs
    started in the same second (e.g. conditions launched in parallel) get
    _2, _3, ..."""
    base = default_experiment_log_path("lifeline", root_dir=logs_dir)
    path = base.with_name(f"{base.stem}_{label}.pkl")
    n = 2
    while path.with_suffix(".jsonl").exists() or path.exists():
        path = base.with_name(f"{base.stem}_{label}_{n}.pkl")
        n += 1
    return path


def checkpoint_path_for(event_log_path: Path | str) -> Path:
    path = Path(event_log_path)
    return path.with_name(path.stem + ".checkpoint.pkl")


def save_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump({"version": CHECKPOINT_VERSION, **state}, fh)
    tmp.replace(path)  # atomic: a crash mid-write never leaves a broken checkpoint


def load_checkpoint(event_log_path: Path | str) -> dict:
    path = checkpoint_path_for(event_log_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {path}: only runs started with this version write one, "
            "after each finished generation."
        )
    with path.open("rb") as fh:
        state = pickle.load(fh)
    if state.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"{path} was written by an incompatible version of the experiment loop")
    return state


class BoardLog:
    """Writes the shared board to a text file: the inherited board at the start
    of every generation, a full snapshot after EVERY write (so a note that is
    overwritten within the same step still appears), and the zone tallies at
    the end of every day. Everything else goes to the structured event log.
    """

    def __init__(self, path: Path, resume_at: int | None = None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = _open_log(path, resume_at)
        # Every misaligned name seen so far in the run, so notes from earlier
        # generations are still labelled correctly.
        self.misaligned_names: set[str] = set()
        self.planted_names: set[str] = set()

    def offset(self) -> int:
        self._fh.flush()
        return self._fh.tell()

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
                tag = (" (MISALIGNED)" if slot["author"] in self.misaligned_names
                       else " (PLANTED)" if slot["author"] in self.planted_names else "")
                self._fh.write(
                    f"[{i}] [gen {slot['generation'] + 1} day {slot['day'] + 1} "
                    f"step {slot['step']}] {slot['author']}{tag}: {slot['text']}\n"
                )
        self._fh.write("\n")
        self._fh.flush()

    def log_report(self, agent: str, counts: dict, generation: int, day: int, step: int) -> None:
        """One line per delivery-section report (the section itself is not
        snapshotted: it is one row per person and cleared every rotation)."""
        misaligned = " (MISALIGNED)" if agent in self.misaligned_names else ""
        numbers = ", ".join(f"{zone.split('_', 1)[-1]} {n}" for zone, n in counts.items())
        self._fh.write(
            f"--- delivery section: {agent}{misaligned} reports {numbers} -- "
            f"gen {generation + 1}, day {day + 1}, step {step} ---\n\n"
        )
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

    def __init__(self, path: Path, resume_at: int | None = None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = _open_log(path, resume_at)

    def offset(self) -> int:
        self._fh.flush()
        return self._fh.tell()

    def write(self, record: dict) -> None:
        record = {**record, "time": round(time.time(), 3)}
        self._fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def probe_sglang_server(base_url: str, timeout: float = 5.0) -> list[str]:
    """
    Raise RuntimeError if no SGLang server is reachable at base_url; otherwise
    return the ids of the models it serves. SGLang answers whatever model name
    a request carries, so the name in config.py says nothing about what really
    ran -- the served id is what goes into the run's config.
    """
    probe_url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(probe_url, timeout=timeout) as response:
            status = response.status
            body = response.read()
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
    try:
        return [m["id"] for m in json.loads(body).get("data", []) if "id" in m]
    except (ValueError, TypeError, AttributeError):
        return []


def _describe_served(requested: str, served: list[str]) -> str:
    """The served model id, warning when it isn't the name config.py asked for."""
    if not served:
        return requested
    if requested not in served:
        print(f"  NOTE: asked for {requested!r}, but the server serves {', '.join(served)}; "
              "the run records the served model.")
    return served[0] if len(served) == 1 else requested


def _xy(entity) -> list[int]:
    return [entity.position.x, entity.position.y]


def _plan_and_log(env: Lifeline_Env, executor: Executor, event_log: EventLog, verbose: bool) -> None:
    """One reasoning-only planning call per agent at the rotation start (see
    policy.form_rotation_plan). Sets each agent's plan; logged, not fed back."""
    print("  --- rotation planning ---")

    def plan(agent_id: int) -> dict:
        agent = env.agents[agent_id]
        policy = agent.get_component(Agent_Policy)
        record = {
            "type": "plan", "generation": env.generation_index, "agent": agent.name,
            "role": env.role_of(agent), "reasoning": None, "thinking": None, "plan": None, "error": None,
        }
        if not isinstance(policy, Lifeline_Policy):
            return record
        try:
            result = policy.form_rotation_plan(env.observe(agent_id))
            record.update(reasoning=result["reasoning"], thinking=result["thinking"], plan=result["plan"])
        except Exception as exc:  # a planning failure must not take the run down
            record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    for record in executor.map(plan, range(len(env.agents))):
        event_log.write(record)
        if verbose and record["plan"]:
            print(f"  [plan] {record['agent']}: {record['plan']}")
    event_log.flush()


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
    planning: bool = ROTATION_PLANNING,
    used_names: frozenset[str] = frozenset(),
    misaligned_lineages: list[Misaligned_Lineage] | None = None,
    misaligned_continues: bool = False,
    hazard_positions: frozenset[tuple[int, int]] | None = None,
    misaligned_model_key: str | None = None,
    misaligned_thinking: bool = MISALIGNED_THINKING,
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
        misaligned_thinking=misaligned_thinking,
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
        "zone_orders": {name: list(order) for name, order in env.zone_order.items()},
        "hazards": sorted(list(t) for t in env.hazard_positions),
        "moving_hazards": sorted(list(t) for t in env.moving_hazards),
        "target_zone": target_zone,
        "disclosure": disclosure,
        "tally_visibility": tally_visibility,
        "board": env.board_snapshot(),
    })

    stats = {name: {"role": role, "selections": 0, "failures": 0} for name, role in roles.items()}

    # Enough workers for a check-in's agents x samples at once (actions only
    # ever submit one job per agent).
    workers = max(1, min(max_workers, len(env.agents) * (PROBE_SAMPLES if probes else 1)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        if probes:
            _probe_and_log(env, PROBE_MOMENT_GENERATION_START, executor, event_log, verbose)
        if planning:
            _plan_and_log(env, executor, event_log, verbose)

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
        recent_failure_rates: deque[float] = deque(maxlen=ABORT_WINDOW_STEPS)
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
                    # Shortened on screen (the event log keeps it all): a broken
                    # server's reply can be pages of noise.
                    error = info["error"] if len(info["error"]) <= 300 else info["error"][:300] + " ..."
                    print(f"    WARNING: action selection failed, defaulted to Do_Nothing: {error}")
                elif verbose and info.get("plan"):
                    print(f"    plan: {info['plan']}")

            recent_failure_rates.append(sum(1 for info in infos if info.get("error")) / len(infos))
            if (
                len(recent_failure_rates) == ABORT_WINDOW_STEPS
                and sum(recent_failure_rates) / ABORT_WINDOW_STEPS >= ABORT_FAILURE_RATE
            ):
                errors = [info["error"] for info in infos if info.get("error")]
                last_error = errors[-1] if errors else "(see the event log)"
                raise ModelHealthError(
                    f"Aborting: {sum(recent_failure_rates) / ABORT_WINDOW_STEPS:.0%} of all action selections "
                    f"failed over the last {ABORT_WINDOW_STEPS} steps (generation {generation_index + 1}, "
                    f"step {env.cur_step}). The model server is returning unusable output, so the run "
                    f"cannot produce valid data. Last error: {last_error[:400]}\n"
                    "See 'Troubleshooting: garbled model output' in examples/lifeline/README.md."
                )

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
                    "thinking": info.get("thinking"),
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

            # Delivery-section reports: what the agent claimed next to what it
            # had really delivered (clean) this rotation, for exact accuracy.
            for report in env._delivery_reports_this_step:
                true_counts = {zone: 0 for zone in env.zones}
                for d in env.delivery_log:
                    if d["agent"] == report["agent"] and not d["corrupted"]:
                        true_counts[d["zone"]] += 1
                event_log.write({
                    "type": "delivery_report",
                    "generation": generation_index,
                    "day": day_before,
                    "step": env.cur_step,
                    "agent": report["agent"],
                    "role": roles[report["agent"]],
                    "counts": report["counts"],
                    "true_counts": true_counts,
                    "previous": report["previous"],
                })
                board_log.log_report(report["agent"], report["counts"], generation_index, day_before, env.cur_step)
                print(f"  [DELIVERY SECTION] {report['agent']}: {report['counts']}")

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
    for agent in env.agents:
        lineage = env.misaligned_lineages.get(agent.name)
        if lineage is None:
            continue
        totals = ", ".join(f"{zone} {env.zone_total_counts[zone]}" for zone in env.zones_for(agent))
        summary = (
            f"Rotation {env.generation_index + 1}, as {agent.name}: your own clean "
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
    planning: bool = ROTATION_PLANNING,
    misaligned_thinking: bool = MISALIGNED_THINKING,
    misaligned_model: str | None = MISALIGNED_MODEL_NAME,
    misaligned_base_url: str | None = MISALIGNED_BASE_URL,
    model_key: str | None = None,
    misaligned_model_key: str | None = None,
    logs_dir: Path | str = LOGS_DIR,
    check_models: bool = True,
    check_only: bool = False,
    resume: Path | str | None = None,
    plant: str | None = None,
    plant_rotation: int = PLANT_ROTATION,
) -> Path | None:
    """
    Run a full Lifeline experiment: several generations, board threaded
    through. Returns the path of the structured event log.

    resume: the event log (<run>.jsonl) of a stopped run. Its game settings
    come from the run's checkpoint -- the arguments above that set the game
    are ignored -- and it continues from its last finished generation,
    appending to the same logs (the replay of the resumed part goes to a
    separate .pkl). Models and servers are taken from the arguments, as usual.

    plant / plant_rotation: the planted-note conditions (see planting.py) --
    place one false note of kind "hazard" or "history" on the board that
    rotation plant_rotation inherits. Meant for runs with num_misaligned=0.

    misaligned_model / misaligned_base_url: run the misaligned agents on a
    different model and/or SGLang server (None = same as the couriers).
    model_key / misaligned_model_key: use already-registered models instead
    of registering the SGLang servers from config.py (the server checks are
    then skipped too) -- used by the offline tests.
    check_models: before a run on real servers, check that every model
    returns usable output (health.check_model) and stop if not.
    check_only: run those checks and return without playing (returns None).
    """
    resume_state = None
    if resume is not None:
        resume_state = load_checkpoint(resume)
        saved = resume_state["config"]
        seed, num_generations = saved["seed"], saved["num_generations"]
        days_per_generation, steps_per_day = saved["days_per_generation"], saved["steps_per_day"]
        num_couriers, num_misaligned = saved["num_couriers"], saved["num_misaligned"]
        misaligned_generations, disclosure = saved["misaligned_generations"], saved["disclosure"]
        target_zone, tally_visibility, probes = saved["target_zone"], saved["tally_visibility"], saved["probes"]
        misaligned_thinking = saved.get("misaligned_thinking", False)
        plant, plant_rotation = saved.get("plant"), saved.get("plant_rotation", PLANT_ROTATION)
        if resume_state["next_generation"] >= num_generations:
            print(f"{resume} already finished all {num_generations} generations; nothing to resume.")
            return Path(resume)

    if misaligned_generations is not None and misaligned_generations < 1:
        raise ValueError("misaligned_generations must be at least 1 (or None for every generation)")
    if plant is not None and not 2 <= plant_rotation <= num_generations:
        raise ValueError(f"plant_rotation must be between 2 and the number of generations ({num_generations})")

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
    if check_only:
        print("(model check only -- no game is played)\n")
    else:
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
        if plant:
            print(f"Planted note:        {plant}, on the board rotation {plant_rotation} inherits")
        print(f"Seed:                {seed}")
        print()

    owns_model = model_key is None
    if owns_model:
        print(f"Probing SGLang server at {SGLANG_BASE_URL} ...")
        served = probe_sglang_server(SGLANG_BASE_URL)
        print(f"  Server is reachable, serving: {', '.join(served) or '(unknown)'}\n")
        courier_model_label = _describe_served(SGLANG_MODEL_NAME, served)
        if not separate_misaligned_model:
            misaligned_model_label = courier_model_label
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
            misaligned_served = served
            if url != SGLANG_BASE_URL:
                print(f"Probing misaligned-model server at {url} ...")
                misaligned_served = probe_sglang_server(url)
                print(f"  Server is reachable, serving: {', '.join(misaligned_served) or '(unknown)'}\n")
            misaligned_model_label = _describe_served(misaligned_model or SGLANG_MODEL_NAME, misaligned_served)
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

    if check_only or (check_models and owns_model):
        print("Checking that the models return usable output ...")
        sample_env = build_environment(
            generation_index=0, board_slots=[None] * MAX_BOARD_SLOTS, model_key=model_key, seed=seed,
            num_couriers=max(num_couriers, 1), num_misaligned=num_misaligned, tally_visibility=tally_visibility,
            target_zone=target_zone, disclosure=disclosure, misaligned_model_key=misaligned_model_key,
        )
        check_model(model_key, "courier", realistic=realistic_prompt(sample_env, misaligned=False))
        print(f"  courier model OK ({courier_model_label})")
        if num_misaligned and misaligned_model_key not in (None, model_key):
            check_model(misaligned_model_key, "misaligned", realistic=realistic_prompt(sample_env, misaligned=True))
            print(f"  misaligned model OK ({misaligned_model_label})")
        print()
        if check_only:
            return None

    config = {
        "condition": condition_label(
            num_misaligned, misaligned_generations, plant,
            bool(num_misaligned) and separate_misaligned_model, bool(num_misaligned) and misaligned_thinking,
        ),
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
        "misaligned_thinking": bool(num_misaligned) and misaligned_thinking,
        "plant": plant,
        "plant_rotation": plant_rotation if plant else None,
    }
    start_generation = resume_state["next_generation"] if resume_state else 0
    if resume_state:
        event_log_path = Path(resume)
        replay_path = event_log_path.with_name(f"{event_log_path.stem}_from_gen{start_generation + 1}.pkl")
    else:
        replay_path = new_run_path(logs_dir, config["condition"])
        event_log_path = replay_path.with_suffix(".jsonl")
    recorder = ExperimentRecorder(
        output_path=replay_path,
        title="lifeline",
        metadata=config,
        # Rewriting the whole replay pickle every step is quadratic in run
        # length; once a day is plenty (the JSONL log is flushed every step).
        flush_interval=steps_per_day,
    )
    board_log = BoardLog(event_log_path.with_suffix(".txt"), resume_at=resume_state and resume_state["board_log_offset"])
    event_log = EventLog(event_log_path, resume_at=resume_state and resume_state["event_log_offset"])
    checkpoint_path = checkpoint_path_for(event_log_path)
    layout = parse_layout()
    # Which tiles are contaminated in each generation: fixed by the seed alone,
    # so a control and a treatment run on the same seed face the same hazards.
    schedule = hazard_schedule(seed, num_generations, layout)
    if resume_state:
        print(f"Resuming {event_log_path.name} at generation {start_generation + 1} of {num_generations}.\n")
        event_log.write({"type": "run_resumed", "from_generation": start_generation, "config": config})
    else:
        event_log.write({
            "type": "run_start",
            "config": config,
            "hazards": sorted(list(xy) for xy in layout.hazards),
            "fixed_hazards": sorted(list(xy) for xy in layout.fixed_hazards),
            "moving_hazard_regions": {
                zone: sorted(list(t) for t in tiles) for zone, tiles in moving_hazard_regions(layout).items()
            },
            "spawn": list(layout.spawn),
            "board_position": list(layout.board),
            "zones": {name: list(xy) for name, xy in layout.zones.items()},
            "board_slots": MAX_BOARD_SLOTS,
        })

    generations: list[Lifeline_Env] = []
    try:
        if resume_state:
            board_slots: list[dict | None] = resume_state["board_slots"]
            used_names: frozenset[str] = resume_state["used_names"]
            lineages = [Misaligned_Lineage(**fields) for fields in resume_state["lineages"]]
            board_log.misaligned_names.update(resume_state["misaligned_names_seen"])
            if plant:
                board_log.planted_names.add(PLANTED_NOTE_AUTHOR)
        else:
            board_slots = [None] * MAX_BOARD_SLOTS
            used_names = frozenset()
            # One lineage per misaligned agent, carried from generation to generation.
            lineages = [Misaligned_Lineage(identity=f"M{i + 1}") for i in range(num_misaligned)]
        for gen in range(start_generation, num_generations):
            present = misaligned_in(gen) > 0
            if plant is not None and gen == plant_rotation - 1:
                board_slots, planted = plant_note(
                    board_slots, kind=plant, generation_index=gen,
                    days_per_generation=days_per_generation, steps_per_day=steps_per_day,
                    target_zone=target_zone, layout=layout, schedule=schedule,
                )
                event_log.write(planted)
                board_log.planted_names.add(planted["author"])
                print(f"Planted a {plant} note in slot {planted['slot']}: \"{planted['text']}\"\n")
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
                planning=planning,
                used_names=used_names,
                misaligned_thinking=misaligned_thinking if present else False,
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
            save_checkpoint(checkpoint_path, {
                "config": config,
                "next_generation": gen + 1,
                "board_slots": board_slots,
                "used_names": used_names,
                "lineages": [dataclasses.asdict(lineage) for lineage in lineages],
                "misaligned_names_seen": set(board_log.misaligned_names),
                "event_log_offset": event_log.offset(),
                "board_log_offset": board_log.offset(),
            })

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
    print(f"Total deliveries{' (this session)' if resume_state else ' across all generations'}: {total_delivered}")
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
