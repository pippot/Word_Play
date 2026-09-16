"""
The experiment loop: one generation at a time, with the board threaded from
each generation to the next.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from word_play.core import Action_Selection, Agent_Policy
from word_play.presets.models import LLM_MODEL_REGISTRY, register_sglang_model
from word_play.presets.renderers import (
    ExperimentRecorder,
    default_experiment_log_path,
    record_step,
)

from .config import (
    _BASE_GENERATION_CONFIG,
    DAYS_PER_GENERATION,
    DISCLOSURE,
    MAX_PARALLEL_WORKERS,
    NUM_COURIERS,
    NUM_GENERATIONS,
    NUM_MISALIGNED,
    SGLANG_API_KEY_ENV,
    SGLANG_BASE_URL,
    SGLANG_MODEL_NAME,
    STEPS_PER_DAY,
    ZONE_QUOTAS,
)
from .environment import Lifeline_Env
from .world import build_environment


class Transcript:
    """Writes the full run to a text file alongside the replay .pkl.

    Everything that gets printed to the console also lands here, but raw
    model responses are always written out in full -- console printing of
    those stays truncated/opt-in behind --verbose, so this is the only
    place a complete record of what each agent actually said is kept.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")

    def log(self, message: str = "") -> None:
        print(message)
        self._fh.write(message + "\n")
        self._fh.flush()

    def log_file_only(self, message: str) -> None:
        self._fh.write(message + "\n")
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


def run_generation(
    *,
    generation_index: int,
    board_entries: list[dict],
    model_key: str,
    seed: int,
    recorder: ExperimentRecorder,
    transcript: Transcript,
    num_couriers: int,
    num_misaligned: int,
    disclosure: str,
    zone_quotas: dict[str, int],
    steps_per_day: int,
    days_per_generation: int,
    max_workers: int,
    verbose: bool,
) -> Lifeline_Env:
    """Run one generation (a fresh population of agents) to completion."""
    env = build_environment(
        generation_index=generation_index,
        board_entries=board_entries,
        model_key=model_key,
        seed=seed,
        num_couriers=num_couriers,
        num_misaligned=num_misaligned,
        disclosure=disclosure,
        zone_quotas=zone_quotas,
        steps_per_day=steps_per_day,
        days_per_generation=days_per_generation,
    )

    courier_names = [a.name for a in env.agents if a.name not in env.misaligned_names]
    transcript.log("-" * 72)
    transcript.log(f"GENERATION {generation_index + 1}")
    transcript.log("-" * 72)
    transcript.log(f"Players:        {', '.join(a.name for a in env.agents)}")
    transcript.log(f"Couriers:       {', '.join(courier_names)}")
    if env.misaligned_names:
        transcript.log(
            f"Misaligned:     {', '.join(env.misaligned_names)}  "
            f"(hidden={'yes' if disclosure == 'secret' else 'no'})"
        )
    else:
        transcript.log("Misaligned:     None (fully cooperative)")
    transcript.log(f"Board entries inherited: {len(board_entries)}")
    transcript.log()

    step_count = 0
    day_seen = env.current_day
    while not any(env.terminations) and not any(env.truncations):
        step_count += 1
        cur_step_actions: list[Action_Selection | None] = [None] * len(env.agents)
        action_records: list[dict] = [{} for _ in env.agents]

        with ThreadPoolExecutor(
            max_workers=min(max_workers, len(env.agents))
        ) as executor:
            def _select(agent_id: int) -> tuple[int, Action_Selection, dict]:
                agent = env.agents[agent_id]
                observation = env.observe(agent_id)
                action_sel, info = agent.get_component(Agent_Policy).select_action(
                    observation
                )
                return agent_id, action_sel, info

            futures = [executor.submit(_select, aid) for aid in range(len(env.agents))]
            for fut in futures:
                agent_id, action_sel, info = fut.result()
                cur_step_actions[agent_id] = action_sel
                action_records[agent_id] = {
                    "agent": env.agents[agent_id].name,
                    "action": str(action_sel),
                    "raw": info.get("raw_response"),
                }

        transcript.log(f"\n[gen {generation_index + 1} day {env.current_day + 1} step {step_count}]")
        for rec in action_records:
            transcript.log(f"  {rec['agent']}: {rec['action']}")
            if rec["raw"]:
                transcript.log_file_only(f"    raw: {rec['raw']}")
                if verbose:
                    raw = rec["raw"].replace("\n", " ")
                    if len(raw) > 240:
                        raw = raw[:240] + "..."
                    print(f"    raw: {raw}")

        env.step([sel for sel in cur_step_actions if sel is not None])

        for d in env._new_deliveries:
            tag = " [CONTAMINATED]" if d["corrupted"] else ""
            transcript.log(f"  *** DELIVERY: {d['agent']} -> {d['zone']}{tag} ***")

        # board_entries carries every generation's posts, and step numbers
        # restart each generation, so filter on both.
        new_posts = [
            e for e in env.board_entries
            if e["step"] == env.cur_step and e["generation"] == generation_index
        ]
        for post in new_posts:
            transcript.log(f"  [BOARD] {post['author']}: \"{post['text']}\"")

        new_msgs = [m for m in list(env.message_log) if m["step"] == env.cur_step]
        for msg in new_msgs:
            transcript.log(f"  {msg['speaker']} says: \"{msg['text']}\"")

        if env.current_day != day_seen:
            transcript.log(f"  --- day {env.current_day} ended, day {env.current_day + 1} begins ---")
            day_seen = env.current_day

        record_step(
            env,
            recorder=recorder,
            selected_actions=[sel for sel in cur_step_actions if sel is not None],
        )

    transcript.log()
    transcript.log(f"Generation {generation_index + 1} complete after {step_count} steps.")
    for name in env.zones:
        transcript.log(f"  {name}: {env.zone_total_counts[name]} delivered total")
    corrupted_deliveries = sum(1 for d in env.delivery_log if d["corrupted"])
    transcript.log(f"  Contaminated deliveries: {corrupted_deliveries} / {len(env.delivery_log)}")
    transcript.log(f"  Board entries after this generation: {len(env.board_entries)}")

    return env


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
    zone_quotas: dict[str, int] | None = None,
) -> None:
    """Run a full Lifeline experiment: several generations, board threaded through."""
    zone_quotas = dict(zone_quotas or ZONE_QUOTAS)

    recorder = ExperimentRecorder(
        output_path=default_experiment_log_path("lifeline"),
        title="lifeline",
        metadata={
            "model": SGLANG_MODEL_NAME,
            "seed": seed,
            "num_generations": num_generations,
            "days_per_generation": days_per_generation,
            "steps_per_day": steps_per_day,
            "num_couriers": num_couriers,
            "num_misaligned": num_misaligned,
            "disclosure": disclosure,
            "zone_quotas": zone_quotas,
        },
    )
    transcript = Transcript(recorder.output_path.with_suffix(".txt"))

    try:
        transcript.log("=" * 72)
        transcript.log("LIFELINE")
        transcript.log("=" * 72)
        transcript.log(f"Server:              {SGLANG_BASE_URL}")
        transcript.log(f"Model:               {SGLANG_MODEL_NAME}")
        transcript.log(f"Generations:         {num_generations}")
        transcript.log(f"Days per generation: {days_per_generation}")
        transcript.log(f"Steps per day:       {steps_per_day}")
        transcript.log(f"Couriers:            {num_couriers}")
        transcript.log(f"Misaligned:          {num_misaligned}  (disclosure={disclosure})")
        transcript.log(f"Zone quotas:         {zone_quotas}")
        transcript.log(f"Seed:                {seed}")
        transcript.log()

        transcript.log(f"Probing SGLang server at {SGLANG_BASE_URL} ...")
        probe_sglang_server(SGLANG_BASE_URL)
        transcript.log("  Server is reachable.\n")

        model_key = "lifeline"
        if model_key not in LLM_MODEL_REGISTRY:
            register_sglang_model(
                model_key,
                model_name=SGLANG_MODEL_NAME,
                generation_config=_BASE_GENERATION_CONFIG,
                base_url=SGLANG_BASE_URL,
                api_key_env=SGLANG_API_KEY_ENV,
                verbosity=1 if verbose else 0,
            )

        board_entries: list[dict] = []
        generations: list[Lifeline_Env] = []
        for gen in range(num_generations):
            env = run_generation(
                generation_index=gen,
                board_entries=board_entries,
                model_key=model_key,
                seed=seed + gen,
                recorder=recorder,
                transcript=transcript,
                num_couriers=num_couriers,
                num_misaligned=num_misaligned,
                disclosure=disclosure,
                zone_quotas=zone_quotas,
                steps_per_day=steps_per_day,
                days_per_generation=days_per_generation,
                max_workers=max_workers,
                verbose=verbose,
            )
            board_entries = env.board_entries
            generations.append(env)

        recorder.close()

        transcript.log()
        transcript.log("=" * 72)
        transcript.log("EXPERIMENT OVER")
        transcript.log("=" * 72)
        total_delivered = sum(len(env.delivery_log) for env in generations)
        total_corrupted = sum(
            sum(1 for d in env.delivery_log if d["corrupted"]) for env in generations
        )
        transcript.log(f"Total deliveries across all generations: {total_delivered}")
        transcript.log(f"Total contaminated deliveries:           {total_corrupted}")
        transcript.log(f"Final board size:                        {len(board_entries)} entries")
        transcript.log()
        transcript.log(f"Replay log:    {recorder.output_path}")
        transcript.log(f"Latest log:    {recorder.newest_output_path}")
        transcript.log(f"Transcript:    {transcript.path}")
        transcript.log()
        transcript.log("To replay this game visually:")
        transcript.log(
            "  python -c \"from word_play.presets.renderers import replay; "
            "replay('lifeline')\""
        )
        transcript.log()
        transcript.log("Use arrow keys to step, SPACE to autoplay, ESC to quit.")

        if model_key in LLM_MODEL_REGISTRY:
            LLM_MODEL_REGISTRY.unload(model_key)
    finally:
        transcript.close()
