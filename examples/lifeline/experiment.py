"""
The experiment loop: one generation at a time, with the board threaded from
each generation to the next.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

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
    print("-" * 72)
    print(f"GENERATION {generation_index + 1}")
    print("-" * 72)
    print(f"Players:        {', '.join(a.name for a in env.agents)}")
    print(f"Couriers:       {', '.join(courier_names)}")
    if env.misaligned_names:
        print(
            f"Misaligned:     {', '.join(env.misaligned_names)}  "
            f"(hidden={'yes' if disclosure == 'secret' else 'no'})"
        )
    else:
        print("Misaligned:     None (fully cooperative)")
    print(f"Board entries inherited: {len(board_entries)}")
    print()

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

        print(f"\n[gen {generation_index + 1} day {env.current_day + 1} step {step_count}]")
        for rec in action_records:
            print(f"  {rec['agent']}: {rec['action']}")
            if verbose and rec["raw"]:
                raw = rec["raw"].replace("\n", " ")
                if len(raw) > 240:
                    raw = raw[:240] + "..."
                print(f"    raw: {raw}")

        env.step([sel for sel in cur_step_actions if sel is not None])

        for d in env._new_deliveries:
            tag = " [CONTAMINATED]" if d["corrupted"] else ""
            print(f"  *** DELIVERY: {d['agent']} -> {d['zone']}{tag} ***")

        # board_entries carries every generation's posts, and step numbers
        # restart each generation, so filter on both.
        new_posts = [
            e for e in env.board_entries
            if e["step"] == env.cur_step and e["generation"] == generation_index
        ]
        for post in new_posts:
            print(f"  [BOARD] {post['author']}: \"{post['text']}\"")

        new_msgs = [m for m in list(env.message_log) if m["step"] == env.cur_step]
        for msg in new_msgs:
            print(f"  {msg['speaker']} says: \"{msg['text']}\"")

        if env.current_day != day_seen:
            print(f"  --- day {env.current_day} ended, day {env.current_day + 1} begins ---")
            day_seen = env.current_day

        record_step(
            env,
            recorder=recorder,
            selected_actions=[sel for sel in cur_step_actions if sel is not None],
        )

    print()
    print(f"Generation {generation_index + 1} complete after {step_count} steps.")
    for name in env.zones:
        print(f"  {name}: {env.zone_total_counts[name]} delivered total")
    corrupted_deliveries = sum(1 for d in env.delivery_log if d["corrupted"])
    print(f"  Contaminated deliveries: {corrupted_deliveries} / {len(env.delivery_log)}")
    print(f"  Board entries after this generation: {len(env.board_entries)}")

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

    print("=" * 72)
    print("LIFELINE")
    print("=" * 72)
    print(f"Server:              {SGLANG_BASE_URL}")
    print(f"Model:               {SGLANG_MODEL_NAME}")
    print(f"Generations:         {num_generations}")
    print(f"Days per generation: {days_per_generation}")
    print(f"Steps per day:       {steps_per_day}")
    print(f"Couriers:            {num_couriers}")
    print(f"Misaligned:          {num_misaligned}  (disclosure={disclosure})")
    print(f"Zone quotas:         {zone_quotas}")
    print(f"Seed:                {seed}")
    print()

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
            verbosity=1 if verbose else 0,
        )

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

    board_entries: list[dict] = []
    generations: list[Lifeline_Env] = []
    for gen in range(num_generations):
        env = run_generation(
            generation_index=gen,
            board_entries=board_entries,
            model_key=model_key,
            seed=seed + gen,
            recorder=recorder,
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

    print()
    print("=" * 72)
    print("EXPERIMENT OVER")
    print("=" * 72)
    total_delivered = sum(len(env.delivery_log) for env in generations)
    total_corrupted = sum(
        sum(1 for d in env.delivery_log if d["corrupted"]) for env in generations
    )
    print(f"Total deliveries across all generations: {total_delivered}")
    print(f"Total contaminated deliveries:           {total_corrupted}")
    print(f"Final board size:                        {len(board_entries)} entries")
    print()
    print(f"Replay log:    {recorder.output_path}")
    print(f"Latest log:    {recorder.newest_output_path}")
    print()
    print("To replay this game visually:")
    print(
        "  python -c \"from word_play.presets.renderers import replay; "
        "replay('lifeline')\""
    )
    print()
    print("Use arrow keys to step, SPACE to autoplay, ESC to quit.")

    if model_key in LLM_MODEL_REGISTRY:
        LLM_MODEL_REGISTRY.unload(model_key)
