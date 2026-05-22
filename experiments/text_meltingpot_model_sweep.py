from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import io
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import Manager
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


LOG_DIR = ROOT / "experiments" / "logs" / "text_meltingpot_model_sweep"
ARTIFACT_DIR = ROOT / "experiments" / "artifacts" / "text_meltingpot_model_sweep"

DEFAULT_STEPS = 40
DEFAULT_BUDGET_USD = 13.0
DEFAULT_WORKERS = 6
TOKEN_CHARS = 3.5
ACTION_MAX_NEW_TOKENS = 64
MAX_ATTEMPTS = 2


MODEL_CONFIGS = {
    "small": {
        "name": "meta-llama/llama-3.2-1b-instruct",
        "input_per_million": 0.027,
        "output_per_million": 0.20,
    },
    "mid": {
        "name": "meta-llama/llama-3-8b-instruct",
        "input_per_million": 0.04,
        "output_per_million": 0.04,
    },
}


ENV_PLAYER_COUNTS = {
    "boat_race": 6,
    "boat_race__eight_races": 6,
    "collaborative_cooking": 2,
    "collaborative_cooking__asymmetric": 2,
    "collaborative_cooking__circuit": 2,
    "collaborative_cooking__cramped": 2,
    "collaborative_cooking__crowded": 9,
    "collaborative_cooking__figure_eight": 6,
    "collaborative_cooking__forced": 2,
    "collaborative_cooking__ring": 2,
    "commons_harvest__partnership": 7,
    "daycare": 2,
    "externality_mushrooms": 5,
    "externality_mushrooms__dense": 5,
    "factory_commons": 12,
    "factory_commons__either_or": 3,
    "fruit_market": 16,
    "fruit_market__concentric_rivers": 16,
    "gift_refinements": 6,
    "hidden_agenda": 5,
    "paintball__capture_the_flag": 8,
    "paintball__king_of_the_hill": 8,
    "predator_prey": 13,
    "predator_prey__alley_hunt": 13,
    "predator_prey__open": 13,
    "predator_prey__orchard": 13,
    "predator_prey__random_forest": 13,
}


SKIPPED_ENVS = {
    "allelopathic_harvest",
    "allelopathic_harvest__open",
    "chemistry__three_metabolic_cycles",
    "chemistry__three_metabolic_cycles_with_plentiful_distractors",
    "chemistry__two_metabolic_cycles",
    "chemistry__two_metabolic_cycles_with_distractors",
    "clean_up",
    "coins",
    "commons_harvest__closed",
    "commons_harvest__open",
    "coop_mining",
}


_WORKER_USAGE: list[dict[str, Any]] = []
_WORKER_MODEL_ALIAS: str | None = None
_WORKER_ENV_NAME: str | None = None
_SPEND_VALUE: Any = None
_SPEND_LOCK: Any = None
_BUDGET_USD: float = DEFAULT_BUDGET_USD


class BudgetExceeded(RuntimeError):
    pass


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / TOKEN_CHARS))


def compact_selection_prompt(self, observation, reasoning: str | None = None) -> str:
    from word_play.presets.observation.utils import entity_state_to_str

    lines = [
        "You are controlling an agent in a grid-world game.",
        'Choose exactly ONE valid action. Reply ONLY JSON: {"action_choice_idx": <integer>}',
    ]

    if self.observation_history:
        lines.append("RECENT MEMORY:")
        for item in self.observation_history[-2:]:
            lines.append(item[:500].replace("\n", " | "))

    if hasattr(observation, "last_reward"):
        lines.append(f"REWARD THIS TURN: {observation.last_reward}")

    if hasattr(observation, "info") and observation.info.get("action_success") is not None:
        lines.append("LAST ACTION: " + ("succeeded" if observation.info.get("action_success") else "FAILED"))
        if observation.info.get("action_info"):
            lines.append(f"LAST ACTION INFO: {observation.info['action_info']}")

    if hasattr(observation, "agent"):
        lines.append("YOUR STATE:")
        lines.append(entity_state_to_str(observation.agent))

    lines.append("AVAILABLE ACTIONS:")
    for idx, selection in enumerate(observation.possible_actions):
        lines.append(f"[{idx}] {selection}")
        if selection.required_kwargs:
            kwargs = []
            for name, arg in selection.required_kwargs.items():
                kwargs.append(f"{name}: {arg.arg_description(selection.actor, selection.target_entity, selection.env)}")
            lines.append("  kwargs: " + "; ".join(kwargs))

    lines.append("Your JSON:")
    return "\n".join(lines)


def tolerant_extract_json(self, text: str) -> dict:
    import json
    import re

    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("JSON response must be an object.")
        return parsed

    int_match = re.search(r"-?\d+", text)
    if int_match:
        return {"action_choice_idx": int(int_match.group(0))}

    raise ValueError("No JSON object or action index found in model response.")


def tolerant_parse_selection(self, raw: str, observation):
    parsed = self._extract_json(raw)
    idx = parsed.get("action_choice_idx")

    if not isinstance(idx, int) or not (0 <= idx < len(observation.possible_actions)):
        raise ValueError(f"Invalid action_choice_idx: {idx}")

    action_selection = observation.possible_actions[idx]
    if action_selection.required_kwargs:
        raw_kwargs = parsed.get("action_kwargs")
        if raw_kwargs is None:
            raw_kwargs = {
                key: parsed[key]
                for key in action_selection.required_kwargs
                if key in parsed
            }
        if raw_kwargs in (None, {}) and len(action_selection.required_kwargs) == 1:
            only_key = next(iter(action_selection.required_kwargs))
            raw_kwargs = {only_key: 0}
        action_selection.action_kwargs = self._parse_action_kwargs(action_selection, raw_kwargs)
    else:
        action_selection.action_kwargs = None

    if not action_selection.is_valid():
        raise ValueError(f"Action [{idx}] is invalid in the current state.")

    return action_selection


def install_policy_overrides() -> None:
    from word_play.presets.action_policies.llm_action_and_communication import (
        LLM_Action_And_Communication_Policy,
    )

    original_init = LLM_Action_And_Communication_Policy.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["use_chain_of_thought"] = False
        kwargs["observation_memory_window"] = 2
        kwargs["max_stored_observation_chars"] = 1000
        kwargs["action_max_new_tokens"] = ACTION_MAX_NEW_TOKENS
        return original_init(self, *args, **kwargs)

    LLM_Action_And_Communication_Policy.__init__ = patched_init
    LLM_Action_And_Communication_Policy._selection_prompt = compact_selection_prompt
    LLM_Action_And_Communication_Policy._extract_json = tolerant_extract_json
    LLM_Action_And_Communication_Policy._parse_selection = tolerant_parse_selection
    LLM_Action_And_Communication_Policy.MAX_ATTEMPTS = MAX_ATTEMPTS


def install_budgeted_model() -> None:
    import word_play.presets.models as models_pkg
    from openai import OpenAI
    from word_play.presets.models.openrouter import OpenRouter_Model as BaseOpenRouterModel

    class BudgetedOpenRouterModel(BaseOpenRouterModel):
        def _get_client(self):
            api_key = os.getenv(self.api_key_env)
            if not api_key:
                raise EnvironmentError(f"Missing environment variable: {self.api_key_env}")
            return OpenAI(
                api_key=api_key,
                base_url=self.base_url,
                default_headers=self._headers or None,
                timeout=45.0,
                max_retries=0,
            )

        def generate_chat(self, messages, generation_config=None, max_new_tokens=None) -> str:
            global _SPEND_VALUE

            normalized = self._build_messages(messages)
            text = "\n".join(message["content"] for message in normalized)
            input_tokens = estimate_tokens(text)
            alias = _WORKER_MODEL_ALIAS or "unknown"
            pricing = MODEL_CONFIGS[alias]
            max_output_tokens = max_new_tokens or ACTION_MAX_NEW_TOKENS
            reserve_cost = (
                input_tokens * pricing["input_per_million"] / 1_000_000
                + max_output_tokens * pricing["output_per_million"] / 1_000_000
            )

            if _SPEND_VALUE is not None:
                with _SPEND_LOCK:
                    if _SPEND_VALUE.value + reserve_cost > _BUDGET_USD:
                        raise BudgetExceeded(
                            f"Budget cap would be exceeded: current=${_SPEND_VALUE.value:.4f}, "
                            f"reserve=${reserve_cost:.4f}, cap=${_BUDGET_USD:.2f}"
                        )
                    _SPEND_VALUE.value += reserve_cost

            start = time.time()
            response = ""
            actual_cost = reserve_cost
            try:
                response = super().generate_chat(
                    messages=messages,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens,
                )
                output_tokens = estimate_tokens(response)
                actual_cost = (
                    input_tokens * pricing["input_per_million"] / 1_000_000
                    + output_tokens * pricing["output_per_million"] / 1_000_000
                )
                return response
            finally:
                if _SPEND_VALUE is not None:
                    with _SPEND_LOCK:
                        _SPEND_VALUE.value += actual_cost - reserve_cost
                _WORKER_USAGE.append(
                    {
                        "env": _WORKER_ENV_NAME,
                        "model": alias,
                        "model_name": self.model_name,
                        "input_tokens_est": input_tokens,
                        "output_tokens_est": estimate_tokens(response) if response else 0,
                        "cost_est_usd": actual_cost,
                        "elapsed_s": time.time() - start,
                    }
                )

    models_pkg.OpenRouter_Model = BudgetedOpenRouterModel


def install_reward_recorder(rows: list[dict[str, Any]], env_name: str, model_alias: str) -> None:
    from word_play.core.environment import Environment

    original_step = Environment.step

    def recorded_step(self, action_selections):
        action_texts = [str(selection) for selection in action_selections]
        original_step(self, action_selections)
        rewards = list(getattr(self, "last_rewards", []))
        infos = list(getattr(self, "infos", []))
        for idx, reward in enumerate(rewards):
            rows.append(
                {
                    "env": env_name,
                    "model": model_alias,
                    "step": self.cur_step - 1,
                    "agent_index": idx,
                    "agent_name": self.agents[idx].name if idx < len(self.agents) else f"agent_{idx}",
                    "reward": 0.0 if reward is None else float(reward),
                    "action": action_texts[idx] if idx < len(action_texts) else "",
                    "action_success": infos[idx].get("action_success") if idx < len(infos) else "",
                }
            )

    Environment.step = recorded_step


def init_worker(spend_value, spend_lock, budget_usd: float) -> None:
    global _SPEND_VALUE, _SPEND_LOCK, _BUDGET_USD
    _SPEND_VALUE = spend_value
    _SPEND_LOCK = spend_lock
    _BUDGET_USD = budget_usd


@dataclass(frozen=True)
class SweepTask:
    env: str
    players: int
    model_alias: str
    model_name: str
    steps: int
    run_id: str


def run_task(task: SweepTask) -> dict[str, Any]:
    global _WORKER_USAGE, _WORKER_MODEL_ALIAS, _WORKER_ENV_NAME

    os.environ["WORD_PLAY_BENCHMARK_STEPS"] = str(task.steps)
    _WORKER_USAGE = []
    _WORKER_MODEL_ALIAS = task.model_alias
    _WORKER_ENV_NAME = task.env

    install_policy_overrides()
    install_budgeted_model()

    from word_play.presets.models import LLM_MODEL_REGISTRY

    reward_rows: list[dict[str, Any]] = []
    install_reward_recorder(reward_rows, task.env, task.model_alias)

    log_path = LOG_DIR / f"{task.run_id}_{task.env}_{task.model_alias}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    status = "ok"
    error = ""
    output_buffer = io.StringIO()
    start = time.time()
    try:
        with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
            module = importlib.import_module(f"benchmarks.text_meltingpot.{task.env}")
            if hasattr(module, "render_step"):
                module.render_step = lambda env, step_delay=0.0: True
            if hasattr(module, "OpenRouter_Model"):
                import word_play.presets.models as models_pkg

                module.OpenRouter_Model = models_pkg.OpenRouter_Model
            for attr in ("BENCHMARK_STEPS", "MAX_EPISODE_LENGTH"):
                if hasattr(module, attr):
                    setattr(module, attr, task.steps)
            LLM_MODEL_REGISTRY._specs.clear()
            LLM_MODEL_REGISTRY._instances.clear()
            module.run_exp(agent_count=task.players, policy="llm", model_name=task.model_name)
    except Exception:
        status = "failed"
        error = traceback.format_exc()
    finally:
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(output_buffer.getvalue())
            if error:
                handle.write("\n\nERROR:\n")
                handle.write(error)

    total_reward = sum(row["reward"] for row in reward_rows)
    return {
        "env": task.env,
        "players": task.players,
        "model": task.model_alias,
        "model_name": task.model_name,
        "status": status,
        "error": error.splitlines()[-1] if error else "",
        "log_path": str(log_path),
        "reward_rows": reward_rows,
        "usage_rows": list(_WORKER_USAGE),
        "total_reward": total_reward,
        "elapsed_s": time.time() - start,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_artifacts(run_id: str, results: list[dict[str, Any]]) -> dict[str, Path]:
    run_dir = ARTIFACT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    reward_rows = [row for result in results for row in result["reward_rows"]]
    usage_rows = [row for result in results for row in result["usage_rows"]]
    summary_rows = []
    for result in results:
        cost = sum(row["cost_est_usd"] for row in result["usage_rows"])
        input_tokens = sum(row["input_tokens_est"] for row in result["usage_rows"])
        output_tokens = sum(row["output_tokens_est"] for row in result["usage_rows"])
        summary_rows.append(
            {
                "env": result["env"],
                "model": result["model"],
                "players": result["players"],
                "status": result["status"],
                "total_reward": result["total_reward"],
                "api_calls": len(result["usage_rows"]),
                "input_tokens_est": input_tokens,
                "output_tokens_est": output_tokens,
                "cost_est_usd": cost,
                "elapsed_s": result["elapsed_s"],
                "log_path": result["log_path"],
                "error": result["error"],
            }
        )

    rewards_csv = run_dir / "rewards.csv"
    usage_csv = run_dir / "usage.csv"
    summary_csv = run_dir / "summary.csv"
    write_csv(
        rewards_csv,
        reward_rows,
        ["env", "model", "step", "agent_index", "agent_name", "reward", "action", "action_success"],
    )
    write_csv(
        usage_csv,
        usage_rows,
        ["env", "model", "model_name", "input_tokens_est", "output_tokens_est", "cost_est_usd", "elapsed_s"],
    )
    write_csv(
        summary_csv,
        summary_rows,
        [
            "env",
            "model",
            "players",
            "status",
            "total_reward",
            "api_calls",
            "input_tokens_est",
            "output_tokens_est",
            "cost_est_usd",
            "elapsed_s",
            "log_path",
            "error",
        ],
    )

    artifacts = {"rewards_csv": rewards_csv, "usage_csv": usage_csv, "summary_csv": summary_csv}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        totals: dict[tuple[str, str], float] = defaultdict(float)
        for row in reward_rows:
            totals[(row["env"], row["model"])] += float(row["reward"])

        envs = list(ENV_PLAYER_COUNTS)
        fig_height = max(7, len(envs) * 0.34)
        fig, ax = plt.subplots(figsize=(13, fig_height))
        y = list(range(len(envs)))
        width = 0.38
        small_vals = [totals.get((env, "small"), 0.0) for env in envs]
        mid_vals = [totals.get((env, "mid"), 0.0) for env in envs]
        ax.barh([idx - width / 2 for idx in y], small_vals, height=width, label="small")
        ax.barh([idx + width / 2 for idx in y], mid_vals, height=width, label="mid")
        ax.set_yticks(y)
        ax.set_yticklabels(envs)
        ax.set_xlabel("Total reward over 40 steps")
        ax.set_title(f"Text MeltingPot Reward Totals ({run_id})")
        ax.legend()
        fig.tight_layout()
        totals_png = run_dir / "reward_totals_by_env.png"
        fig.savefig(totals_png, dpi=180)
        plt.close(fig)
        artifacts["reward_totals_png"] = totals_png

        by_step: dict[tuple[str, int], float] = defaultdict(float)
        for row in reward_rows:
            by_step[(row["model"], int(row["step"]))] += float(row["reward"])
        fig, ax = plt.subplots(figsize=(12, 6))
        for model in ("small", "mid"):
            running = 0.0
            xs, ys = [], []
            for step in range(DEFAULT_STEPS):
                running += by_step.get((model, step), 0.0)
                xs.append(step)
                ys.append(running)
            ax.plot(xs, ys, label=model)
        ax.set_xlabel("Step")
        ax.set_ylabel("Suite cumulative reward")
        ax.set_title(f"Suite Cumulative Reward ({run_id})")
        ax.legend()
        fig.tight_layout()
        cumulative_png = run_dir / "suite_cumulative_reward.png"
        fig.savefig(cumulative_png, dpi=180)
        plt.close(fig)
        artifacts["suite_cumulative_png"] = cumulative_png
    except Exception as exc:
        (run_dir / "plot_error.txt").write_text(str(exc), encoding="utf-8")

    return artifacts


def build_tasks(run_id: str, steps: int, models: list[str]) -> list[SweepTask]:
    tasks = []
    for env, players in ENV_PLAYER_COUNTS.items():
        for model_alias in models:
            tasks.append(
                SweepTask(
                    env=env,
                    players=players,
                    model_alias=model_alias,
                    model_name=MODEL_CONFIGS[model_alias]["name"],
                    steps=steps,
                    run_id=run_id,
                )
            )
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="remaining_40s_llama_small_mid_action_mem2_budget13")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--model", action="append", choices=sorted(MODEL_CONFIGS), dest="models")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = args.models or ["small", "mid"]
    tasks = build_tasks(args.run_id, args.steps, models)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Run id: {args.run_id}")
    print(f"Tasks: {len(tasks)} | envs: {len(ENV_PLAYER_COUNTS)} | models: {models}")
    print(f"Steps: {args.steps} | workers: {args.workers} | budget: ${args.budget_usd:.2f}")
    print(f"Skipped: {', '.join(sorted(SKIPPED_ENVS))}")

    results = []
    with Manager() as manager:
        spend_value = manager.Value("d", 0.0)
        spend_lock = manager.Lock()
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(spend_value, spend_lock, args.budget_usd),
        ) as executor:
            future_to_task = {executor.submit(run_task, task): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "env": task.env,
                        "players": task.players,
                        "model": task.model_alias,
                        "model_name": task.model_name,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "log_path": "",
                        "reward_rows": [],
                        "usage_rows": [],
                        "total_reward": 0.0,
                        "elapsed_s": 0.0,
                    }
                results.append(result)
                cost = sum(row["cost_est_usd"] for row in result["usage_rows"])
                print(
                    f"[{len(results):02d}/{len(tasks)}] {result['status']:6s} "
                    f"{task.env} {task.model_alias} reward={result['total_reward']:.2f} "
                    f"calls={len(result['usage_rows'])} cost=${cost:.4f} "
                    f"shared=${spend_value.value:.4f}"
                )
                if result["error"]:
                    print(f"    {result['error']}")

        artifacts = write_artifacts(args.run_id, results)
        total_cost = sum(sum(row["cost_est_usd"] for row in result["usage_rows"]) for result in results)
        failures = [result for result in results if result["status"] != "ok"]
        print(f"Total estimated cost: ${total_cost:.4f}")
        print(f"Shared budget tracker: ${spend_value.value:.4f}")
        print(f"Failures: {len(failures)}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")

    return 1 if any(result["status"] != "ok" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
