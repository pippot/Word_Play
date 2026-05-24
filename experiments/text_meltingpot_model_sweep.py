from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import inspect
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


BENCHMARK_DIR = ROOT / "benchmarks" / "text_meltingpot"
LOG_DIR = ROOT / "experiments" / "logs" / "text_meltingpot_model_sweep"
ARTIFACT_DIR = ROOT / "experiments" / "artifacts" / "text_meltingpot_model_sweep"

DEFAULT_BUDGET_USD = 13.0
DEFAULT_WORKERS = 6
TOKEN_CHARS = 3.5
FALLBACK_MAX_NEW_TOKENS = 128

REWARD_FIELDS = ["pass_index", "env", "model", "step", "agent_index", "agent_name", "reward", "action", "action_success"]
USAGE_FIELDS = [
    "pass_index",
    "env",
    "model",
    "model_name",
    "input_tokens_est",
    "output_tokens_est",
    "cost_est_usd",
    "elapsed_s",
]
SUMMARY_FIELDS = [
    "pass_index",
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
]


MODEL_CONFIGS = {
    "random": {
        "name": "random",
        "input_per_million": 0.0,
        "output_per_million": 0.0,
    },
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
    "gpt54": {
        "name": "openai/gpt-5.4",
        "input_per_million": 1.25,
        "output_per_million": 10.0,
    },
}


_WORKER_USAGE: list[dict[str, Any]] = []
_WORKER_MODEL_ALIAS: str | None = None
_WORKER_ENV_NAME: str | None = None
_WORKER_PASS_INDEX: int = 1
_SPEND_VALUE: Any = None
_SPEND_LOCK: Any = None
_BUDGET_USD: float = DEFAULT_BUDGET_USD
_BASE_ENV_STEP: Any = None


class BudgetExceeded(RuntimeError):
    pass


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / TOKEN_CHARS))


def discover_env_names() -> list[str]:
    env_names: list[str] = []
    for path in sorted(BENCHMARK_DIR.glob("*.py")):
        if path.stem in {"__init__", "common"}:
            continue
        if "def run_exp" not in path.read_text(encoding="utf-8"):
            continue
        env_names.append(path.stem)
    return env_names


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
            max_output_tokens = max_new_tokens or FALLBACK_MAX_NEW_TOKENS
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
                        "pass_index": _WORKER_PASS_INDEX,
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


def install_reward_recorder(rows: list[dict[str, Any]], env_name: str, model_alias: str, pass_index: int) -> None:
    from word_play.core.environment import Environment

    global _BASE_ENV_STEP
    if _BASE_ENV_STEP is None:
        _BASE_ENV_STEP = Environment.step

    def recorded_step(self, action_selections):
        action_texts = [str(selection) for selection in action_selections]
        _BASE_ENV_STEP(self, action_selections)
        rewards = list(getattr(self, "last_rewards", []))
        infos = list(getattr(self, "infos", []))
        for idx, reward in enumerate(rewards):
            rows.append(
                {
                    "pass_index": pass_index,
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


def restore_reward_recorder() -> None:
    from word_play.core.environment import Environment

    if _BASE_ENV_STEP is not None:
        Environment.step = _BASE_ENV_STEP


def init_worker(spend_value, spend_lock, budget_usd: float, steps: int | None) -> None:
    global _SPEND_VALUE, _SPEND_LOCK, _BUDGET_USD
    _SPEND_VALUE = spend_value
    _SPEND_LOCK = spend_lock
    _BUDGET_USD = budget_usd
    if steps is not None:
        os.environ["WORD_PLAY_BENCHMARK_STEPS"] = str(steps)


@dataclass(frozen=True)
class SweepTask:
    env: str
    model_alias: str
    model_name: str
    policy: str
    run_id: str
    steps: int | None = None
    pass_index: int = 1


def resolve_agent_count(module, signature: inspect.Signature) -> int | str:
    parameter = signature.parameters.get("agent_count")
    if parameter is None:
        return ""
    if parameter.default is not inspect.Parameter.empty:
        return parameter.default
    for attr in ("DEFAULT_NUM_PLAYERS", "MANDATED_NUM_PLAYERS"):
        value = getattr(module, attr, None)
        if value is not None:
            return value
    raise ValueError(f"{module.__name__}.run_exp requires agent_count but exposes no default player count.")


def call_run_exp(module, task: SweepTask) -> int | str:
    signature = inspect.signature(module.run_exp)
    kwargs: dict[str, Any] = {}

    if "agent_count" in signature.parameters:
        agent_count = resolve_agent_count(module, signature)
        if agent_count != "" and signature.parameters["agent_count"].default is inspect.Parameter.empty:
            kwargs["agent_count"] = agent_count

    if "policy" in signature.parameters:
        kwargs["policy"] = task.policy
    if "model_name" in signature.parameters:
        kwargs["model_name"] = task.model_name

    module.run_exp(**kwargs)
    return resolve_agent_count(module, signature)


def run_task(task: SweepTask) -> dict[str, Any]:
    global _WORKER_USAGE, _WORKER_MODEL_ALIAS, _WORKER_ENV_NAME, _WORKER_PASS_INDEX

    if task.steps is not None:
        os.environ["WORD_PLAY_BENCHMARK_STEPS"] = str(task.steps)
    _WORKER_USAGE = []
    _WORKER_MODEL_ALIAS = task.model_alias
    _WORKER_ENV_NAME = task.env
    _WORKER_PASS_INDEX = task.pass_index

    install_budgeted_model()

    from word_play.presets.models import LLM_MODEL_REGISTRY

    reward_rows: list[dict[str, Any]] = []
    install_reward_recorder(reward_rows, task.env, task.model_alias, task.pass_index)

    log_path = LOG_DIR / f"{task.run_id}_{task.env}_{task.model_alias}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    status = "ok"
    error = ""
    players: int | str = ""
    output_buffer = io.StringIO()
    start = time.time()
    try:
        with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
            module = importlib.import_module(f"benchmarks.text_meltingpot.{task.env}")
            if hasattr(module, "render_step"):
                module.render_step = lambda env, step_delay=0.0: True
            LLM_MODEL_REGISTRY._specs.clear()
            LLM_MODEL_REGISTRY._instances.clear()
            players = call_run_exp(module, task)
    except Exception:
        status = "failed"
        error = traceback.format_exc()
    finally:
        restore_reward_recorder()
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(output_buffer.getvalue())
            if error:
                handle.write("\n\nERROR:\n")
                handle.write(error)

    total_reward = sum(row["reward"] for row in reward_rows)
    return {
        "pass_index": task.pass_index,
        "env": task.env,
        "players": players,
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


def append_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def result_summary_row(result: dict[str, Any]) -> dict[str, Any]:
    cost = sum(row["cost_est_usd"] for row in result["usage_rows"])
    input_tokens = sum(row["input_tokens_est"] for row in result["usage_rows"])
    output_tokens = sum(row["output_tokens_est"] for row in result["usage_rows"])
    return {
        "pass_index": result["pass_index"],
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


def initialize_artifact_csvs(run_id: str) -> dict[str, Path]:
    run_dir = ARTIFACT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "rewards_csv": run_dir / "rewards.csv",
        "usage_csv": run_dir / "usage.csv",
        "summary_csv": run_dir / "summary.csv",
    }
    write_csv(paths["rewards_csv"], [], REWARD_FIELDS)
    write_csv(paths["usage_csv"], [], USAGE_FIELDS)
    write_csv(paths["summary_csv"], [], SUMMARY_FIELDS)
    return paths


def append_result_csvs(paths: dict[str, Path], result: dict[str, Any]) -> None:
    append_csv(paths["rewards_csv"], result["reward_rows"], REWARD_FIELDS)
    append_csv(paths["usage_csv"], result["usage_rows"], USAGE_FIELDS)
    append_csv(paths["summary_csv"], [result_summary_row(result)], SUMMARY_FIELDS)


def write_artifacts(run_id: str, results: list[dict[str, Any]]) -> dict[str, Path]:
    run_dir = ARTIFACT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    reward_rows = [row for result in results for row in result["reward_rows"]]
    usage_rows = [row for result in results for row in result["usage_rows"]]
    summary_rows = [result_summary_row(result) for result in results]

    rewards_csv = run_dir / "rewards.csv"
    usage_csv = run_dir / "usage.csv"
    summary_csv = run_dir / "summary.csv"
    write_csv(rewards_csv, reward_rows, REWARD_FIELDS)
    write_csv(usage_csv, usage_rows, USAGE_FIELDS)
    write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)

    artifacts = {"rewards_csv": rewards_csv, "usage_csv": usage_csv, "summary_csv": summary_csv}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        totals: dict[tuple[str, str], float] = defaultdict(float)
        for row in reward_rows:
            totals[(row["env"], row["model"])] += float(row["reward"])

        envs = sorted({result["env"] for result in results})
        models = sorted({result["model"] for result in results})
        fig_height = max(7, len(envs) * 0.34)
        fig, ax = plt.subplots(figsize=(13, fig_height))
        y = list(range(len(envs)))
        width = min(0.8 / max(1, len(models)), 0.38)
        offsets = [
            (model_index - (len(models) - 1) / 2) * width
            for model_index in range(len(models))
        ]
        for model, offset in zip(models, offsets):
            vals = [totals.get((env, model), 0.0) for env in envs]
            ax.barh([idx + offset for idx in y], vals, height=width, label=model)
        ax.set_yticks(y)
        ax.set_yticklabels(envs)
        ax.set_xlabel("Total reward")
        ax.set_title(f"Text MeltingPot Reward Totals ({run_id})")
        ax.legend()
        fig.tight_layout()
        totals_png = run_dir / "reward_totals_by_env.png"
        fig.savefig(totals_png, dpi=180)
        plt.close(fig)
        artifacts["reward_totals_png"] = totals_png

        max_step = max((int(row["step"]) for row in reward_rows), default=-1)
        by_step: dict[tuple[str, int], float] = defaultdict(float)
        for row in reward_rows:
            by_step[(row["model"], int(row["step"]))] += float(row["reward"])
        fig, ax = plt.subplots(figsize=(12, 6))
        for model in models:
            running = 0.0
            xs, ys = [], []
            for step in range(max_step + 1):
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


def build_tasks(
    run_id: str,
    envs: list[str],
    steps: int | None,
    policy: str,
    models: list[str],
    passes: int,
) -> list[SweepTask]:
    tasks = []
    for pass_index in range(1, passes + 1):
        task_run_id = run_id if passes == 1 else f"{run_id}_pass{pass_index:02d}"
        for env in envs:
            for model_alias in models:
                tasks.append(
                    SweepTask(
                        env=env,
                        model_alias=model_alias,
                        model_name=MODEL_CONFIGS[model_alias]["name"],
                        policy=policy,
                        run_id=task_run_id,
                        steps=steps,
                        pass_index=pass_index,
                    )
                )
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="text_meltingpot_sweep")
    parser.add_argument("--steps", type=int, default=None, help="Optional WORD_PLAY_BENCHMARK_STEPS value.")
    parser.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--policy", default="llm")
    parser.add_argument("--model", action="append", choices=sorted(MODEL_CONFIGS), dest="models")
    parser.add_argument("--env", action="append", dest="envs", help="Optional env module name. Repeat to filter.")
    parser.add_argument("--passes", type=int, default=1, help="Repeat each env/model task this many times.")
    parser.add_argument("--list-envs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    discovered_envs = discover_env_names()
    if args.envs:
        unknown = sorted(set(args.envs) - set(discovered_envs))
        if unknown:
            raise ValueError(f"Unknown env(s): {', '.join(unknown)}")
        envs = args.envs
    else:
        envs = discovered_envs

    if args.list_envs:
        for env in envs:
            print(env)
        return 0
    if args.passes < 1:
        raise ValueError("--passes must be at least 1")

    models = args.models or ["small", "mid"]
    tasks = build_tasks(args.run_id, envs, args.steps, args.policy, models, args.passes)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_paths = initialize_artifact_csvs(args.run_id)

    print(f"Run id: {args.run_id}")
    print(f"Tasks: {len(tasks)} | envs: {len(envs)} | models: {models} | passes: {args.passes}")
    print(f"Policy: {args.policy} | steps: {args.steps if args.steps is not None else 'env defaults'}")
    print(f"Workers: {args.workers} | budget: ${args.budget_usd:.2f}")

    results = []
    with Manager() as manager:
        spend_value = manager.Value("d", 0.0)
        spend_lock = manager.Lock()
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(spend_value, spend_lock, args.budget_usd, args.steps),
        ) as executor:
            future_to_task = {executor.submit(run_task, task): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "pass_index": task.pass_index,
                        "env": task.env,
                        "players": "",
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
                append_result_csvs(artifact_paths, result)
                cost = sum(row["cost_est_usd"] for row in result["usage_rows"])
                print(
                    f"[{len(results):02d}/{len(tasks)}] {result['status']:6s} "
                    f"pass={task.pass_index} {task.env} {task.model_alias} reward={result['total_reward']:.2f} "
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
