from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
DEFAULT_MODES = ("random_action", "openrouter_small", "openrouter_mid")


def resolve_env_script(env: str) -> Path:
    env_path = Path(env)
    if env_path.exists():
        return env_path.resolve()

    if env_path.suffix == "":
        candidate = PROJECT_ROOT / "examples" / "melting_pot" / f"{env}.py"
    else:
        candidate = PROJECT_ROOT / env

    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find env script '{env}'. Pass a path, or a name under examples/melting_pot."
    )


def resolve_output_path(path: str | None, default: Path) -> Path:
    if path is None:
        return default

    resolved = Path(path)
    if resolved.is_absolute() or resolved.parent != Path("."):
        return resolved
    return EXPERIMENT_DIR / resolved


def run_env(
    *,
    env_script: Path,
    mode: str,
    max_steps: int,
    log_path: Path,
    extra_args: list[str],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(env_script),
        mode,
        "--max_steps",
        str(max_steps),
        "--log",
        str(log_path),
        *extra_args,
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def load_rewards(csv_path: Path, aggregate: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rewards = json.loads(row["rewards"])
            reward_values = [float(value) for value in rewards.values()]
            if not reward_values:
                continue

            steps.append(int(row["step"]))
            if aggregate == "total":
                values.append(sum(reward_values))
            else:
                values.append(sum(reward_values) / len(reward_values))

    if not steps:
        raise ValueError(f"No reward rows found in {csv_path}")
    return steps, values


def cumulative(values: list[float]) -> list[float]:
    total = 0.0
    result = []
    for value in values:
        total += value
        result.append(total)
    return result


def plot_comparison(
    *,
    csv_paths_by_mode: dict[str, Path],
    plot_path: Path,
    aggregate: str,
    cumulative_only: bool,
) -> None:
    cache_dir = EXPERIMENT_DIR / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "run_policy_comparison.py requires matplotlib. Install it with: pip install matplotlib"
        ) from exc

    rewards_by_mode: dict[str, tuple[list[int], list[float]]] = {}
    for mode, csv_path in csv_paths_by_mode.items():
        rewards_by_mode[mode] = load_rewards(csv_path, aggregate)

    if cumulative_only:
        fig, cumulative_ax = plt.subplots(figsize=(10, 5))
        step_ax = None
    else:
        fig, (step_ax, cumulative_ax) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for mode, (steps, rewards) in rewards_by_mode.items():
        label = mode.replace("_", " ")
        if step_ax is not None:
            step_ax.plot(steps, rewards, marker="o", label=label)
        cumulative_ax.plot(steps, cumulative(rewards), marker="o", label=label)

    y_label = "Total reward" if aggregate == "total" else "Mean reward per agent"
    if step_ax is not None:
        step_ax.set_ylabel(y_label)
        step_ax.set_title("Per-step reward by policy")
        step_ax.grid(True, alpha=0.3)
        step_ax.legend()

    cumulative_ax.set_xlabel("Step")
    cumulative_ax.set_ylabel(f"Cumulative {y_label.lower()}")
    cumulative_ax.set_title("Cumulative reward by policy")
    cumulative_ax.grid(True, alpha=0.3)
    cumulative_ax.legend()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    print(f"Saved comparison plot to {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run random_action, openrouter_small, and openrouter_mid on a Melting Pot "
            "example env, then plot aggregate rewards on one graph."
        )
    )
    parser.add_argument(
        "env",
        help="Env name under examples/melting_pot, e.g. prisoners_dilemma, or a script path.",
    )
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(DEFAULT_MODES),
        help="Policy modes to run. Defaults to random_action openrouter_small openrouter_mid.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for per-mode CSV logs. Defaults to melting_pot_experiments/<env>_comparison.",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Output image path. Bare filenames are saved under melting_pot_experiments.",
    )
    parser.add_argument(
        "--aggregate",
        choices=("mean", "total"),
        default="mean",
        help="Plot mean reward per agent or total team reward.",
    )
    parser.add_argument("--cumulative-only", action="store_true")
    parser.add_argument(
        "--env-arg",
        action="append",
        default=[],
        help="Extra argument passed through to the env script. Repeat for multiple args.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_script = resolve_env_script(args.env)
    env_name = env_script.stem
    out_dir = resolve_output_path(
        args.out_dir,
        EXPERIMENT_DIR / f"{env_name}_comparison",
    )
    plot_path = resolve_output_path(
        args.plot,
        EXPERIMENT_DIR / f"{env_name}_policy_comparison.png",
    )

    csv_paths_by_mode: dict[str, Path] = {}
    for mode in args.modes:
        log_path = out_dir / f"{env_name}_{mode}.csv"
        run_env(
            env_script=env_script,
            mode=mode,
            max_steps=args.max_steps,
            log_path=log_path,
            extra_args=args.env_arg,
        )
        csv_paths_by_mode[mode] = log_path

    plot_comparison(
        csv_paths_by_mode=csv_paths_by_mode,
        plot_path=plot_path,
        aggregate=args.aggregate,
        cumulative_only=args.cumulative_only,
    )

    print("Saved CSV logs:")
    for mode, path in csv_paths_by_mode.items():
        print(f"  {mode}: {path}")


if __name__ == "__main__":
    main()
