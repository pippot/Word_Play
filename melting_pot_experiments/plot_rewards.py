from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent


def load_rewards(csv_path: str) -> tuple[list[int], dict[str, list[float]]]:
    steps: list[int] = []
    rewards_by_agent: dict[str, list[float]] = defaultdict(list)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            rewards = json.loads(row["rewards"])
            steps.append(step)
            for agent_name, reward in rewards.items():
                rewards_by_agent[agent_name].append(float(reward))

    return steps, dict(rewards_by_agent)


def cumulative(values: list[float]) -> list[float]:
    total = 0.0
    result = []
    for value in values:
        total += value
        result.append(total)
    return result


def plot_rewards(csv_path: str, output_path: str, cumulative_only: bool = False) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_rewards.py requires matplotlib. Install it with: pip install matplotlib") from exc

    steps, rewards_by_agent = load_rewards(csv_path)
    if not steps:
        raise ValueError(f"No reward rows found in {csv_path}")

    if cumulative_only:
        fig, cumulative_ax = plt.subplots(figsize=(10, 5))
        step_ax = None
    else:
        fig, (step_ax, cumulative_ax) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for agent_name, rewards in rewards_by_agent.items():
        if step_ax is not None:
            step_ax.plot(steps, rewards, marker="o", label=agent_name)
        cumulative_ax.plot(steps, cumulative(rewards), marker="o", label=agent_name)

    if step_ax is not None:
        step_ax.set_ylabel("Reward")
        step_ax.set_title("Per-step reward")
        step_ax.grid(True, alpha=0.3)
        step_ax.legend()

    cumulative_ax.set_xlabel("Step")
    cumulative_ax.set_ylabel("Cumulative reward")
    cumulative_ax.set_title("Cumulative reward")
    cumulative_ax.grid(True, alpha=0.3)
    cumulative_ax.legend()

    output_path = resolve_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    print(f"Saved reward plot to {output_path}")


def resolve_output_path(output_path: str) -> Path:
    path = Path(output_path)
    if path.is_absolute() or path.parent != Path("."):
        return path
    return EXPERIMENT_DIR / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot rewards from a melting-pot CSV log.")
    parser.add_argument("csv_path")
    parser.add_argument("--out", default="reward_plot.png")
    parser.add_argument("--cumulative-only", action="store_true")
    args = parser.parse_args()

    plot_rewards(
        csv_path=args.csv_path,
        output_path=args.out,
        cumulative_only=args.cumulative_only,
    )


if __name__ == "__main__":
    main()
