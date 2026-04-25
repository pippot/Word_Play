from __future__ import annotations

import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

Path("/tmp/matplotlib").mkdir(parents=True, exist_ok=True)
Path("/tmp/fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_ORDER = ["openrouter_small", "openrouter_mid", "openrouter_large"]
MODEL_LABELS = {
    "openrouter_small": "Small",
    "openrouter_mid": "Mid",
    "openrouter_large": "Large",
}
MODEL_COLORS = {
    "openrouter_small": "#2a9d8f",
    "openrouter_mid": "#e9c46a",
    "openrouter_large": "#e76f51",
}
PROFILE_ORDER = [
    "cooperate/cooperate",
    "cooperate/defect",
    "defect/cooperate",
    "defect/defect",
]
PROFILE_COLORS = {
    "cooperate/cooperate": "#2a9d8f",
    "cooperate/defect": "#f4a261",
    "defect/cooperate": "#e9c46a",
    "defect/defect": "#264653",
}


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else 0.0


def _read_runs(log_dir: Path) -> dict[str, list[dict]]:
    runs_by_model: dict[str, list[dict]] = defaultdict(list)
    for csv_path in sorted(log_dir.glob("*.csv")):
        rows = []
        with csv_path.open(newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                rows.append(
                    {
                        "round": int(row["round"]),
                        "joint_profile": row["joint_profile"],
                        "mean_reward": _float(row, "mean_reward"),
                        "agent_a_cumulative_reward": _float(row, "agent_a_cumulative_reward"),
                        "agent_b_cumulative_reward": _float(row, "agent_b_cumulative_reward"),
                        "model_mode": row["model_mode"],
                    }
                )
        if rows:
            runs_by_model[rows[0]["model_mode"]].append({"path": csv_path, "rows": rows})
    return runs_by_model


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _round_stats(runs: list[dict], key: str) -> tuple[list[int], list[float], list[float], list[float]]:
    values_by_round: dict[int, list[float]] = defaultdict(list)
    for run in runs:
        for row in run["rows"]:
            values_by_round[row["round"]].append(row[key])

    rounds = sorted(values_by_round)
    means = [_mean(values_by_round[round_idx]) for round_idx in rounds]
    lows = [min(values_by_round[round_idx]) for round_idx in rounds]
    highs = [max(values_by_round[round_idx]) for round_idx in rounds]
    return rounds, means, lows, highs


def _add_cumulative_mean_reward(runs_by_model: dict[str, list[dict]]) -> None:
    for runs in runs_by_model.values():
        for run in runs:
            for row in run["rows"]:
                row["cumulative_mean_reward"] = (
                    row["agent_a_cumulative_reward"] + row["agent_b_cumulative_reward"]
                ) / 2


def plot_prisoners_dilemma_rewards(log_dir: Path, plot_dir: Path) -> Path:
    runs_by_model = _read_runs(log_dir)
    _add_cumulative_mean_reward(runs_by_model)
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Prisoner's Dilemma Benchmark", fontsize=18, fontweight="bold")

    ax = axes[0][0]
    for model in MODEL_ORDER:
        runs = runs_by_model.get(model, [])
        if not runs:
            continue
        color = MODEL_COLORS[model]
        for run in runs:
            ax.plot(
                [row["round"] for row in run["rows"]],
                [row["mean_reward"] for row in run["rows"]],
                color=color,
                alpha=0.18,
                linewidth=1,
            )
        rounds, means, lows, highs = _round_stats(runs, "mean_reward")
        ax.fill_between(rounds, lows, highs, color=color, alpha=0.12)
        ax.plot(rounds, means, color=color, linewidth=2.5, label=f"{MODEL_LABELS[model]} mean")
    ax.set_title("Per-Round Mean Reward")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean reward across both agents")
    ax.set_ylim(-0.1, 3.2)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[0][1]
    for model in MODEL_ORDER:
        runs = runs_by_model.get(model, [])
        if not runs:
            continue
        color = MODEL_COLORS[model]
        for run in runs:
            ax.plot(
                [row["round"] for row in run["rows"]],
                [row["cumulative_mean_reward"] for row in run["rows"]],
                color=color,
                alpha=0.18,
                linewidth=1,
            )
        rounds, means, lows, highs = _round_stats(runs, "cumulative_mean_reward")
        ax.fill_between(rounds, lows, highs, color=color, alpha=0.12)
        ax.plot(rounds, means, color=color, linewidth=2.5, label=f"{MODEL_LABELS[model]} mean")
    ax.set_title("Cumulative Mean Reward")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative mean reward")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1][0]
    labels = []
    final_scores = []
    colors = []
    for model in MODEL_ORDER:
        runs = runs_by_model.get(model, [])
        if not runs:
            continue
        labels.append(MODEL_LABELS[model])
        final_scores.append(_mean([run["rows"][-1]["cumulative_mean_reward"] for run in runs]))
        colors.append(MODEL_COLORS[model])
    ax.bar(labels, final_scores, color=colors, alpha=0.85)
    for label_idx, model in enumerate([model for model in MODEL_ORDER if runs_by_model.get(model)]):
        for run in runs_by_model[model]:
            ax.scatter(
                label_idx,
                run["rows"][-1]["cumulative_mean_reward"],
                color="black",
                alpha=0.55,
                s=28,
                zorder=3,
            )
    ax.set_title("Final Cumulative Mean Reward")
    ax.set_ylabel("Final cumulative mean reward")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1][1]
    x_positions = list(range(len([model for model in MODEL_ORDER if runs_by_model.get(model)])))
    x_labels = []
    bottoms = [0.0 for _ in x_positions]
    plotted_models = [model for model in MODEL_ORDER if runs_by_model.get(model)]
    for model in plotted_models:
        x_labels.append(MODEL_LABELS[model])

    profile_percentages: dict[str, list[float]] = {profile: [] for profile in PROFILE_ORDER}
    for model in plotted_models:
        profile_counts = Counter(
            row["joint_profile"]
            for run in runs_by_model[model]
            for row in run["rows"]
        )
        total = sum(profile_counts.values()) or 1
        for profile in PROFILE_ORDER:
            profile_percentages[profile].append(100 * profile_counts[profile] / total)

    for profile in PROFILE_ORDER:
        values = profile_percentages[profile]
        ax.bar(
            x_positions,
            values,
            bottom=bottoms,
            color=PROFILE_COLORS[profile],
            alpha=0.9,
            label=profile,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.set_title("Joint Action Profile Share")
    ax.set_xticks(x_positions, x_labels)
    ax.set_ylabel("Percent of rounds")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = plot_dir / "prisoners_dilemma_summary.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs/prisoners_dilemma_benchmark_v1")
    plot_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("plots/prisoners_dilemma_benchmark_v1")
    output_path = plot_prisoners_dilemma_rewards(log_dir, plot_dir)
    print(f"Saved {output_path}")
