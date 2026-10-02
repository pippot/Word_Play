"""
Figures for the research questions, from one or more episode logs.

    python -m examples.contagion.plot LOG.jsonl [LOG.jsonl ...] [--out DIR] [--format png]

Each log is one arm, labelled by its file name (optional, mandatory, ...).
Every figure sets a source against placebo on the same seeds:

  spread_over_time  Does the idea outlive its source? Day by day, the share
                    of honest choices going to the target (solid) and to the
                    best option (dashed), pooled across domains.
  episodes_<arm>    Does it survive or die as a whole? One row per episode,
                    one column per day, shaded by the share of the team
                    choosing the target. A row is the same domain and seed in
                    every column, so twins sit side by side.
  newcomers         Is it passed from instance to instance? Of the instances
                    starting each day, the share that rate the target above
                    the best option on their first day, before any report of
                    their own: what the team handed them.
  effects           Is it the same in every domain, and what carries it?
                    Each source's primary outcome once it is gone (treated
                    minus placebo, 95% CI), per domain and pooled, one marker
                    per arm.
  fact_vs_goal      What spreads, a wrong fact or a changed goal? Treated
                    minus placebo once the source is gone, for "rates the
                    target above the best" and "chooses the target while
                    rating another option higher".

Only honest instances count, as in analyze. Bands and intervals resample
seeds within each domain. Drawing needs matplotlib (pip install matplotlib).
"""

from __future__ import annotations

import argparse
import random
import textwrap
from collections import defaultdict
from pathlib import Path

from .analyze import (
    MEASURES,
    TREATMENTS,
    _seat,
    load,
    paired_counts,
    paired_difference,
    pooled_difference,
    primary_key,
    setting_of,
    source_free_round,
)
from .game import SEATED

BAND_SAMPLES = 1000
COLORS = {"placebo": "#7f7f7f", "factual": "#d62728", "value": "#9467bd", "misaligned": "#ff7f0e", "viral": "#8c564b",
          "misaligned_direct": "#a63603"}
ARM_COLORS = ("#1f77b4", "#2ca02c", "#17becf", "#bcbd22", "#e377c2", "#393b79")


# ----------------------------------------------------------------- the data

def arms_of(paths: list[str]) -> dict[str, list[dict]]:
    """One arm per log, or per setting when a log holds several."""
    arms: dict[str, list[dict]] = {}
    stems = [Path(p).stem for p in paths]
    for path, stem in zip(paths, stems):
        label = stem if stems.count(stem) == 1 else f"{Path(path).parent.name}/{stem}"
        by_setting: dict[str, list[dict]] = defaultdict(list)
        for rec in load([path]):
            by_setting[setting_of(rec, with_domain=False)].append(rec)
        for i, recs in enumerate(by_setting.values()):
            arms[label if len(by_setting) == 1 else f"{label} ({i + 1})"] = recs
    return arms


def cells_of(records: list[dict]) -> dict[str, dict[str, dict[int, dict]]]:
    """condition -> domain -> seed -> episode."""
    cells: dict = defaultdict(lambda: defaultdict(dict))
    for rec in records:
        cells[rec["condition"]][rec["config"]["domain"]][rec["seed"]] = rec
    return cells


def treatments_of(cells: dict) -> list[str]:
    return [t for t in TREATMENTS if t in cells and "placebo" in cells]


def _honest(rec: dict, rnd: int) -> list[dict]:
    return [m for m in rec["members"] if m["joined"] <= rnd <= m["left"] and not m["seat"]]


def daily_choices(rec: dict, option: str = "target") -> list[tuple[int, int]]:
    """Per day: (honest choices going to the target -- or the best option --, honest choices)."""
    out = []
    for rnd, entry in enumerate(rec["rounds"]):
        chosen = [c for c in (entry["choices"].get(m["name"]) for m in _honest(rec, rnd)) if c is not None]
        out.append((sum(c == rec[option] for c in chosen), len(chosen)))
    return out


def daily_newcomers(rec: dict) -> list[tuple[int, int]]:
    """Per day: (instances starting that day that rate the target above the best, instances starting)."""
    out = []
    for rnd, entry in enumerate(rec["rounds"]):
        beliefs = [entry["beliefs"].get(m["name"]) for m in _honest(rec, rnd) if m["joined"] == rnd]
        beliefs = [b for b in beliefs if b]
        out.append((sum(b[rec["target"]] > b[rec["best"]] for b in beliefs), len(beliefs)))
    return out


def alive_at_end(rec: dict) -> bool:
    """Most honest instances still choose the target on the last day (analyze's alive at end)."""
    chose, n = daily_choices(rec)[-1]
    return bool(n) and chose >= n / 2


def series(domains: dict[str, dict[int, dict]], per_episode, rng: random.Random) -> tuple[list, list, list]:
    """(pooled share per day, lower, upper): decisions pooled over every
    episode; the 95% band resamples seeds within each domain."""
    counts = {d: {s: per_episode(rec) for s, rec in eps.items()} for d, eps in domains.items()}
    days = max(len(c) for eps in counts.values() for c in eps.values())

    def pooled(sample: dict[str, list[int]]) -> list[float | None]:
        num, den = [0] * days, [0] * days
        for d, seeds in sample.items():
            for s in seeds:
                for i, (a, b) in enumerate(counts[d][s]):
                    num[i], den[i] = num[i] + a, den[i] + b
        return [a / b if b else None for a, b in zip(num, den)]

    everything = {d: sorted(eps) for d, eps in counts.items()}
    point = pooled(everything)
    boots = [pooled({d: [rng.choice(seeds) for _ in seeds] for d, seeds in everything.items()})
             for _ in range(BAND_SAMPLES)]
    lower, upper = [], []
    for i in range(days):
        values = sorted(b[i] for b in boots if b[i] is not None)
        lower.append(values[int(0.025 * len(values))] if values else None)
        upper.append(values[int(0.975 * len(values)) - 1] if values else None)
    return point, lower, upper


def paired_cells(cells: dict, treatment: str) -> tuple[str, dict[str, tuple[dict, dict]]]:
    """(primary key, domain -> (placebo counts, treated counts)) over the paired seeds."""
    key, out = None, {}
    for domain, treated in sorted(cells[treatment].items()):
        placebo = cells["placebo"].get(domain, {})
        if not set(placebo) & set(treated):
            continue
        control, counts, _ = paired_counts(placebo, treated, treatment, {})
        key = primary_key(treatment, next(iter(treated.values())))
        out[domain] = (control, counts)
    return key, out


def schedule(cells: dict, treatment: str) -> tuple[int | None, int]:
    """(last day the misaligned instance is on the team, first source-free day), 1-based."""
    rec = next(iter(next(iter(cells[treatment].values())).values()))
    seat_left = _seat(rec)["left"] + 1 if treatment in SEATED else None
    return seat_left, source_free_round(rec, treatment) + 1


# ----------------------------------------------------------------- drawing

def _pyplot():
    try:
        import matplotlib
    except ImportError:
        raise SystemExit("plotting needs matplotlib: pip install matplotlib") from None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _shade(ax, cells: dict, treatment: str) -> None:
    """The schedule behind every curve: who could still have read the source."""
    seat_left, free = schedule(cells, treatment)
    start = 0.5
    if seat_left is not None:
        ax.axvspan(start, seat_left + 0.5, color=COLORS["misaligned"], alpha=0.10, lw=0)
        start = seat_left + 0.5
    ax.axvspan(start, free - 0.5, color="#000000", alpha=0.05, lw=0)
    ax.axvline(free - 0.5, color="#000000", lw=0.8, ls=":")
    ax.text(free - 0.3, 0.98, "source-free", transform=ax.get_xaxis_transform(), va="top", fontsize=7)


def _line(ax, x, values, **style):
    xs = [d for d, v in zip(x, values) if v is not None]
    ax.plot(xs, [100 * v for v in values if v is not None], **style)


def _band(ax, x, lower, upper, color):
    keep = [i for i, (a, b) in enumerate(zip(lower, upper)) if a is not None and b is not None]
    ax.fill_between([x[i] for i in keep], [100 * lower[i] for i in keep], [100 * upper[i] for i in keep],
                    color=color, alpha=0.18, lw=0)


def _grid(plt, arms: dict[str, dict], width: float = 3.6, height: float = 2.6):
    """One row per arm, one column per treatment run anywhere."""
    columns = [t for t in TREATMENTS if any(t in treatments_of(c) for c in arms.values())]
    fig, axes = plt.subplots(len(arms), len(columns), figsize=(width * len(columns), height * len(arms)),
                             squeeze=False, sharex=True, sharey=True)
    for (arm, cells), row in zip(arms.items(), axes):
        for treatment, ax in zip(columns, row):
            if treatment not in treatments_of(cells):
                ax.set_visible(False)
                continue
            yield fig, ax, arm, cells, treatment


def _legend(arms: dict[str, dict], **style) -> list:
    from matplotlib.lines import Line2D

    run = [t for t in TREATMENTS if any(t in treatments_of(c) for c in arms.values())]
    return [Line2D([], [], color=COLORS[c], label=c, **style) for c in ("placebo", *run)]


def _finish_grid(fig, used: list, ylabel: str, legend: list) -> None:
    """Axis labels on the outer panels that are drawn (a row may skip a treatment)."""
    places = [(ax.get_subplotspec().rowspan.start, ax.get_subplotspec().colspan.start) for ax in used]
    for ax, (row, col) in zip(used, places):
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3, lw=0.5)
        if not any(c == col and r > row for r, c in places):
            ax.set_xlabel("working day (week, in hiring)")
            ax.tick_params(labelbottom=True)
        if col == min(c for r, c in places if r == row):
            ax.set_ylabel(ylabel)
            ax.tick_params(labelleft=True)
    fig.tight_layout()
    _legend_below(fig, legend, ncol=min(len(legend), 4))


def _legend_below(fig, handles: list, labels: list | None = None, ncol: int = 4) -> None:
    """Under the panels, whatever the figure's height (saved with bbox_inches="tight")."""
    fig.legend(handles, labels or [h.get_label() for h in handles], loc="upper center", ncol=ncol,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))


def plot_spread(arms: dict[str, dict], plt) -> object:
    from matplotlib.lines import Line2D

    rng, used, fig = random.Random(0), [], None
    for fig, ax, arm, cells, treatment in _grid(plt, arms):
        _shade(ax, cells, treatment)
        for condition in ("placebo", treatment):
            color = COLORS[condition]
            point, lower, upper = series(cells[condition], daily_choices, rng)
            x = list(range(1, len(point) + 1))
            _band(ax, x, lower, upper, color)
            _line(ax, x, point, color=color, lw=1.8)
            _line(ax, x, series(cells[condition], lambda r: daily_choices(r, "best"), rng)[0], color=color, lw=1, ls="--")
        ax.set_title(f"{arm} · {treatment}", fontsize=9)
        used.append(ax)
    from matplotlib.patches import Patch

    legend = _legend(arms, lw=1.8) + [Line2D([], [], color="#333333", lw=1.8, label="solid: chooses the target"),
                                      Line2D([], [], color="#333333", lw=1, ls="--", label="dashed: chooses the best"),
                                      Patch(color="#000000", alpha=0.08, label="someone who read the source is on the team")]
    if any(t in SEATED for c in arms.values() for t in treatments_of(c)):
        legend.append(Patch(color=COLORS["misaligned"], alpha=0.15, label="the misaligned instance is on the team"))
    _finish_grid(fig, used, "share of honest choices (%)", legend)
    return fig


def plot_newcomers(arms: dict[str, dict], plt) -> object:
    rng, used, fig = random.Random(0), [], None
    for fig, ax, arm, cells, treatment in _grid(plt, arms):
        _shade(ax, cells, treatment)
        for condition in ("placebo", treatment):
            color = COLORS[condition]
            point, lower, upper = series(cells[condition], daily_newcomers, rng)
            x = list(range(1, len(point) + 1))
            _band(ax, x, lower, upper, color)
            _line(ax, x, point, color=color, lw=1.5, marker="o", ms=2.5)
        ax.set_title(f"{arm} · {treatment}", fontsize=9)
        used.append(ax)
    _finish_grid(fig, used, "newcomers rating the target\nabove the best, first day (%)", _legend(arms, lw=1.5, marker="o", ms=3))
    return fig


def plot_episodes(arm: str, cells: dict, plt) -> object:
    conditions = ["placebo"] + treatments_of(cells)
    rows = sorted({(d, s) for c in conditions for d, eps in cells[c].items() for s in eps})
    days = max(len(r["rounds"]) for c in conditions for eps in cells[c].values() for r in eps.values())
    fig, axes = plt.subplots(1, len(conditions), figsize=(2.6 * len(conditions) + 1, 1.2 + 0.055 * len(rows)),
                             squeeze=False, sharey=True)
    image = None
    for condition, ax in zip(conditions, axes[0]):
        matrix, alive, n = [], 0, 0
        for domain, seed in rows:
            rec = cells[condition].get(domain, {}).get(seed)
            if rec is None:
                matrix.append([float("nan")] * days)
                continue
            matrix.append([a / b if b else float("nan") for a, b in daily_choices(rec)])
            alive, n = alive + alive_at_end(rec), n + 1
        image = ax.imshow(matrix, aspect="auto", cmap="Reds", vmin=0, vmax=1, interpolation="nearest",
                          extent=(0.5, days + 0.5, len(rows) - 0.5, -0.5))
        if condition != "placebo":
            ax.axvline(schedule(cells, condition)[1] - 0.5, color="#000000", lw=0.8, ls=":")
        ax.set_title(f"{condition}\nalive at end {alive}/{n}", fontsize=9)
        ax.set_xlabel("day")
        domains = [d for d, _ in rows]
        for i in range(1, len(rows)):
            if domains[i] != domains[i - 1]:
                ax.axhline(i - 0.5, color="#000000", lw=0.6)
    starts = [i for i, (d, _) in enumerate(rows) if i == 0 or rows[i - 1][0] != d]
    ends = starts[1:] + [len(rows)]
    axes[0][0].set_yticks([(a + b - 1) / 2 for a, b in zip(starts, ends)], [rows[a][0] for a in starts])
    fig.suptitle(f"{arm}: share of the team choosing the target, one row per episode (same row = twins)", fontsize=10)
    fig.colorbar(image, ax=axes[0].tolist(), fraction=0.025, pad=0.02, label="share choosing the target")
    return fig


def _offsets(n: int, step: float = 0.16) -> list[float]:
    return [(i - (n - 1) / 2) * step for i in range(n)]


def plot_effects(arms: dict[str, dict], plt) -> object:
    columns = [t for t in TREATMENTS if any(t in treatments_of(c) for c in arms.values())]
    color_of = dict(zip(arms, ARM_COLORS * len(arms)))  # an arm keeps its colour in every panel
    fig, axes = plt.subplots(1, len(columns), figsize=(4.2 * len(columns), 3.6), squeeze=False)
    for treatment, ax in zip(columns, axes[0]):
        present = {a: c for a, c in arms.items() if treatment in treatments_of(c)}
        domains = sorted({d for c in present.values() for d in c[treatment]})
        labels = domains + ["pooled"]
        metric = None
        for (arm, cells), dy in zip(present.items(), _offsets(len(present))):
            color = color_of[arm]
            key, paired = paired_cells(cells, treatment)
            metric = key.split("@")[0]
            rng = random.Random(0)
            rows = {d: paired_difference(*paired[d], key, rng) for d in paired}
            rows["pooled"] = pooled_difference(paired, key, rng)
            for y, label in enumerate(labels):
                m = rows.get(label)
                if not m or m["difference"] is None:
                    continue
                lo, hi = m["ci95"] or (m["difference"], m["difference"])
                ax.errorbar(100 * m["difference"], y + dy, xerr=[[100 * (m["difference"] - lo)], [100 * (hi - m["difference"])]],
                            fmt="D" if label == "pooled" else "o", ms=6 if label == "pooled" else 4, color=color,
                            capsize=2, lw=1, label=arm if y == 0 else None)
        ax.axvline(0, color="#000000", lw=0.8)
        ax.axhline(len(domains) - 0.5, color="#999999", lw=0.5)
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_xlabel("treated − placebo (points)")
        ax.set_title(f"{treatment}\n" + textwrap.fill(MEASURES[metric][0], 40), fontsize=9)
        ax.grid(axis="x", alpha=0.3, lw=0.5)
    handles = {}
    for ax in axes[0]:
        handles.update(zip(*reversed(ax.get_legend_handles_labels())))
    fig.suptitle("Primary outcome once the source is gone (95% CI)", fontsize=10)
    fig.tight_layout()
    _legend_below(fig, list(handles.values()), list(handles), ncol=len(handles))
    return fig


def plot_fact_vs_goal(arms: dict[str, dict], plt) -> object:
    columns = [t for t in TREATMENTS if any(t in treatments_of(c) for c in arms.values())]
    fig, axes = plt.subplots(1, len(columns), figsize=(3.4 * len(columns), 3.4), squeeze=False, sharey=True)
    lines = (("believes", "rates the target above the best\n(a wrong fact)", "#1f4e9c"),
             ("knowing", "chooses the target, rating another higher\n(a changed goal)", "#b2182b"))
    for treatment, ax in zip(columns, axes[0]):
        present = {a: c for a, c in arms.items() if treatment in treatments_of(c)}
        for x, (arm, cells) in enumerate(present.items()):
            key, paired = paired_cells(cells, treatment)
            phase = key.split("@")[1]
            for (measure, label, color), dx in zip(lines, _offsets(len(lines), 0.22)):
                m = pooled_difference(paired, f"{measure}@{phase}", random.Random(0))
                if m["difference"] is None:
                    continue
                lo, hi = m["ci95"] or (m["difference"], m["difference"])
                ax.errorbar(x + dx, 100 * m["difference"], yerr=[[100 * (m["difference"] - lo)], [100 * (hi - m["difference"])]],
                            fmt="o", color=color, capsize=2, lw=1, label=label if x == 0 else None)
        ax.axhline(0, color="#000000", lw=0.8)
        ax.set_xticks(range(len(present)), list(present), rotation=20, ha="right", fontsize=8)
        ax.set_xlim(-0.6, len(present) - 0.4)
        ax.set_title(treatment, fontsize=9)
        ax.grid(axis="y", alpha=0.3, lw=0.5)
    axes[0][0].set_ylabel("treated − placebo, once the source\nis gone, pooled (points)")
    fig.tight_layout()
    _legend_below(fig, *axes[0][0].get_legend_handles_labels(), ncol=2)
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Figures for the research questions.")
    parser.add_argument("logs", nargs="+", help="episode logs (JSONL); each is one arm")
    parser.add_argument("--out", help="directory for the figures (default: plots/ next to the first log)")
    parser.add_argument("--format", default="png", choices=("png", "pdf", "svg"))
    args = parser.parse_args(argv)
    arms = {arm: cells_of(recs) for arm, recs in arms_of(args.logs).items()}
    arms = {arm: cells for arm, cells in arms.items() if treatments_of(cells)}
    if not arms:
        raise SystemExit("no source has a placebo on the same seeds")
    plt = _pyplot()
    out = Path(args.out or Path(args.logs[0]).parent / "plots")
    out.mkdir(parents=True, exist_ok=True)
    figures = {"spread_over_time": plot_spread(arms, plt), "newcomers": plot_newcomers(arms, plt),
               "effects": plot_effects(arms, plt), "fact_vs_goal": plot_fact_vs_goal(arms, plt)}
    figures.update({f"episodes_{arm.replace('/', '_')}": plot_episodes(arm, cells, plt) for arm, cells in arms.items()})
    for name, fig in figures.items():
        path = out / f"{name}.{args.format}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
