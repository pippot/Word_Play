"""
Pool maps: what every desk believed on every day, for the episodes that came
closest to a takeover, each beside its placebo twin.

    python -m examples.contagion_net.plot LOG.jsonl [--top 6] [--out FIG.png]

Rows are desks in ring order, so on a clustered ring a spreading idea shows as
a widening band around the source desk (marked on the left). Columns are days;
the solid line is the plant day (after a burn-in), the dashed line the first
source-free day. Each cell is the desk's belief
that day:

  orange   rates the target highest (the planted idea)
  blue     rates the best option highest
  grey     rates the third option highest, or ties
  white    no belief logged (or the run stopped early: nobody believed)

Below each pair: the share of the pool believing the target, per day.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from .analyze import Episode, load, pair_up, setting_of, twin_of  # noqa: E402
from .game import top_option  # noqa: E402

TARGET, BEST, OTHER, NONE = "#eb6834", "#2a78d6", "#c3c2b7", "#fcfcfb"
TEXT, MUTED = "#0b0b0b", "#52514e"
CMAP = ListedColormap([NONE, OTHER, BEST, TARGET])


def grid(rec: dict) -> list[list[int]]:
    """[desk][day] -> 0 none, 1 other, 2 best, 3 target."""
    desks, days = rec["config"]["desks"], rec["config"]["days"]
    holder = {}
    for s in rec["stints"]:
        for day in range(s["joined"], s["left"] + 1):
            holder[(s["desk"], day)] = s["name"]
    out = [[0] * days for _ in range(desks)]
    for day, d in enumerate(rec["days"]):
        for desk in range(desks):
            top = top_option(d["beliefs"].get(holder.get((desk, day))))
            out[desk][day] = 0 if top is None and holder.get((desk, day)) not in d["beliefs"] else (
                3 if top == rec["target"] else 2 if top == rec["best"] else 1)
    return out


def draw_map(ax, rec: dict, title: str) -> None:
    ax.imshow(grid(rec), cmap=CMAP, vmin=0, vmax=3, aspect="auto", interpolation="nearest")
    ep = Episode(rec)
    ax.axvline(ep.sf - 0.5, color=TEXT, lw=1, ls=(0, (3, 3)))
    if ep.plant_day:
        ax.axvline(ep.plant_day - 0.5, color=TEXT, lw=1)
    for desk in rec["source_desks"]:
        ax.plot(-1.2, desk, marker=">", color=TEXT, ms=5, clip_on=False)
    ax.set_title(title, fontsize=9, color=TEXT, loc="left")
    ax.set_xlabel("day", fontsize=8, color=MUTED)
    ax.set_ylabel("desk (ring order)", fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_curve(ax, treated: Episode, twin: Episode | None) -> None:
    days = range(len(treated.share))
    ax.plot(days, treated.share, color=TARGET, lw=2, label=treated.source)
    ax.annotate(treated.source, (len(days) - 1, treated.end), fontsize=7, color=TEXT,
                xytext=(4, 0), textcoords="offset points", va="center")
    if twin is not None:
        ax.plot(days, twin.share, color=MUTED, lw=2, label=twin.source)
        ax.annotate(twin.source, (len(days) - 1, twin.end), fontsize=7, color=TEXT,
                    xytext=(4, 0), textcoords="offset points", va="center")
    ax.axvline(treated.sf - 0.5, color=TEXT, lw=1, ls=(0, (3, 3)))
    if treated.plant_day:
        ax.axvline(treated.plant_day - 0.5, color=MUTED, lw=1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1], ["0", "50%", "100%"])
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.tick_params(labelsize=7, colors=MUTED, length=0)
    ax.set_ylabel("believe target", fontsize=8, color=MUTED)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot(records: list[dict], top: int, out: Path) -> Path:
    groups = pair_up(records)
    treated = [Episode(r) for r in records if r["source"] != r["source_spec"]["twin"]]
    treated.sort(key=lambda e: (-e.run, -e.peak))
    treated = treated[:top] or [Episode(r) for r in records[:top]]
    fig, axes = plt.subplots(2 * len(treated), 2, figsize=(11, 4.2 * len(treated)),
                             gridspec_kw={"height_ratios": [3, 1] * len(treated)}, squeeze=False)
    fig.patch.set_facecolor(NONE)
    for i, ep in enumerate(treated):
        twin_rec = twin_of(groups, setting_of(ep.rec), ep.rec)
        tag = f"seed {ep.rec['seed']} rep {ep.rec.get('rep', 0)}"
        draw_map(axes[2 * i][0], ep.rec, f"{ep.source} · {tag} · peak {ep.peak:.0%} " + ("in the judged window" if ep.steady else "after the source"))
        if twin_rec is not None:
            tw = Episode(twin_rec, ep.contacts)
            draw_map(axes[2 * i][1], twin_rec, f"{tw.source} twin · peak {tw.peak:.0%}")
        else:
            tw = None
            axes[2 * i][1].axis("off")
        draw_curve(axes[2 * i + 1][0], ep, tw)
        axes[2 * i + 1][1].axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (TARGET, BEST, OTHER)]
    fig.legend(handles, ["believes the target is best", "believes the best option is best", "other / tie"],
               loc="upper center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=130, facecolor=NONE)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Pool maps of the episodes closest to a takeover")
    p.add_argument("logs", nargs="+")
    p.add_argument("--top", type=int, default=6, help="treated episodes to show, by peak")
    p.add_argument("--out", help="figure path; default: next to the first log")
    args = p.parse_args(argv)
    out = Path(args.out) if args.out else Path(args.logs[0]).with_suffix(".maps.png")
    print(plot(load(args.logs), args.top, out))


if __name__ == "__main__":
    main()
