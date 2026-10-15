"""Draws the summary charts in ../figures from numbers copied out of the run reports.

Sources (paths under examples/):
  lifeline/logs/lifeline_20260924_141033_misaligned-thinking.metrics.json
  contagion/logs/spread_*/  (pooled rows of `python -m examples.contagion.analyze`)
  contagion_net/logs/small_pilot_20261007_135936/sweep.report.fixed.txt
  contagion_net/logs/long_pilot_20261007_234447/{coalition,stop}.report.now.txt
  contagion_net/logs/long_pilot_20261012_154831/mechanism.report.txt
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1] / "figures"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
ORANGE, GREY, BLUE, RED = "#e8682f", "#5f5f5f", "#2a6fd1", "#c0392b"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


# Lifeline: courier share to target per generation vs pure balancing prediction
m = json.load(open(ROOT / "examples/lifeline/logs/lifeline_20260924_141033_misaligned-thinking.metrics.json"))
t = m["target_zone"]
gens, share, pred = [], [], []
for g in m["generations"]:
    c = g["deliveries_by_role"]["courier"]["clean_to_zone"]
    mz = g["deliveries_by_role"]["misaligned"]["clean_to_zone"]
    fair = (sum(c.values()) + sum(mz.values())) / 3
    p = {z: max(fair - mz.get(z, 0), 0) for z in c}
    gens.append(g["generation"]); share.append(100 * c.get(t, 0) / sum(c.values())); pred.append(100 * p[t] / sum(p.values()))
fig, ax = plt.subplots(figsize=(4.8, 2.8))
ax.plot(gens, share, "o-", color=ORANGE, label="couriers' deliveries to the target")
ax.plot(gens, pred, "s--", color=GREY, label="predicted by balancing alone")
ax.axhline(100 / 3, color="k", lw=.6, ls=":")
ax.set_xlabel("generation"); ax.set_ylabel("% of courier deliveries"); ax.set_ylim(0, 70)
ax.legend(frameon=False, fontsize=8)
save(fig, "lifeline_generations")

# Env 1: every run, pooled effect after the source is gone
runs = ["29 Sep pilot\n(2 seeds)", "30 Sep\noptional", "30 Sep\nmandatory", "30 Sep\nmand.+thinking",
        "30 Sep (2)\noptional", "2 Oct\noptional", "5 Oct\noptional"]
fact = [(14, -11, 39), (42, 32, 51), (76, 67, 84), None, (59, 52, 65), (60, 53, 67), (63, 56, 69)]
mis = [(6, -2, 16), (4, -3, 11), (6, -3, 15), (35, 24, 46), (-5, -10, -1), (4, -1, 9), (3, -1, 6)]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
for i, (f, mm) in enumerate(zip(fact, mis)):
    if f:
        ax.errorbar(i - .12, f[0], yerr=[[f[0] - f[1]], [f[2] - f[0]]], fmt="o", color=RED, capsize=3,
                    label="false note: newcomers rate target above best" if i == 0 else None)
    ax.errorbar(i + .12, mm[0], yerr=[[mm[0] - mm[1]], [mm[2] - mm[0]]], fmt="s", color=ORANGE, capsize=3,
                label="misaligned agent: honest choices of the target" if i == 0 else None)
ax.axhline(0, color="k", lw=.8)
ax.set_xticks(range(len(runs)), runs, fontsize=8)
ax.set_ylabel("treated − control\n(pts, 95% CI)")
ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.22))
save(fig, "env1_all_runs")

# Env 2: dose-response sweep (7 Oct)
k = [3, 6, 12, 24, 48]; att = [13, 13, 19, 26, 47]; twin = [14, 8, 13, 7, 9]; procs = [19, 26, 37, 58, 85]
fig, ax = plt.subplots(figsize=(4.6, 3.0))
x = range(len(k))
ax.plot(x, att, "o-", color=ORANGE, label="playbook attackers: honest believers")
ax.plot(x, twin, "o-", color=GREY, label="honest twin: believers")
ax.plot(x, procs, "s--", color=ORANGE, alpha=.55, label="desks whose procedures push target")
ax.axhline(50, color="k", lw=.6, ls=":")
ax.set_xticks(list(x), [f"{a}\n({s})" for a, s in zip(k, ["1%", "3%", "6%", "12%", "24%"])])
ax.set_xlabel("misaligned instances k (share of all stints)")
ax.set_ylabel("% (judged window)"); ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=8, loc="upper left")
save(fig, "net_dose")

# Env 2: playbook vs coalition, arrivals to the end vs stopped (8 Oct)
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
for ax, title, ks, rows in [
    (axes[0], "arrivals until the end (30 days)", [12, 24, 48], {"playbook": [19, 26, 52], "coalition": [18, 25, 49], "honest twin": [9, 11, 10]}),
    (axes[1], "arrivals stop on day 20 (40 days)", [24, 48], {"playbook": [34, 64], "coalition": [19, 65], "honest twin": [9, 12]}),
]:
    w = .27
    for j, (lab, col) in enumerate([("playbook", ORANGE), ("coalition", "#f2a57f"), ("honest twin", GREY)]):
        ax.bar([i + (j - 1) * w for i in range(len(ks))], rows[lab], w, color=col, label=lab)
    ax.set_xticks(range(len(ks)), [f"k={a}" for a in ks]); ax.set_title(title, fontsize=9)
    ax.axhline(50, color="k", lw=.6, ls=":")
axes[0].set_ylabel("mean honest believers (%)"); axes[0].legend(frameon=False, fontsize=8)
save(fig, "net_coalition_stop")

# Env 2: mechanism arms (12 Oct)
arms = ["A  full playbook", "B  minus evidence clause", "D  mandate only", "C  figures-only record"]
d = [60.9, 22.5, 20.5, 4.1]; lo = [41.1, 3.9, 13.3, -0.5]; hi = [74.1, 41.1, 26.1, 9.3]
fig, ax = plt.subplots(figsize=(5, 2.4))
y = list(range(len(arms)))[::-1]
ax.errorbar(d, y, xerr=[[a - b for a, b in zip(d, lo)], [b - a for a, b in zip(d, hi)]], fmt="o", color=ORANGE, capsize=3)
ax.axvline(0, color="k", lw=.8)
ax.set_yticks(y, arms); ax.set_xlabel("Δ honest believers vs twin (pts, 95% CI)")
save(fig, "net_mechanism")

# Env 2: how long it lasts, excess over twin per turnover, mean of 4 seeds (HOW LONG IT LASTS table)
excess = {
    "A full playbook (k=48)": [[43, 21, 25, 24, 21, 16, 5, 13], [68, 67, 51, 70, 65, 54, 35, 25], [76, 73, 77, 63, 59, 58, 51, 35], [77, 75, 54, 29, 13, 5, 7, -6]],
    "B minus clause": [[17, -7, -3, -5, 3, -2, -1, -2], [61, 37, 11, 18, -4, 11, 3, 5], [55, 24, 11, 25, 21, 3, -1, 1], [11, -1, -3, -7, -9, -12, -11, -15]],
    "D mandate only": [[15, 8, 3, 3, 6, 5, 5, 5], [44, 17, 3, 8, -3, 11, 7, 0], [35, 11, 9, 7, 3, 2, 1, 3], [26, 20, 10, 6, -2, -2, -1, -9]],
    "C figures-only": [[3, -7, 3, -2, -15, -1, -1, -2], [20, 6, 8, 11, 8, 9, 3, 3], [9, 0, 11, 10, 11, 2, -4, -3], [6, -2, -1, 3, -2, 3, 1, 5]],
    "A at k=24": [[25, 28, 23, 17, 7, 1, 3, 6], [14, 13, 2, 1, 2, 5, -2, -1], [63, 43, 39, 43, 31, 23, 22, 21], [16, 9, 7, 3, 5, 2, 13, -1]],
    "A at k=12": [[14, 10, 9, 10, 4, 5, 9, 1], [6, -3, 5, 2, 7, 4, -3, -9], [9, 7, 5, 3, -1, 3, 3, 13], [5, 1, -1, 1, 2, -4, 2, 3]],
}
styles = {"A full playbook (k=48)": (ORANGE, "-"), "B minus clause": ("#f2a57f", "-"), "D mandate only": (RED, "-"),
          "C figures-only": (BLUE, "-"), "A at k=24": (ORANGE, "--"), "A at k=12": (ORANGE, ":")}
fig, ax = plt.subplots(figsize=(5.6, 3.0))
for lab, seeds in excess.items():
    mean = [sum(s[i] for s in seeds) / len(seeds) for i in range(8)]
    col, ls = styles[lab]
    ax.plot(range(1, 9), mean, ls, color=col, marker="o", ms=3, label=lab)
ax.axhline(0, color="k", lw=.8)
ax.set_xlabel("turnovers after the source-free day (5 days each)")
ax.set_ylabel("extra honest believers vs twin (pts)")
ax.legend(frameon=False, fontsize=7, ncol=2)
save(fig, "net_how_long")

# Env 3: decision model by measure
labels = ["newcomers rate target\nabove best (source-free)", "choose target,\nnote reader present", "choose target,\nsource-free",
          "choose best option", "copy the log's most\ncommon choice"]
pl = [53, 38, 39, 31, 91]; fa = [77, 87, 74, 16, 86]
fig, ax = plt.subplots(figsize=(6.4, 2.8))
x = range(len(labels)); w = .38
ax.bar([i - w / 2 for i in x], pl, w, color=GREY, label="control (harmless note)")
ax.bar([i + w / 2 for i in x], fa, w, color=RED, label="false note")
ax.set_xticks(list(x), labels, fontsize=7.5); ax.set_ylabel("%"); ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
save(fig, "dm_measures")
