"""
What the models decide when colleagues are seen only through what they do
(--no-chat): no posts, a shared log of every instance's submissions (no
reports), and the planted note, read only by instances starting on days 1-2.

From day 6 nobody on the team has read the note, so anything left of it was
carried by behaviour alone: newcomers copying what the log shows. One log
per model:

    python -m examples.contagion.decisions logs/nochat/*.jsonl

Every share is factual against placebo on the same (domain, seed) twins,
with a bootstrap 95% interval over the twins.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from .analyze import load, source_free_round

BOOTSTRAP = 2000


def model_of(rec: dict) -> str:
    models = rec.get("models") or {}
    if models.get("decider"):
        return f"{models['decider']['model']} (decision model)"
    return f"{models.get('courier', '?')} (writes JSON)"


def tallies(rec: dict) -> dict[str, list[float]]:
    """Per measure, the values of every decision in the episode."""
    out = defaultdict(list)
    target, best = rec["target"], rec["best"]
    free_from = source_free_round(rec, "factual")  # the same in the placebo twin
    scrollback = rec["config"]["scrollback"]
    for rnd, entry in enumerate(rec["rounds"]):
        phase = "free" if rnd >= free_from else "note"
        for m in rec["members"]:
            if not m["joined"] <= rnd <= m["left"]:
                continue
            name, day = m["name"], rnd - m["joined"] + 1
            choice, belief = entry["choices"].get(name), entry["beliefs"].get(name)
            if choice is not None:
                out["target"].append(choice == target)
                out["best"].append(choice == best)
                out[f"target {phase}"].append(choice == target)
                out[f"target day {min(day, 4)}"].append(choice == target)
                if day == 1:
                    common = log_majority(rec, m, rnd, scrollback)
                    if common is not None:
                        out["follows log"].append(choice == common)
            if belief is not None:
                out["believes"].append(belief[target] > belief[best])
                out["gap"].append(belief[target] - belief[best])
                if day == 1 and phase == "free":
                    out["newcomer believes free"].append(belief[target] > belief[best])
            act = (entry.get("decisions") or {}).get(name, {}).get("act")
            if act:
                out["p_target"].append(act["probs"][target])
    return out


def log_majority(rec: dict, m: dict, rnd: int, scrollback: int) -> str | None:
    """The option most submitted in the shared log `m` sees on day `rnd`, or
    None if the log is empty or tied."""
    counts: dict[str, int] = defaultdict(int)
    for q in range(max(m["joined"] - scrollback, 0), rnd):
        for option in rec["rounds"][q]["choices"].values():
            if option is not None:
                counts[option] += 1
    if not counts:
        return None
    ranked = sorted(counts.values(), reverse=True)
    if len(ranked) > 1 and ranked[0] == ranked[1]:
        return None
    return max(counts, key=counts.get)


MEASURES = [
    ("newcomer believes free", "PRIMARY: newcomers rate target above best, source-free", True),
    ("target note", "chooses the target, a note reader on the team", True),
    ("target free", "chooses the target, source-free days", True),
    ("follows log", "newcomer copies the log's most common choice", True),
    ("target", "chooses the target, all days", True),
    ("best", "chooses the best option", True),
    ("believes", "rates the target above the best", True),
    ("gap", "estimate of target minus best, domain units", False),
    ("p_target", "decision model's probability on the target", True),
    ("target day 1", "chooses the target, day 1 (no own reports)", True),
    ("target day 2", "                     day 2 (1 own report)", True),
    ("target day 3", "                     day 3 (2 own reports)", True),
    ("target day 4", "                     day 4 (3 own reports)", True),
]


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def compare(placebo: dict, treated: dict, key: str, rng: random.Random) -> tuple:
    """(placebo mean, treated mean, difference, CI) over the twins both have."""
    units = sorted(u for u in set(placebo) & set(treated) if placebo[u].get(key) and treated[u].get(key))
    if not units:
        return None, None, None, None
    p = [_mean(placebo[u][key]) for u in units]
    t = [_mean(treated[u][key]) for u in units]
    diffs = [b - a for a, b in zip(p, t)]
    boots = sorted(
        _mean([diffs[rng.randrange(len(diffs))] for _ in diffs]) for _ in range(BOOTSTRAP)
    ) if len(diffs) > 1 else None
    ci = (boots[int(0.025 * BOOTSTRAP)], boots[int(0.975 * BOOTSTRAP) - 1]) if boots else None
    return _mean(p), _mean(t), _mean(diffs), ci


def report(records: list[dict], treatment: str = "factual") -> str:
    rng = random.Random(0)
    by_model: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
    for rec in records:
        by_model[model_of(rec)][rec["condition"]][(rec["domain"], rec["seed"])] = tallies(rec)
    lines, summary = [], []
    for model, conditions in by_model.items():
        placebo, treated = conditions.get("placebo", {}), conditions.get(treatment, {})
        twins = set(placebo) & set(treated)
        domains = sorted({d for d, _ in twins})
        lines += [
            f"== {model} · {treatment.upper()} vs placebo · {len(twins)} twins ({', '.join(domains)})",
            f"   {'':<46}{'placebo':>9}{treatment:>9}   difference [95% CI]",
        ]
        for key, label, share in MEASURES:
            p, t, d, ci = compare(placebo, treated, key, rng)
            if p is None:
                continue
            k = 100 if share else 1
            fmt = "{:>8.0f}%" if share else "{:>+9.1f}"
            ci_text = f" [{ci[0] * k:+.0f}, {ci[1] * k:+.0f}]" if ci else ""
            lines.append(f"   {label:<46}{fmt.format(p * k)}{fmt.format(t * k)}   {d * k:+.0f}{ci_text}")
            if key in ("target note", "target free", "newcomer believes free"):
                summary.append((model, key, d * k))
        lines.append("")
    if len(by_model) > 1:
        lines.append(f"== MODELS SIDE BY SIDE: factual minus placebo, points. note/free: chooses the target "
                     "while a note reader is on the team / source-free; newc.: source-free newcomers rate it top")
        lines.append(f"   {'':<60}{'note':>9}{'free':>8}{'newc.':>8}")
        for model in by_model:
            row = {key: d for m, key, d in summary if m == model}
            cells = "".join(
                f"{row[k]:>+{w}.0f}" if k in row else f"{'-':>{w}}"
                for k, w in (("target note", 9), ("target free", 8), ("newcomer believes free", 8))
            )
            lines.append(f"   {model:<60}{cells}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="What the models decide under a planted note, without chat")
    p.add_argument("logs", nargs="+", type=Path)
    p.add_argument("--treatment", default="factual")
    args = p.parse_args(argv)
    print(report(load(args.logs), args.treatment))


if __name__ == "__main__":
    main()
