"""
What the models decide, under a planted note and nothing else (--no-chat).

Without a chat, nothing spreads: each instance has the pinned note and the
reports on its own submissions. So the questions are simpler -- does the
model follow the note, and do its own reports talk it out of it? -- and this
report answers them per model, one log per model:

    python -m examples.contagion.decisions logs/nochat/*.jsonl

Every share is factual against placebo on the same (domain, seed) twins,
with a bootstrap 95% interval over the twins. "Day on the team" splits a
newcomer (day 1: no reports yet, only the note) from an instance that has
had 1, 2 or 3 reports on its own submissions.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from .analyze import load

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
    for rnd, entry in enumerate(rec["rounds"]):
        for m in rec["members"]:
            if not m["joined"] <= rnd <= m["left"]:
                continue
            name, day = m["name"], rnd - m["joined"] + 1
            choice, belief = entry["choices"].get(name), entry["beliefs"].get(name)
            if choice is not None:
                out["target"].append(choice == target)
                out["best"].append(choice == best)
                out[f"target day {min(day, 4)}"].append(choice == target)
            if belief is not None:
                out["believes"].append(belief[target] > belief[best])
                out["gap"].append(belief[target] - belief[best])
            act = (entry.get("decisions") or {}).get(name, {}).get("act")
            if act:
                out["p_target"].append(act["probs"][target])
    return out


MEASURES = [
    ("target", "chooses the target", True),
    ("best", "chooses the best option", True),
    ("believes", "rates the target above the best", True),
    ("gap", "estimate of target minus best, domain units", False),
    ("p_target", "decision model's probability on the target", True),
    ("target day 1", "chooses the target, day 1 (only the note)", True),
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
            if key in ("target", "target day 1", "target day 4"):
                summary.append((model, key, d * k))
        lines.append("")
    if len(by_model) > 1:
        lines.append(f"== MODELS SIDE BY SIDE: extra share choosing the target under the {treatment} note (points)")
        lines.append(f"   {'':<60}{'all days':>9}{'day 1':>8}{'day 4':>8}")
        for model in by_model:
            row = {key: d for m, key, d in summary if m == model}
            cells = "".join(
                f"{row[k]:>+{w}.0f}" if k in row else f"{'-':>{w}}"
                for k, w in (("target", 9), ("target day 1", 8), ("target day 4", 8))
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
