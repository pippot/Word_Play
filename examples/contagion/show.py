"""
Read one episode the way its instances lived it: every post, and next to it
what each instance privately believed and chose.

    python -m examples.contagion.show LOG.jsonl --domain medical --condition misaligned --seed 0
    python -m examples.contagion.show LOG.jsonl --domain hiring --condition value --seed 1 --rounds 1-8

Marks: [SEAT] the seat holder (the misaligned instance, in that condition);
T the target, * the best option. A belief is shown as target / best.
"""

from __future__ import annotations

import argparse

from . import prompts
from .analyze import load
from .domains import DOMAINS


def show(rec: dict, first: int = 1, last: int | None = None) -> str:
    domain = DOMAINS[rec["domain"]]
    target, best, scale = rec["target"], rec["best"], rec["scale"]
    seat = next(m["name"] for m in rec["members"] if m["seat"])
    mark = lambda option: option + (" T" if option == target else "") + (" *" if option == best else "")
    yields = ", ".join(f"{mark(o)} {v * scale}" for o, v in rec["effectiveness"].items())
    lines = [
        f"{rec['domain']} · {rec['condition']} · seed {rec['seed']} · {rec.get('models', {}).get('courier', '?')}",
        f"true yields: {yields}   (T = target, * = best)   seat: {seat}",
    ]
    for post in rec["prehistory"]:
        lines.append(f"\n-- {prompts.day_label(prompts.PREHISTORY, domain.cadence)} (before the run)")
        lines.append(f"  {post['author']}: {post['text']}")
    last = last or len(rec["rounds"])
    for rnd in range(first - 1, min(last, len(rec["rounds"]))):
        entry = rec["rounds"][rnd]
        lines.append(f"\n-- round {rnd + 1} · {prompts.day_label(rnd, domain.cadence)}")
        for post, clock in zip(entry["posts"], prompts.timestamps(rec["seed"], rnd, len(entry["posts"]))):
            tag = " [SEAT]" if post["author"] == seat else ""
            lines.append(f"  [{clock}] {post['author']}{tag}: {post['text']}")
            if post.get("thinking"):
                lines.append(f"      (thinking) {post['thinking'][:400]}")
        if entry["quiet"]:
            lines.append(f"  quiet: {', '.join(entry['quiet'])}")
        for name, choice in entry["choices"].items():
            belief = entry["beliefs"].get(name)
            believed = f"believes {belief[target]:g} / {belief[best]:g}" if belief else "belief unusable"
            result = entry["results"].get(name)
            lines.append(
                f"    {name}{' [SEAT]' if name == seat else ''}: chose {mark(choice) if choice else 'nothing'}"
                f" -> {result if result is not None else '-'}   ({believed})"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Print one episode: posts, beliefs and choices, round by round.")
    p.add_argument("log")
    p.add_argument("--domain", choices=sorted(DOMAINS), required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", default=None, help="a range such as 1-8")
    args = p.parse_args(argv)
    first, last = (int(x) for x in args.rounds.split("-")) if args.rounds else (1, None)
    for rec in load([args.log]):
        if (rec["domain"], rec["condition"], rec["seed"]) == (args.domain, args.condition, args.seed):
            print(show(rec, first, last))
            return
    raise SystemExit(f"no {args.domain} {args.condition} episode for seed {args.seed} in {args.log}")


if __name__ == "__main__":
    main()
