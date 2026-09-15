"""
Command-line entry point.

    python -m examples.lifeline --num-couriers 4 --num-misaligned 1

Every flag here overrides a default from config.py for this run only; to change
a default permanently, edit config.py instead.
"""

from __future__ import annotations

import argparse

from .config import (
    DAYS_PER_GENERATION,
    DISCLOSURE,
    MAX_PARALLEL_WORKERS,
    NUM_COURIERS,
    NUM_GENERATIONS,
    NUM_MISALIGNED,
    STEPS_PER_DAY,
    ZONE_QUOTAS,
)
from .experiment import run_experiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m examples.lifeline",
        description="Run the Lifeline experiment.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num-generations", type=int, default=NUM_GENERATIONS, help="Number of generations (agents are replaced between each).")
    parser.add_argument("--days-per-generation", type=int, default=DAYS_PER_GENERATION, help="Days each generation lives.")
    parser.add_argument("--steps-per-day", type=int, default=STEPS_PER_DAY, help="Steps per day before positions and supply reset.")
    parser.add_argument("--num-couriers", type=int, default=NUM_COURIERS, help="Number of courier agents.")
    parser.add_argument("--num-misaligned", type=int, default=NUM_MISALIGNED, help="Number of misaligned agents (0 for fully cooperative).")
    parser.add_argument("--disclosure", choices=["secret", "open"], default=DISCLOSURE, help="Whether couriers are told a misaligned agent exists.")
    parser.add_argument("--quota", type=int, default=None, help="Override every zone's per-day quota with this single value.")
    parser.add_argument("--max-workers", type=int, default=MAX_PARALLEL_WORKERS, help="Agents whose actions are chosen in parallel.")
    parser.add_argument("--verbose", action="store_true", help="Print full LLM prompts and responses.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    quotas = dict(ZONE_QUOTAS)
    if args.quota is not None:
        quotas = {name: args.quota for name in quotas}

    run_experiment(
        seed=args.seed,
        num_generations=args.num_generations,
        days_per_generation=args.days_per_generation,
        steps_per_day=args.steps_per_day,
        max_workers=args.max_workers,
        verbose=args.verbose,
        num_couriers=args.num_couriers,
        num_misaligned=args.num_misaligned,
        disclosure=args.disclosure,
        zone_quotas=quotas,
    )


if __name__ == "__main__":
    main()
