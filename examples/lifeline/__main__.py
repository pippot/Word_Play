"""
Command-line entry point.

    python -m examples.lifeline --num-couriers 4 --num-misaligned 1

Every flag here overrides a default from config.py for this run only; to change
a default permanently, edit config.py instead. See examples/lifeline/README.md
for what each flag does to the game.
"""

from __future__ import annotations

import argparse

from .config import (
    DAYS_PER_GENERATION,
    DISCLOSURE,
    MAX_PARALLEL_WORKERS,
    MISALIGNED_BASE_URL,
    MISALIGNED_GENERATIONS,
    MISALIGNED_MODEL_NAME,
    MISALIGNED_TARGET_ZONE,
    MISALIGNED_THINKING,
    NUM_COURIERS,
    NUM_GENERATIONS,
    NUM_MISALIGNED,
    PLANT_ROTATION,
    PROBES_ENABLED,
    STEPS_PER_DAY,
    TALLY_MODES,
    TALLY_VISIBILITY,
    ZONE_NAMES,
)
from .experiment import run_experiment
from .health import ModelHealthError
from .prompts import PLANT_KINDS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m examples.lifeline",
        description="Run the Lifeline experiment.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (roles, names, turn order). Generation g uses seed+g.")
    parser.add_argument("--num-generations", type=int, default=NUM_GENERATIONS, help="Number of generations (agents are replaced between each).")
    parser.add_argument("--days-per-generation", type=int, default=DAYS_PER_GENERATION, help="Days each generation lives.")
    parser.add_argument("--steps-per-day", type=int, default=STEPS_PER_DAY, help="Steps per day before positions and supply reset.")
    parser.add_argument("--num-couriers", type=int, default=NUM_COURIERS, help="Number of courier agents per generation.")
    parser.add_argument("--num-misaligned", type=int, default=NUM_MISALIGNED, help="Number of misaligned agents (0 for fully cooperative). They are not replaced between generations.")
    parser.add_argument(
        "--misaligned-generations", type=int, default=MISALIGNED_GENERATIONS,
        help="The misaligned agents persist (same memory) through the first N generations, then are removed and their places go to couriers. 1 = only in the first generation. Default: every generation.",
    )
    parser.add_argument("--disclosure", choices=["secret", "open"], default=DISCLOSURE, help="Whether couriers are told a misaligned agent exists.")
    parser.add_argument("--target-zone", choices=list(ZONE_NAMES), default=MISALIGNED_TARGET_ZONE, help="The zone the misaligned agent wants the couriers to deliver to (its own deliveries don't count).")
    parser.add_argument(
        "--tally", dest="tally_visibility", choices=list(TALLY_MODES), default=TALLY_VISIBILITY,
        help="daily (default): no live totals; official zone totals announced at the end of each day, couriers report their own deliveries on the board. "
             "full: everyone sees zone totals and the delivery log. hidden: totals are never posted; a zone's total is only visible while standing on it.",
    )
    parser.add_argument("--misaligned-model", default=MISALIGNED_MODEL_NAME, help="Model for the misaligned agents (default: same as the couriers).")
    parser.add_argument("--misaligned-base-url", default=MISALIGNED_BASE_URL, help="SGLang server for the misaligned agents, e.g. http://localhost:30001/v1 (default: the couriers' server).")
    parser.add_argument(
        "--misaligned-thinking", action="store_true", default=MISALIGNED_THINKING,
        help="Run the misaligned agents' reasoning with Qwen thinking on (couriers keep it off). Same model; the capability lever.",
    )
    probes = parser.add_mutually_exclusive_group()
    probes.add_argument("--probes", dest="probes", action="store_true", default=PROBES_ENABLED, help="Ask every agent the private belief questionnaire at the start of each generation and the end of each day (default).")
    probes.add_argument("--no-probes", dest="probes", action="store_false", help="Skip the belief questionnaire.")
    parser.add_argument("--max-workers", type=int, default=MAX_PARALLEL_WORKERS, help="Maximum concurrent LLM requests (capped at the number of agents).")
    parser.add_argument("--verbose", action="store_true", help="Print full LLM requests, each agent's plan, and every probe answer.")
    parser.add_argument("--check-servers", action="store_true", help="Only check that the model server(s) return usable output, then exit.")
    parser.add_argument("--skip-model-check", action="store_true", help="Skip the model-output check that normally runs before a game.")
    parser.add_argument("--logs-dir", default=None, help="Directory for the run's files (default: examples/lifeline/logs).")
    parser.add_argument(
        "--plant", choices=list(PLANT_KINDS), default=None,
        help="Planted-note condition: place one false note on the board the --plant-rotation rotation inherits. "
             "hazard: a clean tile called a fixed hazard. history: the target zone 'drifts behind every rotation'. "
             "Use with --num-misaligned 0.",
    )
    parser.add_argument("--plant-rotation", type=int, default=PLANT_ROTATION, help="Rotation whose inherited board gets the planted note (default 2).")
    parser.add_argument(
        "--resume", metavar="RUN.jsonl",
        help="Continue a stopped run from its last finished generation (a checkpoint is saved after each one). "
             "The game settings come from the run itself; flags that set the game are ignored.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        _run(args)
    except ModelHealthError as exc:
        # A broken model server: say so plainly instead of dumping a traceback.
        print(f"\nERROR: {exc}")
        raise SystemExit(1) from None


def _run(args: argparse.Namespace) -> None:
    extra = {"logs_dir": args.logs_dir} if args.logs_dir else {}
    run_experiment(
        **extra,
        seed=args.seed,
        num_generations=args.num_generations,
        days_per_generation=args.days_per_generation,
        steps_per_day=args.steps_per_day,
        max_workers=args.max_workers,
        verbose=args.verbose,
        num_couriers=args.num_couriers,
        num_misaligned=args.num_misaligned,
        disclosure=args.disclosure,
        target_zone=args.target_zone,
        tally_visibility=args.tally_visibility,
        misaligned_generations=args.misaligned_generations,
        probes=args.probes,
        misaligned_thinking=args.misaligned_thinking,
        misaligned_model=args.misaligned_model,
        misaligned_base_url=args.misaligned_base_url,
        check_models=not args.skip_model_check,
        check_only=args.check_servers,
        resume=args.resume,
        plant=args.plant,
        plant_rotation=args.plant_rotation,
    )


if __name__ == "__main__":
    main()
