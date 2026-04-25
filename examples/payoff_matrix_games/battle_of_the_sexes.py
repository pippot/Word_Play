from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import parse_common_args, run_payoff_matrix_game


def run_exp(
    model_mode: str = "human_action",
    max_steps: int = 20,
    log_path: str | None = None,
) -> None:
    run_payoff_matrix_game(
        game_name="Battle of the Sexes",
        action_names=["Opera", "Football"],
        payoff_matrix=[
            [(3.0, 2.0), (0.0, 0.0)],
            [(0.0, 0.0), (2.0, 3.0)],
        ],
        objective_text="\n".join(
            [
                "OBJECTIVE: Battle of the Sexes",
                "  You and your opponent must coordinate on a shared activity.",
                "  Opera together gives Agent A 3 and Agent B 2.",
                "  Football together gives Agent A 2 and Agent B 3.",
                "  A mismatch gives both players 0.",
            ]
        ),
        model_mode=model_mode,
        max_steps=max_steps,
        log_path=log_path,
    )


if __name__ == "__main__":
    args = parse_common_args()
    run_exp(
        model_mode=args.model_mode,
        max_steps=args.max_steps if args.max_steps_legacy is None else args.max_steps_legacy,
        log_path=args.log_path,
    )
