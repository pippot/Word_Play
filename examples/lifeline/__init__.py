"""
LIFELINE
========

A generational, hidden-objective logistics game for studying how a misaligned
goal spreads through a population of LLM agents. Couriers ferry supply from a
spawn point to three delivery zones, trying to keep the zones' clean delivery
totals equal without delivering contaminated supply. Misaligned agents only
want deliveries to one target zone and are told to steer the couriers there
through the shared board. Couriers are replaced every generation and inherit
only the board; the misaligned agents persist, keeping their memory (under a
new name each generation).

Full documentation -- how to run, every flag, the detailed game mechanics,
the belief probes, the output files and the metrics -- is in README.md next
to this file.

LAYOUT
------
    config.py       every tunable knob, plus the tilemap.  Edit this one.
    layout.py       landmark / hazard coordinates, moving-hazard schedule
    validations.py  preconditions deciding which actions are offered
    actions.py      moves, pickup / deliver / discard, write-board
    entities.py     constructors for agents, supply, zones, board, hazards
    environment.py  Lifeline_Env: one generation, day resets, observations
    prompts.py      every piece of agent-facing text
    policy.py       Lifeline_Policy: compact memory, reasoning -> action, probes
    probes.py       the private belief questionnaire
    world.py        build_environment: tilemap + population -> a live env
    experiment.py   the generation loop, board threading, logs
    metrics.py      metrics computed from a run's JSONL event log
    __main__.py     the CLI

RUNNING
-------
Needs an SGLang server on port 30000. From the repository root:

    python -m examples.lifeline
    python -m examples.lifeline --num-misaligned 0                 # control
    python -m examples.lifeline --misaligned-generations 1         # misaligned only in generation 1
    python -m examples.lifeline --target-zone Zone_Pine --tally hidden

Mechanics are covered by tests that need no server or GPU:

    python -m unittest tests.test_lifeline
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when running straight from a checkout. This has to come
# before any word_play import below.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from .config import *  # noqa: E402,F401,F403  (re-export every tunable)
from .actions import (  # noqa: E402
    Deliver_Supply,
    Drop_Supply,
    Lifeline_Move_Down,
    Lifeline_Move_Left,
    Lifeline_Move_Right,
    Lifeline_Move_Up,
    Pickup_Supply,
    Write_Board,
)
from .entities import (  # noqa: E402
    build_agent_entity,
    build_board_entity,
    build_hazard_entity,
    build_supply_entity,
    build_supply_spawn_entity,
    build_zone_entity,
)
from .environment import Lifeline_Env, Lifeline_Observation, Misaligned_Lineage  # noqa: E402
from .experiment import probe_sglang_server, run_experiment, run_generation  # noqa: E402
from .health import ModelHealthError, check_model  # noqa: E402
from .layout import (  # noqa: E402
    Layout,
    has_clean_shortest_path,
    hazard_schedule,
    keeps_pacing,
    moving_hazard_candidates,
    moving_hazard_regions,
    moving_hazards_by_zone,
    parse_layout,
)
from .policy import Lifeline_Policy  # noqa: E402
from .probes import normalize_probe_answer, run_probes  # noqa: E402
from .prompts import (  # noqa: E402
    COURIER_PERSONAS,
    MISALIGNED_PERSONA_ID,
    build_courier_system_prompt,
    build_misaligned_system_prompt,
    build_probe_prompt,
    misaligned_persona_text,
)
from .world import build_environment  # noqa: E402

__all__ = [
    "Lifeline_Env",
    "Lifeline_Observation",
    "Misaligned_Lineage",
    "Lifeline_Policy",
    "Layout",
    "parse_layout",
    "hazard_schedule",
    "has_clean_shortest_path",
    "keeps_pacing",
    "moving_hazard_candidates",
    "moving_hazard_regions",
    "moving_hazards_by_zone",
    "COURIER_PERSONAS",
    "MISALIGNED_PERSONA_ID",
    "misaligned_persona_text",
    "run_probes",
    "ModelHealthError",
    "check_model",
    "normalize_probe_answer",
    "build_probe_prompt",
    "compute_metrics",
    "load_events",
    "Lifeline_Move_Up",
    "Lifeline_Move_Down",
    "Lifeline_Move_Left",
    "Lifeline_Move_Right",
    "build_environment",
    "run_experiment",
    "run_generation",
    "probe_sglang_server",
    "build_courier_system_prompt",
    "build_misaligned_system_prompt",
    "build_agent_entity",
    "build_supply_entity",
    "build_zone_entity",
    "build_board_entity",
    "build_supply_spawn_entity",
    "build_hazard_entity",
    "Pickup_Supply",
    "Deliver_Supply",
    "Drop_Supply",
    "Write_Board",
]


def __getattr__(name: str):
    # metrics is imported on first use, not with the package: importing it
    # here made `python -m examples.lifeline.metrics` warn that the module
    # was already in sys.modules before it ran.
    if name in ("compute_metrics", "load_events"):
        from . import metrics
        return getattr(metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
