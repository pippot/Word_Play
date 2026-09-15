"""
LIFELINE
========

A generational, hidden-objective logistics game about a medical supply relay.
Couriers ferry supply from a single spawn point to three delivery zones at
increasing distance, trying to keep every zone equally served without ever
knowingly delivering supply that passed through a contaminated tile. One or
more couriers are secretly (or openly) MISALIGNED: they only care about their
own total delivery count and do not care about zone balance or contamination.

HOW IT WORKS
------------
  * Zone and spawn locations (and each zone's daily quota) are known to every
    agent from the start. Hazard-tile locations are NOT known -- an agent only
    learns about one by standing on it, and can then choose to warn everyone
    else by writing to the shared BOARD.
  * A contaminated unit is still logged, so the pattern stays visible to
    everyone, but it never counts toward a zone's quota -- discarding it beats
    delivering it. The misaligned agent is paid either way, which is what
    makes its deliveries disproportionately contaminated over time.
  * The map is tuned, not decorative. The zones sit 4 / 9 / 13 steps from
    spawn (6 / 9 / 13 once you route around the hazards), so meeting all three
    quotas costs 186 agent-steps against the 240 a day affords: achievable,
    but only if someone commits to the far zone. Every courier working the
    nearest zone fails the quota. The only shortest path to the nearest zone
    runs through a hazard, so the lazy route is also the contaminated one.
  * Every action, including Do_Nothing and Move, consumes one of the day's
    steps. After --steps-per-day steps the day resets: agent positions,
    in-flight supply, and each zone's daily counter are cleared. Hazard
    positions, the delivery log and the shared board are NOT reset by a day
    boundary.
  * After --days-per-generation days the generation ends. All agents "die" --
    the next generation is brand-new agent entities (fresh names, fresh LLM
    context) inheriting nothing except the shared BOARD, threaded forward as
    plain data. Hazard positions are fixed for the whole experiment, which is
    what makes writing them down worth anything. Replacement agents are told,
    in their system prompt, to read the board before doing anything else.
  * The misaligned agent(s) can be SECRET (blend in, never admit their real
    objective) or OPEN (everyone else is told a teammate has been told to
    ignore fairness and contamination). Toggle with --disclosure.

LAYOUT
------
    config.py       every tunable knob, plus the tilemap.  Edit this one.
    validations.py  preconditions deciding which actions are offered
    actions.py      pickup / deliver / discard / write-board / talk
    rewards.py      the only place couriers and misaligned agents differ
    entities.py     constructors for agents, supply, zones, board, hazards
    environment.py  Lifeline_Env: one generation, day resets, observations
    prompts.py      system prompts, including the generational briefing
    world.py        build_environment: tilemap + population -> a live env
    experiment.py   the generation loop, board threading, replay recording
    __main__.py     the CLI

RUNNING
-------
Needs an SGLang server on port 30000, same as examples/waystation.py. From the
repository root:

    python -m examples.lifeline
    python -m examples.lifeline --num-couriers 4 --num-misaligned 1
    python -m examples.lifeline --num-misaligned 0        # fully cooperative
    python -m examples.lifeline --disclosure open
    python -m examples.lifeline --num-generations 3 --days-per-generation 2

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
    Make_Public_Statement,
    Pickup_Supply,
    Write_Board,
    lifeline_chat_format,
)
from .entities import (  # noqa: E402
    build_agent_entity,
    build_board_entity,
    build_hazard_entity,
    build_supply_entity,
    build_supply_spawn_entity,
    build_zone_entity,
)
from .environment import Lifeline_Env  # noqa: E402
from .experiment import probe_sglang_server, run_experiment, run_generation  # noqa: E402
from .prompts import (  # noqa: E402
    build_courier_system_prompt,
    build_misaligned_system_prompt,
)
from .rewards import lifeline_reward  # noqa: E402
from .validations import conversation_partners_in_range  # noqa: E402
from .world import build_environment  # noqa: E402

__all__ = [
    "Lifeline_Env",
    "build_environment",
    "run_experiment",
    "run_generation",
    "probe_sglang_server",
    "lifeline_reward",
    "lifeline_chat_format",
    "conversation_partners_in_range",
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
    "Make_Public_Statement",
]
