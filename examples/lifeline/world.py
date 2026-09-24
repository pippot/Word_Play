"""
build_environment: turns the tilemap plus a population into one generation's
ready-to-run Lifeline_Env.
"""

from __future__ import annotations

import itertools
import random

from word_play.core import Agent_Policy, Entity
from word_play.presets.movement.common import Collidable
from word_play.presets.movement.simple_2d_grid import Position_2D
from word_play.presets.renderers import Renderable
from word_play.utils import tilemap_to_entities

from .config import (
    AGENT_SPRITES,
    BOARD_SPRITE,
    DAYS_PER_GENERATION,
    DISCLOSURE,
    ENTITY_TILEMAP,
    HAZARD_SPRITE,
    MISALIGNED_NEW_NAME_EACH_GENERATION,
    MISALIGNED_THINKING,
    MISALIGNED_TARGET_ZONE,
    NUM_COURIERS,
    NUM_MISALIGNED,
    OBSERVATION_RADIUS,
    REASONING_GENERATION_CONFIG,
    THINKING_REASONING_GENERATION_CONFIG,
    PLAYER_NAMES,
    STEPS_PER_DAY,
    SUPPLY_SPRITE,
    TALLY_MODES,
    TALLY_VISIBILITY,
    WALL_SET,
    WALL_SPRITE,
    ZONE_NAMES,
    ZONE_SPRITES,
)
from .entities import (
    build_agent_entity,
    build_board_entity,
    build_hazard_entity,
    build_supply_spawn_entity,
    build_zone_entity,
)
from .environment import Lifeline_Env, Misaligned_Lineage
from .layout import parse_layout
from .prompts import (
    COURIER_PERSONAS,
    MISALIGNED_PERSONA_ID,
    build_courier_system_prompt,
    build_misaligned_system_prompt,
)


def pick_names(
    total: int, rng: random.Random, generation_index: int, used_names: frozenset[str]
) -> list[str]:
    """
    Names are never shared by two agents within a run: board notes are signed,
    so a reused name would carry an earlier agent's reputation onto a stranger.
    When the pool runs out, names get a generation suffix ("Alice-7").
    """
    pool = [name for name in PLAYER_NAMES if name not in used_names]
    if len(pool) < total:
        suffixed = [f"{name}-{generation_index + 1}" for name in PLAYER_NAMES]
        pool = [name for name in suffixed if name not in used_names]
    if len(pool) < total:
        raise ValueError(
            f"{total} agents per generation needs more names than config.PLAYER_NAMES "
            f"({len(PLAYER_NAMES)}) provides"
        )
    rng.shuffle(pool)
    return pool[:total]


def build_walls() -> list[Entity]:
    """Wall entities from the tilemap. Every other symbol is placed from the
    parsed Layout instead, so it only needs a throwaway placeholder here."""
    placeholder = {"name": "placeholder", "tags": [], "components": []}
    tileset = {symbol: placeholder for symbol in "XB123HM"}
    tileset["W"] = {
        "name": "Wall",
        "tags": ["wall"],
        "components": [Collidable(), Renderable(sprite_path=WALL_SPRITE, wall_set=WALL_SET)],
    }
    return [e for e in tilemap_to_entities(ENTITY_TILEMAP, tileset) if "wall" in e.tags]


def build_environment(
    *,
    generation_index: int,
    board_slots: list[dict | None],
    model_key: str,
    seed: int,
    num_couriers: int = NUM_COURIERS,
    num_misaligned: int = NUM_MISALIGNED,
    disclosure: str = DISCLOSURE,
    steps_per_day: int = STEPS_PER_DAY,
    days_per_generation: int = DAYS_PER_GENERATION,
    observation_radius: int = OBSERVATION_RADIUS,
    target_zone: str = MISALIGNED_TARGET_ZONE,
    tally_visibility: str = TALLY_VISIBILITY,
    used_names: frozenset[str] = frozenset(),
    misaligned_lineages: list[Misaligned_Lineage] | None = None,
    misaligned_continues: bool = False,
    rename_misaligned: bool = MISALIGNED_NEW_NAME_EACH_GENERATION,
    hazard_positions: frozenset[tuple[int, int]] | None = None,
    misaligned_model_key: str | None = None,
    misaligned_thinking: bool = MISALIGNED_THINKING,
) -> Lifeline_Env:
    """
    Build one generation's environment. Hazard layout is fixed by the tilemap,
    so it is identical across every generation of a run.

    misaligned_lineages: one per misaligned agent, for agents that persist
    across generations (their memory is restored into the new entity).
    misaligned_continues: whether those agents will survive the end of this
    generation (only affects what they and, under open disclosure, couriers
    are told).
    hazard_positions: this generation's contaminated tiles (see
    layout.hazard_schedule); None means the map as drawn.
    misaligned_model_key: a different registered model for the misaligned
    agents; None means the couriers' model_key.
    misaligned_thinking: run the misaligned agents' reasoning/planning calls
    with thinking on (config._with_thinking); couriers keep it off.
    """
    if target_zone not in ZONE_NAMES:
        raise ValueError(f"target_zone must be one of {ZONE_NAMES}, got {target_zone!r}")
    if tally_visibility not in TALLY_MODES:
        raise ValueError(f"tally_visibility must be one of {TALLY_MODES}, got {tally_visibility!r}")
    if disclosure not in ("secret", "open"):
        raise ValueError(f"disclosure must be 'secret' or 'open', got {disclosure!r}")
    if num_couriers < 0 or num_misaligned < 0 or num_couriers + num_misaligned < 1:
        raise ValueError("need at least one agent and no negative counts")
    if steps_per_day < 1 or days_per_generation < 1:
        raise ValueError("steps_per_day and days_per_generation must be at least 1")
    lineages = misaligned_lineages or []
    if lineages and len(lineages) != num_misaligned:
        raise ValueError(f"got {len(lineages)} misaligned lineages for {num_misaligned} misaligned agents")

    rng = random.Random(seed)
    # entity_orderings.randomize_agent_order shuffles via the global random
    # module, so a local Random() alone would not make a run reproducible.
    random.seed(seed)

    # A persistent misaligned agent keeps its last name unless it is renamed
    # every generation; everyone else gets a name never used before.
    kept_names = [
        lineage.names[-1][1] if (lineage.names and not rename_misaligned) else None
        for lineage in lineages
    ] or [None] * num_misaligned
    fresh = pick_names(
        num_couriers + kept_names.count(None), rng, generation_index,
        used_names | {name for name in kept_names if name},
    )
    courier_names = fresh[:num_couriers]
    remaining = iter(fresh[num_couriers:])
    misaligned_order = [name or next(remaining) for name in kept_names]
    lineage_of = dict(zip(misaligned_order, lineages))
    misaligned_names = sorted(misaligned_order)
    misaligned_persistent = misaligned_continues or any(lineage.names for lineage in lineages)

    # Courier personas go to the courier slots in a seeded order that changes
    # every generation (with 4 couriers each persona appears exactly once;
    # with more they repeat). A separate Random keeps name draws unchanged.
    persona_ids = [persona.id for persona in COURIER_PERSONAS]
    random.Random(f"personas-{seed}").shuffle(persona_ids)
    personas = {name: persona_ids[i % len(persona_ids)] for i, name in enumerate(courier_names)}
    personas.update({name: MISALIGNED_PERSONA_ID for name in misaligned_names})

    # Sprites are handed out round-robin within each role, so with at least
    # len(AGENT_SPRITES) couriers every misaligned sprite is also worn by a
    # courier and nobody can be picked out by eye in the replay.
    roster = [
        (name, AGENT_SPRITES[i % len(AGENT_SPRITES)]) for i, name in enumerate(courier_names)
    ] + [
        (name, AGENT_SPRITES[i % len(AGENT_SPRITES)]) for i, name in enumerate(misaligned_names)
    ]
    # Entity order leaks into observation listings and logs; never let the
    # misaligned agents sit in a fixed position.
    rng.shuffle(roster)

    # Every agent sees the zones listed in its own order. With all zones tied
    # (every rotation's start), agents break the tie by list order -- in the
    # pilot every courier picked the first-listed zone, which was also the
    # target. Seeded permutations dealt round-robin over the shuffled roster
    # spread the first-listed zone evenly across agents instead of by chance.
    orders = list(itertools.permutations(ZONE_NAMES))
    random.Random(f"zone-order-{seed}").shuffle(orders)
    zone_order = {name: orders[i % len(orders)] for i, (name, _) in enumerate(roster)}

    layout = parse_layout(ENTITY_TILEMAP)
    final_entities: list[Entity] = build_walls()

    supply_spawn = build_supply_spawn_entity("Supply_Depot", Position_2D(*layout.spawn), SUPPLY_SPRITE)
    board = build_board_entity("Board", Position_2D(*layout.board), BOARD_SPRITE)
    zones = {
        name: build_zone_entity(name, Position_2D(*xy), ZONE_SPRITES[name])
        for name, xy in layout.zones.items()
    }
    final_entities += [supply_spawn, board, *zones.values()]
    hazards = frozenset(hazard_positions) if hazard_positions is not None else layout.hazards
    final_entities += [
        build_hazard_entity(f"Hazard_{i + 1}", Position_2D(*xy), HAZARD_SPRITE)
        for i, xy in enumerate(sorted(hazards))
    ]

    common = dict(
        disclosure=disclosure,
        steps_per_day=steps_per_day,
        days_per_generation=days_per_generation,
        num_misaligned=num_misaligned,
        generation_index=generation_index,
        # Nothing has been written this generation yet, so this is exactly
        # what the previous generations managed to pass on.
        inherited_board_count=sum(1 for slot in board_slots if slot is not None),
        target_zone=target_zone,
        tally_visibility=tally_visibility,
        layout=layout,
    )
    agents: list[Entity] = []
    for name, sprite in roster:
        lineage = lineage_of.get(name)
        if name in misaligned_names:
            prompt = build_misaligned_system_prompt(
                name,
                **common,
                zone_order=zone_order[name],
                first_generation=lineage.first_generation if lineage else None,
                previous_names=tuple(lineage.names) if lineage else (),
                continues_next_generation=misaligned_continues,
                renamed=rename_misaligned,
            )
        else:
            prompt = build_courier_system_prompt(
                name,
                **common,
                zone_order=zone_order[name],
                misaligned_persistent=misaligned_persistent,
                misaligned_renamed=rename_misaligned,
                persona=personas[name],
            )
        is_misaligned = name in misaligned_names
        agent_model = (misaligned_model_key or model_key) if is_misaligned else model_key
        reasoning_config = (
            THINKING_REASONING_GENERATION_CONFIG if is_misaligned and misaligned_thinking
            else REASONING_GENERATION_CONFIG
        )
        # Every agent starts at the depot, same as every day reset (see
        # Lifeline_Env._day_reset) -- not a random map-wide point.
        agent = build_agent_entity(
            name, Position_2D(*layout.spawn), sprite, agent_model, prompt,
            reasoning_generation_config=reasoning_config,
        )
        if lineage is not None:
            policy = agent.get_component(Agent_Policy)
            policy.persistent = misaligned_continues or bool(lineage.names)
            if lineage.memory is not None:
                policy.import_memory(lineage.memory, past_names=lineage.names, generation_index=generation_index)
        agents.append(agent)
    final_entities += agents

    env = Lifeline_Env(
        description=(
            "A medical-supply relay: couriers carry supply from a depot to three "
            "equally distant zones, keeping each zone's clean delivery total roughly "
            "equal without delivering hazard-contaminated supply."
        ),
        entities=final_entities,
        misaligned_names=misaligned_names,
        zones=zones,
        supply_spawn=supply_spawn,
        board=board,
        hazard_positions=set(hazards),
        steps_per_day=steps_per_day,
        days_per_generation=days_per_generation,
        generation_index=generation_index,
        board_slots=board_slots,
        disclosure=disclosure,
        observation_radius=observation_radius,
        target_zone=target_zone,
        tally_visibility=tally_visibility,
        misaligned_lineages=lineage_of,
    )
    env.personas = personas
    env.zone_order = zone_order
    env.misaligned_thinking = misaligned_thinking
    env.moving_hazards = hazards - layout.fixed_hazards
    return env
