from __future__ import annotations

import random

from word_play.core import Entity
from word_play.presets.entity_orderings import randomize_agent_order
from word_play.presets.environments.simple_2d_grid_world import Simple_2D_Grid_World
from word_play.presets.movement.simple_2d_grid import (
    Collidable,
    Move_Down,
    Move_Left,
    Move_Right,
    Move_Up,
    Position_2D,
)
from word_play.presets.systems.cooldown import Cooldown
from word_play.presets.systems.do_nothing import Do_Nothing
from word_play.presets.systems.freezable import Freezable
from word_play.utils import tilemap_to_entities
from word_play.utils.tilemap import find_tile_positions

from benchmarks.text_mp.core.policies import PolicyKind, make_policy, register_llm_model
from benchmarks.text_mp.core.timing import normalized_steps
from benchmarks.text_mp.substrates.externality_mushrooms.mechanics import (
    MUSHROOM_DIGESTION_STEPS,
    ExternalityMushroom,
    ExternalityZap,
    ExternalityZapState,
)
from benchmarks.text_mp.substrates.externality_mushrooms.variants import ExternalityMushroomsVariant


def build_env(
    *,
    variant: ExternalityMushroomsVariant,
    agent_count: int | None,
    policy_kind: PolicyKind,
    model_name: str,
    generation_config: dict | None = None,
) -> Simple_2D_Grid_World:
    if policy_kind == "llm":
        register_llm_model(
            variant.model_key,
            model_name=model_name,
            generation_config=generation_config,
        )

    entities = tilemap_to_entities(variant.tilemap, _tileset())
    spawn_positions = find_tile_positions(variant.tilemap, ".")
    total_agents = variant.default_agent_count if agent_count is None else agent_count
    if total_agents > len(spawn_positions):
        raise ValueError(f"Map only has {len(spawn_positions)} spawn positions.")

    random.shuffle(spawn_positions)
    for agent_id, (x, y) in enumerate(spawn_positions[:total_agents], start=1):
        entities.append(_build_agent(agent_id, x, y, variant, policy_kind))

    env = Simple_2D_Grid_World(
        description=variant.description,
        entities=entities,
        entity_order=randomize_agent_order,
        observation_radius=variant.observation_radius,
    )
    env.externality_mushroom_potential_positions = spawn_positions.copy()
    env.mushroom_events = []
    return env


def _tileset() -> dict:
    return {
        "W": {
            "name": "Wall",
            "tags": ["wall"],
            "components": [
                Collidable(collidable_tags=["wall"]),
            ],
        },
        "R": {
            "name": "Red Mushroom",
            "components": [
                ExternalityMushroom("red"),
            ],
        },
        "G": {
            "name": "Green Mushroom",
            "components": [
                ExternalityMushroom("green"),
            ],
        },
        "B": {
            "name": "Blue Mushroom",
            "components": [
                ExternalityMushroom("blue"),
            ],
        },
        "O": {
            "name": "Orange Mushroom",
            "components": [
                ExternalityMushroom("orange"),
            ],
        },
    }


def _build_agent(
    agent_id: int,
    x: int,
    y: int,
    variant: ExternalityMushroomsVariant,
    policy_kind: PolicyKind,
) -> Entity:
    agent_policy = make_policy(
        policy_kind=policy_kind,
        model_key=variant.model_key,
        system_prompt=(
            f"You are Player {agent_id} in Externality Mushrooms: {variant.title}. "
            "Red mushrooms give you +1. "
            f"Green mushrooms split +2 across everyone and freeze you for {MUSHROOM_DIGESTION_STEPS['green']} turn(s). "
            f"Blue mushrooms split +3 across everyone except you and freeze you for {MUSHROOM_DIGESTION_STEPS['blue']} turn(s). "
            f"Orange mushrooms give you -1, freeze you for {MUSHROOM_DIGESTION_STEPS['orange']} turn(s), and destroy red mushrooms. "
            "Shape mushroom regrowth by choosing what to eat and when to zap nearby players."
        ),
        observation_memory_window=4,
        conversation_memory_window=4,
    )
    return Entity(
        name=f"Player {agent_id}",
        position=Position_2D(x, y),
        tags=["agent", "player", "default"],
        actions=[
            Do_Nothing(),
            Move_Up(),
            Move_Down(),
            Move_Left(),
            Move_Right(),
            ExternalityZap(),
        ],
        components=[
            agent_policy,
            Cooldown(cooldowns={"zap": normalized_steps(2)}),
            ExternalityZapState(),
            Freezable(default_duration=normalized_steps(25)),
            Collidable(collidable_tags=["wall"]),
        ],
    )
