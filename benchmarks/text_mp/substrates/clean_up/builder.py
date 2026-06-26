from __future__ import annotations

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
from benchmarks.text_mp.substrates.clean_up.mechanics import (
    ZAP_DURATION,
    ApplePatch,
    CleanDirt,
    CleanUpZap,
    PollutionManager,
)
from benchmarks.text_mp.substrates.clean_up.variants import CleanUpVariant


def build_env(
    *,
    variant: CleanUpVariant,
    agent_count: int | None,
    policy_kind: PolicyKind,
    model_name: str,
    generation_config: dict | None = None,
) -> Simple_2D_Grid_World:
    if policy_kind == "llm":
        register_llm_model(
            variant.model_key,
            model_name=model_name,
            generation_config=generation_config or {"temperature": 0.3},
        )

    entities = tilemap_to_entities(variant.tilemap.replace("P", "."), _tileset())
    spawn_positions = find_tile_positions(variant.tilemap, "P")
    river_positions = find_tile_positions(variant.tilemap, {"H", "F"})
    total_agents = variant.default_agent_count if agent_count is None else agent_count
    if total_agents > len(spawn_positions):
        raise ValueError(f"Map only has {len(spawn_positions)} spawn positions.")

    entities.append(
        Entity(
            name="Pollution Manager",
            position=Position_2D(-1, -1),
            components=[PollutionManager(river_positions=river_positions)],
        )
    )

    for agent_id, (x, y) in enumerate(spawn_positions[:total_agents], start=1):
        entities.append(_build_agent(agent_id, x, y, variant, policy_kind))

    env = Simple_2D_Grid_World(
        description=variant.description,
        entities=entities,
        entity_order=randomize_agent_order,
        observation_radius=variant.observation_radius,
    )
    env.clean_up_events = []
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
        "H": {
            "name": "River Tile",
            "tags": ["wall", "water"],
            "components": [
                Collidable(collidable_tags=["wall"]),
            ],
        },
        "F": {
            "name": "Dirty River Tile",
            "tags": ["wall", "water", "dirt_patch"],
            "components": [
                Collidable(collidable_tags=["wall"]),
            ],
        },
        "a": {
            "name": "Apple",
            "components": [
                ApplePatch(),
            ],
        },
    }


def _build_agent(
    agent_id: int,
    x: int,
    y: int,
    variant: CleanUpVariant,
    policy_kind: PolicyKind,
) -> Entity:
    agent_policy = make_policy(
        policy_kind=policy_kind,
        model_key=variant.model_key,
        system_prompt=f"You are Player {agent_id} in Clean Up: {variant.title}. {variant.prompt}",
        observation_memory_window=4,
        conversation_memory_window=8,
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
            CleanUpZap(),
            CleanDirt(),
        ],
        components=[
            agent_policy,
            Cooldown(cooldowns={"zap": normalized_steps(10)}),
            Freezable(default_duration=ZAP_DURATION),
            Collidable(collidable_tags=["wall"]),
        ],
    )
