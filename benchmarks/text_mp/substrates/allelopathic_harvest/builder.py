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
from word_play.presets.systems.containers.presets import Take_From_Infinite_Source
from word_play.presets.systems.cooldown import Cooldown
from word_play.presets.systems.do_nothing import Do_Nothing
from word_play.presets.systems.freezable import Freezable
from word_play.presets.systems.health import Health
from word_play.presets.systems.inventory import Inventory
from word_play.presets.systems.preferences import Preference
from word_play.presets.systems.zap import ZapMarking, Zap_Player
from word_play.utils import tilemap_to_entities
from word_play.utils.tilemap import find_tile_positions

from benchmarks.text_mp.core.policies import PolicyKind, make_policy, register_llm_model
from benchmarks.text_mp.core.timing import normalized_steps
from benchmarks.text_mp.substrates.allelopathic_harvest.mechanics import (
    AllelopathicBerryPatch,
    AllelopathicPlantBerry,
    AllelopathicZapState,
    PaintBerry,
    standard_berry_source,
)
from benchmarks.text_mp.substrates.allelopathic_harvest.variants import AllelopathicHarvestVariant


def build_env(
    *,
    variant: AllelopathicHarvestVariant,
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

    entities = tilemap_to_entities(_entity_tilemap(variant), _tileset_for_variant(variant))
    spawn_positions = find_tile_positions(variant.tilemap, "P")
    random.shuffle(spawn_positions)
    total_agents = variant.default_agent_count if agent_count is None else agent_count
    if total_agents > len(spawn_positions):
        raise ValueError(f"Map only has {len(spawn_positions)} spawn positions.")

    for agent_id, (x, y) in enumerate(spawn_positions[:total_agents], start=1):
        entities.append(_build_agent(agent_id, x, y, total_agents, variant, policy_kind))

    env = Simple_2D_Grid_World(
        description=variant.description,
        entities=entities,
        entity_order=randomize_agent_order,
        observation_radius=variant.observation_radius,
    )
    env.allelopathic_events = []
    return env


def _entity_tilemap(variant: AllelopathicHarvestVariant) -> str:
    return variant.tilemap if variant.open_patch_mode else variant.tilemap.replace("P", ".")


def _tileset_for_variant(variant: AllelopathicHarvestVariant) -> dict:
    if variant.open_patch_mode:
        return {
            "P": {
                "name": "Empty Berry Patch",
                "components": [
                    AllelopathicBerryPatch(auto_consume=True, grows=True),
                ],
            },
            "1": {
                "name": "Red Berry Patch",
                "components": [
                    AllelopathicBerryPatch("red", auto_consume=True, grows=True),
                ],
            },
            "2": {
                "name": "Green Berry Patch",
                "components": [
                    AllelopathicBerryPatch("green", auto_consume=True, grows=True),
                ],
            },
            "3": {
                "name": "Blue Berry Patch",
                "components": [
                    AllelopathicBerryPatch("blue", auto_consume=True, grows=True),
                ],
            },
        }

    return {
        "W": {
            "name": "Wall",
            "tags": ["wall"],
            "components": [
                Collidable(collidable_tags=["wall"]),
            ],
        },
        "1": {
            "name": "Red Berry Patch",
            "tags": ["berry_patch", "berry", "red"],
            "components": [
                standard_berry_source("red"),
            ],
        },
        "2": {
            "name": "Green Berry Patch",
            "tags": ["berry_patch", "berry", "green"],
            "components": [
                standard_berry_source("green"),
            ],
        },
        "3": {
            "name": "Blue Berry Patch",
            "tags": ["berry_patch", "berry", "blue"],
            "components": [
                standard_berry_source("blue"),
            ],
        },
    }


def _build_agent(
    agent_id: int,
    x: int,
    y: int,
    agent_count: int,
    variant: AllelopathicHarvestVariant,
    policy_kind: PolicyKind,
) -> Entity:
    preferred_color = _preferred_color(agent_id, variant)
    agent_policy = make_policy(
        policy_kind=policy_kind,
        model_key=variant.model_key,
        system_prompt=(
            f"You are Player {agent_id} in Allelopathic Harvest: {variant.title}. "
            f"You get +2 from {preferred_color} berries and +1 from other berries. "
            f"{variant.prompt}"
        ),
        observation_memory_window=0 if variant.open_patch_mode else 4,
        conversation_memory_window=0 if variant.open_patch_mode else 4,
    )

    actions = [
        Do_Nothing(),
        Move_Up(),
        Move_Down(),
        Move_Left(),
        Move_Right(),
        Zap_Player(),
    ]
    components = [
        agent_policy,
        Preference(reward_map={preferred_color: 2.0}, default_reward=1.0),
        Cooldown(cooldowns=_cooldowns(variant)),
        Freezable(default_duration=normalized_steps(25)),
    ]

    tags = ["agent", "player", "default", f"player_who_likes_{preferred_color}"]
    if variant.open_patch_mode:
        actions.extend([
            AllelopathicPlantBerry("red"),
            AllelopathicPlantBerry("green"),
            AllelopathicPlantBerry("blue"),
        ])
        components.append(AllelopathicZapState())
    else:
        actions.extend([
            Take_From_Infinite_Source(),
            PaintBerry("red"),
            PaintBerry("green"),
            PaintBerry("blue"),
        ])
        components.extend([
            Inventory(max_size=5, accepted_tags=["berry"]),
            ZapMarking(freeze_duration=25, penalty=-10.0, recovery_time=50),
            Health(max_health=2, starting_health=2),
            Collidable(collidable_tags=["wall"]),
        ])

    return Entity(
        name=f"Player {agent_id}",
        position=Position_2D(x, y),
        tags=tags,
        actions=actions,
        components=components,
    )


def _cooldowns(variant: AllelopathicHarvestVariant) -> dict[str, int]:
    if variant.open_patch_mode:
        return {"zap": normalized_steps(4), "plant": normalized_steps(2)}
    return {"zap": 4, "color": 2}


def _preferred_color(agent_id: int, variant: AllelopathicHarvestVariant) -> str:
    if variant.open_patch_mode:
        return "red" if agent_id <= variant.default_agent_count // 2 else "green"
    return random.choice(["red", "green", "blue"])
