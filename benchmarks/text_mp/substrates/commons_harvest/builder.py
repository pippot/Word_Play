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
from word_play.presets.systems.reward import Rewardable
from word_play.utils import tilemap_to_entities
from word_play.utils.tilemap import find_tile_positions

from benchmarks.text_mp.core.policies import PolicyKind, make_policy, register_llm_model
from benchmarks.text_mp.core.timing import normalized_steps
from benchmarks.text_mp.substrates.commons_harvest.mechanics import (
    Commons_Apple_Patch,
    Commons_Freezable,
    Commons_Zap,
    Role_Based_Punishment_Tile,
)
from benchmarks.text_mp.substrates.commons_harvest.variants import CommonsHarvestVariant


def build_env(
    *,
    variant: CommonsHarvestVariant,
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

    entities = tilemap_to_entities(
        variant.tilemap.replace("P", ".").replace("Q", "."),
        _tileset_for_variant(variant),
    )
    spawn_positions = _spawn_positions(variant.tilemap)
    total_agents = variant.default_agent_count if agent_count is None else agent_count
    if total_agents > len(spawn_positions):
        raise ValueError(f"Map only has {len(spawn_positions)} spawn positions.")

    for agent_id, (x, y) in enumerate(spawn_positions[:total_agents], start=1):
        tags = ["agent", "player", "default"]
        if agent_id <= variant.putative_cooperator_count:
            tags.append("putative_cooperator")
        entities.append(_build_agent(agent_id, x, y, tags, variant, policy_kind))

    env = Simple_2D_Grid_World(
        description=variant.description,
        entities=entities,
        entity_order=randomize_agent_order,
        observation_radius=variant.observation_radius,
    )
    env.commons_events = []
    return env


def _tileset_for_variant(variant: CommonsHarvestVariant) -> dict:
    tileset = {
        "W": {
            "name": "Wall",
            "tags": ["wall"],
            "components": [
                Collidable(collidable_tags=["wall"]),
            ],
        },
        "A": {
            "name": "Apple",
            "components": [
                Commons_Apple_Patch(apple_regrowth_probs=list(variant.apple_regrowth_probs)),
                Rewardable(amount=1.0),
            ],
        },
    }
    if variant.has_punishment_tiles:
        tileset["I"] = {
            "name": "Hidden Punishment Tile",
            "components": [
                Role_Based_Punishment_Tile(punished_tag="putative_cooperator", penalty=-10.0),
            ],
        }
    return tileset


def _spawn_positions(tilemap: str) -> list[tuple[int, int]]:
    inside_spawns = find_tile_positions(tilemap, "Q")
    outside_spawns = find_tile_positions(tilemap, "P")
    random.shuffle(inside_spawns)
    random.shuffle(outside_spawns)
    return inside_spawns[:2] + outside_spawns


def _build_agent(
    agent_id: int,
    x: int,
    y: int,
    tags: list[str],
    variant: CommonsHarvestVariant,
    policy_kind: PolicyKind,
) -> Entity:
    agent_policy = make_policy(
        policy_kind=policy_kind,
        model_key=variant.model_key,
        system_prompt=(
            f"You are Player {agent_id} in Commons Harvest: {variant.title}. "
            f"{variant.prompt}"
        ),
        observation_memory_window=4,
        conversation_memory_window=4,
    )
    return Entity(
        name=f"Player {agent_id}",
        position=Position_2D(x, y),
        tags=tags,
        actions=[
            Do_Nothing(),
            Move_Up(),
            Move_Down(),
            Move_Left(),
            Move_Right(),
            Commons_Zap(),
        ],
        components=[
            agent_policy,
            Cooldown(cooldowns={"zap": normalized_steps(variant.zap_cooldown_steps)}),
            Commons_Freezable(default_duration=normalized_steps(variant.freeze_duration_steps)),
            Collidable(collidable_tags=["wall"]),
        ],
    )
