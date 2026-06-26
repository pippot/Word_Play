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
from word_play.presets.systems.do_nothing import Do_Nothing
from word_play.presets.systems.inventory import Inventory, Pick_Up_Item
from word_play.utils import tilemap_to_entities
from word_play.utils.tilemap import find_tile_positions

from benchmarks.text_mp.core.policies import PolicyKind, make_policy, register_llm_model
from benchmarks.text_mp.substrates.chemistry.mechanics import (
    CHEMISTRY_START_POSITIONS,
    CycleSite,
    Molecule,
)
from benchmarks.text_mp.substrates.chemistry.variants import ChemistryVariant


def build_env(
    *,
    variant: ChemistryVariant,
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
    spawn_positions = _spawn_positions(variant.tilemap)
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
    env.chemistry_events = []
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
        "a": {
            "name": "Blue Compound AX",
            "components": [
                CycleSite("blue"),
            ],
        },
        "b": {
            "name": "Blue Compound BX",
            "components": [
                CycleSite("blue"),
            ],
        },
        "c": {
            "name": "Blue Compound CX",
            "components": [
                CycleSite("blue"),
            ],
        },
        "h": {
            "name": "Energy",
            "components": [
                Molecule("energy"),
            ],
        },
        "x": {
            "name": "Distractor",
            "components": [
                Molecule("distractor", holding_reward=0.1),
            ],
        },
        "1": {
            "name": "Green Compound AY",
            "components": [
                CycleSite("green"),
            ],
        },
        "2": {
            "name": "Green Compound BY",
            "components": [
                CycleSite("green"),
            ],
        },
        "3": {
            "name": "Green Compound CY",
            "components": [
                CycleSite("green"),
            ],
        },
        "4": {
            "name": "Red Compound AZ",
            "components": [
                CycleSite("red"),
            ],
        },
        "5": {
            "name": "Red Compound BZ",
            "components": [
                CycleSite("red"),
            ],
        },
        "6": {
            "name": "Red Compound CZ",
            "components": [
                CycleSite("red"),
            ],
        },
    }


def _spawn_positions(tilemap: str) -> list[tuple[int, int]]:
    floor_positions = find_tile_positions(tilemap, ".")
    return [
        *[position for position in CHEMISTRY_START_POSITIONS if position in floor_positions],
        *[position for position in floor_positions if position not in CHEMISTRY_START_POSITIONS],
    ]


def _build_agent(
    agent_id: int,
    x: int,
    y: int,
    agent_count: int,
    variant: ChemistryVariant,
    policy_kind: PolicyKind,
) -> Entity:
    agent_policy = make_policy(
        policy_kind=policy_kind,
        model_key=variant.model_key,
        system_prompt=(
            f"You are Player {agent_id} in Chemistry: {variant.title}. "
            f"{variant.role_prompt(agent_id, agent_count)} "
            "Do not all go to the same cycle. "
            f"{variant.prompt}"
        ),
        observation_memory_window=2,
        conversation_memory_window=0,
    )
    return Entity(
        name=f"Player {agent_id}",
        position=Position_2D(x, y),
        tags=["agent", "player", "default"],
        actions=[
            Do_Nothing(),
            Pick_Up_Item(["molecule"]),
            Move_Up(),
            Move_Down(),
            Move_Left(),
            Move_Right(),
        ],
        components=[
            agent_policy,
            Inventory(accepted_tags=[], max_size=1),
            Collidable(collidable_tags=["wall"]),
        ],
    )
