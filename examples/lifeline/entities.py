"""
Constructors for every entity in the world. These are deliberately dumb: they
take a position and a sprite and return an Entity, with no knowledge of the
environment they will end up in.
"""

from __future__ import annotations

from word_play.core import Entity
from word_play.presets.movement.common import Collidable
from word_play.presets.movement.simple_2d_grid import Position_2D
from word_play.presets.renderers import Renderable
from word_play.presets.systems.do_nothing import Do_Nothing

from .actions import (
    Deliver_Supply,
    Drop_Supply,
    Lifeline_Move_Down,
    Lifeline_Move_Left,
    Lifeline_Move_Right,
    Lifeline_Move_Up,
    Pickup_Supply,
    Write_Board,
)
from .config import (
    ACTION_GENERATION_CONFIG,
    PROBE_GENERATION_CONFIG,
    REASONING_GENERATION_CONFIG,
)
from .policy import Lifeline_Policy

def build_agent_entity(
    name: str, position: Position_2D, sprite: str, model_key: str, system_prompt: str
) -> Entity:
    """Create a courier or misaligned agent entity (identical capabilities)."""
    return Entity(
        name=name,
        position=position,
        actions=[
            Do_Nothing(),
            Lifeline_Move_Up(),
            Lifeline_Move_Down(),
            Lifeline_Move_Left(),
            Lifeline_Move_Right(),
            Pickup_Supply(),
            Deliver_Supply(),
            Drop_Supply(),
            Write_Board(),
        ],
        components=[
            # The board is the only communication channel in Lifeline, so the
            # Communication_Policy half of this component is unused.
            Lifeline_Policy(
                model_key=model_key,
                system_prompt=system_prompt,
                action_generation_config=ACTION_GENERATION_CONFIG,
                reasoning_generation_config=REASONING_GENERATION_CONFIG,
                probe_generation_config=PROBE_GENERATION_CONFIG,
            ),
            Collidable(collidable_tags=["wall"]),
            Renderable(sprite_path=sprite, z_index=10),
        ],
    )


def build_supply_entity(name: str, position: Position_2D, sprite: str) -> Entity:
    """Create a non-agent supply-unit entity."""
    return Entity(
        name=name,
        position=position,
        tags=["supply"],
        components=[Renderable(sprite_path=sprite, z_index=5)],
    )


def build_zone_entity(name: str, position: Position_2D, sprite: str) -> Entity:
    return Entity(
        name=name,
        position=position,
        tags=["zone"],
        components=[Renderable(sprite_path=sprite, z_index=3)],
    )


def build_board_entity(name: str, position: Position_2D, sprite: str) -> Entity:
    return Entity(
        name=name,
        position=position,
        tags=["board"],
        components=[Renderable(sprite_path=sprite, z_index=3)],
    )


def build_supply_spawn_entity(name: str, position: Position_2D, sprite: str) -> Entity:
    return Entity(
        name=name,
        position=position,
        tags=["spawn"],
        components=[Renderable(sprite_path=sprite, z_index=2)],
    )


def build_hazard_entity(name: str, position: Position_2D, sprite: str) -> Entity:
    """
    Non-interactive marker used only so the pygame replay can show hazard
    tiles to a human afterwards. Excluded from every agent's observation
    (see Lifeline_Env.observe) so it never leaks to the LLM.
    """
    return Entity(
        name=name,
        position=position,
        tags=["hazard"],
        components=[Renderable(sprite_path=sprite, z_index=1)],
    )
