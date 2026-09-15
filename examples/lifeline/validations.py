"""
Action_Validation rules: the preconditions that decide which actions an agent
is even offered on a given step.
"""

from __future__ import annotations

from word_play.core import Action_Validation, Entity
from word_play.presets.systems.communication.core import Communication_Policy

from .config import TALK_RADIUS

class Is_Adjacent_To(Action_Validation):
    """True if the actor is adjacent (Manhattan distance <= 1) to the target."""
    def is_valid(self, actor: Entity, target_entity: Entity, env) -> bool:
        dist = abs(actor.position.x - target_entity.position.x) + \
               abs(actor.position.y - target_entity.position.y)
        return dist <= 1


class Target_Is_Supply(Action_Validation):
    """True if the target entity is an unclaimed supply unit."""
    def is_valid(self, actor: Entity, target_entity: Entity, env) -> bool:
        return "supply" in target_entity.tags


class Supply_Not_Carried(Action_Validation):
    """
    True if the target supply unit is available: not carried, and not already
    delivered/discarded earlier in this same step (those are destroyed at the
    end of the step, so they're still in state.entities when a later-acting
    agent evaluates its options).
    """
    def is_valid(self, actor: Entity, target_entity: Entity, env) -> bool:
        return (
            target_entity not in env.carrying.values()
            and target_entity not in env._supplies_awaiting_removal
        )


class Not_Already_Carrying(Action_Validation):
    """True if the actor isn't already carrying a supply unit."""
    def is_valid(self, actor: Entity, target_entity: Entity, env) -> bool:
        return actor not in env.carrying


class Is_Carrying_Supply(Action_Validation):
    """True if the actor is currently carrying a supply unit."""
    def is_valid(self, actor: Entity, target_entity: Entity, env) -> bool:
        return actor in env.carrying


class At_A_Zone(Action_Validation):
    """True if the actor is standing exactly on a zone tile."""
    def is_valid(self, actor: Entity, target_entity: Entity, env) -> bool:
        return any(
            actor.position.x == z.position.x and actor.position.y == z.position.y
            for z in env.zones.values()
        )


class Near_The_Board(Action_Validation):
    """True if the actor is adjacent to the shared board."""
    def is_valid(self, actor: Entity, target_entity: Entity, env) -> bool:
        dist = abs(actor.position.x - env.board.position.x) + \
               abs(actor.position.y - env.board.position.y)
        return dist <= 1


def conversation_partners_in_range(actor: Entity, env) -> list[Entity]:
    """
    Agents within TALK_RADIUS of the actor.

    The engine's own nearby_conversation_partners() uses the movement system's
    positions_are_close, which for this grid means *the exact same tile*. On a
    map where nothing forces agents to converge, that makes conversation
    almost impossible, so this environment uses a Manhattan radius instead.
    """
    return [
        entity for entity in env.state.entities
        if entity is not actor
        and entity.has_component(Communication_Policy)
        and abs(entity.position.x - actor.position.x)
        + abs(entity.position.y - actor.position.y) <= TALK_RADIUS
    ]


class A_Partner_Is_In_Talk_Range(Action_Validation):
    """True if at least one other agent is within TALK_RADIUS."""
    def is_valid(self, actor: Entity, target_entity: Entity, env) -> bool:
        return len(conversation_partners_in_range(actor, env)) > 0


class Talk_Not_On_Cooldown(Action_Validation):
    """True if enough steps have passed since the agent last talked."""
    TALK_COOLDOWN = 4

    def is_valid(self, actor: Entity, target_entity: Entity, env) -> bool:
        last = env.last_talk_step.get(actor.name, -self.TALK_COOLDOWN)
        return (env.cur_step - last) >= self.TALK_COOLDOWN
