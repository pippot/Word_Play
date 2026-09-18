"""
Action_Validation rules: the preconditions that decide which actions an agent
is even offered on a given step.
"""

from __future__ import annotations

from word_play.core import Action_Validation, Entity

def available_supplies(actor: Entity, env) -> list[Entity]:
    """
    Supply units the actor could pick up right now, lowest-numbered first:
    within reach (Manhattan distance <= 1), not carried by anyone, and not
    delivered/discarded earlier in this same step (those are destroyed at the
    end of the step, so they're still in state.entities when a later-acting
    agent's action is checked).
    """
    carried = set(env.carrying.values())
    units = [
        e for e in env.state.entities
        if "supply" in e.tags
        and e not in carried
        and e not in env._supplies_awaiting_removal
        and abs(actor.position.x - e.position.x) + abs(actor.position.y - e.position.y) <= 1
    ]
    return sorted(units, key=lambda e: int(e.name.rsplit("_", 1)[-1]))


class Supply_Within_Reach(Action_Validation):
    """True if at least one supply unit is available to the actor (see available_supplies)."""
    def is_valid(self, actor: Entity, target_entity: Entity, env) -> bool:
        return bool(available_supplies(actor, env))


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
