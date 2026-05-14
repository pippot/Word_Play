"""Ownership system - claim and give ownership.

Everything is public by default. Only add Owner + Ownable when you
want to restrict access to a specific agent.

Exports:
- Owner: Track who owns an entity
- Ownable: Marker for claimable entities (opt-in restriction)
- Target_Is_Unowned: Validation rule for unowned targets
- Claim: Claim ownership of an unowned entity
- Give_Ownership: Give your owned entity to another agent
"""
from __future__ import annotations

from word_play.core import Action, Action_Validation, Component, Entity, Environment, Target_Is_Nearby, Target_Not_Self
from word_play.presets.action_validations import Target_Has_Component


class Owner(Component):
    """Track who owns this entity."""

    def __init__(self, owner: Entity | None = None):
        super().__init__()
        self.owner = owner

    def is_owned(self) -> bool:
        return self.owner is not None

    def is_owned_by(self, entity: Entity) -> bool:
        return self.owner is entity

    def set_owner(self, entity: Entity | None) -> None:
        self.owner = entity


class Ownable(Component):
    """Marker component for claimable entities. Attach alongside Owner.

    By default, entities are public — anyone can interact. Only add
    Ownable + Owner when an entity should be restricted to one agent.
    """

    def __init__(self):
        super().__init__()


class Target_Is_Unowned(Action_Validation):
    """Validate target has no owner."""

    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        owner = target_entity.get_component(Owner)
        return owner is not None and not owner.is_owned()


class Claim(Action):
    """Claim ownership of an unowned entity."""

    def __init__(self):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(),
                Target_Has_Component(Owner),
                Target_Is_Unowned(),
            ]
        )

    def exec_action(self, actor: Entity, target_entity: Entity, env: Environment, kwargs: dict | None) -> dict | None:
        owner = target_entity.get_component(Owner)
        assert owner is not None
        owner.set_owner(actor)
        return {"claimed": target_entity.name, "owner": actor.name}

    def action_description_text(self, actor: Entity, target_entity: Entity, env: Environment) -> str:
        return f"Claim ownership of {target_entity.name}."


class Give_Ownership(Action):
    """Give your owned entity to another agent."""

    def __init__(self):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(),
                Target_Has_Component(Owner),
            ]
        )

    def is_valid(
        self, actor: Entity, target_entity: Entity, env: Environment, kwargs: dict | None | str = "unconsidered"
    ) -> bool:
        if not super().is_valid(actor, target_entity, env, kwargs=kwargs):
            return False
        owner = target_entity.get_component(Owner)
        # Must be owned by actor to give it away
        return owner is not None and owner.is_owned_by(actor)

    def exec_action(self, actor: Entity, target_entity: Entity, env: Environment, kwargs: dict | None) -> dict | None:
        owner = target_entity.get_component(Owner)
        assert owner is not None
        recipient = kwargs.get("recipient") if kwargs else None
        owner.set_owner(recipient)
        return {"transferred": target_entity.name, "to": recipient.name if recipient else None}

    def action_description_text(self, actor: Entity, target_entity: Entity, env: Environment) -> str:
        return f"Give ownership of {target_entity.name} to them."
