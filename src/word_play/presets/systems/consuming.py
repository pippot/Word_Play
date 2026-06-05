from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from word_play.core import Action, Component, Entity, Environment, Target_Is_Nearby, Target_Not_Self
from word_play.presets.action_validations import Target_Has_Component
from word_play.presets.systems.health import Health
from word_play.presets.systems.inventory import In_Actor_Inventory, Inventory

Consume_Effect = Callable[[Entity, Entity, Environment], None]


def heal(amount: float) -> Consume_Effect:
    def effect(actor: Entity, consumed_entity: Entity, env: Environment) -> None:
        h = actor.get_component(Health)
        h.health = min(h.health + amount, h.max_health)
    return effect


def damage(amount: float) -> Consume_Effect:
    def effect(actor: Entity, consumed_entity: Entity, env: Environment) -> None:
        actor.get_component(Health).health -= amount
    return effect


def full_heal() -> Consume_Effect:
    def effect(actor: Entity, consumed_entity: Entity, env: Environment) -> None:
        h = actor.get_component(Health)
        h.health = h.max_health
    return effect


class Consumable(Component):

    def __init__(self, on_consume: Consume_Effect | list[Consume_Effect]):
        """on_consume(actor, consumed_entity, env) — one effect or a list of effects applied in order."""
        super().__init__()
        self._effects: list[Consume_Effect] = on_consume if isinstance(on_consume, list) else [on_consume]

    def on_consume(self, actor: Entity, consumed_entity: Entity, env: Environment) -> None:
        for effect in self._effects:
            effect(actor, consumed_entity, env)


class Consume(Action):

    def __init__(self):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(),
                In_Actor_Inventory(),
                Target_Has_Component(Consumable),
            ]
        )

    def exec_action(self, actor: Entity, target_entity: Entity, env: Environment, kwargs: dict | None) -> dict | None:
        target_entity.get_component(Consumable).on_consume(actor, target_entity, env)
        inventory = actor.get_component(Inventory)
        inventory.inventory.remove(target_entity)
        target_entity.tags.remove("in_inventory")
        env.destroy_entity(target_entity)

    def action_description_text(self, actor: Entity, target_entity: Entity, env: Environment) -> str:
        return f"Consume {target_entity.name}."
