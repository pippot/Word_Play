"""Regrowable system — consumable entities that reappear on a timer.

Exports:
- Regrowable: Component for entities that can be consumed and regrow
- Consume_Regrowable: Action to consume a nearby regrowable entity
- Harvest_Regrowable: Action to harvest a nearby regrowable into inventory
"""

from __future__ import annotations

import random as _random
from typing import Callable

from word_play.core import Action, Component, Entity, Environment
from word_play.presets.action_validations import (
    Target_Has_Component,
    Target_Is_Nearby,
)
from word_play.presets.renderers.pygame_renderer.renderable import Renderable
from word_play.presets.systems.inventory import Inventory, materialize_item
from word_play.presets.systems.reward import Rewardable, award_reward


class Regrowable(Component):
    """An entity that can be consumed and regrows after a cooldown.

    Handles visibility toggling and regrowth timing. Supports three
    regrowth conditions: fixed interval, probability-per-tick, and
    density-dependent probability tables. It can also swap active/inactive
    tags while toggling visibility.
    """

    def __init__(
        self,
        *,
        regrow_tick_interval: int = 100,
        regrow_prob: float = 1.0,
        density_dependent: bool = False,
        density_radius: int = 2,
        density_regrow_probs: list[float] | None = None,
        active_tags: list[str] | None = None,
        inactive_tags: list[str] | None = None,
        consumed: bool = False,
    ):
        self.active_tags = active_tags or []
        self.inactive_tags = inactive_tags or []
        super().__init__(tags=list(self.inactive_tags if consumed else self.active_tags))
        self.regrow_tick_interval = regrow_tick_interval
        self.regrow_prob = regrow_prob
        self.density_dependent = density_dependent
        self.density_radius = density_radius
        self.density_regrow_probs = density_regrow_probs or [0.0, 0.0025, 0.005, 0.025]
        self._consumed = consumed

    @property
    def consumed(self) -> bool:
        return self._consumed

    def consume(self) -> bool:
        if self._consumed:
            return False
        self._consumed = True
        self._sync_tags()
        renderable = self.entity.get_component(Renderable) if hasattr(self, "entity") else None
        if renderable is not None:
            renderable.visible = False
        return True

    def regrow(self) -> bool:
        if not self._consumed:
            return False
        self._consumed = False
        self._sync_tags()
        renderable = self.entity.get_component(Renderable) if hasattr(self, "entity") else None
        if renderable is not None:
            renderable.visible = True
        return True

    def _sync_tags(self) -> None:
        if not hasattr(self, "entity"):
            return
        removed_tags = self.active_tags if self._consumed else self.inactive_tags
        added_tags = self.inactive_tags if self._consumed else self.active_tags
        for tag in removed_tags:
            while tag in self.entity.tags:
                self.entity.tags.remove(tag)
        for tag in added_tags:
            if tag not in self.entity.tags:
                self.entity.tags.append(tag)

    def _regrow_prob_this_tick(self, env: Environment) -> float:
        if not self.density_dependent:
            return self.regrow_prob
        nearby = sum(
            1 for e in env.state.entities
            if (r := e.get_component(Regrowable)) and not r.consumed and e is not self.entity
            and abs(self.entity.position.x - e.position.x) + abs(self.entity.position.y - e.position.y) <= self.density_radius
        )
        return self.density_regrow_probs[min(nearby, len(self.density_regrow_probs) - 1)]

    def pre_actions_step(self, env: Environment) -> None:
        if not self._consumed:
            return
        if env.tick % self.regrow_tick_interval != 0:
            return
        prob = self._regrow_prob_this_tick(env)
        if prob >= 1.0 or (prob > 0 and _random.random() < prob):
            self.regrow()


class Consume_Regrowable(Action):
    """Consume a nearby regrowable entity.

    If destroy=True (default), the entity is removed from the world after
    consuming (one-shot pickup). If destroy=False, the entity stays but is
    hidden and will regrow on its timer (common-pool style).
    """

    def __init__(self, *, reward: float | None = None, destroy: bool = True):
        super().__init__(validation_rules=[Target_Is_Nearby(), Target_Has_Component(Regrowable)])
        self.reward = reward
        self.destroy = destroy

    def exec_action(self, actor, target, env, kwargs=None):
        regrowable = target.get_component(Regrowable)
        if regrowable is None or regrowable.consumed:
            return {"success": False, "reason": "already_consumed" if regrowable else "not_regrowable"}
        regrowable.consume()
        reward = self.reward
        if reward is not None:
            award_reward(env, actor, reward)
        elif (rewardable := target.get_component(Rewardable)) is not None:
            reward = rewardable.reward_for(actor, env)
        result = {"success": True}
        if reward is not None:
            result["reward"] = reward
        if self.destroy:
            env.destroy_entity(target)
        return result

    def action_description_text(self, actor, target, env):
        return f"Consume {target.name}."


class Harvest_Regrowable(Action):
    """Harvest a nearby regrowable entity into inventory (destroy it, get item).

    Args:
        item_factory: Callable that returns an Entity to store in inventory.
            Can be either a zero-arg factory or a factory that takes the
            target entity as its argument (for target-dependent items).
    """

    def __init__(self, item_factory: Callable[[], Entity] | Callable[[Entity], Entity] | None = None):
        super().__init__(validation_rules=[Target_Is_Nearby(), Target_Has_Component(Regrowable)])
        self.item_factory = item_factory

    def exec_action(self, actor, target, env, kwargs=None):
        regrowable = target.get_component(Regrowable)
        inventory = actor.get_component(Inventory)
        if regrowable is None or inventory is None:
            return {"success": False, "reason": "missing_component"}
        if regrowable.consumed:
            return {"success": False, "reason": "already_consumed"}
        if not inventory.has_space():
            return {"success": False, "reason": "inventory_full"}
        regrowable.consume()
        if self.item_factory:
            try:
                item = self.item_factory(target)
            except TypeError:
                item = materialize_item(self.item_factory)
            inventory.store(item, env)
        env.destroy_entity(target)
        return {"success": True, "harvested": target.name}

    def action_description_text(self, actor, target, env):
        return f"Harvest {target.name}."
