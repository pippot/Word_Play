"""Container-adjacent presets: infinite item sources and regenerating pools."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

from word_play.core import Action, Component, Entity, Environment, Target_Is_Nearby, Target_Not_Self
from word_play.presets.action_validations import Target_Has_Component
from word_play.presets.systems.inventory import Inventory, materialize_item
from word_play.presets.systems.reward import award_reward


class Regrowable_Item_Source(Component):
    """Source that dispenses items through the inventory interface."""

    def __init__(
        self,
        item_factory: Entity | Callable[[], Entity],
        *,
        max_capacity: int | None = None,
        regen_rate: int = 0,
        degen_rate: int = 0,
    ):
        super().__init__()
        self.item_factory = item_factory
        self.max_capacity = max_capacity
        self.regen_rate = regen_rate
        self.degen_rate = degen_rate
        self._dispensed = 0

    @property
    def available(self) -> int | None:
        if self.max_capacity is None:
            return None
        return self.max_capacity - self._dispensed

    def create_item(self) -> Entity | None:
        if self.max_capacity is not None and self._dispensed >= self.max_capacity:
            return None
        self._dispensed += 1
        return materialize_item(self.item_factory)

    def dispense(self, env: Environment) -> dict:
        item = self.create_item()
        if item is None:
            return {"error": "Source empty"}

        item.position = deepcopy(self.entity.position)
        if item not in env.state.entities:
            env.instantiate_entity(item)
        return {"dispensed": item.name}

    def post_actions_step(self, env: Environment) -> None:
        self._dispensed = max(0, self._dispensed - self.regen_rate + self.degen_rate)

    def reset(self) -> None:
        self._dispensed = 0


class Take_From_Infinite_Source(Action):
    """Take an item from a Regrowable_Item_Source directly into inventory."""

    def __init__(self, reward: float | None = None):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(),
                Target_Has_Component(Regrowable_Item_Source),
            ],
        )
        self.reward = reward

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False
        source = target.get_component(Regrowable_Item_Source)
        if source is None or (source.available is not None and source.available <= 0):
            return False
        inventory = actor.get_component(Inventory)
        return inventory is not None and inventory.has_space()

    def exec_action(self, actor, target, env, kwargs) -> dict:
        from word_play.presets.systems.preferences import Preference

        source = target.get_component(Regrowable_Item_Source)
        item = source.create_item()
        if item is None:
            return {"error": "Source empty"}

        inventory = actor.get_component(Inventory)
        if inventory is None:
            return {"error": "No inventory"}
        if not inventory.store(item, env):
            return {"error": "Inventory full"}

        reward = self.reward
        preference = actor.get_component(Preference) if reward is None else None
        if preference is not None:
            reward = preference.reward_for(item)
        if reward is not None and reward != 0:
            award_reward(env, actor, reward)
        if preference is not None:
            preference.apply_mismatch_penalty(actor, item, env)

        result = {"taken": item.name, "from": target.name}
        if reward is not None:
            result["reward"] = reward
        return result

    def action_description_text(self, actor, target, env) -> str:
        source = target.get_component(Regrowable_Item_Source)
        item_factory = source.item_factory if source is not None else None
        item_name = item_factory.name if isinstance(item_factory, Entity) else "item"
        return f"Take {item_name} from {target.name} into your inventory."


class Regen_Pool(Component):
    """Shared finite source with regeneration dynamics."""

    def __init__(
        self,
        item_factory: Entity | Callable[[], Entity],
        *,
        max_capacity: int = 10,
        regen_rate: int = 2,
        restock_below: int = 0,
        track_contributions: bool = False,
    ):
        super().__init__()
        self.item_factory = item_factory
        self.max_capacity = max_capacity
        self.regen_rate = regen_rate
        self.restock_below = restock_below
        self.track_contributions = track_contributions
        self._available = max_capacity
        self._contributions: dict[str, int] | None = {} if track_contributions else None

    @property
    def available(self) -> int:
        return self._available

    @property
    def depletion_ratio(self) -> float:
        return 1.0 - (self._available / self.max_capacity)

    def create_item(self) -> Entity | None:
        if self._available <= 0:
            return None
        self._available -= 1
        return materialize_item(self.item_factory)

    def deposit(self, amount: int = 1) -> int:
        actual = min(amount, self.max_capacity - self._available)
        self._available += actual
        return actual

    def pre_actions_step(self, env: Environment) -> None:
        if self.restock_below > 0 and self._available < self.restock_below:
            self.deposit(self.restock_below - self._available)

    def post_actions_step(self, env: Environment) -> None:
        if self._available > 0:
            growth_rate = self._available / self.max_capacity
            regen = max(0, int(self.regen_rate * growth_rate))
            self._available = min(self.max_capacity, self._available + regen)

    def reset(self) -> None:
        self._available = self.max_capacity
        if self._contributions is not None:
            self._contributions = {}


class Take_From_Regen_Pool(Action):
    """Take an item from a Regen_Pool directly into inventory."""

    def __init__(self, reward: float | None = None):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(),
                Target_Has_Component(Regen_Pool),
            ],
        )
        self.reward = reward

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False
        pool = target.get_component(Regen_Pool)
        inventory = actor.get_component(Inventory)
        return pool is not None and pool.available > 0 and inventory is not None and inventory.has_space()

    def exec_action(self, actor, target, env, kwargs) -> dict:
        from word_play.presets.systems.preferences import Preference

        pool = target.get_component(Regen_Pool)
        item = pool.create_item()
        if item is None:
            return {"error": "Pool depleted"}

        inventory = actor.get_component(Inventory)
        if inventory is None:
            pool.deposit(1)
            return {"error": "No inventory"}
        if not inventory.store(item, env):
            pool.deposit(1)
            return {"error": "Inventory full"}

        if pool.track_contributions and pool._contributions is not None:
            pool._contributions[actor.name] = pool._contributions.get(actor.name, 0) + 1

        reward = self.reward
        preference = actor.get_component(Preference) if reward is None else None
        if preference is not None:
            reward = preference.reward_for(item)
        if reward is not None and reward != 0:
            award_reward(env, actor, reward)
        if preference is not None:
            preference.apply_mismatch_penalty(actor, item, env)

        result = {"taken": item.name, "from": target.name, "remaining": pool.available}
        if reward is not None:
            result["reward"] = reward
        return result

    def action_description_text(self, actor, target, env) -> str:
        pool = target.get_component(Regen_Pool)
        if pool and pool.available > 0:
            return f"Take from {target.name} ({pool.available} remaining)."
        return f"{target.name} is depleted."


class Clean_Pool(Action):
    """Restore items to a Regen_Pool."""

    def __init__(self, *, efficiency: int = 10, reward_per_unit: float = 1.0):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(),
                Target_Has_Component(Regen_Pool),
            ],
        )
        self.efficiency = efficiency
        self.reward_per_unit = reward_per_unit

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False
        pool = target.get_component(Regen_Pool)
        return pool is not None and pool.available < pool.max_capacity

    def exec_action(self, actor, target, env, kwargs) -> dict:
        pool = target.get_component(Regen_Pool)
        if pool is None:
            return {"success": False, "reason": "not_pool"}
        restored = pool.deposit(self.efficiency)
        reward = restored * self.reward_per_unit
        if reward != 0:
            award_reward(env, actor, reward)
        return {"success": True, "restored": restored, "remaining": pool.available, "reward": reward}

    def action_description_text(self, actor, target, env) -> str:
        pool = target.get_component(Regen_Pool)
        if pool:
            return f"Clean {target.name} ({pool.max_capacity - pool.available} depleted)."
        return f"Clean {target.name}."
