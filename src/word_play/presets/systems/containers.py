"""Container system — chests, infinite sources, regen pools, and single-item holders.

Exports:
- Container: Chest-style storage with visibility control
- Infinite_Item_Source: Source that produces items on demand
- Regen_Pool: Shared resource pool with regeneration (tragedy of the commons)
- Single_Item_Holder: Container that holds exactly one item
- Open_Container: Action to open a container
- Take_From_Infinite_Source: Action to take from an infinite source into inventory
- Take_From_Regen_Pool: Action to take from a regen pool into inventory
"""
from __future__ import annotations

from copy import deepcopy
from typing import Callable

from word_play.core import Action, Component, Entity, Environment
from word_play.core.actions import Target_Not_Self, Target_Is_Nearby
from word_play.presets.action_validations import Target_Has_Component
from word_play.presets.systems.inventory import Inventory, _set_item_visibility, materialize_item


class Open_Container(Action):
    """Open a nearby container to reveal its contents."""

    def __init__(self):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(),
                Target_Has_Component(Container),
            ],
        )

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False
        container = target.get_component(Container)
        return container is not None and not container.is_open

    def exec_action(self, actor, target, env, kwargs) -> dict:
        container = target.get_component(Container)
        return container.open()

    def action_description_text(self, actor, target, env) -> str:
        return f"Open {target.name}."


class Take_From_Infinite_Source(Action):
    """Take an item from an Infinite_Item_Source directly into inventory.

    Args:
        reward: Fixed reward for taking, or None to check actor's
            Preference component for tag-based reward.
    """

    def __init__(self, reward: float | None = None):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(),
                Target_Has_Component(Infinite_Item_Source),
            ],
        )
        self.reward = reward

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False
        source = target.get_component(Infinite_Item_Source)
        if source is None:
            return False
        if source.available is not None and source.available <= 0:
            return False
        inv = actor.get_component(Inventory)
        if inv is None:
            return False
        return inv.has_space()

    def exec_action(self, actor, target, env, kwargs) -> dict:
        from word_play.presets.systems.preferences import Preference

        source = target.get_component(Infinite_Item_Source)
        item = source.create_item()
        if item is None:
            return {"error": "Source empty"}
        inv = actor.get_component(Inventory)
        if inv is None:
            return {"error": "No inventory"}
        if not inv.store(item, env):
            return {"error": "Inventory full"}

        reward = self.reward
        pref = actor.get_component(Preference) if reward is None else None
        if pref is not None:
            reward = pref.reward_for(item)
        if reward is not None and reward != 0:
            if hasattr(env, "last_step_rewards"):
                for idx, agent in enumerate(env.agents):
                    if agent is actor and idx < len(env.last_step_rewards):
                        env.last_step_rewards[idx] += reward
        if pref is not None:
            pref.apply_mismatch_penalty(actor, item, env)

        result = {"taken": item.name, "from": target.name}
        if reward is not None:
            result["reward"] = reward
        return result

    def action_description_text(self, actor, target, env) -> str:
        return f"Take item from {target.name}."


class Container(Inventory):
    """A chest-style container with hidden contents.

    Extends Inventory with visibility control.
    Items can be hidden (invisible until opened) or visible.
    """

    def __init__(
        self,
        contents: list[Entity] | None = None,
        *,
        max_size: int | None = None,
        accepted_tags: list[str] | None = None,
        starts_open: bool = False,
    ):
        super().__init__(contents=contents, max_size=max_size, accepted_tags=accepted_tags)
        self.is_open = starts_open

    def open(self) -> dict:
        """Open container and reveal hidden contents."""
        self.is_open = True
        for item in self.contents:
            _set_item_visibility(item, visible=True)
        return {"opened": True}


class Infinite_Item_Source(Component):
    """Source that dispenses items via inventory interface."""

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

    def create_item(self) -> Entity | None:
        """Create item. Returns None if at capacity."""
        if self.max_capacity is not None and self._dispensed >= self.max_capacity:
            return None
        self._dispensed += 1
        return materialize_item(self.item_factory)

    def dispense(self, env: Environment) -> dict:
        """Dispense an item into the environment."""
        item = self.create_item()
        if item is None:
            return {"error": "Source empty"}

        item.position = deepcopy(self.entity.position)
        if item not in env.state.entities:
            env.instantiate_entity(item)

        return {"dispensed": item.name}

    @property
    def available(self) -> int | None:
        if self.max_capacity is None:
            return None
        return self.max_capacity - self._dispensed

    def post_actions_step(self, env: "Environment") -> None:
        self._dispensed = max(0, self._dispensed - self.regen_rate + self.degen_rate)

    def reset(self) -> None:
        self._dispensed = 0


class Regen_Pool(Component):
    """Shared resource pool with depletion/regeneration dynamics.

    Unlike Infinite_Item_Source which has unlimited items, Regen_Pool has
    a finite stock that regrows proportionally to remaining population
    (logistic growth). Over-harvesting depletes the pool — tragedy of
    the commons.

    Args:
        item_factory: Callable that creates an Entity for each item taken.
        max_capacity: Maximum items the pool can hold.
        regen_rate: Base regrowth per tick (scaled by remaining fraction).
        restock_below: If > 0, auto-restock to this level if below it.
        track_contributions: If True, record who harvested how many.
    """

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
    """Take an item from a Regen_Pool directly into inventory.

    Args:
        reward: Fixed reward for taking, or None to check actor's
            Preference component for tag-based reward.
    """

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
        if pool is None or pool.available <= 0:
            return False
        inv = actor.get_component(Inventory)
        if inv is None:
            return False
        return inv.has_space()

    def exec_action(self, actor, target, env, kwargs) -> dict:
        from word_play.presets.systems.preferences import Preference

        pool = target.get_component(Regen_Pool)
        item = pool.create_item()
        if item is None:
            return {"error": "Pool depleted"}

        inv = actor.get_component(Inventory)
        if inv is None:
            pool.deposit(1)
            return {"error": "No inventory"}
        if not inv.store(item, env):
            pool.deposit(1)
            return {"error": "Inventory full"}

        if pool.track_contributions and pool._contributions is not None:
            pool._contributions[actor.name] = pool._contributions.get(actor.name, 0) + 1

        reward = self.reward
        pref = actor.get_component(Preference) if reward is None else None
        if pref is not None:
            reward = pref.reward_for(item)
        if reward is not None and reward != 0:
            if hasattr(env, "last_step_rewards"):
                for idx, agent in enumerate(env.agents):
                    if agent is actor and idx < len(env.last_step_rewards):
                        env.last_step_rewards[idx] += reward
        if pref is not None:
            pref.apply_mismatch_penalty(actor, item, env)

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
    """Restore items to a Regen_Pool (inverse of Take_From_Regen_Pool).

    Used for tragedy-of-the-commons scenarios where agents can spend
    effort to clean/restore a shared resource. Gives reward proportional
    to the amount actually restored.

    Args:
        efficiency: How many items to attempt restoring per action.
        reward_per_unit: Reward per unit actually restored.
    """

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
        if reward != 0 and hasattr(env, "last_step_rewards"):
            for idx, agent in enumerate(env.agents):
                if agent is actor and idx < len(env.last_step_rewards):
                    env.last_step_rewards[idx] += reward
        return {"success": True, "restored": restored, "remaining": pool.available, "reward": reward}

    def action_description_text(self, actor, target, env) -> str:
        pool = target.get_component(Regen_Pool)
        if pool:
            return f"Clean {target.name} ({pool.max_capacity - pool.available} depleted)."
        return f"Clean {target.name}."


class Single_Item_Holder(Container):
    """A container that can hold exactly one item."""

    def __init__(self, *, accepted_tags: list[str] | None = None):
        super().__init__(contents=[], max_size=1, accepted_tags=accepted_tags or [])

    @property
    def stored_item(self) -> Entity | None:
        return self.contents[0] if self.contents else None
