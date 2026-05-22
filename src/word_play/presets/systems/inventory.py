"""Inventory system — base storage class, item arguments, and inventory actions.

Exports:
- Inventory: Generic storage component with capacity and tag filtering
- Inventory_Item_Index_Arg: Action arg for single item selection
- Inventory_Items_Arg: Action arg for multi-item selection
- Pick_Up_Item: Action to pick up a nearby item
- Put_In_Container: Action to transfer item to a container (also covers delivery/deposit)
- Drop_Item: Action to drop an inventory item onto the ground
- materialize_item: Create an Entity from a spec or factory
"""
from __future__ import annotations

from copy import deepcopy
from typing import Callable

from word_play.core import Action, Action_Validation, Component, Entity, Environment, Action_Arg
from word_play.presets.action_validations import (
    Target_Has_Component,
    Target_Has_Tag,
    Target_Is_Nearby,
    Target_Is_Self,
    Target_Not_Self,
)
from word_play.presets.renderers.renderer import Renderable


class Target_Not_In_Inventory(Action_Validation):
    """Target entity is not stored in any agent's inventory."""

    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        for entity in env.state.entities:
            inv = entity.get_component(Inventory)
            if inv and target_entity in inv.contents:
                return False
        return True


class Inventory_Item_Index_Arg(Action_Arg):
    """Argument for selecting an item from inventory by index."""

    def parse(self, input: str) -> int:
        return int(input)

    def arg_description(self, actor: Entity, target_entity: Entity, env: Environment) -> str:
        inv = actor.get_component(Inventory)
        if inv and inv.contents:
            items = ", ".join(f"{i}:{item.name}" for i, item in enumerate(inv.contents))
            return f"Inventory items: {items}"
        return "No items in inventory"


class Inventory_Items_Arg(Action_Arg):
    """Argument for selecting multiple items by comma-separated indices."""

    def parse(self, input: str | list) -> list[int]:
        if not input:
            return []
        if isinstance(input, list):
            return [int(i) for i in input]
        return [int(i.strip()) for i in str(input).split(",") if i.strip()]

    def arg_description(self, actor: Entity, target_entity: Entity, env: Environment) -> str:
        inv = actor.get_component(Inventory)
        if inv and inv.contents:
            items = ", ".join(f"{i}:{item.name}" for i, item in enumerate(inv.contents))
            return f"Inventory items (comma-separated indices like '0,1'): {items}"
        return "No items in inventory"


def _set_item_visibility(item: Entity, *, visible: bool) -> None:
    """Set item visibility via its Renderable component."""
    renderable = item.get_component(Renderable)
    if renderable is not None:
        renderable.visible = visible


def materialize_item(spec: Entity | Callable[[], Entity]) -> Entity:
    """Create an item from spec (entity or factory)."""
    entity = spec() if callable(spec) else deepcopy(spec)
    if not isinstance(entity, Entity):
        raise TypeError("Expected an Entity or a factory that returns an Entity.")
    return entity


class Inventory(Component):
    """Generic storage for items. Base class for inventories, containers, sources.

    Pure data component — all interaction logic lives in actions.
    """

    def __init__(
        self,
        contents: list[Entity] | None = None,
        *,
        max_size: int | None = None,
        accepted_tags: list[str] | None = None,
    ):
        super().__init__()
        self.contents = contents or []
        self.max_size = max_size
        self.accepted_tags = accepted_tags or []

    def can_accept(self, item: Entity) -> bool:
        """Check if item can be stored (capacity + tag check)."""
        if not self.has_space():
            return False
        if self.accepted_tags:
            return any(t in item.tags for t in self.accepted_tags)
        return True

    def has_space(self) -> bool:
        """Check if inventory has room for more items."""
        if self.max_size is None:
            return True
        return len(self.contents) < self.max_size

    def store(self, item: Entity, env: Environment | None = None) -> bool:
        """Store an item. Returns False if rejected."""
        if not self.can_accept(item):
            return False

        self.contents.append(item)

        if env is not None and hasattr(self, 'entity'):
            item.position = deepcopy(self.entity.position)
            if item not in env.state.entities:
                env.instantiate_entity(item)

        return True

    def remove(self, item: Entity) -> Entity | None:
        """Remove item from inventory. Returns item or None."""
        if item in self.contents:
            self.contents.remove(item)
            return item
        return None

    def remove_by_index(self, index: int) -> Entity | None:
        """Remove item by index. Returns item or None."""
        if 0 <= index < len(self.contents):
            return self.remove(self.contents[index])
        return None

    def post_actions_step(self, env: Environment) -> None:
        """Keep items synced with holder's position."""
        for item in self.contents:
            if item in env.state.entities:
                item.position = deepcopy(self.entity.position)


class Pick_Up_Item(Action):
    """Pick up an item from the environment into inventory.

    Args:
        item_tags: Target must have at least one of these tags.
        reward: Fixed reward for picking up, or None to check actor's Preference component for tag-based reward.
    """

    def __init__(self, item_tags: list[str] | None = None, reward: float | None = None):
        self.item_tags = item_tags or ["collectable"]
        super().__init__(
            validation_rules=[
                Target_Has_Tag(self.item_tags),
                Target_Not_In_Inventory(),
                Target_Not_Self(),
                Target_Is_Nearby(),
            ]
        )
        self.reward = reward

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False
        inv = actor.get_component(Inventory)
        if inv is None:
            return False
        return any(tag in target.tags for tag in self.item_tags)

    def exec_action(self, actor, target, env, kwargs) -> dict:
        from word_play.presets.systems.preferences import Preference

        inv = actor.get_component(Inventory)
        if inv is None:
            return {"error": f"{actor.name} has no inventory"}
        if not inv.can_accept(target):
            return {"error": "Cannot pick up item"}

        # Remove from other inventories if present
        for other in env.state.entities:
            other_inv = other.get_component(Inventory)
            if other_inv and target in other_inv.contents:
                other_inv.remove(target)
                break

        # Store in actor's inventory and sync position
        inv.store(target, env)

        _set_item_visibility(target, visible=False)

        # Determine reward
        reward = self.reward
        pref = actor.get_component(Preference) if reward is None else None
        if pref is not None:
            reward = pref.reward_for(target)
        if reward is not None and reward != 0:
            if hasattr(env, "last_step_rewards"):
                for idx, agent in enumerate(env.agents):
                    if agent is actor and idx < len(env.last_step_rewards):
                        env.last_step_rewards[idx] += reward
        if pref is not None:
            pref.apply_mismatch_penalty(actor, target, env)

        result = {"picked_up": target.name}
        if reward is not None:
            result["reward"] = reward
        return result

    def action_description_text(self, actor, target, env) -> str:
        return f"Pick up {target.name}."


class Put_In_Container(Action):
    """Put an inventory item into a container.

    Also serves as the delivery/deposit action when configured with reward, target_tags, and on_stored.
    With destroy_item=True (default when reward is set), the item is consumed on delivery rather than kept
    in the target's inventory — matching the common pattern where delivering an item to a counter or grate
    removes it from the world.

    Args:
        target_tags: If set, target must have one of these tags instead of just having an Inventory component.
        reward: Fixed reward, or None to check actor's Preference component.
        on_stored: Callback(actor, target, env, item, reward) after successful store.
        destroy_item: If True, the item is destroyed after storing (consumed on delivery).
            Default True when reward is set, False otherwise.
    """

    def __init__(
        self,
        target_tags: list[str] | None = None,
        reward: float | None = None,
        on_stored: Callable[[Entity, Entity, Environment, Entity, float], None] | None = None,
        destroy_item: bool | None = None,
    ):
        if target_tags:
            validation = [Target_Not_Self(), Target_Is_Nearby(), Target_Has_Tag(target_tags)]
        else:
            validation = [Target_Not_Self(), Target_Is_Nearby(), Target_Has_Component(Inventory)]
        super().__init__(
            validation_rules=validation,
            required_kwargs={"inventory_index": Inventory_Item_Index_Arg()},
        )
        self.target_tags = target_tags
        self.reward = reward
        self.on_stored = on_stored
        self.destroy_item = destroy_item if destroy_item is not None else (reward is not None)

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False
        inv = actor.get_component(Inventory)
        if inv is None or len(inv.contents) == 0:
            return False
        # When not using tag-based validation, check target has Inventory
        if not self.target_tags:
            container = target.get_component(Inventory)
            if container is None:
                return False
        if kwargs is None or kwargs == "unconsidered":
            return True
        idx = int(kwargs.get("inventory_index", -1))
        if not (0 <= idx < len(inv.contents)):
            return False
        # Tag-based targets skip container capacity check
        if not self.target_tags:
            container = target.get_component(Inventory)
            if container and not container.can_accept(inv.contents[idx]):
                return False
        return True

    def exec_action(self, actor, target, env, kwargs) -> dict:
        from word_play.presets.systems.preferences import Preference
        from word_play.presets.systems.containers import Regen_Pool

        inv = actor.get_component(Inventory)
        assert inv is not None
        idx = int(kwargs["inventory_index"])
        item = inv.contents[idx]
        item_name = item.name

        # Determine reward before removing item
        reward = self.reward
        pref = actor.get_component(Preference) if reward is None else None
        if pref is not None:
            reward = pref.reward_for(item)

        # Try Regen_Pool deposit first (just increments count, doesn't store item)
        pool = target.get_component(Regen_Pool)
        if pool is not None:
            inv.remove(item)
            pool.deposit(1)
            if item in env.state.entities:
                env.destroy_entity(item)
        elif self.destroy_item:
            # Remove from inventory and destroy (delivery pattern)
            inv.remove(item)
            if item in env.state.entities:
                env.destroy_entity(item)
        else:
            # Standard container transfer
            container = target.get_component(Inventory)
            if container is None:
                return {"error": f"{target.name} has no inventory"}
            if not container.can_accept(item):
                return {"error": f"Receiver cannot accept {item.name}"}
            inv.remove(item)
            container.store(item, env)

        if reward is not None and reward != 0:
            if hasattr(env, "last_step_rewards"):
                for idx_a, agent in enumerate(env.agents):
                    if agent is actor and idx_a < len(env.last_step_rewards):
                        env.last_step_rewards[idx_a] += reward
        if pref is not None:
            pref.apply_mismatch_penalty(actor, item, env)
        if self.on_stored is not None:
            self.on_stored(actor, target, env, item, reward if reward is not None else 0.0)

        result = {"stored": item_name, "container": target.name}
        if reward is not None:
            result["reward"] = reward
        return result

    def action_description_text(self, actor, target, env) -> str:
        return f"Put item into {target.name}."


class Drop_Item(Action):
    """Drop an inventory item onto the ground at the actor's position."""

    def __init__(self):
        super().__init__(
            validation_rules=[Target_Is_Self()],
            required_kwargs={"inventory_index": Inventory_Item_Index_Arg()},
        )

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False
        inv = actor.get_component(Inventory)
        return inv is not None and len(inv.contents) > 0

    def exec_action(self, actor, target, env, kwargs) -> dict:
        inv = actor.get_component(Inventory)
        idx = int(kwargs["inventory_index"])
        if not (0 <= idx < len(inv.contents)):
            return {"error": "Invalid index"}
        item = inv.remove_by_index(idx)
        assert item is not None
        item.position = deepcopy(actor.position)
        _set_item_visibility(item, visible=True)
        return {"dropped": item.name}

    def action_description_text(self, actor, target, env) -> str:
        return "Drop an inventory item."
