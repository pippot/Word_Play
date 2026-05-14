"""Delivery helpers — item validation, reward, and side-effect callables.

These are plain callables passed to Put_In_Container(reward=, on_stored=)
to configure delivery/deposit behavior. Deliver_Item is a thin convenience
subclass of Put_In_Container that wires these together.

Exports:
- Deliver_Item: Convenience action = Put_In_Container with tag validation + reward + callback
- Front_Order_Match: Validator that checks item matches front of order queue
- Named_Item_Reward: Reward function that pays based on item name
- Record_Delivery: Callback that updates score, order queue, and log
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Callable

from word_play.core import Entity, Environment
from word_play.presets.systems.inventory import (
    Inventory,
    Put_In_Container,
)


class Front_Order_Match:
    """Item validator: the item name must match the front of the order queue."""

    def __init__(self, order_attr: str = "order_queue"):
        self.order_attr = order_attr

    def __call__(self, actor: Entity, target_entity: Entity, env: Environment, item: Entity) -> bool:
        order_queue = getattr(env, self.order_attr, None)
        return isinstance(order_queue, Sequence) and bool(order_queue) and item.name == order_queue[0]


class Named_Item_Reward:
    """Reward function: pays by item name, with a default fallback."""

    def __init__(self, rewards_by_name: dict[str, float], default_reward: float = 0.0):
        self.rewards_by_name = rewards_by_name
        self.default_reward = default_reward

    def __call__(self, actor: Entity, target_entity: Entity, env: Environment, item: Entity) -> float:
        return float(self.rewards_by_name.get(item.name, self.default_reward))


class Record_Delivery:
    """Side-effect callback: updates score, order queue, and event log on delivery.

    All attributes are optional — set only the ones you need.
    """

    def __init__(
        self,
        score_attr: str = "score",
        step_score_attr: str | None = "step_score_delta",
        count_attr: str | None = None,
        order_attr: str | None = "order_queue",
        log_attr: str | None = None,
        log_limit: int | None = None,
        message_builder: Callable[[Entity, Entity, Environment, Entity, float], str] | None = None,
    ):
        self.score_attr = score_attr
        self.step_score_attr = step_score_attr
        self.count_attr = count_attr
        self.order_attr = order_attr
        self.log_attr = log_attr
        self.log_limit = log_limit
        self.message_builder = message_builder

    def __call__(self, actor: Entity, target_entity: Entity, env: Environment, item: Entity, reward: float) -> None:
        setattr(env, self.score_attr, int(getattr(env, self.score_attr, 0)) + int(reward))
        if self.step_score_attr is not None:
            setattr(env, self.step_score_attr, int(getattr(env, self.step_score_attr, 0)) + int(reward))
        if self.count_attr is not None:
            setattr(env, self.count_attr, int(getattr(env, self.count_attr, 0)) + 1)
        if self.order_attr is not None:
            order_queue = getattr(env, self.order_attr, None)
            if isinstance(order_queue, list) and order_queue:
                order_queue.pop(0)
        if self.log_attr is not None:
            existing_log = list(getattr(env, self.log_attr, []))
            message = (
                self.message_builder(actor, target_entity, env, item, reward)
                if self.message_builder is not None
                else f"{actor.name} delivered {item.name.replace('_', ' ')}."
            )
            updated_log = [message, *existing_log]
            setattr(env, self.log_attr, updated_log if self.log_limit is None else updated_log[:self.log_limit])


class Deliver_Item(Put_In_Container):
    """Deliver an inventory item to a tagged target for reward.

    Thin convenience wrapper over Put_In_Container that adds tag-based
    target validation, item validation, reward, and side-effect callback.
    """

    def __init__(
        self,
        target_tags: list[str],
        item_is_valid: Callable[[Entity, Entity, Environment, Entity], bool] | None = None,
        reward_for_item: Callable[[Entity, Entity, Environment, Entity], float] | None = None,
        on_delivered: Callable[[Entity, Entity, Environment, Entity, float], None] | None = None,
    ):
        super().__init__(
            target_tags=target_tags,
            destroy_item=True,
        )
        self.item_is_valid = item_is_valid
        self.reward_for_item = reward_for_item
        self.on_delivered = on_delivered

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False
        if kwargs is None or kwargs == "unconsidered":
            return True
        inv = actor.get_component(Inventory)
        if inv is None:
            return False
        idx = int(kwargs.get("inventory_index", -1))
        if not (0 <= idx < len(inv.contents)):
            return False
        if self.item_is_valid and not self.item_is_valid(actor, target, env, inv.contents[idx]):
            return False
        return True

    def exec_action(self, actor, target, env, kwargs) -> dict | None:
        assert kwargs is not None
        inv = actor.get_component(Inventory)
        assert inv is not None
        idx = int(kwargs["inventory_index"])
        item = inv.contents[idx]

        # Compute reward via callback if provided, else 0
        reward = 0.0
        if self.reward_for_item is not None:
            reward = float(self.reward_for_item(actor, target, env, item))

        # Remove and destroy the item (delivery consumes it)
        inv.remove(item)
        if item in env.state.entities:
            env.destroy_entity(item)

        # Apply reward
        if reward != 0 and hasattr(env, "last_step_rewards"):
            for idx_a, agent in enumerate(env.agents):
                if agent is actor and idx_a < len(env.last_step_rewards):
                    env.last_step_rewards[idx_a] += reward

        # Side effects
        if self.on_delivered is not None:
            self.on_delivered(actor, target, env, item, reward)

        return {"delivered_item": item.name, "reward": reward}

    def action_description_text(self, actor, target, env) -> str:
        return f"Deliver an inventory item to {target.name}."
