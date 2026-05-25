"""Reward system — target-side reward definition and awarding.

Exports:
- Rewardable: Component on target entities that defines the reward for interacting with them
- award_reward: Helper to add reward to an actor's last_step_rewards
"""
from __future__ import annotations

from typing import Callable, Literal

from word_play.core import Component, Entity, Environment


class Rewardable(Component):
    """Defines the reward an actor receives for interacting with this entity.

    Put this on the *target* — the item being picked up, the pool being
    taken from, the counter being delivered to, etc. Actions call
    `target.get_component(Rewardable).reward_for(actor)` to determine
    and apply the reward.

    Resolution order:
    1. If this Rewardable has a fixed `amount`, use that.
    2. Else if the actor has a Preference component, use `pref.reward_for(target)`.
    3. Else no reward.

    After awarding, also applies the actor's Preference mismatch_penalty
    if the target wasn't preferred.

    Args:
        amount: Fixed reward amount, or None to delegate to actor's Preference.
        on_reward: Optional callback(actor, target, env, reward) after reward is applied.
        recipients: Who receives the reward: the actor, everyone except the actor, or all agents.
        counter_attr: Optional env attribute to increment after a reward interaction.
    """

    def __init__(
        self,
        amount: float | None = None,
        on_reward: Callable[[Entity, Entity, Environment, float], None] | None = None,
        *,
        recipients: Literal["actor", "others", "all"] = "actor",
        counter_attr: str | None = None,
    ):
        super().__init__()
        self.amount = amount
        self.on_reward = on_reward
        self.recipients = recipients
        self.counter_attr = counter_attr

    def reward_for(self, actor: Entity, env: Environment) -> float | None:
        """Compute and apply reward for the actor interacting with this entity."""
        from word_play.presets.systems.preferences import Preference

        reward = self.amount
        pref = actor.get_component(Preference) if reward is None else None
        if pref is not None:
            reward = pref.reward_for(self.entity)

        if reward is not None and reward != 0:
            self._award_recipients(actor, env, reward)

        if pref is not None:
            pref.apply_mismatch_penalty(actor, self.entity, env)

        if self.counter_attr is not None:
            setattr(env, self.counter_attr, getattr(env, self.counter_attr, 0) + 1)

        if reward is not None and self.on_reward is not None:
            self.on_reward(actor, self.entity, env, reward)

        return reward

    def _award_recipients(self, actor: Entity, env: Environment, reward: float) -> None:
        if self.recipients == "actor":
            award_reward(env, actor, reward)
            return
        for agent in env.agents:
            if self.recipients == "others" and agent is actor:
                continue
            award_reward(env, agent, reward)


def award_reward(env: Environment, actor: Entity, reward: float) -> None:
    """Add reward to the actor's entry in env.last_step_rewards."""
    if not hasattr(env, "last_step_rewards"):
        return
    for idx, agent in enumerate(env.agents):
        if agent is actor and idx < len(env.last_step_rewards):
            env.last_step_rewards[idx] += reward
            return
