"""Preference system — tag-based reward preferences for agents.

Exports:
- Preference: Component mapping target tags to reward amounts
"""
from __future__ import annotations

from word_play.core import Component, Entity, Environment


class Preference(Component):
    """Maps target entity tags to reward amounts.

    When an agent picks up or takes an item, the action can check the
    actor's Preference component to determine the reward based on the
    target's tags.

    Args:
        reward_map: Dict mapping tag strings to reward values.
            If a target has multiple matching tags, the first match wins.
        default_reward: Reward when no tag matches (default 1.0).
        mismatch_penalty: Penalty applied to *other* agents when this
            agent collects a non-preferred target. Set > 0 to enable
            (e.g., CoinPreference's mismatch logic).
    """

    def __init__(
        self,
        reward_map: dict[str, float] | None = None,
        default_reward: float = 1.0,
        mismatch_penalty: float = 0.0,
    ):
        super().__init__()
        self.reward_map = reward_map or {}
        self.default_reward = default_reward
        self.mismatch_penalty = mismatch_penalty

    def reward_for(self, target: Entity) -> float:
        """Return the reward for picking up / taking the given target."""
        for tag, reward in self.reward_map.items():
            if tag in target.tags:
                return reward
        return self.default_reward

    def is_preferred(self, target: Entity) -> bool:
        """Return whether the target's tags match any preferred tag."""
        for tag in self.reward_map:
            if tag in target.tags:
                return True
        return not self.reward_map

    def apply_mismatch_penalty(self, actor: Entity, target: Entity, env: Environment) -> None:
        """If the actor collected a non-preferred target, penalize other agents."""
        if self.mismatch_penalty <= 0:
            return
        if self.is_preferred(target):
            return
        if not hasattr(env, "last_step_rewards"):
            return
        for idx, agent in enumerate(env.agents):
            if agent is not actor and idx < len(env.last_step_rewards):
                env.last_step_rewards[idx] += self.mismatch_penalty
