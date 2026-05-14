"""Cooldown system — per-action cooldown tracking on entities.

Exports:
- Cooldown: Component tracking remaining cooldown ticks per action key
- Action_On_Cooldown: Validation rule that blocks actions while on cooldown
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from word_play.core import Action_Validation, Component

if TYPE_CHECKING:
    from word_play.core import Entity, Environment


class Cooldown(Component):
    """Tracks per-action cooldown ticks on an entity.

    Add this component to any entity that needs cooldowns.
    Each action is identified by a string key. Tick the cooldowns
    down automatically each step via pre_actions_step.
    """

    def __init__(self, cooldowns: dict[str, int] | None = None):
        """Args:
            cooldowns: Map of action key → cooldown duration in ticks.
                Durations are templates; the actual remaining ticks are
                tracked in _remaining and reset from these values when
                an action fires.
        """
        super().__init__()
        self.durations: dict[str, int] = cooldowns or {}
        self._remaining: dict[str, int] = {}

    def start(self, key: str) -> None:
        """Put `key` on cooldown using its configured duration."""
        self._remaining[key] = self.durations.get(key, 0)

    def start_custom(self, key: str, duration: int) -> None:
        """Put `key` on cooldown with an arbitrary duration."""
        self._remaining[key] = duration

    def remaining(self, key: str) -> int:
        return self._remaining.get(key, 0)

    def on_cooldown(self, key: str) -> bool:
        return self.remaining(key) > 0

    def pre_actions_step(self, env: Environment) -> None:
        for key in list(self._remaining):
            if self._remaining[key] > 0:
                self._remaining[key] -= 1


class Action_On_Cooldown(Action_Validation):
    """Validation rule that fails when the named action key is on cooldown.

    Add this to an action's validation_rules list and give it the same
    key you register on the actor's Cooldown component. The action will
    be filtered out of possible_actions while on cooldown.
    """

    def __init__(self, key: str):
        self.key = key

    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        cooldown = actor.get_component(Cooldown)
        if cooldown is None:
            return True
        return not cooldown.on_cooldown(self.key)
