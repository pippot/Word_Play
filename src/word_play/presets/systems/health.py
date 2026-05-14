"""Health system — HP tracking and damage/death."""
from __future__ import annotations

from word_play.core import Component, Environment


class Health(Component):
    """Health component with max/current health tracking."""

    def __init__(self, max_health: float, starting_health: float | None = None):
        super().__init__()
        self.max_health = max_health
        self.health = starting_health if starting_health is not None else max_health

    def heal(self, amount: float) -> float:
        """Heal by amount. Returns actual amount healed."""
        old_health = self.health
        self.health = min(self.health + amount, self.max_health)
        return self.health - old_health

    def damage(self, amount: float) -> float:
        """Take damage. Returns actual damage taken."""
        old_health = self.health
        self.health -= amount
        return old_health - self.health

    def post_actions_step(self, env: Environment) -> None:
        if self.health <= 0:
            env.destroy_entity(self.entity)
