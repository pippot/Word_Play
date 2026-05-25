"""Stamina system - gate actions on a reusable resource pool."""
from __future__ import annotations

from word_play.core import Action_Validation, Component, Entity, Environment


class Stamina(Component):
    def __init__(
        self,
        maximum: int = 10,
        *,
        current: int | None = None,
        action_costs: dict[str, int] | list[tuple[str, int]] | None = None,
        recover_per_step: int = 1,
    ):
        super().__init__()
        self.maximum = maximum
        self.current = maximum if current is None else min(current, maximum)
        self.action_costs = dict(action_costs or {})
        self.recover_per_step = recover_per_step
        self._spent_this_step = False

    def cost(self, action_key: str) -> int:
        return self.action_costs.get(action_key, 0)

    def can_spend(self, action_key: str) -> bool:
        return self.current >= self.cost(action_key)

    def spend(self, action_key: str) -> None:
        cost = self.cost(action_key)
        if cost <= 0:
            return
        self.current = max(0, self.current - cost)
        self._spent_this_step = True

    def recover(self, amount: int | None = None) -> None:
        self.current = min(self.maximum, self.current + (self.recover_per_step if amount is None else amount))

    def post_actions_step(self, env: Environment) -> None:
        if not self._spent_this_step:
            self.recover()
        self._spent_this_step = False


class Has_Stamina(Action_Validation):
    def __init__(self, action_key: str):
        self.action_key = action_key

    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        stamina = actor.get_component(Stamina)
        return stamina is None or stamina.can_spend(self.action_key)

    def on_action_success(self, actor: Entity, target_entity: Entity, env: Environment, action_result: dict | None):
        stamina = actor.get_component(Stamina)
        if stamina is not None:
            stamina.spend(self.action_key)
