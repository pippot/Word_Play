"""Freezable system — temporarily hold an entity in place.

Exports:
- Freezable: Component for entities that can be frozen for a duration
- Freeze: Action to freeze a nearby freezable entity
"""

from __future__ import annotations

from word_play.core import Action, Component, Entity, Environment
from word_play.presets.action_validations import Target_Has_Component, Target_Is_Nearby
from word_play.presets.renderers import Renderable
from word_play.presets.movement.simple_2d_grid import Position_2D


class Freezable(Component):
    """An entity that can be frozen in place for a duration.

    When frozen, the entity is teleported to a hold position and
    returns to its spawn position when the freeze expires.
    """

    def __init__(
        self,
        *,
        hold_position: tuple[int, int] | None = None,
        spawn_position: tuple[int, int] | None = None,
        default_duration: int = 25,
    ):
        super().__init__()
        self.hold_position = hold_position
        self._spawn_position = spawn_position
        self.default_duration = default_duration
        self.frozen_ticks = 0
        self.eliminated = False

    @property
    def is_frozen(self) -> bool:
        return self.frozen_ticks > 0

    def freeze(self, duration: int | None = None) -> None:
        dur = duration if duration is not None else self.default_duration
        self.frozen_ticks = max(self.frozen_ticks, dur)
        if not hasattr(self, "entity"):
            return
        if self.hold_position is not None:
            self.entity.position = Position_2D(*self.hold_position)

    def eliminate(self) -> None:
        self.eliminated = True
        self.frozen_ticks = 0
        if not hasattr(self, "entity"):
            return
        self.entity.position = Position_2D(-1000, -1000)
        renderable = self.entity.get_component(Renderable)
        if renderable is not None:
            renderable.visible = False

    def post_actions_step(self, env: Environment) -> None:
        if self.eliminated or not hasattr(self, "entity"):
            return
        if self.frozen_ticks > 0:
            self.frozen_ticks -= 1
            if self.hold_position is not None:
                self.entity.position = Position_2D(*self.hold_position)
            if self.frozen_ticks == 0 and self._spawn_position is not None:
                self.entity.position = Position_2D(*self._spawn_position)


class Freeze(Action):
    """Freeze a nearby freezable entity for a duration."""

    def __init__(self, *, duration: int | None = None, reward: float = 1.0):
        super().__init__(validation_rules=[Target_Is_Nearby(), Target_Has_Component(Freezable)])
        self.duration = duration
        self.reward = reward

    def exec_action(self, actor, target, env, kwargs=None):
        freezable = target.get_component(Freezable)
        if freezable is None or freezable.eliminated:
            return {"success": False, "reason": "not_freezable"}
        freezable.freeze(duration=self.duration)
        return {"success": True, "frozen": target.name, "duration": freezable.frozen_ticks}

    def action_description_text(self, actor, target, env):
        return f"Freeze {target.name}."
