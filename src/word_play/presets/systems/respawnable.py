"""Respawnable system — temporarily remove and restore an entity."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

from word_play.core import Action_Validation, Component, Entity, Environment, Position
from word_play.presets.renderers import Renderable


class Respawnable(Component):
    """Tracks active/inactive state and restores an entity after a timer."""

    def __init__(
        self,
        respawn_steps: int = 0,
        *,
        starts_active: bool = True,
        respawn_position: Position | None = None,
        inactive_position: Position | None = None,
        inactive_tag: str = "respawning",
        hide_while_inactive: bool = True,
        on_respawn: Callable[[Entity, Environment], None] | None = None,
    ):
        super().__init__(tags=[] if starts_active else [inactive_tag])
        self.respawn_steps = respawn_steps
        self.active = starts_active
        self.respawn_position = deepcopy(respawn_position)
        self.inactive_position = deepcopy(inactive_position)
        self.inactive_tag = inactive_tag
        self.hide_while_inactive = hide_while_inactive
        self.on_respawn = on_respawn
        self.remaining_steps = 0 if starts_active else respawn_steps

    def post_initialization(self) -> None:
        if self.respawn_position is None and self.entity is not None:
            self.respawn_position = deepcopy(self.entity.position)
        self._sync()

    def remove_temporarily(self, duration: int | None = None) -> None:
        self.active = False
        self.remaining_steps = self.respawn_steps if duration is None else duration
        if self.entity is not None and self.inactive_position is not None:
            self.entity.position = deepcopy(self.inactive_position)
        self._sync()

    def respawn(self, env: Environment | None = None, position: Position | None = None) -> None:
        self.active = True
        self.remaining_steps = 0
        if self.entity is not None:
            target_position = position if position is not None else self.respawn_position
            if target_position is not None:
                self.entity.position = deepcopy(target_position)
            if env is not None and self.on_respawn is not None:
                self.on_respawn(self.entity, env)
        self._sync()

    def pre_actions_step(self, env: Environment) -> None:
        if self.active:
            return
        if self.remaining_steps > 0:
            self.remaining_steps -= 1
        if self.remaining_steps <= 0:
            self.respawn(env)

    def _sync(self) -> None:
        if self.entity is None:
            return
        if self.active:
            while self.inactive_tag in self.entity.tags:
                self.entity.tags.remove(self.inactive_tag)
        elif self.inactive_tag not in self.entity.tags:
            self.entity.tags.append(self.inactive_tag)

        if self.hide_while_inactive:
            renderable = self.entity.get_component(Renderable)
            if renderable is not None:
                renderable.visible = self.active


class Actor_Is_Active(Action_Validation):
    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        respawnable = actor.get_component(Respawnable)
        return respawnable is None or respawnable.active


class Target_Is_Active(Action_Validation):
    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        respawnable = target_entity.get_component(Respawnable)
        return respawnable is None or respawnable.active
