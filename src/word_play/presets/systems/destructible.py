"""Destructible system — entities that transform when their HP hits zero.

Exports:
- Destructible: Component for entities that transform on destruction
"""

from __future__ import annotations

from typing import Callable

from word_play.core import Component, Entity, Environment
from word_play.presets.renderers.pygame_renderer.renderable import Renderable
from word_play.presets.systems.health import Health


class Destructible(Component):
    """An entity that transforms into another form when its HP reaches zero.

    Instead of simply being destroyed, the entity can change its tags,
    components, and renderer to represent a destroyed/ruined state.
    """

    def __init__(
        self,
        *,
        destroyed_tags: list[str] | None = None,
        destroyed_sprite: str | None = None,
        on_destroyed: Callable[[Entity, Environment], None] | None = None,
    ):
        super().__init__()
        self.destroyed_tags = destroyed_tags or ["floor", "destroyed"]
        self.destroyed_sprite = destroyed_sprite
        self.on_destroyed = on_destroyed
        self._destroyed = False

    @property
    def is_destroyed(self) -> bool:
        return self._destroyed

    def post_actions_step(self, env: Environment) -> None:
        if self._destroyed or not hasattr(self, "entity"):
            return
        health = self.entity.get_component(Health)
        if health is not None and health.current_health <= 0:
            self._do_destroy(env)

    def _do_destroy(self, env: Environment) -> None:
        self._destroyed = True
        entity = self.entity
        entity.tags = list(self.destroyed_tags)
        if self.destroyed_sprite is not None:
            renderable = entity.get_component(Renderable)
            if renderable is not None:
                renderable.sprite_path = self.destroyed_sprite
                renderable.wall_set = None
        health = entity.get_component(Health)
        if health is not None:
            entity.remove_component(Health)
        if self.on_destroyed is not None:
            self.on_destroyed(entity, env)
