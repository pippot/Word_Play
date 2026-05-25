"""Container primitives."""

from __future__ import annotations

from word_play.core import Action, Target_Is_Nearby, Target_Not_Self
from word_play.presets.action_validations import Target_Has_Component
from word_play.presets.systems.inventory import Inventory, _set_item_visibility


class Open_Container(Action):
    """Open a nearby container to reveal its contents."""

    def __init__(self):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(),
                Target_Has_Component(Container),
            ],
        )

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False
        container = target.get_component(Container)
        return container is not None and not container.is_open

    def exec_action(self, actor, target, env, kwargs) -> dict:
        container = target.get_component(Container)
        return container.open()

    def action_description_text(self, actor, target, env) -> str:
        return f"Open {target.name}."


class Container(Inventory):
    """A chest-style container with hidden contents."""

    def __init__(
        self,
        contents=None,
        *,
        max_size: int | None = None,
        accepted_tags: list[str] | None = None,
        starts_open: bool = False,
    ):
        super().__init__(contents=contents, max_size=max_size, accepted_tags=accepted_tags)
        self.is_open = starts_open

    def open(self) -> dict:
        self.is_open = True
        for item in self.contents:
            _set_item_visibility(item, visible=True)
        return {"opened": True}


class Single_Item_Holder(Container):
    """A container that can hold exactly one item."""

    def __init__(self, *, accepted_tags: list[str] | None = None):
        super().__init__(contents=[], max_size=1, accepted_tags=accepted_tags or [])

    @property
    def stored_item(self):
        return self.contents[0] if self.contents else None
