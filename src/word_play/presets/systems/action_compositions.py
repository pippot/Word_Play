"""Action compositions — action chaining with per-step validation.

Exports:
- Action_Comp: Compose multiple actions into one selectable action
- Step: A step in the chain (required or optional)
- Required: Shorthand for Step(action, required=True)
- Optional: Shorthand for Step(action, required=False)
"""
from __future__ import annotations

from word_play.core import Action
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from word_play.core import Entity, Environment


class Step:
    """A single step in Action_Comp."""

    def __init__(self, action: Action, *, required: bool = True):
        self.action = action
        self.required = required


class Required(Step):
    """Required step — chain fails if invalid."""

    def __init__(self, action: Action):
        super().__init__(action, required=True)


class Optional(Step):
    """Optional step — skip if invalid."""

    def __init__(self, action: Action):
        super().__init__(action, required=False)


class Action_Comp(Action):
    """Compose multiple actions into a single selectable action.

    Validation follows the 2D movement pattern: each step's is_valid()
    checks the hypothetical outcome (like No_Collision_Will_Occur_Left
    computes the target position and checks it). No deep-copy simulation
    needed — each step's validation already encodes what it needs.
    """

    def __init__(
        self,
        name: str,
        *steps: Step,
        validation_rules: list | None = None,
    ):
        super().__init__(validation_rules=validation_rules or [])
        self._name = name
        self.steps = steps

    def action_description_text(self, actor, target, env):
        return self._name

    def is_valid(self, actor, target, env, kwargs: dict | None | str = "unconsidered") -> bool:
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False
        for step in self.steps:
            if not step.action.is_valid(actor, target, env, {}):
                if step.required:
                    return False
        return True

    def exec_action(self, actor, target, env, kwargs) -> dict:
        results = []
        for step_num, step in enumerate(self.steps, 1):
            if not step.action.is_valid(actor, target, env, {}):
                if step.required:
                    return {"success": False, "failed_at": step_num, "results": results}
                results.append({"step": step_num, "skipped": True})
                continue
            result = step.action.exec_action(actor, target, env, kwargs)
            results.append({"step": step_num, "success": True, "result": result})
        return {"success": True, "results": results}
