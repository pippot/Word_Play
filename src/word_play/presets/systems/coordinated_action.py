"""Coordinated actions — require multiple actors to choose the same action together."""

from __future__ import annotations

from word_play.core import Action, Action_Selection, Entity, Environment


class Coordinated_Action(Action):
    """Base action that executes once when enough actors choose it this step."""

    def __init__(
        self,
        action_name: str,
        required_actors: int,
        *,
        coordination_key: str | None = None,
        same_target: bool = True,
        validation_rules: list | None = None,
    ):
        super().__init__(validation_rules=validation_rules or [])
        self.action_name = action_name
        self.required_actors = required_actors
        self.coordination_key = coordination_key or type(self).__qualname__
        self.same_target = same_target

    def _local_is_valid(self, actor: Entity, target: Entity, env: Environment, kwargs=None) -> bool:
        return super().is_valid(actor, target, env, kwargs=kwargs)

    def _selected_actions(self, env: Environment) -> list[Action_Selection]:
        if getattr(env, "_current_action_selections_step", None) != env.cur_step:
            return []
        return getattr(env, "current_action_selections", None) or []

    def _coordination_key(self, target: Entity) -> tuple:
        return (
            self.coordination_key,
            id(target) if self.same_target else None,
        )

    def _step_cache(self, env: Environment) -> dict:
        if getattr(env, "_coordinated_action_step", None) != env.cur_step:
            env._coordinated_action_step = env.cur_step
            env._coordinated_action_results = {}
        return env._coordinated_action_results

    def _cached_result(self, actor: Entity, target: Entity, env: Environment) -> dict | None:
        cached = self._step_cache(env).get(self._coordination_key(target))
        if cached is None or actor not in cached["participants"]:
            return None
        return cached["result"]

    def participants(self, actor: Entity, target: Entity, env: Environment) -> list[Entity]:
        participants = []
        for selection in self._selected_actions(env):
            action = selection.action
            if not isinstance(action, Coordinated_Action):
                continue
            if action.coordination_key != self.coordination_key:
                continue
            if self.same_target and selection.target_entity is not target:
                continue
            if not action._local_is_valid(selection.actor, selection.target_entity, env, selection.action_kwargs):
                continue
            if selection.actor not in participants:
                participants.append(selection.actor)
        return participants

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if self._cached_result(actor, target, env) is not None:
            return True
        if not self._local_is_valid(actor, target, env, kwargs):
            return False
        if kwargs == "unconsidered" or not self._selected_actions(env):
            return True
        return len(self.participants(actor, target, env)) >= self.required_actors

    def exec_action(self, actor, target, env, kwargs=None) -> dict:
        cached = self._cached_result(actor, target, env)
        if cached is not None:
            return cached

        participants = self.participants(actor, target, env)
        if len(participants) < self.required_actors:
            return {"success": False, "participants": [entity.name for entity in participants]}

        result = self.exec_coordinated_action(actor, target, env, participants, kwargs) or {}
        result = {
            "success": True,
            "participants": [entity.name for entity in participants],
            **result,
        }
        self._step_cache(env)[self._coordination_key(target)] = {
            "participants": set(participants),
            "result": result,
        }
        return result

    def exec_coordinated_action(
        self,
        actor: Entity,
        target: Entity,
        env: Environment,
        participants: list[Entity],
        kwargs: dict | None,
    ) -> dict | None:
        return None

    def action_description_text(self, actor, target, env) -> str:
        return self.action_name
