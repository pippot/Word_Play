from __future__ import annotations

from typing import Iterable

from word_play.core import Action, Action_Selection, Entity, Environment, Non_Agent_Policy
from word_play.presets.systems.do_nothing import Do_Nothing


def matching_actions(
    action_type: type[Action], target_entity: Entity | str | None, action_selections: Iterable[Action_Selection]
) -> list[Action_Selection]:
    """
    Return all Action_Selection objects whose action matches action_type and whose target matches target_entity. If
    target_entity is None, any target is accepted. If target_entity is a string, match by entity name.
    """
    def target_matches(sel_target, expected):
        if expected is None:
            return True
        if isinstance(expected, str):
            # Match by entity name
            return hasattr(sel_target, 'name') and sel_target.name == expected
        # Match by identity (same object)
        return sel_target is expected

    return [
        action_selection
        for action_selection in action_selections
        if isinstance(action_selection.action, action_type)
        and target_matches(action_selection.target_entity, target_entity)
    ]


# TODO: ANDREI: use this to make a zap-on-sight comp--this should be a preset (or maybe just show it as an example?). I.e., it's just Follow_Action_Sequence([Attack("Zap", damage=1)])
# TODO: this class can be made much more general. E.g., the target should likely be a func or a list of valid target tags or something
class Follow_Action_Sequence(Non_Agent_Policy):
    """
    Entity will continuously iterate through a sequence of actions. E.g., imagine an entity patrolling an area or doing
    a repetitive routine.
    """

    def __init__(self, action_sequence: list[tuple[type[Action], Entity | None]], skip_invalid_actions: bool = True):
        """
        action_sequence is a list of (Action type, target_entity). If target_entity is None, we consider this to mean
        that all target entities are valid.
        """
        super().__init__()
        assert len(action_sequence) > 0, "action_sequence cannot be empty."
        self.action_sequence = action_sequence
        self.cur_action_index = 0
        self.skip_invalid_actions = skip_invalid_actions

    def post_initialization(self) -> None:
        for action_type, _ in self.action_sequence:
            assert self.entity.has_action_type(action_type), (
                "The Follow_Action_Sequence class expects all action in action_sequence to be present in the parent"
                f" entity. '{action_type}' is missing"
            )

        if not any(isinstance(action, Do_Nothing) for action in self.entity.actions):
            self.entity.actions.append(Do_Nothing())

    def _increment_cur_action_index(self) -> None:
        self.cur_action_index += 1
        if self.cur_action_index >= len(self.action_sequence):
            self.cur_action_index = 0

    def _iterate_over_actions(self, start_index):
        n = len(self.action_sequence)
        for i in range(n):
            yield self.action_sequence[(start_index + i) % n]

    def _auto_fill_kwargs(self, action_selection: Action_Selection) -> Action_Selection:
        """Auto-fill required kwargs for actions like Load_Crafter that need them."""
        from word_play.presets.systems.crafter import Load_Crafter
        if not action_selection.required_kwargs:
            return action_selection

        if action_selection.action_kwargs:
            return action_selection

        action_kwargs = {}
        for arg_name, arg in action_selection.required_kwargs.items():
            # For inventory index, default to 0 (first item)
            if "index" in arg_name.lower():
                action_kwargs[arg_name] = 0
            # For other args, try to get default from choices
            elif hasattr(arg, 'choices') and arg.choices:
                action_kwargs[arg_name] = list(arg.choices)[0]
            else:
                action_kwargs[arg_name] = 0  # Default fallback

        if action_kwargs:
            # Create new selection with populated kwargs
            return Action_Selection(
                action=action_selection.action,
                action_kwargs=action_kwargs,
                actor=action_selection.actor,
                target_entity=action_selection.target_entity,
                env=action_selection.env,
            )
        return action_selection

    def select_action(self, possible_actions: list[Action_Selection], env: Environment) -> Action_Selection:
        action_selection = Action_Selection(
            action=Do_Nothing(), action_kwargs=None, actor=self.entity, target_entity=self.entity, env=env
        )

        for cur_action in self._iterate_over_actions(self.cur_action_index):
            all_matching_actions = matching_actions(cur_action[0], cur_action[1], possible_actions)
            if all_matching_actions:
                selected = all_matching_actions[0]
                # Auto-fill any required kwargs
                action_selection = self._auto_fill_kwargs(selected)
                break

            if not self.skip_invalid_actions:
                break
            self._increment_cur_action_index()

        self._increment_cur_action_index()

        return action_selection
