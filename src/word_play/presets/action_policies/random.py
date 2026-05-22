from __future__ import annotations

import random

from word_play.core import Agent_Policy, Observation
from word_play.core.actions import Action_Selection
from word_play.presets.systems.do_nothing import Do_Nothing


class Random_Takes_Action(Agent_Policy):
    """
    Agent policy that samples uniformly from currently valid actions.

    Actions requiring kwargs are skipped by default because there is no general
    way to sample valid arbitrary arguments for every possible Action_Arg.
    """

    def __init__(
        self,
        seed: int | None = None,
        include_actions_with_kwargs: bool = False,
        prefer_non_do_nothing: bool = True,
    ):
        super().__init__()
        self.rng = random.Random(seed)
        self.include_actions_with_kwargs = include_actions_with_kwargs
        self.prefer_non_do_nothing = prefer_non_do_nothing

    def select_action(self, observation: Observation) -> tuple[Action_Selection, dict | None]:
        possible_actions = observation.possible_actions
        if not self.include_actions_with_kwargs:
            possible_actions = [
                action_selection
                for action_selection in possible_actions
                if not action_selection.required_kwargs
            ]

        if not possible_actions:
            raise RuntimeError("Random_Takes_Action found no selectable actions.")

        if self.prefer_non_do_nothing:
            non_do_nothing_actions = [
                action_selection
                for action_selection in possible_actions
                if not isinstance(action_selection.action, Do_Nothing)
            ]
            if non_do_nothing_actions:
                possible_actions = non_do_nothing_actions

        selected_action = self.rng.choice(possible_actions)
        return selected_action, {"policy": "random"}
