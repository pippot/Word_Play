"""Team marker system — team identification and ally/enemy validators.

Exports:
- Team: Component for marking an entity's team
- Target_Is_Enemy: Validation rule requiring target to be on a different team
- Target_Is_Ally: Validation rule requiring target to be on the same team
"""

from __future__ import annotations

from word_play.core import Component, Entity, Environment
from word_play.presets.action_validations import Action_Validation


class Team(Component):
    """Marks an entity as belonging to a team."""

    def __init__(self, *, team_id: str):
        super().__init__()
        self.team_id = team_id


class Target_Is_Enemy(Action_Validation):
    """Validation: target must have a Team component with a different team_id."""

    def is_valid(self, actor: Entity, target: Entity, env: Environment) -> bool:
        actor_team = actor.get_component(Team)
        target_team = target.get_component(Team)
        if actor_team is None or target_team is None:
            return False
        return actor_team.team_id != target_team.team_id

    def __repr__(self) -> str:
        return "Target_Is_Enemy()"


class Target_Is_Ally(Action_Validation):
    """Validation: target must have a Team component with the same team_id."""

    def is_valid(self, actor: Entity, target: Entity, env: Environment) -> bool:
        actor_team = actor.get_component(Team)
        target_team = target.get_component(Team)
        if actor_team is None or target_team is None:
            return False
        return actor_team.team_id == target_team.team_id

    def __repr__(self) -> str:
        return "Target_Is_Ally()"
