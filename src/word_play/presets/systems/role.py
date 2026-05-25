"""Role system — lightweight role labels and role-based validations."""

from __future__ import annotations

from collections.abc import Iterable

from word_play.core import Action_Validation, Component, Entity, Environment


def _role_set(roles: str | Iterable[str]) -> set[str]:
    if isinstance(roles, str):
        return {roles}
    return set(roles)


class Role(Component):
    """Marks an entity as having a gameplay role."""

    def __init__(self, role: str, *, add_tag: bool = True):
        super().__init__(tags=[role] if add_tag else [])
        self.role = role
        self.add_tag = add_tag

    def post_initialization(self) -> None:
        if self.entity is not None and self.add_tag and self.role not in self.entity.tags:
            self.entity.tags.append(self.role)

    def set_role(self, role: str) -> None:
        if self.entity is not None and self.add_tag:
            while self.role in self.entity.tags:
                self.entity.tags.remove(self.role)
            if role not in self.entity.tags:
                self.entity.tags.append(role)
        self.role = role

    def has_role(self, roles: str | Iterable[str]) -> bool:
        return self.role in _role_set(roles)


class Actor_Has_Role(Action_Validation):
    def __init__(self, roles: str | Iterable[str]):
        self.roles = _role_set(roles)

    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        role = actor.get_component(Role)
        return role is not None and role.has_role(self.roles)


class Target_Has_Role(Action_Validation):
    def __init__(self, roles: str | Iterable[str]):
        self.roles = _role_set(roles)

    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        role = target_entity.get_component(Role)
        return role is not None and role.has_role(self.roles)


class Target_Doesnt_Have_Role(Action_Validation):
    def __init__(self, roles: str | Iterable[str]):
        self.roles = _role_set(roles)

    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        role = target_entity.get_component(Role)
        return role is None or not role.has_role(self.roles)
