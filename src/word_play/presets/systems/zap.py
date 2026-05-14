"""Zap system — instant entity transforms, player attacks, and inventory consumes.

Exports:
- Zap_Change: Instantly transform a nearby entity into a different form
- Zap_Change_Inventory: Instantly transform a held inventory item
- Zap_Player: Zap a nearby player with graduated sanctions (freeze → eliminate)
- ZapMarking: Component tracking zap-hit level on a player
"""
from __future__ import annotations

from typing import Callable

from word_play.core import Action, Component, Entity, Environment, Target_Is_Nearby, Target_Not_Self
from word_play.presets.action_validations import Target_Has_Component, Target_Has_Tag
from word_play.presets.systems.cooldown import Action_On_Cooldown, Cooldown
from word_play.presets.systems.freezable import Freezable
from word_play.presets.systems.health import Health
from word_play.presets.systems.inventory import Inventory, materialize_item


class ZapMarking(Component):
    """Graduated sanctions marking: tracks zap-hit level on a player.

    Level 1: freeze (delegates to Freezable component).
    Level 2: penalty reward + eliminate.
    Recovery clears marking back to 0 after recovery_time ticks.
    """

    def __init__(self, freeze_duration: int = 25, penalty: float = -10.0, recovery_time: int = 50):
        super().__init__()
        self.mark_level = 0
        self.freeze_duration = freeze_duration
        self.penalty = penalty
        self.recovery_time = recovery_time
        self.recovery_counter = 0

    def zap_hit(self, env) -> dict:
        freezable = self.entity.get_component(Freezable) if hasattr(self, "entity") else None
        if self.mark_level == 0:
            self.mark_level = 1
            if freezable is not None:
                freezable.freeze(duration=self.freeze_duration)
            return {"level": 1, "effect": "freeze", "duration": self.freeze_duration}
        else:
            self.mark_level = 2
            if freezable is not None:
                freezable.eliminate()
            from word_play.presets.systems.reward import award_reward
            award_reward(env, self.entity, self.penalty)
            return {"level": 2, "effect": "remove", "penalty": self.penalty}

    def post_actions_step(self, env) -> None:
        if self.mark_level > 0:
            self.recovery_counter += 1
            if self.recovery_counter >= self.recovery_time:
                self.mark_level = 0
                self.recovery_counter = 0


class Zap_Player(Action):
    """Zap a nearby player with graduated sanctions.

    Cooldown is managed by the actor's Cooldown component (key="zap").
    First hit freezes target; second hit within recovery eliminates them.
    """

    def __init__(self):
        super().__init__(validation_rules=[
            Action_On_Cooldown("zap"),
            Target_Is_Nearby(),
            Target_Not_Self(),
            Target_Has_Component(ZapMarking),
        ])

    def exec_action(self, actor, target, env, kwargs=None):
        marking = target.get_component(ZapMarking)
        if marking is None:
            return {"success": False, "reason": "no_marking"}
        cooldown = actor.get_component(Cooldown)
        if cooldown is not None:
            cooldown.start("zap")
        result = marking.zap_hit(env)
        return {"success": True, "zapped": target.name, **result}

    def action_description_text(self, actor, target, env):
        return f"Zap {target.name}."


class Zap_Change(Action):
    """Instantly transform a nearby entity into a new form.

    Common uses:
    - Change berry color: Zap_Change(allowed_tag="zappable", output=lambda: Entity("Berry_Blue", ...))
    - Destroy wall: Zap_Change(allowed_tag="wall", damage=1, damage_only=True)
    - Turn resource into item: Zap_Change(allowed_tag="node", output=lambda: Entity("Gold_Nugget", ...))
    - Remove dirt tile: Zap_Change(allowed_tag="dirt")
    """

    def __init__(
        self,
        allowed_tag: str = "zappable",
        output: Callable[[], Entity] | Entity | None = None,
        *,
        damage: int = 0,
        damage_only: bool = False,
        reward: float = 0.0,
    ):
        rules = [Target_Not_Self(), Target_Is_Nearby()]
        if allowed_tag:
            rules.append(Target_Has_Tag([allowed_tag]))
        super().__init__(validation_rules=rules)
        self.allowed_tag = allowed_tag
        self.output = output
        self.damage = damage
        self.damage_only = damage_only
        self.reward = reward

    def exec_action(self, actor, target, env, kwargs=None):
        # If the original target was already destroyed this step (e.g. by
        # another agent's Zap_Change), there's a new entity at the same
        # position — act on that instead.
        if target not in env.state.entities:
            target = next(
                (e for e in env.state.entities
                 if e.position == target.position
                 and e is not actor
                 and (not self.allowed_tag or self.allowed_tag in e.tags)),
                None,
            )
            if target is None:
                return {"success": False, "reason": "already_destroyed"}

        result = {"success": True, "target": target.name}

        if self.damage > 0:
            health = target.get_component(Health)
            if health is not None:
                health.damage(self.damage)
                result["damage"] = self.damage

        if self.damage_only:
            return result

        if self.output is not None:
            pos = target.position
            new_entity = materialize_item(self.output)
            new_entity.position = pos
            env.destroy_entity(target)
            env.instantiate_entity(new_entity)
            result["created"] = new_entity.name
        else:
            env.destroy_entity(target)

        if self.reward != 0:
            from word_play.presets.systems.reward import award_reward
            award_reward(env, actor, self.reward)

        return result

    def action_description_text(self, actor, target, env):
        return f"Transform {target.name}."


class Zap_Change_Inventory(Action):
    """Instantly transform a held inventory item into a different form.

    Removes the first matching item from the actor's inventory and optionally
    creates a replacement item in its place. Use this for any "consume / eat /
    deposit / refine what you're holding" pattern.

    Common uses:
    - Eat item: Zap_Change_Inventory("fruit", reward=1.0)
    - Refine item: Zap_Change_Inventory("gem", output=make_refined_gem)
    - Deposit item: Zap_Change_Inventory("collectable", on_deposited=deposit_callback)
    - Consume food: Zap_Change_Inventory(("food", "fruit"), reward=3.0)
    """

    def __init__(
        self,
        item_spec: str | tuple[str, ...],
        output: Callable[[], Entity] | None = None,
        *,
        reward: float = 0.0,
        on_consumed: Callable[[Entity, Entity, Environment, Entity, float], None] | None = None,
    ):
        super().__init__(validation_rules=[Target_Is_Nearby(), Target_Not_Self()])
        if isinstance(item_spec, str):
            self.item_tags = (item_spec,)
        else:
            self.item_tags = item_spec
        self.output = output
        self.reward = reward
        self.on_consumed = on_consumed

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False
        inv = actor.get_component(Inventory)
        if inv is None:
            return False
        return any(any(tag in item.tags for tag in self.item_tags) for item in inv.contents)

    def exec_action(self, actor, target, env, kwargs=None):
        inv = actor.get_component(Inventory)
        if inv is None:
            return {"success": False, "reason": "no_inventory"}

        for item in list(inv.contents):
            if any(tag in item.tags for tag in self.item_tags):
                inv.remove(item)
                result = {"success": True, "consumed": item.name, "reward": self.reward}

                if self.output is not None:
                    new_item = materialize_item(self.output)
                    if inv.store(new_item, env):
                        result["created"] = new_item.name
                    else:
                        result["created"] = None
                        result["reason"] = "inventory_full_for_output"

                if item in env.state.entities:
                    env.destroy_entity(item)

                if self.on_consumed is not None:
                    self.on_consumed(actor, target, env, item, self.reward)

                return result

        return {"success": False, "reason": "no_matching_item"}

    def action_description_text(self, actor, target, env):
        return f"Transform held {self.item_tags[0]} item."
