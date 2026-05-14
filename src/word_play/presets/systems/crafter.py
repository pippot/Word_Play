"""Crafter system - crafting recipes, stations, and instant transforms.

Exports:
- Crafter: Component with recipes (multi-input, timed)
- Crafter_Recipe: Input/output specification
- Load_Crafter / Load_First_Into_Crafter / Collect_From_Crafter: Station actions
- Zap_Change: Instantly transform a nearby entity into a different form
- Zap_Change_Inventory: Instantly transform a held inventory item
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from word_play.core import Action, Component, Entity, Environment, Target_Is_Nearby, Target_Not_Self
from word_play.presets.action_validations import Target_Has_Component
from word_play.presets.systems.health import Health
from word_play.presets.systems.inventory import Inventory, Inventory_Item_Index_Arg, materialize_item


@dataclass(slots=True)
class Crafter_Recipe:
    input_names: tuple[str, ...]
    output: Entity | Callable[[], Entity]
    duration: int = 0


class Crafter(Component):
    """A crafter that acts as a container - items are loaded in, processed, and collected out."""

    def __init__(self, recipes: list[Crafter_Recipe]):
        super().__init__(actions=[Load_Crafter(), Collect_From_Crafter()])
        self.recipes = recipes
        self.staged_inputs: list[str] = []
        self.staged_items: list[Entity] = []
        self.active_recipe: Crafter_Recipe | None = None
        self.remaining_steps: int | None = None
        self.output_item: Entity | None = None

    def can_accept(self, item: Entity) -> bool:
        """Check if item can be loaded (part of a valid recipe)."""
        if self.output_item is not None:
            return False

        next_inputs = self.staged_inputs + [item.name]
        return any(Crafter._is_partial_match(r, next_inputs) for r in self.recipes)

    @staticmethod
    def _is_partial_match(recipe: Crafter_Recipe, candidate_inputs: list[str]) -> bool:
        """Check if candidate is a subset of recipe inputs (partial match allowed)."""
        recipe_names = list(recipe.input_names)
        for name in candidate_inputs:
            if name in recipe_names:
                recipe_names.remove(name)
            else:
                return False
        return True

    def load_item(self, item: Entity) -> None:
        """Load item into the crafter."""
        self.staged_inputs.append(item.name)
        self.staged_items.append(item)

        # Check if recipe is complete
        for recipe in self.recipes:
            if sorted(self.staged_inputs) == sorted(recipe.input_names):
                self.active_recipe = recipe
                if recipe.duration <= 0:
                    self._complete_craft()
                else:
                    self.remaining_steps = recipe.duration
                return

    def _complete_craft(self) -> None:
        """Produce output and store it."""
        if self.active_recipe is None:
            return

        self.output_item = materialize_item(self.active_recipe.output)
        self.active_recipe = None
        self.remaining_steps = None
        self.staged_inputs = []
        self.staged_items = []

    def advance(self) -> bool:
        """Advance crafting by one step."""
        if self.remaining_steps is None:
            return False

        self.remaining_steps -= 1
        if self.remaining_steps <= 0:
            self._complete_craft()
            return True
        return False

    def post_actions_step(self, env: Environment) -> None:
        self.advance()

    def has_output(self) -> bool:
        return self.output_item is not None

    @property
    def ready_item(self) -> Entity | None:
        """Return the finished output item if ready."""
        return self.output_item

    @property
    def loaded_items(self) -> list[str]:
        return list(self.staged_inputs)

    def collect_output(self) -> Entity:
        """Take the output from the container."""
        if self.output_item is None:
            raise ValueError("Crafter has no output to collect.")
        item = self.output_item
        self.output_item = None
        return item


class Load_Crafter(Action):
    """Load an item into the crafter's container."""

    def __init__(self):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(),
                Target_Has_Component(Crafter),
            ],
            required_kwargs={"inventory index": Inventory_Item_Index_Arg()},
        )

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        result = super().is_valid(actor, target, env, kwargs=kwargs)
        if not result:
            return False
        if actor.get_component(Inventory) is None:
            return False
        if kwargs is None or kwargs == "unconsidered":
            return True

        inventory = actor.get_component(Inventory)
        crafter = target.get_component(Crafter)
        idx = int(kwargs["inventory index"])
        if idx >= len(inventory.contents):
            return False
        item = inventory.contents[idx]
        return crafter.can_accept(item)

    def exec_action(self, actor, target, env, kwargs) -> dict | None:
        inventory = actor.get_component(Inventory)
        crafter = target.get_component(Crafter)

        item = inventory.contents.pop(int(kwargs["inventory index"]))

        crafter.load_item(item)
        return {"loaded_item": item.name}

    def action_description_text(self, actor, target, env) -> str:
        return f"Load an inventory item into {target.name}."


class Load_First_Into_Crafter(Action):
    """Load the first compatible held item into a nearby crafter.

    Unlike Load_Crafter which requires an inventory index, this action
    auto-finds the first item that the crafter can accept. Better UX for
    LLM agents that don't manage indices.
    """

    def __init__(self):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(),
                Target_Has_Component(Crafter),
            ],
        )

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False
        inventory = actor.get_component(Inventory)
        crafter = target.get_component(Crafter)
        if inventory is None or crafter is None:
            return False
        return any(crafter.can_accept(item) for item in inventory.contents)

    def exec_action(self, actor, target, env, kwargs=None):
        inventory = actor.get_component(Inventory)
        crafter = target.get_component(Crafter)
        if inventory is None or crafter is None:
            return {"success": False, "reason": "missing_components"}
        for item in list(inventory.contents):
            if crafter.can_accept(item):
                inventory.remove(item)
                crafter.load_item(item)
                return {"success": True, "loaded_item": item.name, "into": target.name}
        return {"success": False, "reason": "no_matching_item"}

    def action_description_text(self, actor, target, env) -> str:
        return f"Load a held ingredient into {target.name}."


class Collect_From_Crafter(Action):
    """Collect the output from a crafter's container."""

    def __init__(self):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(),
                Target_Has_Component(Crafter),
            ],
        )

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False
        crafter = target.get_component(Crafter)
        return crafter.has_output()

    def exec_action(self, actor, target, env, kwargs) -> dict | None:
        crafter = target.get_component(Crafter)
        item = crafter.collect_output()
        inventory = actor.get_component(Inventory)
        if inventory is None:
            raise ValueError(f"{actor.name} has no inventory.")

        inventory.contents.append(item)
        return {"collected_item": item.name, "from": target.name}

    def action_description_text(self, actor, target, env) -> str:
        return f"Collect the finished item from {target.name}."


class Zap_Change(Action):
    """Instantly transform a nearby zappable entity into a new form.

    Common uses:
    - Change berry color: Zap_Change("zappable", lambda: Entity("Berry_Blue", ...))
    - Destroy wall: Zap_Change("wall", damage=1, damage_only=True)
    - Turn resource into item: Zap_Change("node", lambda: Entity("Gold_Nugget", ...))
    """

    def __init__(
        self,
        target_tag: str = "zappable",
        output: Callable[[], Entity] | Entity | None = None,
        *,
        damage: int = 0,
        damage_only: bool = False,
        reward: float = 0.0,
    ):
        from word_play.presets.action_validations import Target_Has_Tag
        rules = [Target_Not_Self(), Target_Is_Nearby()]
        if target_tag:
            rules.append(Target_Has_Tag([target_tag]))
        super().__init__(validation_rules=rules)
        self.target_tag = target_tag
        self.output = output
        self.damage = damage
        self.damage_only = damage_only
        self.reward = reward

    def exec_action(self, actor, target, env, kwargs=None):
        result = {"success": True, "target": target.name}

        if self.damage > 0:
            health = target.get_component(Health)
            if health is not None:
                health.damage(self.damage)
                result["damage"] = self.damage

        if self.damage_only:
            return result

        if self.output is not None:
            from word_play.presets.systems.inventory import materialize_item
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
