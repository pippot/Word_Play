from __future__ import annotations

import os
import random

from word_play.core import (
    Action,
    Action_Validation,
    Component,
    Entity,
    Environment,
    Target_Is_Nearby,
    Target_Is_Self,
    Target_Not_Self,
)
from word_play.presets.action_validations import Target_Has_Component, Target_Has_Tag
from word_play.presets.systems.combat import Attack
from word_play.presets.systems.cooldown import Action_On_Cooldown, Cooldown
from word_play.presets.systems.crafter import Load_First_Into_Crafter
from word_play.presets.systems.freezable import Freezable
from word_play.presets.movement.simple_2d_grid import Collidable, Move_Down, Move_Left, Move_Right, Move_Up, Position_2D
from word_play.presets.renderers import Renderable
from word_play.presets.systems.inventory import Inventory, Pick_Up_Item, Put_In_Container
from word_play.presets.systems.preferences import Preference
from word_play.presets.systems.reward import award_reward
from word_play.presets.systems.zap import ZapMarking

BENCHMARK_TIMING_BASE_STEPS = 500
BENCHMARK_STEPS = max(1, int(os.environ.get("WORD_PLAY_BENCHMARK_STEPS", "40")))


def normalized_steps(steps: int, *, minimum: int = 1) -> int:
    return max(minimum, round(steps * BENCHMARK_STEPS / BENCHMARK_TIMING_BASE_STEPS))


def normalized_probability(probability: float) -> float:
    if probability <= 0:
        return 0.0
    if probability >= 1:
        return 1.0
    return 1 - (1 - probability) ** (BENCHMARK_TIMING_BASE_STEPS / BENCHMARK_STEPS)


class Room_In_Inventory(Action_Validation):
    """Validate that the actor has an inventory with free capacity."""

    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        inventory = actor.get_component(Inventory)
        return inventory is not None and inventory.has_space()


CHEMISTRY_START_POSITIONS = [
    (9, 9),
    (10, 9),
    (11, 9),
    (12, 9),
    (13, 9),
    (14, 9),
    (15, 9),
    (8, 8),
]


ALLELOPATHIC_COLORS = ("red", "green", "blue")


class Allelopathic_Berry_Patch(Component):
    minimum_time_to_ripen = normalized_steps(10)
    base_growth_rate = normalized_probability(5e-6)

    def __init__(
        self,
        color: str | None = None,
        *,
        ripe: bool = True,
        auto_consume: bool = False,
        grows: bool = False,
    ):
        super().__init__(tags=["berry_patch"])
        self.color = color
        self.ripe = ripe if color is not None else False
        self.age = self.minimum_time_to_ripen if self.ripe else 0
        self.harvested = False
        self.auto_consume = auto_consume
        self.grows = grows

    def post_initialization(self) -> None:
        self._sync()

    def pre_actions_step(self, env: Environment) -> None:
        if getattr(env, "_allelopathic_events_step", None) != env.cur_step:
            env.tick = env.cur_step
            env.allelopathic_events = []
            env._allelopathic_events_step = env.cur_step
        if self.grows:
            self._grow(env)

    def post_actions_step(self, env: Environment) -> None:
        if self.harvested:
            self.harvested = False
            self._sync()
            return
        if not self.auto_consume or not self.available:
            return
        agent = next(
            (
                agent for agent in env.agents
                if agent.position == self.entity.position and not self._is_removed(agent)
            ),
            None,
        )
        if agent is None:
            return
        preference = agent.get_component(Preference)
        reward = preference.reward_for(self.entity) if preference is not None else 1.0
        color = self.consume()
        award_reward(env, agent, reward)
        env.allelopathic_events.append(f"{agent.name} ate a {color} berry for +{reward:g}.")

    @property
    def available(self) -> bool:
        return self.color is not None and self.ripe and not self.harvested

    def set_color(self, color: str) -> None:
        self.color = color
        self.ripe = True
        self.age = self.minimum_time_to_ripen
        self.harvested = False
        self._sync()

    def plant(self, color: str) -> None:
        self.color = color
        self.ripe = False
        self.age = 0
        self.harvested = False
        self._sync()

    def ripen(self) -> None:
        if self.color is None:
            return
        self.ripe = True
        self._sync()

    def consume(self) -> str | None:
        if not self.available:
            return None
        color = self.color
        self.color = None
        self.ripe = False
        self.age = 0
        self.harvested = False
        self._sync()
        return color

    def harvest_color(self) -> str | None:
        if not self.available:
            return None
        self.harvested = True
        self._sync()
        return self.color

    def _sync(self) -> None:
        for tag in (*ALLELOPATHIC_COLORS, "berry", "ripe_berry", "unripe_berry", "empty_patch"):
            while tag in self.entity.tags:
                self.entity.tags.remove(tag)

        if self.color is None:
            self.entity.name = "Empty Berry Patch"
            self.entity.tags.append("empty_patch")
        else:
            self.entity.name = f"{self.color.title()} Berry Patch"
            self.entity.tags.extend(["berry", self.color, "ripe_berry" if self.ripe else "unripe_berry"])

        renderable = self.entity.get_component(Renderable)
        if renderable is not None:
            renderable.visible = self.color is not None and not self.harvested
            if self.color is not None:
                renderable.sprite_path = (
                    "src/items/consumables/vegetables/mushroom_red.png"
                    if self.color == "red"
                    else "src/items/consumables/vegetables/mushroom_green.png"
                    if self.color == "green"
                    else "src/items/consumables/vegetables/mushroom_blue.png"
                )

    def _grow(self, env: Environment) -> None:
        if self.color is not None:
            if not self.ripe:
                self.age += 1
                if self.age >= self.minimum_time_to_ripen:
                    self.ripen()
                    env.allelopathic_events.append(f"A {self.color} berry ripened.")
            return

        for color, weight in self._growth_weights(env).items():
            if random.random() < self.base_growth_rate * weight:
                self.plant(color)
                env.allelopathic_events.append(f"A new {color} berry sprouted.")
                return

    def _growth_weights(self, env: Environment) -> dict[str, float]:
        if getattr(env, "_allelopathic_growth_step", None) == env.cur_step:
            return env._allelopathic_growth_weights

        active_counts = {color: 0 for color in ALLELOPATHIC_COLORS}
        for entity in env.state.entities:
            patch = entity.get_component(Allelopathic_Berry_Patch)
            if patch is not None and patch.color is not None:
                active_counts[patch.color] += 1

        total_active = sum(active_counts.values())
        env._allelopathic_growth_weights = (
            {color: 1 / len(ALLELOPATHIC_COLORS) for color in ALLELOPATHIC_COLORS}
            if total_active == 0
            else {color: active_counts[color] / total_active for color in ALLELOPATHIC_COLORS}
        )
        env._allelopathic_growth_step = env.cur_step
        return env._allelopathic_growth_weights

    def _is_removed(self, agent: Entity) -> bool:
        zap_state = agent.get_component(Allelopathic_Zap_State)
        return zap_state is not None and zap_state.removed


class Target_Has_Available_Berry(Action_Validation):
    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        patch = target_entity.get_component(Allelopathic_Berry_Patch)
        return patch is not None and patch.available


class Harvest_Berry(Action):
    def __init__(self):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(),
                Target_Has_Available_Berry(),
                Room_In_Inventory(),
            ]
        )
        self.pick_up = Pick_Up_Item(["berry"])

    def exec_action(self, actor: Entity, target_entity: Entity, env: Environment, kwargs=None) -> dict:
        color = target_entity.get_component(Allelopathic_Berry_Patch).harvest_color()
        if color is None:
            return {"success": False, "reason": "berry_unavailable"}
        item = Entity(
            name=f"{color.title()} Berry",
            position=Position_2D(target_entity.position.x, target_entity.position.y),
            tags=["berry", color],
            components=[
                Renderable(
                    sprite_path=(
                        "src/items/consumables/vegetables/mushroom_red.png"
                        if color == "red"
                        else "src/items/consumables/vegetables/mushroom_green.png"
                        if color == "green"
                        else "src/items/consumables/vegetables/mushroom_blue.png"
                    ),
                    z_index=5,
                )
            ],
        )
        env.instantiate_entity(item)
        self.pick_up(actor, item, env, None)
        preference = actor.get_component(Preference)
        reward = preference.reward_for(item) if preference is not None else 1.0
        award_reward(env, actor, reward)
        return {"harvested": item.name, "reward": reward}

    def action_description_text(self, actor: Entity, target_entity: Entity, env: Environment) -> str:
        return f"Harvest {target_entity.name}."


class Paint_Berry(Action):
    def __init__(self, color: str):
        super().__init__(
            validation_rules=[
                Action_On_Cooldown("color"),
                Target_Not_Self(),
                Target_Is_Nearby(),
                Target_Has_Component(Allelopathic_Berry_Patch),
            ]
        )
        self.color = color

    def exec_action(self, actor: Entity, target_entity: Entity, env: Environment, kwargs=None) -> dict:
        actor.get_component(Cooldown).start("color")
        target_entity.get_component(Allelopathic_Berry_Patch).set_color(self.color)
        return {"painted": target_entity.name, "color": self.color}

    def action_description_text(self, actor: Entity, target_entity: Entity, env: Environment) -> str:
        return f"Paint {target_entity.name} {self.color}."


class Allelopathic_Plant_Berry(Action):
    def __init__(self, color: str, freeze_duration: int = normalized_steps(2)):
        super().__init__(
            validation_rules=[
                Action_On_Cooldown("plant"),
                Target_Not_Self(),
                Target_Is_Nearby(),
                Target_Has_Component(Allelopathic_Berry_Patch),
            ]
        )
        self.color = color
        self.freeze_duration = freeze_duration

    def exec_action(self, actor, target, env, kwargs=None):
        target_name = target.name
        cooldown = actor.get_component(Cooldown)
        if cooldown is not None:
            cooldown.start("plant")
        freezable = actor.get_component(Freezable)
        if freezable is not None:
            freezable.freeze(self.freeze_duration)
        target.get_component(Allelopathic_Berry_Patch).plant(self.color)
        return {"planted": self.color, "target": target_name, "position": str(target.position)}

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        patch = target.get_component(Allelopathic_Berry_Patch)
        return super().is_valid(actor, target, env, kwargs=kwargs) and patch is not None and patch.color is None

    def action_description_text(self, actor, target, env):
        return f"Plant {self.color} berry on {target.name}."


class Allelopathic_Zap_State(ZapMarking):
    def __init__(
        self,
        freeze_duration: int = normalized_steps(25),
        remove_duration: int = normalized_steps(50),
        recovery_time: int = normalized_steps(50),
    ):
        super().__init__(freeze_duration=freeze_duration, penalty=0.0, recovery_time=recovery_time)
        self.remove_duration = remove_duration
        self.removed = False

    def zap_hit(self, env: Environment) -> dict:
        if self.mark_level == 0:
            return super().zap_hit(env)

        self.mark_level = 0
        self.recovery_counter = 0
        self.removed = True
        freezable = self.entity.get_component(Freezable)
        if freezable is not None:
            freezable.freeze(self.remove_duration)
        renderable = self.entity.get_component(Renderable)
        if renderable is not None:
            renderable.visible = False
        return {"level": 2, "effect": "removed", "duration": self.remove_duration}

    def post_actions_step(self, env: Environment) -> None:
        super().post_actions_step(env)
        if not self.removed:
            return
        freezable = self.entity.get_component(Freezable)
        if freezable is not None and freezable.is_frozen:
            return
        self.removed = False
        renderable = self.entity.get_component(Renderable)
        if renderable is not None:
            renderable.visible = True


MOLECULE_NAMES = {
    "energy": "Energy",
    "x": "Molecule X",
    "y": "Molecule Y",
    "ix": "Inhibitor IX",
    "iy": "Inhibitor IY",
    "food1": "Food 1",
    "food2": "Food 2",
    "food3": "Food 3",
    "distractor": "Distractor",
}

MOLECULE_SPRITE_PATHS = {
    "energy": "src/items/consumables/magic/golden_potion.png",
    "x": "src/items/consumables/magic/brilliant_blue_potion.png",
    "y": "src/items/consumables/magic/dark_green_potion.png",
    "ix": "src/items/consumables/magic/dusty_green_scroll.png",
    "iy": "src/items/consumables/magic/dusty_blue_scroll.png",
    "food1": "src/items/consumables/vegetables/berry_blue.png",
    "food2": "src/items/consumables/vegetables/berry_green.png",
    "food3": "src/items/consumables/vegetables/berry_red.png",
    "distractor": "src/items/consumables/magic/purple_red_potion.png",
}


class Molecule(Component):
    def __init__(self, kind: str, holding_reward: float = 0.0):
        super().__init__(tags=["item", "molecule", kind])
        self.kind = kind
        self.holding_reward = holding_reward

    def pre_actions_step(self, env: Environment) -> None:
        chemistry_events(env)
        if carrier_of(env, self.entity) is not None:
            return
        for agent in env.agents:
            inventory = agent.get_component(Inventory)
            if inventory is None or inventory.contents:
                continue
            if abs(agent.position.x - self.entity.position.x) + abs(agent.position.y - self.entity.position.y) <= 1:
                inventory.store(self.entity, env)
                chemistry_events(env).append(f"{agent.name} picked up {self.entity.name}.")
                return

    def post_actions_step(self, env: Environment) -> None:
        if self.entity not in env.state.entities:
            return
        maybe_end_chemistry_episode(env)
        if self.kind in {"food1", "food2", "food3"}:
            self._metabolize_if_carried(env)
        elif self.kind == "distractor":
            self._reward_if_carried(env)
        elif self.kind == "x":
            self._combine_with_y(env)

    def _metabolize_if_carried(self, env: Environment) -> None:
        carrier = carrier_of(env, self.entity)
        if carrier is None:
            return
        reward = 10.0 if self.kind == "food3" else 1.0
        remove_molecule(env, self.entity)
        award_reward(env, carrier, reward)
        chemistry_events(env).append(f"{carrier.name} metabolized {self.entity.name} for +{reward:g}.")

    def _reward_if_carried(self, env: Environment) -> None:
        if self.holding_reward <= 0:
            return
        carrier = carrier_of(env, self.entity)
        if carrier is None:
            return
        award_reward(env, carrier, self.holding_reward)
        chemistry_events(env).append(f"{carrier.name} held {self.entity.name} for +{self.holding_reward:g}.")

    def _combine_with_y(self, env: Environment) -> None:
        position = molecule_position(self.entity, env)
        y_molecule = first_molecule_near(env, "y", position, exclude=self.entity)
        if y_molecule is None:
            return

        carriers = []
        for carrier in (carrier_of(env, self.entity), carrier_of(env, y_molecule)):
            if carrier is not None and carrier not in carriers:
                carriers.append(carrier)

        remove_molecule(env, self.entity)
        remove_molecule(env, y_molecule)
        env.instantiate_entity(
            Entity(
                name=MOLECULE_NAMES["energy"],
                position=Position_2D(position.x, position.y),
                components=[
                    Molecule("energy"),
                    Renderable(sprite_path=MOLECULE_SPRITE_PATHS["energy"], z_index=4),
                ],
            )
        )

        for carrier in carriers:
            award_reward(env, carrier, 10.0)

        reward_text = ", ".join(f"{carrier.name} +10" for carrier in carriers) or "no carried molecule reward"
        chemistry_events(env).append(f"MetabolizeXY at {position}: x + y -> energy ({reward_text}).")


class Cycle_Site(Component):
    def __init__(self, cycle: str):
        super().__init__(tags=["cycle_site", f"{cycle}_cycle"])
        self.cycle = cycle

    def pre_actions_step(self, env: Environment) -> None:
        chemistry_events(env)

    def post_actions_step(self, env: Environment) -> None:
        if self.entity not in env.state.entities:
            return
        maybe_end_chemistry_episode(env)
        energy = first_molecule_near(env, "energy", self.entity.position)
        if energy is None:
            return

        energy_carrier = carrier_of(env, energy)
        reaction_position = molecule_position(energy, env)
        remove_molecule(env, energy)
        products = self.products(env)
        for product in products:
            env.instantiate_entity(
                Entity(
                    name=MOLECULE_NAMES[product],
                    position=Position_2D(reaction_position.x, reaction_position.y),
                    components=[
                        Molecule(product),
                        Renderable(sprite_path=MOLECULE_SPRITE_PATHS[product], z_index=4),
                    ],
                )
            )

        carrier_text = f" using {energy_carrier.name}'s carried energy" if energy_carrier else ""
        chemistry_events(env).append(
            f"{self.cycle.title()} cycle fired near {self.entity.position}{carrier_text}: "
            f"energy -> {', '.join(products)}."
        )

    def products(self, env: Environment) -> list[str]:
        products = (
            ["food1", "x", "iy"] if self.cycle == "blue" else
            ["food2", "y", "ix"] if self.cycle == "green" else
            ["food3", "food1", "food2"]
        )
        inhibitor = "iy" if self.cycle == "blue" else "ix" if self.cycle == "green" else None
        blocked_product = "x" if self.cycle == "blue" else "y" if self.cycle == "green" else None
        if inhibitor is not None and first_molecule_at(env, inhibitor, self.entity.position) is not None:
            products.remove(blocked_product)
            chemistry_events(env).append(
                f"{self.cycle.title()} cycle inhibited at {self.entity.position}: "
                f"{inhibitor} blocked {blocked_product}."
            )
        return products


def chemistry_events(env: Environment) -> list[str]:
    if getattr(env, "_chemistry_events_step", None) != env.cur_step:
        env.tick = env.cur_step
        env.chemistry_events = []
        env._chemistry_events_step = env.cur_step
    return env.chemistry_events


def maybe_end_chemistry_episode(env: Environment) -> None:
    if getattr(env, "_chemistry_end_checked_step", None) == env.cur_step:
        return
    env._chemistry_end_checked_step = env.cur_step
    if env.cur_step + 1 >= BENCHMARK_STEPS:
        for idx in range(len(env.agents)):
            env.truncations[idx] = True


def molecule_kind(entity: Entity) -> str | None:
    molecule = entity.get_component(Molecule)
    return molecule.kind if molecule is not None else None


def carrier_of(env: Environment, item: Entity) -> Entity | None:
    for agent in env.agents:
        inventory = agent.get_component(Inventory)
        if inventory is not None and item in inventory.contents:
            return agent
    return None


def molecule_position(entity: Entity, env: Environment) -> Position_2D:
    carrier = carrier_of(env, entity)
    return carrier.position if carrier is not None else entity.position


def first_molecule_at(
    env: Environment,
    kind: str,
    position: Position_2D,
    exclude: Entity | None = None,
) -> Entity | None:
    for entity in env.state.entities:
        if entity is not exclude and molecule_kind(entity) == kind and molecule_position(entity, env) == position:
            return entity
    return None


def first_molecule_near(
    env: Environment,
    kind: str,
    position: Position_2D,
    exclude: Entity | None = None,
) -> Entity | None:
    for entity in env.state.entities:
        entity_position = molecule_position(entity, env)
        if (
            entity is not exclude
            and molecule_kind(entity) == kind
            and abs(entity_position.x - position.x) + abs(entity_position.y - position.y) <= 1
        ):
            return entity
    return None


def remove_molecule(env: Environment, molecule: Entity) -> None:
    carrier = carrier_of(env, molecule)
    if carrier is not None:
        inventory = carrier.get_component(Inventory)
        if molecule in inventory.contents:
            inventory.contents.remove(molecule)
    while "in_inventory" in molecule.tags:
        molecule.tags.remove("in_inventory")
    if molecule in env.state.entities:
        env.destroy_entity(molecule)


class Commons_Apple_Patch(Component):
    def __init__(self, apple_regrowth_probs: list[float] | None = None):
        super().__init__(tags=["apple_patch", "apple"])
        self.apple_regrowth_probs = [
            normalized_probability(prob)
            for prob in (apple_regrowth_probs or [0.0, 0.0025, 0.005, 0.025])
        ]
        self.consumed = False

    def pre_actions_step(self, env: Environment) -> None:
        if getattr(env, "_commons_events_step", None) != env.cur_step:
            env.tick = env.cur_step
            env.commons_events = []
            env._commons_events_step = env.cur_step

    def post_actions_step(self, env: Environment) -> None:
        self._collect(env)
        self._regrow(env)
        self._maybe_end_episode(env)

    def consume(self) -> None:
        self.consumed = True
        if "apple" in self.entity.tags:
            self.entity.tags.remove("apple")
        renderable = self.entity.get_component(Renderable)
        if renderable is not None:
            renderable.visible = False

    def regrow(self) -> None:
        self.consumed = False
        if "apple" not in self.entity.tags:
            self.entity.tags.append("apple")
        renderable = self.entity.get_component(Renderable)
        if renderable is not None:
            renderable.visible = True

    def _collect(self, env: Environment) -> None:
        if self.consumed:
            return
        for agent in env.agents:
            if agent.position == self.entity.position:
                self.consume()
                award_reward(env, agent, 1.0)
                env.commons_events.append(f"{agent.name} ate {self.entity.name} for +1.")
                break

    def _regrow(self, env: Environment) -> None:
        if not self.consumed:
            return

        ripe_apples = [
            entity for entity in env.state.entities
            if (
                (patch := entity.get_component(Commons_Apple_Patch)) is not None
                and not patch.consumed
            )
        ]
        nearby_ripe_count = sum(
            1
            for ripe in ripe_apples
            if (ripe.position.x - self.entity.position.x) ** 2
            + (ripe.position.y - self.entity.position.y) ** 2
            <= 4
        )
        regrowth_prob = self.apple_regrowth_probs[
            min(nearby_ripe_count, len(self.apple_regrowth_probs) - 1)
        ]
        if random.random() < regrowth_prob:
            self.regrow()
            env.commons_events.append(f"{self.entity.name} regrew.")

    def _maybe_end_episode(self, env: Environment) -> None:
        if getattr(env, "_commons_end_step", None) == env.cur_step:
            return
        env._commons_end_step = env.cur_step
        next_step = env.cur_step + 1
        if next_step >= BENCHMARK_STEPS:
            for idx in range(len(env.agents)):
                env.truncations[idx] = True
            return
        if (
            next_step >= normalized_steps(1000)
            and next_step % normalized_steps(100) == 0
            and random.random() < 0.15
        ):
            for idx in range(len(env.agents)):
                env.truncations[idx] = True


class Role_Based_Punishment_Tile(Component):
    def __init__(self, punished_tag: str = "putative_cooperator", penalty: float = -10.0):
        super().__init__(tags=["punishment_tile"])
        self.punished_tag = punished_tag
        self.penalty = penalty

    def pre_actions_step(self, env: Environment) -> None:
        if getattr(env, "_commons_events_step", None) != env.cur_step:
            env.tick = env.cur_step
            env.commons_events = []
            env._commons_events_step = env.cur_step

    def post_actions_step(self, env: Environment) -> None:
        for agent in env.agents:
            if agent.position == self.entity.position and self.punished_tag in agent.tags:
                award_reward(env, agent, self.penalty)
                env.commons_events.append(f"{agent.name} crossed a punishment tile for {self.penalty:g}.")


class Commons_Zap(Action):
    def __init__(self):
        super().__init__(
            validation_rules=[
                Action_On_Cooldown("zap"),
                Target_Is_Nearby(
                    lambda actor, target, env: abs(actor.position.x - target.position.x)
                    + abs(actor.position.y - target.position.y)
                    <= 1
                ),
                Target_Not_Self(),
                Target_Has_Tag(["player"]),
            ],
        )

    def exec_action(self, actor, target, env, kwargs=None):
        cooldown = actor.get_component(Cooldown)
        if cooldown is not None:
            cooldown.start("zap")

        freezable = target.get_component(Freezable)
        if freezable is None:
            return {"success": False, "reason": "not_freezable"}
        freezable.freeze()
        return {"success": True, "zapped": target.name}

    def action_description_text(self, actor, target, env):
        return f"Zap {target.name}."


# ============================================================================
# Shared helpers for the remaining Melting Pot substrate ports
# ============================================================================


def grid_distance(first: Entity, second: Entity) -> int:
    return abs(first.position.x - second.position.x) + abs(first.position.y - second.position.y)


def nearby_within(distance: int):
    def _nearby(actor: Entity, target: Entity, env: Environment) -> bool:
        return grid_distance(actor, target) <= distance

    return _nearby


# ---------------------------------------------------------------------------
# Collaborative Cooking
# ---------------------------------------------------------------------------


COOKING_DELIVERY_REWARD = 20.0


def soup_is_valid(item: Entity) -> bool:
    return "soup" in item.tags


def shared_delivery_bonus(actor, target, env, item: Entity, reward: float) -> None:
    for idx, agent in enumerate(env.agents):
        if agent is not actor and idx < len(env.last_step_rewards):
            env.last_step_rewards[idx] += reward
    env.completed_orders = getattr(env, "completed_orders", 0) + 1


class Load_Held_Item_Into_Crafter(Load_First_Into_Crafter):
    def action_description_text(self, actor, target, env) -> str:
        return f"Load your held item into {target.name}."


class Deliver_Held_Soup(Put_In_Container):
    def __init__(self):
        super().__init__(
            target_tags=["delivery"],
            reward=COOKING_DELIVERY_REWARD,
            on_stored=shared_delivery_bonus,
            destroy_item=True,
        )
        self.required_kwargs = None

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        inventory = actor.get_component(Inventory)
        if inventory is None or not inventory.contents:
            return False
        if not soup_is_valid(inventory.contents[0]):
            return False
        return super().is_valid(actor, target, env, kwargs={"inventory_index": 0})

    def exec_action(self, actor, target, env, kwargs=None):
        return super().exec_action(actor, target, env, kwargs={"inventory_index": 0})

    def action_description_text(self, actor, target, env) -> str:
        return f"Deliver your held soup to {target.name}."


# ---------------------------------------------------------------------------
# Boat Race
# ---------------------------------------------------------------------------


class BoatRower(Component):
    def __init__(self):
        super().__init__()
        self.intent = "none"
        self.row_count = 0

    def pre_actions_step(self, env: Environment) -> None:
        self.intent = "none"


class Paddle(Action):
    def __init__(self):
        super().__init__(validation_rules=[Target_Is_Self()])

    def exec_action(self, actor, target, env, kwargs=None):
        rower = actor.get_component(BoatRower)
        if rower is None:
            return {"success": False}
        rower.intent = "paddle"
        rower.row_count += 1
        return {"paddle": True}

    def action_description_text(self, actor, target, env):
        return "Paddle the boat."


class Flail(Action):
    def __init__(self):
        super().__init__(validation_rules=[Target_Is_Self()])

    def exec_action(self, actor, target, env, kwargs=None):
        rower = actor.get_component(BoatRower)
        if rower is None:
            return {"success": False}
        rower.intent = "flail"
        return {"flail": True}

    def action_description_text(self, actor, target, env):
        return "Flail the oar."


class BoatFood(Component):
    def __init__(self, reward: float = 1.0):
        super().__init__(tags=["boat_food"])
        self.reward = reward
        self.consumed = False

    def post_actions_step(self, env: Environment) -> None:
        if self.consumed:
            return
        for agent in env.agents:
            if agent.position == self.entity.position:
                self.consumed = True
                award_reward(env, agent, self.reward)
                renderable = self.entity.get_component(Renderable)
                if renderable is not None:
                    renderable.visible = False
                env.boat_events.append(f"{agent.name} ate {self.entity.name} for +{self.reward:g}.")
                return


class BoatRaceManager(Component):
    def __init__(self, num_races: int = 8):
        super().__init__()
        self.num_races = num_races
        self.progress = 0

    def pre_actions_step(self, env: Environment) -> None:
        env.boat_events = []
        env.tick = env.cur_step

    def post_actions_step(self, env: Environment) -> None:
        paddlers = [agent for agent in env.agents if (r := agent.get_component(BoatRower)) and r.intent == "paddle"]
        flailers = [agent for agent in env.agents if (r := agent.get_component(BoatRower)) and r.intent == "flail"]
        if len(paddlers) >= 2:
            self.progress += 1
            for agent in paddlers:
                award_reward(env, agent, 0.2)
            env.boat_events.append(f"{len(paddlers)} paddlers coordinated; race progress is {self.progress}.")
        elif flailers and random.random() < 0.25:
            self.progress += 1
            for agent in flailers:
                award_reward(env, agent, 0.05)
            env.boat_events.append(f"A flail moved a boat; race progress is {self.progress}.")
        if self.progress >= self.num_races:
            for idx in range(len(env.agents)):
                env.terminations[idx] = True
        if env.cur_step + 1 >= BENCHMARK_STEPS:
            for idx in range(len(env.agents)):
                env.truncations[idx] = True


# ---------------------------------------------------------------------------
# Factory Commons
# ---------------------------------------------------------------------------


class FactoryStamina(Component):
    def __init__(self, maximum: int = 10):
        super().__init__()
        self.maximum = maximum
        self.current = maximum
        self._before: Position_2D | None = None

    def pre_actions_step(self, env: Environment) -> None:
        self._before = Position_2D(self.entity.position.x, self.entity.position.y)

    def post_actions_step(self, env: Environment) -> None:
        if self._before is None:
            return
        moved = self.entity.position != self._before
        self.current = max(0, self.current - 1) if moved else min(self.maximum, self.current + 1)


class HasFactoryStamina(Action_Validation):
    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        stamina = actor.get_component(FactoryStamina)
        return stamina is None or stamina.current > 0


class FactoryMoveUp(Move_Up):
    def __init__(self):
        super().__init__()
        self.validation_rules.append(HasFactoryStamina())


class FactoryMoveDown(Move_Down):
    def __init__(self):
        super().__init__()
        self.validation_rules.append(HasFactoryStamina())


class FactoryMoveLeft(Move_Left):
    def __init__(self):
        super().__init__()
        self.validation_rules.append(HasFactoryStamina())


class FactoryMoveRight(Move_Right):
    def __init__(self):
        super().__init__()
        self.validation_rules.append(HasFactoryStamina())


class FactoryCube(Component):
    def __init__(self, cube_type: str = "blue", live: bool = True):
        super().__init__(tags=["cube", cube_type])
        self.cube_type = cube_type
        self.live = live

    def post_initialization(self) -> None:
        self.sync()

    def sync(self) -> None:
        for tag in ("cube", self.cube_type):
            if self.live and tag not in self.entity.tags:
                self.entity.tags.append(tag)
            while not self.live and tag in self.entity.tags:
                self.entity.tags.remove(tag)
        renderable = self.entity.get_component(Renderable)
        if renderable is not None:
            renderable.visible = self.live


class FactoryHopper(Component):
    def __init__(self, output_count: int = 1):
        super().__init__(tags=["hopper"])
        self.output_count = output_count


def carrying_factory_cube(actor: Entity) -> Entity | None:
    inventory = actor.get_component(Inventory)
    if inventory is None:
        return None
    return next((item for item in inventory.contents if item.get_component(FactoryCube) is not None), None)


def factory_position_is_open(env: Environment, position: Position_2D) -> bool:
    return not any(
        entity.position == position
        and (entity.has_component(Collidable) or "apple" in entity.tags)
        for entity in env.state.entities
    )


def factory_apple_spawn_position(env: Environment, hopper_position: Position_2D, offset: int) -> Position_2D:
    candidates = [
        Position_2D(hopper_position.x + 1, hopper_position.y + offset),
        Position_2D(hopper_position.x - 1, hopper_position.y + offset),
        Position_2D(hopper_position.x, hopper_position.y + 1 + offset),
        Position_2D(hopper_position.x, hopper_position.y - 1 - offset),
        Position_2D(hopper_position.x + 1 + offset, hopper_position.y),
        Position_2D(hopper_position.x - 1 - offset, hopper_position.y),
    ]
    return next((position for position in candidates if factory_position_is_open(env, position)), hopper_position)


class DepositFactoryCube(Action):
    def __init__(self):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(nearby_within(1)),
                Target_Has_Component(FactoryHopper),
            ]
        )

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        return super().is_valid(actor, target, env, kwargs=kwargs) and carrying_factory_cube(actor) is not None

    def exec_action(self, actor, target, env, kwargs=None):
        inventory = actor.get_component(Inventory)
        cube = carrying_factory_cube(actor)
        if inventory is None or cube is None:
            return {"success": False}
        inventory.remove(cube)
        if cube in env.state.entities:
            env.destroy_entity(cube)
        hopper = target.get_component(FactoryHopper)
        for offset in range(hopper.output_count):
            position = factory_apple_spawn_position(env, target.position, offset)
            env.instantiate_entity(
                Entity(
                    name="Apple",
                    position=position,
                    tags=["apple", "food"],
                    components=[Renderable(sprite_path="src/items/consumables/lpc_food/apple.png", z_index=4)],
                )
            )
        env.factory_events.append(f"{actor.name} deposited a cube into {target.name}.")
        return {"deposited": cube.name, "apples": hopper.output_count}

    def action_description_text(self, actor, target, env):
        return f"Deposit a cube into {target.name}."


class FactoryManager(Component):
    def pre_actions_step(self, env: Environment) -> None:
        env.factory_events = []
        env.tick = env.cur_step

    def post_actions_step(self, env: Environment) -> None:
        if env.cur_step + 1 >= BENCHMARK_STEPS:
            for idx in range(len(env.agents)):
                env.truncations[idx] = True


# ---------------------------------------------------------------------------
# Paintball
# ---------------------------------------------------------------------------


class Flag(Component):
    def __init__(self, team: str):
        super().__init__(tags=["flag", f"{team}_flag"])
        self.team = team


class HillZone(Component):
    def __init__(self):
        super().__init__(tags=["hill"])


class PaintballZap(Action):
    def __init__(self):
        super().__init__(
            validation_rules=[
                Action_On_Cooldown("paintball"),
                Target_Not_Self(),
                Target_Is_Nearby(nearby_within(3)),
            ]
        )
        self.wall_attack = Attack("Shoot paintball at", 1, target_is_nearby=nearby_within(3))

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False
        if ("red" in actor.tags and "red" in target.tags) or ("blue" in actor.tags and "blue" in target.tags):
            return False
        return "player" in target.tags or "destructible" in target.tags

    def exec_action(self, actor, target, env, kwargs=None):
        cooldown = actor.get_component(Cooldown)
        if cooldown is not None:
            cooldown.start("paintball")
        if "player" in target.tags:
            freezable = target.get_component(Freezable)
            if freezable is not None:
                freezable.freeze(normalized_steps(5))
            env.paintball_events.append(f"{actor.name} tagged {target.name}.")
            return {"tagged": target.name}
        if self.wall_attack.is_valid(actor, target, env):
            result = self.wall_attack.exec_action(actor, target, env, kwargs)
            env.paintball_events.append(f"{actor.name} damaged {target.name}.")
            return result
        return {"success": False}

    def action_description_text(self, actor, target, env):
        return f"Shoot paintball at {target.name}."


class GrabFlag(Action):
    def __init__(self):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(nearby_within(1)),
                Target_Has_Component(Flag),
            ]
        )

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False
        flag = target.get_component(Flag)
        if flag is None or flag.team in actor.tags:
            return False
        inventory = actor.get_component(Inventory)
        return inventory is not None and inventory.has_space()

    def exec_action(self, actor, target, env, kwargs=None):
        inventory = actor.get_component(Inventory)
        inventory.store(target, env)
        renderable = target.get_component(Renderable)
        if renderable is not None:
            renderable.visible = False
        env.paintball_events.append(f"{actor.name} grabbed {target.name}.")
        return {"grabbed": target.name}

    def action_description_text(self, actor, target, env):
        return f"Grab {target.name}."


class PaintballManager(Component):
    def __init__(self, mode: str, red_base: tuple[int, int] | None = None, blue_base: tuple[int, int] | None = None):
        super().__init__()
        self.mode = mode
        self.red_base = red_base
        self.blue_base = blue_base
        self.red_score = 0.0
        self.blue_score = 0.0

    def pre_actions_step(self, env: Environment) -> None:
        env.paintball_events = []
        env.tick = env.cur_step

    def post_actions_step(self, env: Environment) -> None:
        if self.mode == "ctf":
            self._score_ctf(env)
        else:
            self._score_hill(env)
        if env.cur_step + 1 >= BENCHMARK_STEPS:
            for idx in range(len(env.agents)):
                env.truncations[idx] = True

    def _score_ctf(self, env: Environment) -> None:
        for idx, agent in enumerate(env.agents):
            inventory = agent.get_component(Inventory)
            if inventory is None:
                continue
            for item in list(inventory.contents):
                flag = item.get_component(Flag)
                if flag is None:
                    continue
                pos = (agent.position.x, agent.position.y)
                if "red" in agent.tags and flag.team == "blue" and pos == self.red_base:
                    env.last_step_rewards[idx] += 5.0
                    self.red_score += 1
                    env.terminations = [True] * len(env.agents)
                if "blue" in agent.tags and flag.team == "red" and pos == self.blue_base:
                    env.last_step_rewards[idx] += 5.0
                    self.blue_score += 1
                    env.terminations = [True] * len(env.agents)

    def _score_hill(self, env: Environment) -> None:
        hill_positions = [
            (entity.position.x, entity.position.y)
            for entity in env.state.entities
            if entity.get_component(HillZone) is not None
        ]
        if not hill_positions:
            return
        red_on_hill = [agent for agent in env.agents if "red" in agent.tags and (agent.position.x, agent.position.y) in hill_positions]
        blue_on_hill = [agent for agent in env.agents if "blue" in agent.tags and (agent.position.x, agent.position.y) in hill_positions]
        if red_on_hill and not blue_on_hill:
            self.red_score += 1
            for agent in red_on_hill:
                award_reward(env, agent, 0.5)
        if blue_on_hill and not red_on_hill:
            self.blue_score += 1
            for agent in blue_on_hill:
                award_reward(env, agent, 0.5)
        if self.red_score >= normalized_steps(30) or self.blue_score >= normalized_steps(30):
            env.terminations = [True] * len(env.agents)


# ---------------------------------------------------------------------------
# Predator-Prey
# ---------------------------------------------------------------------------


class PredatorPreyRole(Component):
    def __init__(self, role: str, spawn_position: tuple[int, int]):
        super().__init__(tags=[role])
        self.role = role
        self.spawn_position = spawn_position
        self.respawn_counter = 0

    @property
    def active(self) -> bool:
        return self.respawn_counter <= 0

    def remove_temporarily(self, duration: int = 40) -> None:
        self.respawn_counter = duration
        if hasattr(self, "entity"):
            self.entity.position = Position_2D(-1000, -1000)
            renderable = self.entity.get_component(Renderable)
            if renderable is not None:
                renderable.visible = False

    def post_actions_step(self, env: Environment) -> None:
        if self.respawn_counter <= 0:
            return
        self.respawn_counter -= 1
        if self.respawn_counter <= 0 and hasattr(self, "entity"):
            self.entity.position = Position_2D(*self.spawn_position)
            renderable = self.entity.get_component(Renderable)
            if renderable is not None:
                renderable.visible = True


class PredatorPreyFood(Component):
    def __init__(self, kind: str, reward: float):
        super().__init__(tags=[kind, "food"])
        self.kind = kind
        self.reward = reward
        self.consumed = False

    def post_actions_step(self, env: Environment) -> None:
        if self.consumed:
            return
        for agent in env.agents:
            role = agent.get_component(PredatorPreyRole)
            if role is None or role.role != "prey" or not role.active:
                continue
            if agent.position == self.entity.position:
                self.consumed = True
                award_reward(env, agent, self.reward)
                renderable = self.entity.get_component(Renderable)
                if renderable is not None:
                    renderable.visible = False
                env.predator_prey_events.append(f"{agent.name} ate {self.entity.name} for +{self.reward:g}.")
                return


class CatchPrey(Action):
    def __init__(self):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(nearby_within(1)),
            ]
        )

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False
        actor_role = actor.get_component(PredatorPreyRole)
        target_role = target.get_component(PredatorPreyRole)
        return (
            actor_role is not None
            and actor_role.role == "predator"
            and actor_role.active
            and target_role is not None
            and target_role.role == "prey"
            and target_role.active
        )

    def exec_action(self, actor, target, env, kwargs=None):
        target_role = target.get_component(PredatorPreyRole)
        target_role.remove_temporarily(normalized_steps(200))
        award_reward(env, actor, 10.0)
        env.predator_prey_events.append(f"{actor.name} caught {target.name}.")
        return {"caught": target.name}

    def action_description_text(self, actor, target, env):
        return f"Catch {target.name}."


class PredatorPreyManager(Component):
    def pre_actions_step(self, env: Environment) -> None:
        env.predator_prey_events = []
        env.tick = env.cur_step

    def post_actions_step(self, env: Environment) -> None:
        if env.cur_step + 1 >= BENCHMARK_STEPS:
            for idx in range(len(env.agents)):
                env.truncations[idx] = True
