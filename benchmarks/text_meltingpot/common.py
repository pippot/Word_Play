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
from word_play.presets.environments.simple_2d_grid_world import Simple_2D_Grid_World
from word_play.presets.systems.combat import Attack
from word_play.presets.systems.containers import Regrowable_Item_Source
from word_play.presets.systems.cooldown import Action_On_Cooldown, Cooldown
from word_play.presets.systems.crafter import Crafter, Load_First_Into_Crafter
from word_play.presets.systems.freezable import Freezable
from word_play.presets.movement.simple_2d_grid import Collidable, Move_Down, Move_Left, Move_Right, Move_Up, Position_2D
from word_play.presets.observation.simple_observation import Simple_Observation
from word_play.presets.renderers import Renderable
from word_play.presets.systems.inventory import Inventory, Pick_Up_Item, Put_In_Container, Target_Not_In_Inventory
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


def held_soup_index(inventory: Inventory) -> int | None:
    for idx, item in enumerate(inventory.contents):
        if soup_is_valid(item):
            return idx
    return None


def held_item_index_with_tag(inventory: Inventory, tag: str) -> int | None:
    for idx, item in enumerate(inventory.contents):
        if tag in item.tags:
            return idx
    return None


def shared_delivery_bonus(actor, target, env, item: Entity, reward: float) -> None:
    for idx, agent in enumerate(env.agents):
        if agent is not actor and idx < len(env.last_step_rewards):
            env.last_step_rewards[idx] += reward
    env.completed_orders = getattr(env, "completed_orders", 0) + 1


class Load_Held_Item_Into_Crafter(Load_First_Into_Crafter):
    def action_description_text(self, actor, target, env) -> str:
        return super().action_description_text(actor, target, env)


class Plate_Ready_Soup_With_Dish(Action):
    def __init__(self):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(),
                Target_Has_Component(Crafter),
            ]
        )

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False
        inventory = actor.get_component(Inventory)
        crafter = target.get_component(Crafter)
        if inventory is None or crafter is None or not crafter.has_output():
            return False
        if crafter.ready_item is None or not soup_is_valid(crafter.ready_item):
            return False
        return held_item_index_with_tag(inventory, "dish") is not None

    def exec_action(self, actor, target, env, kwargs=None):
        inventory = actor.get_component(Inventory)
        crafter = target.get_component(Crafter)
        dish_idx = held_item_index_with_tag(inventory, "dish")
        if dish_idx is None or crafter is None or not crafter.has_output():
            return {"success": False, "reason": "missing_dish_or_soup"}

        dish = inventory.remove_by_index(dish_idx)
        if dish in env.state.entities:
            env.destroy_entity(dish)

        soup = crafter.collect_output()
        inventory.store(soup, env)
        return {"success": True, "plated": soup.name, "used": "Dish", "from": target.name}

    def action_description_text(self, actor, target, env) -> str:
        return f"Use held Dish to collect ready Soup from {target.name}."


class Put_Held_Item_On_Counter(Put_In_Container):
    def __init__(self):
        super().__init__(target_tags=["counter"], destroy_item=False)
        self.required_kwargs = None

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        inventory = actor.get_component(Inventory)
        counter_inventory = target.get_component(Inventory)
        if inventory is None or counter_inventory is None or not inventory.contents:
            return False
        if not counter_inventory.can_accept(inventory.contents[0]):
            return False
        return super().is_valid(actor, target, env, kwargs={"inventory_index": 0})

    def exec_action(self, actor, target, env, kwargs=None):
        return super().exec_action(actor, target, env, kwargs={"inventory_index": 0})

    def action_description_text(self, actor, target, env) -> str:
        inventory = actor.get_component(Inventory)
        item_name = inventory.contents[0].name if inventory is not None and inventory.contents else "held item"
        return f"Put held {item_name} on {target.name}."


class Take_First_Item_From_Counter(Action):
    def __init__(self):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(),
                Target_Has_Tag(["counter"]),
            ]
        )

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False
        inventory = actor.get_component(Inventory)
        counter_inventory = target.get_component(Inventory)
        if inventory is None or counter_inventory is None or not counter_inventory.contents:
            return False
        return inventory.can_accept(counter_inventory.contents[0])

    def exec_action(self, actor, target, env, kwargs=None):
        inventory = actor.get_component(Inventory)
        counter_inventory = target.get_component(Inventory)
        item = counter_inventory.remove_by_index(0)
        if item is None:
            return {"success": False, "reason": "counter_empty"}
        inventory.store(item, env)
        return {"success": True, "taken": item.name, "from": target.name}

    def action_description_text(self, actor, target, env) -> str:
        counter_inventory = target.get_component(Inventory)
        item_name = (
            counter_inventory.contents[0].name
            if counter_inventory is not None and counter_inventory.contents
            else "item"
        )
        return f"Take {item_name} from {target.name}."


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
        soup_idx = held_soup_index(inventory)
        if soup_idx is None:
            return False
        return super().is_valid(actor, target, env, kwargs={"inventory_index": soup_idx})

    def exec_action(self, actor, target, env, kwargs=None):
        inventory = actor.get_component(Inventory)
        soup_idx = held_soup_index(inventory)
        if soup_idx is None:
            return {"success": False, "reason": "no_held_soup"}
        return super().exec_action(actor, target, env, kwargs={"inventory_index": soup_idx})

    def action_description_text(self, actor, target, env) -> str:
        inventory = actor.get_component(Inventory)
        soup_idx = held_soup_index(inventory) if inventory is not None else None
        item_name = inventory.contents[soup_idx].name if inventory is not None and soup_idx is not None else "soup"
        return f"Turn in your held {item_name} at {target.name} for shared reward."


def cooking_inventory_names(entity: Entity) -> list[str]:
    inventory = entity.get_component(Inventory)
    return [item.name for item in inventory.contents] if inventory is not None else []


def cooking_relative_position(actor: Entity, target: Entity) -> str:
    dx = target.position.x - actor.position.x
    dy = target.position.y - actor.position.y
    return f"relative=({dx:+d},{dy:+d}), distance={abs(dx) + abs(dy)}"


def cooking_entity_in_inventory(entity: Entity, env: Environment) -> bool:
    for holder in env.state.entities:
        inventory = holder.get_component(Inventory)
        if inventory is not None and entity in inventory.contents:
            return True
    return False


def cooking_source_item_name(source: Regrowable_Item_Source | None) -> str:
    if source is None:
        return "item"
    item_factory = source.item_factory
    return item_factory.name if isinstance(item_factory, Entity) else "item"


def cooking_pot_status(crafter: Crafter | None) -> str:
    if crafter is None:
        return "state unknown"
    if crafter.ready_item is not None:
        return f"READY with {crafter.ready_item.name}; requires held Dish to collect"
    if crafter.remaining_steps is not None:
        return f"cooking; {crafter.remaining_steps} steps until Soup is ready"

    recipe = crafter.active_recipe or (crafter.recipes[0] if crafter.recipes else None)
    total = len(recipe.input_names) if recipe is not None else 0
    loaded = len(crafter.loaded_items)
    if recipe is not None and recipe.input_names and len(set(recipe.input_names)) == 1:
        item_name = recipe.input_names[0]
        return f"loaded {loaded}/{total} {item_name}; accepts more {item_name}" if loaded < total else "loaded"
    return f"loaded inputs: {crafter.loaded_items or []}"


def cooking_entity_symbol(entity: Entity, agent: Entity, env: Environment) -> str:
    if entity is agent:
        return "A"
    if entity.is_agent:
        return "P"
    if cooking_entity_in_inventory(entity, env):
        return ""
    tags = set(entity.tags)
    if entity.get_component(Crafter) is not None:
        crafter = entity.get_component(Crafter)
        return "S" if crafter is not None and crafter.ready_item is not None else "C"
    if "tomato_source" in tags:
        return "O"
    if "dish_source" in tags:
        return "D"
    if "delivery" in tags:
        return "T"
    if "counter" in tags:
        return "-"
    if "tomato" in tags:
        return "o"
    if "dish" in tags:
        return "d"
    if "soup" in tags:
        return "s"
    if "wall" in tags:
        return "#"
    return "?"


def cooking_local_map(env: Simple_2D_Grid_World, agent: Entity) -> str:
    radius = env.observation_radius
    visible_entities = env.entities_in_observation_square(agent.position)
    by_position: dict[tuple[int, int], str] = {}
    priority = {"A": 9, "P": 8, "S": 7, "C": 6, "O": 6, "D": 6, "T": 6, "s": 5, "o": 5, "d": 5, "-": 4, "#": 3, "?": 1}
    for entity in visible_entities:
        symbol = cooking_entity_symbol(entity, agent, env)
        if not symbol:
            continue
        key = (entity.position.x, entity.position.y)
        if priority.get(symbol, 0) >= priority.get(by_position.get(key, "."), 0):
            by_position[key] = symbol

    rows = []
    for y in range(agent.position.y + radius, agent.position.y - radius - 1, -1):
        row = []
        for x in range(agent.position.x - radius, agent.position.x + radius + 1):
            row.append(by_position.get((x, y), "."))
        rows.append("  " + " ".join(row))
    return (
        "LOCAL KITCHEN MAP:\n"
        "  legend: A=you, P=other player, C=pot, S=ready soup pot, O=tomato source, D=dish source, "
        "T=delivery, -=counter, #=wall, o/d/s=dropped item\n"
        + "\n".join(rows)
    )


def cooking_status_section(env: Simple_2D_Grid_World, agent: Entity) -> str:
    visible_entities = env.entities_in_observation_square(agent.position)
    inventory = agent.get_component(Inventory)
    held = cooking_inventory_names(agent)
    lines = [
        "KITCHEN STATUS:",
        f"  completed_orders: {getattr(env, 'completed_orders', 0)}",
        f"  you_hold: {held or ['empty']}",
    ]

    visible_pots = [entity for entity in visible_entities if entity.get_component(Crafter) is not None]
    if visible_pots:
        lines.append("  visible_pots:")
        for pot in visible_pots:
            lines.append(
                f"    - {pot.name} at {pot.position} ({cooking_relative_position(agent, pot)}): "
                f"{cooking_pot_status(pot.get_component(Crafter))}"
            )

    visible_sources = [
        entity for entity in visible_entities if entity.get_component(Regrowable_Item_Source) is not None
    ]
    if visible_sources:
        lines.append("  visible_sources:")
        for source_entity in visible_sources:
            source = source_entity.get_component(Regrowable_Item_Source)
            lines.append(
                f"    - {source_entity.name} at {source_entity.position} "
                f"({cooking_relative_position(agent, source_entity)}): gives {cooking_source_item_name(source)}"
            )

    visible_delivery = [entity for entity in visible_entities if "delivery" in entity.tags]
    if visible_delivery:
        lines.append("  visible_delivery:")
        for delivery in visible_delivery:
            lines.append(f"    - {delivery.name} at {delivery.position} ({cooking_relative_position(agent, delivery)}): accepts held Soup")

    ready_pot_visible = any(
        pot.get_component(Crafter) is not None and pot.get_component(Crafter).ready_item is not None
        for pot in visible_pots
    )
    if inventory is not None and held_soup_index(inventory) is not None:
        lines.append("  relevant_mechanic: holding Soup means the delivery action is the reward action when next to Delivery Window.")
    elif ready_pot_visible:
        if inventory is not None and held_item_index_with_tag(inventory, "dish") is not None:
            lines.append("  relevant_mechanic: visible pot has ready Soup and you hold Dish, so plating Soup is available when adjacent to the pot.")
        else:
            lines.append("  relevant_mechanic: visible pot has ready Soup; hold Dish, then use Dish on the ready pot to collect Soup.")
    elif inventory is not None and any(held_item_index_with_tag(inventory, tag) is not None for tag in ("tomato", "ingredient")):
        lines.append("  relevant_mechanic: holding Tomato means you can load it into a visible Cooking Pot when adjacent.")
    else:
        lines.append("  relevant_mechanic: Soup needs 3 Tomatoes loaded into one pot, then a Dish to collect, then delivery.")

    return "\n".join(lines)


def format_cooking_nearby_entities(nearby_entities: list[Entity], agent: Entity, env: Environment) -> str:
    lines = ["VISIBLE KITCHEN ENTITIES:"]
    for entity in sorted(
        nearby_entities,
        key=lambda candidate: (abs(candidate.position.x - agent.position.x) + abs(candidate.position.y - agent.position.y), candidate.name),
    ):
        if entity is agent or cooking_entity_in_inventory(entity, env):
            continue
        tags = set(entity.tags)
        if entity.is_agent:
            lines.append(
                f"- {entity.name} at {entity.position} ({cooking_relative_position(agent, entity)}), "
                f"holding {cooking_inventory_names(entity) or ['empty']}"
            )
        elif entity.get_component(Crafter) is not None:
            lines.append(
                f"- {entity.name} at {entity.position} ({cooking_relative_position(agent, entity)}): "
                f"{cooking_pot_status(entity.get_component(Crafter))}"
            )
        elif entity.get_component(Regrowable_Item_Source) is not None:
            source = entity.get_component(Regrowable_Item_Source)
            lines.append(
                f"- {entity.name} at {entity.position} ({cooking_relative_position(agent, entity)}): "
                f"gives {cooking_source_item_name(source)}"
            )
        elif "delivery" in tags:
            lines.append(f"- {entity.name} at {entity.position} ({cooking_relative_position(agent, entity)}): accepts held Soup")
        elif "counter" in tags:
            contents = cooking_inventory_names(entity)
            if contents:
                lines.append(
                    f"- {entity.name} at {entity.position} ({cooking_relative_position(agent, entity)}): holding {contents}"
                )
        elif tags.intersection({"tomato", "dish", "soup", "ingredient"}):
            lines.append(f"- {entity.name} item at {entity.position} ({cooking_relative_position(agent, entity)})")
    if len(lines) == 1:
        return "VISIBLE KITCHEN ENTITIES: None"
    return "\n".join(lines)


class Collaborative_Cooking_Grid_World(Simple_2D_Grid_World):
    def observe(self, agent_id: int):
        agent = self.agents[agent_id]
        return Simple_Observation(
            possible_actions=self.possible_actions(agent),
            nearby_entities=self.entities_in_observation_square(agent.position),
            agent=agent,
            last_reward=self.last_rewards[agent_id],
            info=self.infos[agent_id],
            observation_radius=self.observation_radius,
            extra_sections=(cooking_local_map(self, agent), cooking_status_section(self, agent)),
            nearby_entities_formatter=lambda nearby_entities, current_agent: format_cooking_nearby_entities(
                nearby_entities, current_agent, self
            ),
        )

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
                Target_Not_In_Inventory(),
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

PREDATOR_PREY_RESPAWN_STEPS = normalized_steps(200)
PREDATOR_PREY_GROUP_DEFENSE_RADIUS = 3
PREDATOR_PREY_PREDATOR_REWARD = 1.0
PREDATOR_PREY_ACORN_EAT_STEPS = normalized_steps(40, minimum=3)
PREDATOR_PREY_FOOD_RESPAWN_STEPS = normalized_steps(120)


def predator_prey_distance(entity_a: Entity, entity_b: Entity) -> int:
    return abs(entity_a.position.x - entity_b.position.x) + abs(entity_a.position.y - entity_b.position.y)


def predator_prey_set_renderable(entity: Entity, visible: bool) -> None:
    renderable = entity.get_component(Renderable)
    if renderable is not None:
        renderable.visible = visible


def predator_prey_begin_step(env: Environment) -> None:
    if getattr(env, "_predator_prey_events_step", None) == env.cur_step:
        return
    env.predator_prey_events = []
    env.tick = env.cur_step
    env._predator_prey_events_step = env.cur_step


def predator_prey_finish_step(env: Environment) -> None:
    if env.cur_step + 1 >= BENCHMARK_STEPS:
        for idx in range(len(env.agents)):
            env.truncations[idx] = True


class PredatorPreyRole(Component):
    def __init__(self, role: str, spawn_position: tuple[int, int]):
        if role not in {"predator", "prey"}:
            raise ValueError(f"Unknown predator-prey role: {role}")
        super().__init__(tags=[role])
        self.role = role
        self.spawn_position = spawn_position
        self.respawn_counter = 0
        self.max_stamina = 6 if role == "predator" else 9
        self.stamina = self.max_stamina
        self.eating_food_name: str | None = None
        self.eating_timer = 0
        self.eating_started_step: int | None = None
        self.eating_reward_per_step = 0.0
        self._before: Position_2D | None = None

    @property
    def active(self) -> bool:
        return self.respawn_counter <= 0

    @property
    def eating(self) -> bool:
        return self.eating_timer > 0

    def post_initialization(self) -> None:
        self._sync_tags()

    def pre_actions_step(self, env: Environment) -> None:
        predator_prey_begin_step(env)
        self._before = Position_2D(self.entity.position.x, self.entity.position.y)
        self._sync_tags()

    def can_move(self) -> bool:
        return self.active and not self.eating and self.stamina > 0

    def can_interact(self) -> bool:
        return self.active and not self.eating and self.stamina > 0

    def remove_temporarily(self, duration: int = PREDATOR_PREY_RESPAWN_STEPS) -> None:
        self.respawn_counter = duration
        self.eating_food_name = None
        self.eating_timer = 0
        self.eating_started_step = None
        self.eating_reward_per_step = 0.0
        if hasattr(self, "entity"):
            self.entity.position = Position_2D(-1000, -1000)
            predator_prey_set_renderable(self.entity, False)
            self._sync_tags()

    def start_eating(self, food: Entity, reward: float, eat_steps: int, env: Environment) -> None:
        self.eating_food_name = food.name
        self.eating_timer = max(1, eat_steps)
        self.eating_started_step = env.cur_step
        self.eating_reward_per_step = reward / self.eating_timer

    def _sync_tags(self) -> None:
        if not hasattr(self, "entity"):
            return
        while self.role not in self.entity.tags:
            self.entity.tags.append(self.role)
        if self.active:
            while "respawning" in self.entity.tags:
                self.entity.tags.remove("respawning")
        elif "respawning" not in self.entity.tags:
            self.entity.tags.append("respawning")
        if self.eating:
            if "eating" not in self.entity.tags:
                self.entity.tags.append("eating")
        else:
            while "eating" in self.entity.tags:
                self.entity.tags.remove("eating")
        self._sync_actions()

    def _sync_actions(self) -> None:
        movement_types = (Move_Up, Move_Down, Move_Left, Move_Right)
        self.entity.actions = [action for action in self.entity.actions if not isinstance(action, movement_types)]
        if not self.can_move():
            return
        catch_actions = [action for action in self.entity.actions if isinstance(action, CatchPrey)]
        non_catch_actions = [action for action in self.entity.actions if not isinstance(action, CatchPrey)]
        self.entity.actions = [
            *non_catch_actions,
            Move_Up(),
            Move_Down(),
            Move_Left(),
            Move_Right(),
            *catch_actions,
        ]

    def post_actions_step(self, env: Environment) -> None:
        predator_prey_finish_step(env)
        if self.respawn_counter > 0:
            self.respawn_counter -= 1
            if self.respawn_counter <= 0 and hasattr(self, "entity"):
                self.entity.position = Position_2D(*self.spawn_position)
                self.stamina = self.max_stamina
                predator_prey_set_renderable(self.entity, True)
            self._sync_tags()
            return

        if self.eating:
            if self.eating_started_step == env.cur_step:
                self._sync_tags()
                return
            award_reward(env, self.entity, self.eating_reward_per_step)
            self.eating_timer -= 1
            if self.eating_timer <= 0:
                env.predator_prey_events.append(f"{self.entity.name} finished eating {self.eating_food_name}.")
                self.eating_food_name = None
                self.eating_started_step = None
                self.eating_reward_per_step = 0.0
            self._sync_tags()
            return

        if self._before is None:
            return
        moved = self.entity.position != self._before
        self.stamina = max(0, self.stamina - 1) if moved else min(self.max_stamina, self.stamina + 1)
        self._sync_tags()


class PredatorPreyFood(Component):
    def __init__(
        self,
        kind: str,
        reward: float,
        eat_steps: int | None = None,
        respawn_steps: int = PREDATOR_PREY_FOOD_RESPAWN_STEPS,
    ):
        if kind not in {"apple", "acorn"}:
            raise ValueError(f"Unknown predator-prey food kind: {kind}")
        super().__init__(tags=[kind, "food"])
        self.kind = kind
        self.reward = reward
        self.eat_steps = eat_steps if eat_steps is not None else (PREDATOR_PREY_ACORN_EAT_STEPS if kind == "acorn" else 1)
        self.respawn_steps = respawn_steps
        self.available = True
        self.respawn_counter = 0

    def post_initialization(self) -> None:
        self._sync()

    def _sync(self) -> None:
        if not hasattr(self, "entity"):
            return
        predator_prey_set_renderable(self.entity, self.available)
        for tag in (self.kind, "food"):
            if self.available and tag not in self.entity.tags:
                self.entity.tags.append(tag)
            while not self.available and tag in self.entity.tags:
                self.entity.tags.remove(tag)
        base_name = self.kind.title()
        self.entity.name = base_name if self.available else f"Empty {base_name} Site"

    def _consume(self) -> None:
        self.available = False
        self.respawn_counter = self.respawn_steps
        self._sync()

    def post_actions_step(self, env: Environment) -> None:
        if not self.available:
            self.respawn_counter -= 1
            if self.respawn_counter <= 0:
                self.available = True
                self._sync()
            return
        for agent in env.agents:
            role = agent.get_component(PredatorPreyRole)
            if role is None or role.role != "prey" or not role.active:
                continue
            if role.eating:
                continue
            if agent.position == self.entity.position:
                if self.kind == "acorn":
                    role.start_eating(self.entity, self.reward, self.eat_steps, env)
                    env.predator_prey_events.append(
                        f"{agent.name} started eating {self.entity.name} for +{self.reward:g} over {self.eat_steps} steps."
                    )
                else:
                    award_reward(env, agent, self.reward)
                    env.predator_prey_events.append(f"{agent.name} ate {self.entity.name} for +{self.reward:g}.")
                self._consume()
                return


class ActorCanCatch(Action_Validation):
    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        role = actor.get_component(PredatorPreyRole)
        return role is not None and role.role == "predator" and role.can_interact()


class TargetCanBeCaught(Action_Validation):
    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        target_role = target_entity.get_component(PredatorPreyRole)
        return target_role is not None and target_role.active


def prey_defends_against_catch(prey: Entity, env: Environment) -> bool:
    active_prey = 0
    active_predators = 0
    for agent in env.agents:
        role = agent.get_component(PredatorPreyRole)
        if role is None or not role.active:
            continue
        if predator_prey_distance(agent, prey) > PREDATOR_PREY_GROUP_DEFENSE_RADIUS:
            continue
        if role.role == "prey" and not role.eating:
            active_prey += 1
        elif role.role == "predator":
            active_predators += 1
    return active_prey > active_predators


class CatchPrey(Action):
    def __init__(self):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(nearby_within(1)),
                ActorCanCatch(),
                TargetCanBeCaught(),
            ]
        )

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False
        actor_role = actor.get_component(PredatorPreyRole)
        target_role = target.get_component(PredatorPreyRole)
        if actor_role is None or target_role is None:
            return False
        if target_role.role == "prey" and prey_defends_against_catch(target, env):
            return False
        return target_role.role in {"prey", "predator"}

    def exec_action(self, actor, target, env, kwargs=None):
        target_role = target.get_component(PredatorPreyRole)
        target_role.remove_temporarily(PREDATOR_PREY_RESPAWN_STEPS)
        reward = PREDATOR_PREY_PREDATOR_REWARD if target_role.role == "prey" else 0.0
        if reward:
            award_reward(env, actor, reward)
        env.predator_prey_events.append(f"{actor.name} caught {target.name} for +{reward:g}.")
        return {"caught": target.name, "reward": reward}

    def action_description_text(self, actor, target, env):
        return f"Catch {target.name}."
