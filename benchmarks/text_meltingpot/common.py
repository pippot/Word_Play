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
from word_play.presets.action_validations import Target_Has_Component, Target_Has_Tag, Target_Within_Range
from word_play.presets.systems.combat import Attack
from word_play.presets.systems.cooldown import Action_On_Cooldown, Cooldown
from word_play.presets.systems.coordinated_action import Coordinated_Action
from word_play.presets.systems.crafter import Crafter
from word_play.presets.systems.freezable import Freezable
from word_play.presets.movement.simple_2d_grid import (
    Collidable,
    Position_2D,
)
from word_play.presets.renderers import Renderable
from word_play.presets.systems.inventory import Inventory, Target_Not_In_Inventory
from word_play.presets.systems.preferences import Preference
from word_play.presets.systems.regrowable import Regrowable
from word_play.presets.systems.respawnable import Respawnable
from word_play.presets.systems.reward import Rewardable, award_reward
from word_play.presets.systems.role import Role
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
        if inhibitor is not None and first_molecule_near(env, inhibitor, self.entity.position, distance=0) is not None:
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


def first_molecule_near(
    env: Environment,
    kind: str,
    position: Position_2D,
    exclude: Entity | None = None,
    distance: int = 1,
) -> Entity | None:
    for entity in env.state.entities:
        entity_position = molecule_position(entity, env)
        if (
            entity is not exclude
            and molecule_kind(entity) == kind
            and abs(entity_position.x - position.x) + abs(entity_position.y - position.y) <= distance
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


class Commons_Apple_Patch(Regrowable):
    def __init__(self, apple_regrowth_probs: list[float] | None = None):
        super().__init__(
            regrow_tick_interval=1,
            density_dependent=True,
            density_radius=2,
            density_regrow_probs=[
                normalized_probability(prob)
                for prob in (apple_regrowth_probs or [0.0, 0.0025, 0.005, 0.025])
            ],
            active_tags=["apple"],
            inactive_tags=["depleted_apple_patch"],
        )
        self.tags.append("apple_patch")

    def pre_actions_step(self, env: Environment) -> None:
        if getattr(env, "_commons_events_step", None) != env.cur_step:
            env.tick = env.cur_step
            env.commons_events = []
            env._commons_events_step = env.cur_step
        was_consumed = self.consumed
        super().pre_actions_step(env)
        if was_consumed and not self.consumed:
            env.commons_events.append(f"{self.entity.name} regrew.")

    def post_actions_step(self, env: Environment) -> None:
        self._collect(env)
        self._maybe_end_episode(env)

    def _collect(self, env: Environment) -> None:
        if self.consumed:
            return
        for agent in env.agents:
            if agent.position == self.entity.position:
                self.consume()
                rewardable = self.entity.get_component(Rewardable)
                reward = rewardable.reward_for(agent, env) if rewardable is not None else 1.0
                if rewardable is None:
                    award_reward(env, agent, reward)
                env.commons_events.append(f"{agent.name} ate {self.entity.name} for +{reward:g}.")
                break

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
                Target_Is_Nearby(),
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


# ---------------------------------------------------------------------------
# Collaborative Cooking
# ---------------------------------------------------------------------------


COOKING_DELIVERY_REWARD = 20.0


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
        return (
            inventory is not None
            and crafter is not None
            and crafter.ready_item is not None
            and "soup" in crafter.ready_item.tags
            and inventory.first_index_with_tags("dish") is not None
        )

    def exec_action(self, actor, target, env, kwargs=None):
        inventory = actor.get_component(Inventory)
        crafter = target.get_component(Crafter)
        dish = inventory.remove_by_index(inventory.first_index_with_tags("dish"))
        if dish in env.state.entities:
            env.destroy_entity(dish)
        soup = crafter.collect_output()
        inventory.store(soup, env)
        return {"success": True, "plated": soup.name, "used": "Dish", "from": target.name}

    def action_description_text(self, actor, target, env) -> str:
        return f"Use held Dish to collect ready Soup from {target.name}."


# ---------------------------------------------------------------------------
# Boat Race
# ---------------------------------------------------------------------------


def advance_boat_race(env: Environment, participants: list[Entity], reward: float, event: str) -> None:
    if not hasattr(env, "boat_events"):
        env.boat_events = []
    env.boat_progress = getattr(env, "boat_progress", 0) + 1
    for participant in participants:
        award_reward(env, participant, reward)
    env.boat_events.append(event.format(progress=env.boat_progress, participants=len(participants)))
    if env.boat_progress >= getattr(env, "boat_num_races", 8):
        env.terminations = [True] * len(env.agents)


class Paddle(Coordinated_Action):
    def __init__(self):
        super().__init__(
            "Paddle the boat.",
            2,
            coordination_key="boat_paddle",
            same_target=False,
            validation_rules=[Target_Is_Self()],
        )

    def exec_coordinated_action(self, actor, target, env, participants, kwargs=None):
        advance_boat_race(
            env,
            participants,
            0.2,
            "{participants} paddlers coordinated; race progress is {progress}.",
        )
        return {"paddle": True, "progress": env.boat_progress, "reward_each": 0.2}

    def action_description_text(self, actor, target, env):
        return "Paddle the boat."


class Flail(Coordinated_Action):
    def __init__(self):
        super().__init__(
            "Flail the oar.",
            1,
            coordination_key="boat_flail",
            same_target=False,
            validation_rules=[Target_Is_Self()],
        )

    def exec_coordinated_action(self, actor, target, env, participants, kwargs=None):
        paddle_count = sum(
            1
            for selection in self._selected_actions(env)
            if isinstance(selection.action, Paddle)
            and selection.action._local_is_valid(
                selection.actor, selection.target_entity, env, selection.action_kwargs
            )
        )
        if paddle_count >= 2 or random.random() >= 0.25:
            return {"flail": True, "moved": False}

        advance_boat_race(
            env,
            participants,
            0.05,
            "A flail moved a boat; race progress is {progress}.",
        )
        return {"flail": True, "moved": True, "progress": env.boat_progress, "reward_each": 0.05}

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
                rewardable = self.entity.get_component(Rewardable)
                reward = rewardable.reward_for(agent, env) if rewardable is not None else self.reward
                if rewardable is None:
                    award_reward(env, agent, reward)
                renderable = self.entity.get_component(Renderable)
                if renderable is not None:
                    renderable.visible = False
                if not hasattr(env, "boat_events"):
                    env.boat_events = []
                env.boat_events.append(f"{agent.name} ate {self.entity.name} for +{reward:g}.")
                return


# ---------------------------------------------------------------------------
# Factory Commons
# ---------------------------------------------------------------------------


class FactoryCube(Component):
    def __init__(self, cube_type: str = "blue", live: bool = True):
        super().__init__(tags=["cube", cube_type] if live else [])
        self.cube_type = cube_type
        self.live = live


class FactoryHopper(Component):
    def __init__(self, output_count: int = 1):
        super().__init__(tags=["hopper"])
        self.output_count = output_count


class DepositFactoryCube(Action):
    def __init__(self):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(),
                Target_Has_Component(FactoryHopper),
            ]
        )

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        return super().is_valid(actor, target, env, kwargs=kwargs) and self._carried_cube(actor) is not None

    def exec_action(self, actor, target, env, kwargs=None):
        inventory = actor.get_component(Inventory)
        cube = self._carried_cube(actor)
        if inventory is None or cube is None:
            return {"success": False}
        inventory.remove(cube)
        if cube in env.state.entities:
            env.destroy_entity(cube)
        hopper = target.get_component(FactoryHopper)
        for offset in range(hopper.output_count):
            position = self._apple_spawn_position(env, target.position, offset)
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

    def _carried_cube(self, actor: Entity) -> Entity | None:
        inventory = actor.get_component(Inventory)
        if inventory is None:
            return None
        return next((item for item in inventory.contents if item.get_component(FactoryCube) is not None), None)

    def _apple_spawn_position(self, env: Environment, hopper_position: Position_2D, offset: int) -> Position_2D:
        candidates = (
            Position_2D(hopper_position.x + 1, hopper_position.y + offset),
            Position_2D(hopper_position.x - 1, hopper_position.y + offset),
            Position_2D(hopper_position.x, hopper_position.y + 1 + offset),
            Position_2D(hopper_position.x, hopper_position.y - 1 - offset),
            Position_2D(hopper_position.x + 1 + offset, hopper_position.y),
            Position_2D(hopper_position.x - 1 - offset, hopper_position.y),
        )
        return next(
            (
                position for position in candidates
                if not any(
                    entity.position == position
                    and (entity.has_component(Collidable) or "apple" in entity.tags)
                    for entity in env.state.entities
                )
            ),
            hopper_position,
        )

    def action_description_text(self, actor, target, env):
        return f"Deposit a cube into {target.name}."


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
                Target_Within_Range(3),
            ]
        )
        self.wall_attack = Attack("Shoot paintball at", 1, target_is_nearby=Target_Within_Range(3).is_valid)

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
                Target_Is_Nearby(),
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
                    award_reward(env, agent, 5.0)
                    self.red_score += 1
                    env.terminations = [True] * len(env.agents)
                if "blue" in agent.tags and flag.team == "red" and pos == self.blue_base:
                    award_reward(env, agent, 5.0)
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


class PredatorPreyRole(Role):
    def __init__(self, role: str):
        if role not in {"predator", "prey"}:
            raise ValueError(f"Unknown predator-prey role: {role}")
        super().__init__(role)
        self.max_stamina = 6 if role == "predator" else 9
        self.stamina = self.max_stamina
        self.eating_food_name: str | None = None
        self.eating_timer = 0
        self.eating_started_step: int | None = None
        self.eating_reward_per_step = 0.0
        self._before: Position_2D | None = None
        self._was_active = True

    @property
    def active(self) -> bool:
        respawnable = self.entity.get_component(Respawnable)
        return respawnable is None or respawnable.active

    @property
    def eating(self) -> bool:
        return self.eating_timer > 0

    def pre_actions_step(self, env: Environment) -> None:
        if getattr(env, "_predator_prey_events_step", None) != env.cur_step:
            env.predator_prey_events = []
            env.tick = env.cur_step
            env._predator_prey_events_step = env.cur_step
        if self.active and not self._was_active:
            self.stamina = self.max_stamina
        self._was_active = self.active
        self._before = Position_2D(self.entity.position.x, self.entity.position.y)

    def can_move(self) -> bool:
        return self.active and not self.eating and self.stamina > 0

    def can_interact(self) -> bool:
        return self.active and not self.eating and self.stamina > 0

    def remove_temporarily(self, duration: int = PREDATOR_PREY_RESPAWN_STEPS) -> None:
        self.eating_food_name = None
        self.eating_timer = 0
        self.eating_started_step = None
        self.eating_reward_per_step = 0.0
        respawnable = self.entity.get_component(Respawnable)
        if respawnable is not None:
            respawnable.remove_temporarily(duration)

    def start_eating(self, food: Entity, reward: float, eat_steps: int, env: Environment) -> None:
        self.eating_food_name = food.name
        self.eating_timer = max(1, eat_steps)
        self.eating_started_step = env.cur_step
        self.eating_reward_per_step = reward / self.eating_timer

    def post_actions_step(self, env: Environment) -> None:
        if env.cur_step + 1 >= BENCHMARK_STEPS:
            for idx in range(len(env.agents)):
                env.truncations[idx] = True
        if not self.active:
            return

        if self.eating:
            if self.eating_started_step == env.cur_step:
                return
            award_reward(env, self.entity, self.eating_reward_per_step)
            self.eating_timer -= 1
            if self.eating_timer <= 0:
                env.predator_prey_events.append(f"{self.entity.name} finished eating {self.eating_food_name}.")
                self.eating_food_name = None
                self.eating_started_step = None
                self.eating_reward_per_step = 0.0
            return

        if self._before is None:
            return
        moved = self.entity.position != self._before
        self.stamina = max(0, self.stamina - 1) if moved else min(self.max_stamina, self.stamina + 1)


class PredatorPreyFood(Respawnable):
    def __init__(
        self,
        kind: str,
        reward: float,
        eat_steps: int | None = None,
        respawn_steps: int = PREDATOR_PREY_FOOD_RESPAWN_STEPS,
    ):
        if kind not in {"apple", "acorn"}:
            raise ValueError(f"Unknown predator-prey food kind: {kind}")
        super().__init__(
            respawn_steps,
            inactive_tag="eaten_food",
            inactive_position=Position_2D(-1000, -1000),
        )
        self.tags.extend([kind, "food"])
        self.kind = kind
        self.reward = reward
        self.eat_steps = eat_steps if eat_steps is not None else (PREDATOR_PREY_ACORN_EAT_STEPS if kind == "acorn" else 1)
        self.respawn_steps = respawn_steps

    def _consume(self) -> None:
        self.remove_temporarily(self.respawn_steps)

    def post_actions_step(self, env: Environment) -> None:
        if not self.active:
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


class ActorCanMove(Action_Validation):
    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        role = actor.get_component(PredatorPreyRole)
        return role is None or role.can_move()


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
        if (
            abs(agent.position.x - prey.position.x)
            + abs(agent.position.y - prey.position.y)
            > PREDATOR_PREY_GROUP_DEFENSE_RADIUS
        ):
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
                Target_Is_Nearby(),
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
