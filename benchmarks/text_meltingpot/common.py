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
