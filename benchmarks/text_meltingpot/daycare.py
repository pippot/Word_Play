from __future__ import annotations

import argparse
import random

from word_play.core import (
    Action,
    Agent_Policy,
    Component,
    Entity,
    Environment,
    Target_Is_Nearby,
    Target_Not_Self,
    Target_Is_Self,
)
from word_play.presets.action_policies.llm_action_and_communication import (
    LLM_Action_And_Communication_Policy,
)
from word_play.presets.action_policies.human import Human_Takes_Action
from word_play.presets.action_policies.random_policy import Random_Policy
from word_play.presets.action_validations import Target_Has_Component
from word_play.presets.entity_orderings import randomize_agent_order
from word_play.presets.environments.simple_2d_grid_world import Simple_2D_Grid_World
from word_play.presets.models import (
    LLM_MODEL_REGISTRY,
    OpenRouter_Model,
)
from word_play.presets.movement.simple_2d_grid import (
    Collidable,
    Move_Down,
    Move_Left,
    Move_Right,
    Move_Up,
    Position_2D,
)
from word_play.presets.renderers import (
    Renderable,
    render_step,
)
from word_play.presets.systems.do_nothing import Do_Nothing
from word_play.presets.systems.inventory import Inventory
from word_play.presets.systems.reward import award_reward
from benchmarks.text_meltingpot.common import BENCHMARK_STEPS, normalized_probability, normalized_steps
from word_play.utils import tilemap_to_entities
from word_play.utils.tilemap import find_tile_positions


DEFAULT_NUM_PLAYERS = 2
MAX_EPISODE_LENGTH = BENCHMARK_STEPS
FRAMES_TILL_APPLE_RESPAWN = normalized_steps(50)
FRAMES_TILL_RESPAWN = normalized_steps(100)
FRAMES_TILL_HUNGRY = normalized_steps(200)
FLORA_PROBABILITY = normalized_probability(0.2)
FRUIT_KIND_WEIGHTS = [
    1.0 - FLORA_PROBABILITY,
    FLORA_PROBABILITY * 0.75,
    FLORA_PROBABILITY * 0.05,
    FLORA_PROBABILITY * 0.15,
    FLORA_PROBABILITY * 0.05,
]


ASCII_MAP = """
WWWWWWWWWWWWWWWWWWWW
WFFFFFFFFFFFFFFFFFFW
WFFFFFFFFFFFFFFFFFFW
WFFFFFFFFFFFFFFFFFFW
WFFFFFFFFFFFFFFFFFFW
WFFFFFFFPPPFFFFFFFFW
WFFFFFFFPPPFFFFFFFFW
WFFFFFFFPPPFFFFFFFFW
WFFFFFFFFFFFFFFFFFFW
WFFFFFFFFFFFFFFFFFFW
WFFFFFFFFFFFFFFFFFFW
WFFFFFFFFFFFFFFFFFFW
WWWWWWWWWWWWWWWWWWWW
"""

class FruitPatch(Component):
    def __init__(self):
        super().__init__(tags=["fruit_patch"])
        self.kind: str | None = None
        self.fruit_available = False
        self.respawn_timer = 0

    @property
    def fruit_name(self) -> str | None:
        if not self.kind:
            return None
        if "banana" in self.kind:
            return "Banana"
        if "apple" in self.kind:
            return "Apple"
        return None

    @property
    def is_tree(self) -> bool:
        return bool(self.kind and "tree" in self.kind)

    def post_initialization(self) -> None:
        if self.kind is None:
            self.kind = random.choices(
                ["empty", "apple_tree", "apple_shrub", "banana_tree", "banana_shrub"],
                weights=FRUIT_KIND_WEIGHTS,
                k=1,
            )[0]
            self.fruit_available = self.kind != "empty"
        self._sync_state()

    def _sync_state(self) -> None:
        if self.kind == "empty":
            self.entity.name = "Empty Fruit Patch"
        else:
            fruit = self.fruit_name or "Fruit"
            plant = "Tree" if self.is_tree else "Shrub"
            self.entity.name = f"{fruit} {plant}" + (" with fruit" if self.fruit_available else " without fruit")

    def can_grasp(self, actor: Entity) -> bool:
        if not self.fruit_available or self.kind == "empty":
            return False
        role = actor.get_component(AvatarState).role
        if role == "parent":
            return True
        return not self.is_tree

    def grasp_success_probability(self, actor: Entity) -> float:
        role = actor.get_component(AvatarState).role
        return 1.0 if role == "parent" else 0.3

    def harvest(self) -> str | None:
        fruit_name = self.fruit_name
        if fruit_name is None or not self.fruit_available:
            return None
        self.fruit_available = False
        self.respawn_timer = FRAMES_TILL_APPLE_RESPAWN
        self._sync_state()
        return fruit_name

    def pre_actions_step(self, env: Environment) -> None:
        if self.kind == "empty" or self.fruit_available or self.respawn_timer <= 0:
            return
        self.respawn_timer -= 1
        if self.respawn_timer <= 0:
            self.fruit_available = True
            self._sync_state()


class AvatarState(Component):
    def __init__(self, role: str, spawn_position: Position_2D):
        super().__init__(tags=[role])
        self.role = role
        self.spawn_position = Position_2D(spawn_position.x, spawn_position.y)
        self.hunger_timer = 0
        self.is_hungry = False
        self.active = True
        self.respawn_timer = 0

    def enter_respawn(self, env: Environment) -> None:
        self.active = False
        self.respawn_timer = FRAMES_TILL_RESPAWN
        self.hunger_timer = 0
        self.is_hungry = False
        self.entity.position = Position_2D(self.spawn_position.x, self.spawn_position.y)

        inventory = self.entity.get_component(Inventory)
        if inventory is not None:
            for item in inventory.contents.copy():
                inventory.contents.remove(item)
                if "in_inventory" in item.tags:
                    item.tags.remove("in_inventory")
                if item in env.state.entities:
                    env.destroy_entity(item)

        renderable = self.entity.get_component(Renderable)
        if renderable is not None:
            renderable.visible = False

    def reset_after_eating(self) -> None:
        self.hunger_timer = 0
        self.is_hungry = False

    def pre_actions_step(self, env: Environment) -> None:
        if not self.active:
            if self.respawn_timer > 0:
                self.respawn_timer -= 1
            if self.respawn_timer <= 0:
                self.active = True
                self.entity.position = Position_2D(self.spawn_position.x, self.spawn_position.y)
                renderable = self.entity.get_component(Renderable)
                if renderable is not None:
                    renderable.visible = True
            return

        self.hunger_timer += 1
        self.is_hungry = self.hunger_timer >= FRAMES_TILL_HUNGRY
        if self.is_hungry:
            self.enter_respawn(env)


def actor_can_act(actor: Entity) -> bool:
    state = actor.get_component(AvatarState)
    return state is not None and state.active


def daycare_is_adjacent(actor: Entity, target: Entity, env: Environment) -> bool:
    return abs(actor.position.x - target.position.x) + abs(actor.position.y - target.position.y) <= 1


class Grasp(Action):
    def __init__(self):
        super().__init__()
        self.validation_rules = [
            Target_Not_Self(),
            Target_Is_Nearby(target_is_nearby=daycare_is_adjacent),
            Target_Has_Component(FruitPatch),
        ]

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if not actor_can_act(actor):
            return False
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False

        actor_inventory = actor.get_component(Inventory)
        patch = target.get_component(FruitPatch)
        return (
            actor_inventory is not None
            and patch is not None
            and actor_inventory.has_space()
            and patch.can_grasp(actor)
        )

    def exec_action(self, actor, target, env, kwargs=None):
        actor_inventory = actor.get_component(Inventory)
        assert actor_inventory is not None

        patch = target.get_component(FruitPatch)
        assert patch is not None
        if random.random() > patch.grasp_success_probability(actor):
            return {"success": False, "reason": "grasp_failed"}
        fruit_name = patch.harvest()
        if fruit_name is None:
            return {"success": False, "reason": "no_fruit"}
        fruit = Entity(
            name=fruit_name,
            position=Position_2D(actor.position.x, actor.position.y),
            tags=["fruit", fruit_name.lower()],
            components=[
                Renderable(
                    sprite_path=(
                        "src/items/consumables/lpc_food/banana.png"
                        if fruit_name == "Banana"
                        else "src/items/consumables/lpc_food/apple.png"
                    ),
                    z_index=5,
                )
            ],
        )
        fruit.tags.append("in_inventory")
        env.instantiate_entity(fruit)
        actor_inventory.contents.append(fruit)
        return {"success": True, "grasped": fruit_name, "from": target.name}

    def action_description_text(self, actor, target, env) -> str:
        return f"Grasp fruit from {target.name}."


class Eat_Held_Fruit(Action):
    def __init__(self):
        super().__init__(validation_rules=[Target_Is_Self()])

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if not actor_can_act(actor):
            return False
        inventory = actor.get_component(Inventory)
        return (
            super().is_valid(actor, target, env, kwargs=kwargs)
            and inventory is not None
            and any("fruit" in item.tags for item in inventory.contents)
        )

    def exec_action(self, actor, target, env, kwargs=None):
        inventory = actor.get_component(Inventory)
        item = next(item for item in inventory.contents if "fruit" in item.tags)
        inventory.contents.remove(item)
        if "in_inventory" in item.tags:
            item.tags.remove("in_inventory")
        if item in env.state.entities:
            env.destroy_entity(item)
        Eating.on_consumed(actor, target, env, item, 0.0)
        return {"success": True, "ate": item.name}

    def action_description_text(self, actor, target, env) -> str:
        return "Eat your held fruit."


class Eating(Component):
    def __init__(self):
        super().__init__(actions=[Eat_Held_Fruit()])

    @staticmethod
    def on_consumed(actor: Entity, target: Entity, env: Environment, item: Entity, reward: float) -> None:
        state = actor.get_component(AvatarState)
        if state is None:
            return
        actual_reward = 1.0 if state.role == "parent" else 1.0 if "banana" in item.tags else 0.0
        award_reward(env, actor, actual_reward)
        state.reset_after_eating()


def run_exp(agent_count: int = DEFAULT_NUM_PLAYERS, policy: str = "random", model_name: str = "openai/gpt-4o-mini"):
    assert agent_count == 2, "Daycare is intended for exactly 2 players."
    exp_steps = MAX_EPISODE_LENGTH

    if policy == "llm":
        LLM_MODEL_REGISTRY.unload("daycare")
        LLM_MODEL_REGISTRY.register(
            "daycare",
            OpenRouter_Model,
            model_name=model_name,
            generation_config={"temperature": 0.3},
        )

    entity_tilemap = ASCII_MAP
    entity_tileset = {
        "W": {
            "name": "Wall",
            "tags": ["wall"],
            "components": [
                Collidable(collidable_tags=["wall"]),
                Renderable(sprite_path="", wall_set="src/world_tiles/indoors/wall_sets/bright_brick_wall", z_index=3),
            ],
        },
        "F": {
            "name": "Fruit Patch",
            "components": [
                FruitPatch(),
                Renderable(
                    sprite_path="src/world_tiles/outdoors/nature/tree.png",
                    z_index=1,
                ),
            ],
        },
    }

    entities = tilemap_to_entities(entity_tilemap.replace("P", "."), entity_tileset)
    spawn_positions = find_tile_positions(entity_tilemap, "P")
    roles = ["child", "parent"]

    for agent_id, role in enumerate(roles, start=1):
        x, y = spawn_positions[agent_id - 1]
        agent_policy = (
            LLM_Action_And_Communication_Policy(
                model_key="daycare",
                system_prompt=(
                    f"You are the {role} in Daycare. "
                    "Manage hunger, grasp fruit, eat when holding fruit, and cooperate. "
                    "If you are holding fruit, use Eat your held fruit immediately. "
                    "The child only gets reward from bananas, cannot grasp fruit from trees, and grasps ground fruit "
                    "unreliably. The parent gets reward from apples or bananas and can grasp any fruit reliably."
                ),
                use_chain_of_thought=True,
                observation_memory_window=4,
                conversation_memory_window=8,
            )
            if policy == "llm"
            else Human_Takes_Action() if policy == "human" else Random_Policy()
        )
        entities.append(
            Entity(
                name=f"{role.title()}",
                position=Position_2D(x, y),
                tags=["agent", "player", role],
                actions=[
                    Do_Nothing(),
                    Move_Up(),
                    Move_Down(),
                    Move_Left(),
                    Move_Right(),
                    Grasp(),
                ],
                components=[
                    agent_policy,
                    AvatarState(role=role, spawn_position=Position_2D(x, y)),
                    Eating(),
                    Inventory(accepted_tags=["fruit"], max_size=1),
                    Collidable(collidable_tags=["wall"]),
                    Renderable(
                        sprite_path="src/characters/humanoids/human/farmer_man.png",
                        z_index=10,
                    ),
                ],
            )
        )

    env = Simple_2D_Grid_World(
        description="Daycare, adapted from MeltingPot into a single-file Word Play environment.",
        entities=entities,
        entity_order=randomize_agent_order,
        observation_radius=3,
    )
    env.reward_func = lambda action_selections, current_env: list(current_env.last_step_rewards)

    for step in range(exp_steps):
        if policy != "human" and not render_step(env, step_delay=0.0):
            break

        env.tick = env.cur_step
        env.last_step_rewards = [0.0] * len(env.agents)

        cur_step_actions = []
        for agent_id, agent in enumerate(env.agents):
            if not actor_can_act(agent):
                action = next(
                    selection
                    for selection in env.possible_actions(agent)
                    if isinstance(selection.action, Do_Nothing)
                )
                print(f"[step {step}] {agent.name} -> {action}")
                cur_step_actions.append(action)
                continue
            observation = env.observe(agent_id)
            action, info = agent.get_component(Agent_Policy).select_action(observation)
            print(f"[step {step}] {agent.name} -> {action}")
            cur_step_actions.append(action)

        env.step(cur_step_actions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-count", type=int, default=DEFAULT_NUM_PLAYERS)
    parser.add_argument("--policy", choices=["random", "llm", "human"], default="random")
    parser.add_argument("--model-name", default="openai/gpt-4o-mini")
    args = parser.parse_args()
    run_exp(
        agent_count=args.agent_count,
        policy=args.policy,
        model_name=args.model_name,
    )
