from __future__ import annotations

import random

from word_play.core import Action, Agent_Policy, Component, Entity, Environment, Target_Is_Nearby, Target_Not_Self
from word_play.presets.action_policies.llm_action_and_communication import LLM_Action_And_Communication_Policy
from word_play.presets.action_policies.human import Human_Takes_Action
from word_play.presets.action_policies.random_policy import Random_Policy
from word_play.presets.action_validations import Target_Has_Tag
from word_play.presets.entity_orderings import randomize_agent_order
from word_play.presets.environments.simple_2d_grid_world import Simple_2D_Grid_World
from word_play.presets.models import LLM_MODEL_REGISTRY, OpenRouter_Model
from word_play.presets.movement.simple_2d_grid import (
    Collidable,
    Move_Down,
    Move_Left,
    Move_Right,
    Move_Up,
    Position_2D,
)
from word_play.presets.renderers import Renderable, render_step
from word_play.presets.systems.cooldown import Action_On_Cooldown, Cooldown
from word_play.presets.systems.do_nothing import Do_Nothing
from word_play.presets.systems.freezable import Freezable
from word_play.presets.systems.respawnable import Respawnable
from word_play.presets.systems.reward import award_reward
from benchmarks.text_meltingpot.common import BENCHMARK_STEPS, normalized_steps
from word_play.utils import tilemap_to_entities
from word_play.utils.tilemap import find_tile_positions


MAX_EPISODE_LENGTH = BENCHMARK_STEPS
DEFAULT_NUM_PLAYERS = 5
MIN_EPISODE_LENGTH = normalized_steps(1000)
INTERVAL_LENGTH = normalized_steps(100)
TERMINATION_PROBABILITY = 0.2
MUSHROOM_SPOIL_STEPS = {
    "red": normalized_steps(200),
    "green": normalized_steps(100),
    "blue": normalized_steps(75),
    "orange": normalized_steps(10_000_000),
}
MUSHROOM_DIGESTION_STEPS = {
    "green": normalized_steps(10),
    "blue": normalized_steps(15),
    "orange": normalized_steps(15),
}


def configure_externality_mushrooms(env: Environment, potential_positions: list[tuple[int, int]]) -> None:
    env.externality_mushroom_potential_positions = potential_positions
    env.mushroom_events = []
    if getattr(env, "_externality_mushrooms_configured", False):
        return

    env._externality_mushrooms_configured = True
    original_start_of_step = env.environment_start_of_step
    original_end_of_step = env.environment_end_of_step

    def environment_start_of_step(action_selections):
        original_start_of_step(action_selections)
        env.tick = env.cur_step
        env.mushroom_events = []

    def environment_end_of_step(action_selections):
        original_end_of_step(action_selections)
        next_step = env.cur_step + 1
        if next_step >= MAX_EPISODE_LENGTH or (
            next_step >= MIN_EPISODE_LENGTH
            and next_step % INTERVAL_LENGTH == 0
            and random.random() < TERMINATION_PROBABILITY
        ):
            for idx in range(len(env.agents)):
                env.truncations[idx] = True

    env.environment_start_of_step = environment_start_of_step
    env.environment_end_of_step = environment_end_of_step


class Externality_Zap_State(Respawnable):
    def __init__(self, recovery_time: int = normalized_steps(50)):
        super().__init__(inactive_tag="removed")
        self.mark_level = 0
        self.recovery_time = recovery_time
        self.recovery_ticks = 0

    def hit(self, freeze_duration: int, remove_duration: int) -> dict:
        freezable = self.entity.get_component(Freezable)
        if self.mark_level == 0:
            self.mark_level = 1
            self.recovery_ticks = self.recovery_time
            if freezable is not None:
                freezable.freeze(freeze_duration)
            return {"level": 1, "effect": "freeze", "duration": freeze_duration}

        self.mark_level = 0
        self.recovery_ticks = 0
        self.remove_temporarily(remove_duration)
        if freezable is not None:
            freezable.freeze(remove_duration)
        return {"level": 2, "effect": "removed", "duration": remove_duration}

    def post_actions_step(self, env: Environment) -> None:
        if self.recovery_ticks > 0:
            self.recovery_ticks -= 1
            if self.recovery_ticks == 0:
                self.mark_level = 0


class Externality_Zap(Action):
    def __init__(
        self,
        freeze_duration: int = normalized_steps(25),
        remove_duration: int = normalized_steps(50),
    ):
        super().__init__(
            validation_rules=[
                Action_On_Cooldown("zap"),
                Target_Is_Nearby(),
                Target_Not_Self(),
                Target_Has_Tag(["player"]),
            ]
        )
        self.freeze_duration = freeze_duration
        self.remove_duration = remove_duration

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        return (
            super().is_valid(actor, target, env, kwargs=kwargs)
            and target.get_component(Externality_Zap_State) is not None
        )

    def exec_action(self, actor, target, env, kwargs=None):
        cooldown = actor.get_component(Cooldown)
        if cooldown is not None:
            cooldown.start("zap")
        state = target.get_component(Externality_Zap_State)
        return {
            "success": True,
            "zapped": target.name,
            **state.hit(self.freeze_duration, self.remove_duration),
        }

    def action_description_text(self, actor, target, env):
        return f"Zap {target.name}."


class Externality_Mushroom(Component):
    def __init__(self, kind: str):
        super().__init__(tags=["mushroom", kind])
        self.kind = kind
        self.age = 0

    def post_actions_step(self, env: Environment) -> None:
        if self.entity not in env.state.entities:
            return

        eater = next(
            (
                agent
                for agent in env.agents
                if agent.position == self.entity.position and not self._is_removed(agent)
            ),
            None,
        )
        if eater is not None:
            self._eat(env, eater)
            return

        self.age += 1
        if self.age >= MUSHROOM_SPOIL_STEPS[self.kind]:
            kind = self.kind
            env.destroy_entity(self.entity)
            env.mushroom_events.append(f"A {kind} mushroom spoiled.")

    def _eat(self, env: Environment, eater: Entity) -> None:
        kind = self.kind
        if kind == "red":
            award_reward(env, eater, 1.0)
            env.mushroom_events.append(f"{eater.name} ate a red mushroom for +1.")
        elif kind == "green":
            reward = 2.0 / len(env.agents)
            for agent in env.agents:
                award_reward(env, agent, reward)
            self._freeze(eater, MUSHROOM_DIGESTION_STEPS["green"])
            env.mushroom_events.append(f"{eater.name} ate a green mushroom; everyone got +{reward:g}.")
        elif kind == "blue":
            others = [agent for agent in env.agents if agent is not eater]
            reward = 3.0 / len(others) if others else 0.0
            for agent in others:
                award_reward(env, agent, reward)
            self._freeze(eater, MUSHROOM_DIGESTION_STEPS["blue"])
            env.mushroom_events.append(f"{eater.name} ate a blue mushroom; others got +{reward:g}.")
        elif kind == "orange":
            award_reward(env, eater, -1.0)
            self._freeze(eater, MUSHROOM_DIGESTION_STEPS["orange"])
            destroyed = self._destroy_red_mushrooms(env)
            env.mushroom_events.append(f"{eater.name} ate an orange mushroom for -1; destroyed {destroyed} red mushrooms.")

        env.destroy_entity(self.entity)
        self._regrow_after_eating(env, kind)

    def _regrow_after_eating(self, env: Environment, eaten_kind: str) -> None:
        if eaten_kind in {"red", "green", "blue"} and random.random() < 0.25:
            self._spawn_mushroom(env, "red")
        if eaten_kind in {"green", "blue"} and random.random() < 0.40:
            self._spawn_mushroom(env, "green")
        if eaten_kind == "blue" and random.random() < 0.60:
            self._spawn_mushroom(env, "blue")
        if eaten_kind == "orange":
            self._spawn_mushroom(env, "orange")

    def _spawn_mushroom(self, env: Environment, kind: str) -> None:
        occupied = {
            (entity.position.x, entity.position.y)
            for entity in env.state.entities
            if entity.get_component(Externality_Mushroom) is not None or entity in env.agents
        }
        open_positions = [
            position
            for position in getattr(env, "externality_mushroom_potential_positions", [])
            if position not in occupied
        ]
        if not open_positions:
            return

        x, y = random.choice(open_positions)
        env.instantiate_entity(
            Entity(
                name=f"{kind.title()} Mushroom",
                position=Position_2D(x, y),
                components=[
                    Externality_Mushroom(kind),
                    Renderable(
                        sprite_path=(
                            "src/items/consumables/vegetables/mushroom_red.png"
                            if kind == "red"
                            else "src/items/consumables/vegetables/mushroom_green.png"
                            if kind == "green"
                            else "src/items/consumables/vegetables/mushroom_blue.png"
                            if kind == "blue"
                            else "src/items/consumables/vegetables/mushroom_1.png"
                        ),
                        z_index=4,
                    ),
                ],
            )
        )
        env.mushroom_events.append(f"A {kind} mushroom regrew.")

    def _destroy_red_mushrooms(self, env: Environment) -> int:
        red_mushrooms = [
            entity
            for entity in env.state.entities
            if (mushroom := entity.get_component(Externality_Mushroom)) is not None and mushroom.kind == "red"
        ]
        destroy_count = max(1, int(len(red_mushrooms) * 0.25)) if red_mushrooms else 0
        for mushroom in random.sample(red_mushrooms, destroy_count):
            env.destroy_entity(mushroom)
        return destroy_count

    def _freeze(self, entity: Entity, duration: int) -> None:
        freezable = entity.get_component(Freezable)
        if freezable is not None:
            freezable.freeze(duration)

    def _is_removed(self, entity: Entity) -> bool:
        zap_state = entity.get_component(Externality_Zap_State)
        return zap_state is not None and not zap_state.active


def run_exp(
    agent_count: int = DEFAULT_NUM_PLAYERS,
    policy: str = "random",
    model_name: str = "openai/gpt-4o-mini",
) -> None:
    entity_tilemap = """
    WWWWWWWWWWWWWWWWWWWWWWW
    W....R....WWW.....G...W
    W.........W.W.........W
    W..WWW....W.W....WWW..W
    W..W......W.W......W..W
    W..W..B.......O....W..W
    W..W......WWW......W..W
    W..WWW....W.W....WWW..W
    W.........W.W.........W
    W...G.....WWW....R....W
    W.....................W
    WWWWWWWWWWWWWWWWWWWWWWW
    """

    if policy == "llm":
        LLM_MODEL_REGISTRY.unload("externality_mushrooms")
        LLM_MODEL_REGISTRY.register(
            "externality_mushrooms",
            OpenRouter_Model,
            model_name=model_name,
            generation_config={"temperature": 0.2},
        )

    entity_tileset = {
        "W": {
            "name": "Wall",
            "tags": ["wall"],
            "components": [
                Collidable(collidable_tags=["wall"]),
                Renderable(
                    sprite_path="",
                    wall_set="src/world_tiles/indoors/wall_sets/bright_brick_wall",
                    z_index=3,
                ),
            ],
        },
        "R": {
            "name": "Red Mushroom",
            "components": [
                Externality_Mushroom("red"),
                Renderable(sprite_path="src/items/consumables/vegetables/mushroom_red.png", z_index=4),
            ],
        },
        "G": {
            "name": "Green Mushroom",
            "components": [
                Externality_Mushroom("green"),
                Renderable(sprite_path="src/items/consumables/vegetables/mushroom_green.png", z_index=4),
            ],
        },
        "B": {
            "name": "Blue Mushroom",
            "components": [
                Externality_Mushroom("blue"),
                Renderable(sprite_path="src/items/consumables/vegetables/mushroom_blue.png", z_index=4),
            ],
        },
        "O": {
            "name": "Orange Mushroom",
            "components": [
                Externality_Mushroom("orange"),
                Renderable(sprite_path="src/items/consumables/vegetables/mushroom_1.png", z_index=4),
            ],
        },
    }

    entities = tilemap_to_entities(entity_tilemap, entity_tileset)
    spawn_positions = find_tile_positions(entity_tilemap, ".")
    if agent_count > len(spawn_positions):
        raise ValueError(f"Map only has {len(spawn_positions)} spawn positions.")

    random.shuffle(spawn_positions)
    for agent_id, (x, y) in enumerate(spawn_positions[:agent_count], start=1):
        agent_policy = (
            LLM_Action_And_Communication_Policy(
                model_key="externality_mushrooms",
                system_prompt=(
                    f"You are Player {agent_id} in Externality Mushrooms. "
                    f"Red mushrooms give you +1. Green mushrooms split +2 across everyone and freeze you for "
                    f"{MUSHROOM_DIGESTION_STEPS['green']} turn(s). Blue mushrooms split +3 across everyone except you "
                    f"and freeze you for {MUSHROOM_DIGESTION_STEPS['blue']} turn(s). Orange mushrooms give you -1, "
                    f"freeze you for {MUSHROOM_DIGESTION_STEPS['orange']} turn(s), and destroy red mushrooms. Shape mushroom regrowth by choosing "
                    "what to eat and when to zap nearby players."
                ),
                use_chain_of_thought=True,
                observation_memory_window=4,
                conversation_memory_window=4,
            )
            if policy == "llm"
            else Human_Takes_Action() if policy == "human" else Random_Policy()
        )
        entities.append(
            Entity(
                name=f"Player {agent_id}",
                position=Position_2D(x, y),
                tags=["agent", "player", "default"],
                actions=[
                    Do_Nothing(),
                    Move_Up(),
                    Move_Down(),
                    Move_Left(),
                    Move_Right(),
                    Externality_Zap(),
                ],
                components=[
                    agent_policy,
                    Cooldown(cooldowns={"zap": normalized_steps(2)}),
                    Externality_Zap_State(),
                    Freezable(default_duration=normalized_steps(25)),
                    Collidable(collidable_tags=["wall"]),
                    Renderable(sprite_path="src/characters/humanoids/human/farmer_man.png", z_index=10),
                ],
            )
        )

    env = Simple_2D_Grid_World(
        description="Externality Mushrooms",
        entities=entities,
        entity_order=randomize_agent_order,
        observation_radius=4,
    )
    configure_externality_mushrooms(env, spawn_positions.copy())

    for step in range(MAX_EPISODE_LENGTH):
        if policy != "human" and not render_step(env, step_delay=0.0):
            break

        env.last_step_rewards = [0.0] * len(env.agents)
        env.reward_func = lambda action_selections, current_env: list(current_env.last_step_rewards)

        cur_step_actions = []
        for agent_id, agent in enumerate(env.agents):
            observation = env.observe(agent_id)
            action, info = agent.get_component(Agent_Policy).select_action(observation)
            info = info or {}
            if info.get("raw_response"):
                print(f"[step {step}] {agent.name} raw -> {info['raw_response'].strip()}")
            print(f"[step {step}] {agent.name} -> {action}")
            cur_step_actions.append(action)

        env.step(cur_step_actions)
        for event in getattr(env, "mushroom_events", []):
            print(f"[step {step}] mushrooms -> {event}")

        if all(env.terminations) or all(env.truncations):
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-count", type=int, default=DEFAULT_NUM_PLAYERS)
    parser.add_argument("--policy", choices=["random", "llm", "human"], default="random")
    parser.add_argument("--model-name", default="openai/gpt-4o-mini")
    args = parser.parse_args()
    run_exp(agent_count=args.agent_count, policy=args.policy, model_name=args.model_name)
