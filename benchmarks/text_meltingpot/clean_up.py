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
)
from word_play.presets.action_policies.llm_action_and_communication import (
    LLM_Action_And_Communication_Policy,
)
from word_play.presets.action_policies.human import Human_Takes_Action
from word_play.presets.action_policies.random_policy import Random_Policy
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
from word_play.presets.systems.cooldown import Cooldown
from word_play.presets.systems.do_nothing import Do_Nothing
from word_play.presets.systems.freezable import Freezable
from word_play.presets.systems.reward import award_reward
from benchmarks.text_meltingpot.common import (
    BENCHMARK_STEPS,
    Commons_Zap,
    normalized_probability,
    normalized_steps,
)
from word_play.utils import tilemap_to_entities
from word_play.utils.tilemap import find_tile_positions


DEFAULT_NUM_PLAYERS = 7
MAX_EPISODE_LENGTH = BENCHMARK_STEPS
DIRT_SPAWN_PROBABILITY = 0.5
DIRT_SPAWN_DELAY = normalized_steps(50)
MIN_EPISODE_LENGTH = normalized_steps(1000)
INTERVAL_LENGTH = normalized_steps(100)
TERMINATION_PROBABILITY = 0.2
MAX_APPLE_GROWTH_RATE = normalized_probability(0.05)
THRESHOLD_DEPLETION = 0.4
THRESHOLD_RESTORATION = 0.0
ZAP_DURATION = normalized_steps(50)


class ApplePatch(Component):
    def __init__(self):
        super().__init__(tags=["apple"])
        self.consumed = False
        self.regrow_prob = MAX_APPLE_GROWTH_RATE

    def consume(self) -> bool:
        if self.consumed:
            return False
        self.consumed = True
        renderable = self.entity.get_component(Renderable)
        if renderable is not None:
            renderable.visible = False
        return True

    def pre_actions_step(self, env: Environment) -> None:
        if self.consumed and random.random() < self.regrow_prob:
            self.consumed = False
            renderable = self.entity.get_component(Renderable)
            if renderable is not None:
                renderable.visible = True


class Clean_Dirt(Action):
    def __init__(self):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(),
            ]
        )

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        return super().is_valid(actor, target, env, kwargs=kwargs) and "dirt_patch" in target.tags

    def exec_action(self, actor, target, env, kwargs=None):
        target.tags.remove("dirt_patch")
        target.name = "River Tile"
        renderable = target.get_component(Renderable)
        if renderable is not None:
            renderable.sprite_path = "src/world_tiles/indoors/floors/shallow_water_tile.png"
        return {"cleaned": target.name}

    def action_description_text(self, actor, target, env) -> str:
        return f"Clean {target.name}."


class PollutionManager(Component):
    def __init__(self, river_positions: list[tuple[int, int]]):
        super().__init__()
        self.river_positions = river_positions

    def pre_actions_step(self, env: Environment) -> None:
        env.tick = env.cur_step

    def post_actions_step(self, env: Environment) -> None:
        self._collect_apples(env)
        self._update_apple_regrowth(env)
        self._spawn_dirt(env)
        self._maybe_end_episode(env)

    def _collect_apples(self, env: Environment) -> None:
        apples = [entity for entity in env.state.entities if entity.get_component(ApplePatch) is not None]
        for agent in env.agents:
            for apple in apples:
                if apple.position != agent.position:
                    continue
                patch = apple.get_component(ApplePatch)
                if patch.consume():
                    award_reward(env, agent, 1.0)

    def _update_apple_regrowth(self, env: Environment) -> None:
        pollution = self.current_pollution_fraction(env)
        if pollution <= THRESHOLD_RESTORATION:
            regrow_prob = MAX_APPLE_GROWTH_RATE
        elif pollution >= THRESHOLD_DEPLETION:
            regrow_prob = 0.0
        else:
            remaining = 1.0 - (pollution - THRESHOLD_RESTORATION) / (THRESHOLD_DEPLETION - THRESHOLD_RESTORATION)
            regrow_prob = MAX_APPLE_GROWTH_RATE * max(0.0, remaining)

        for entity in env.state.entities:
            patch = entity.get_component(ApplePatch)
            if patch is not None:
                patch.regrow_prob = regrow_prob

    def _spawn_dirt(self, env: Environment) -> None:
        if env.cur_step < DIRT_SPAWN_DELAY or random.random() >= DIRT_SPAWN_PROBABILITY:
            return

        occupied = {
            (entity.position.x, entity.position.y)
            for entity in env.state.entities
            if "dirt_patch" in entity.tags
        }
        available = [position for position in self.river_positions if position not in occupied]
        if not available:
            return

        x, y = random.choice(available)
        env.instantiate_entity(
            Entity(
                name="Dirty River Tile",
                position=Position_2D(x, y),
                tags=["wall", "water", "dirt_patch"],
                components=[
                    Collidable(collidable_tags=["wall"]),
                    Renderable(
                        sprite_path="src/world_tiles/water/polluted_water.png",
                        z_index=1,
                    ),
                ],
            )
        )

    def _maybe_end_episode(self, env: Environment) -> None:
        next_step = env.cur_step + 1
        if next_step >= MAX_EPISODE_LENGTH:
            for idx in range(len(env.agents)):
                env.truncations[idx] = True
            return
        if next_step < MIN_EPISODE_LENGTH:
            return
        if next_step % INTERVAL_LENGTH == 0 and random.random() < TERMINATION_PROBABILITY:
            for idx in range(len(env.agents)):
                env.truncations[idx] = True

    def current_pollution_fraction(self, env: Environment) -> float:
        dirt_count = sum(1 for entity in env.state.entities if "dirt_patch" in entity.tags)
        if not self.river_positions:
            return 0.0
        return dirt_count / len(self.river_positions)


def run_exp(agent_count: int = DEFAULT_NUM_PLAYERS, policy: str = "random", model_name: str = "openai/gpt-4o-mini"):
    exp_steps = MAX_EPISODE_LENGTH

    if policy == "llm":
        LLM_MODEL_REGISTRY.unload("clean_up")
        LLM_MODEL_REGISTRY.register(
            "clean_up",
            OpenRouter_Model,
            model_name=model_name,
            generation_config={"temperature": 0.3},
        )

    entity_tilemap = """
    WWWWWWWWWWWWWWWWWWWWWWWWW
    W...a.a.a.a.a.a.a.a.....W
    W..a.a.a.a.a.a.a.a.a....W
    W.......................W
    W.P.P.P.P.P.P.P.........W
    W......HHHHFHHHHH.......W
    W......HHHHHHHHHH.......W
    W......HHFHHHHHHH.......W
    W.......................W
    W..a.a.a.a.a.a.a.a.a....W
    W...a.a.a.a.a.a.a.a.....W
    WWWWWWWWWWWWWWWWWWWWWWWWW
    """
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
        "H": {
            "name": "River Tile",
            "tags": ["wall", "water"],
            "components": [
                Collidable(collidable_tags=["wall"]),
                Renderable(
                    sprite_path="src/world_tiles/indoors/floors/shallow_water_tile.png",
                    z_index=1,
                ),
            ],
        },
        "F": {
            "name": "Dirty River Tile",
            "tags": ["wall", "water", "dirt_patch"],
            "components": [
                Collidable(collidable_tags=["wall"]),
                Renderable(
                    sprite_path="src/world_tiles/water/polluted_water.png",
                    z_index=1,
                ),
            ],
        },
        "a": {
            "name": "Apple",
            "components": [
                ApplePatch(),
                Renderable(
                    sprite_path="src/items/consumables/lpc_food/apple.png",
                    z_index=4,
                ),
            ],
        },
    }

    entities = tilemap_to_entities(entity_tilemap.replace("P", "."), entity_tileset)
    spawn_positions = find_tile_positions(entity_tilemap, "P")
    river_positions = find_tile_positions(entity_tilemap, {"H", "F"})

    entities.append(
        Entity(
            name="Pollution Manager",
            position=Position_2D(-1, -1),
            components=[PollutionManager(river_positions=river_positions)],
        )
    )

    for agent_id, (x, y) in enumerate(spawn_positions[:agent_count], start=1):
        agent_policy = (
            LLM_Action_And_Communication_Policy(
                model_key="clean_up",
                system_prompt=(
                    f"You are Player {agent_id} in Clean Up. "
                    "Collect apples, keep the river clean, and balance short-term reward with public-good maintenance."
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
                name=f"Player {agent_id}",
                position=Position_2D(x, y),
                tags=["agent", "player", "default"],
                actions=[
                    Do_Nothing(),
                    Move_Up(),
                    Move_Down(),
                    Move_Left(),
                    Move_Right(),
                    Commons_Zap(),
                    Clean_Dirt(),
                ],
                components=[
                    agent_policy,
                    Cooldown(cooldowns={"zap": normalized_steps(10)}),
                    Freezable(default_duration=ZAP_DURATION),
                    Collidable(collidable_tags=["wall"]),
                    Renderable(
                        sprite_path="src/characters/humanoids/human/farmer_man.png",
                        z_index=10,
                    ),
                ],
            )
        )

    env = Simple_2D_Grid_World(
        description="Clean Up, adapted from MeltingPot into a single-file Word Play environment.",
        entities=entities,
        entity_order=randomize_agent_order,
        observation_radius=4,
    )

    for step in range(exp_steps):
        if policy != "human" and not render_step(env, step_delay=0.0):
            break

        env.last_step_rewards = [0.0] * len(env.agents)
        env.reward_func = lambda action_selections, current_env: list(current_env.last_step_rewards)

        pollution = next(
            entity.get_component(PollutionManager).current_pollution_fraction(env)
            for entity in env.state.entities
            if entity.name == "Pollution Manager"
        )
        cur_step_actions = []
        for agent_id, agent in enumerate(env.agents):
            observation = env.observe(agent_id)
            action, info = agent.get_component(Agent_Policy).select_action(observation)
            print(f"[step {step}] {agent.name} -> {action} | pollution={pollution:.2f}")
            cur_step_actions.append(action)

        env.step(cur_step_actions)
        if all(env.truncations):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-count", type=int, default=DEFAULT_NUM_PLAYERS)
    parser.add_argument("--policy", choices=["random", "llm", "human"], default="random")
    parser.add_argument("--model-name", default="openai/gpt-4o-mini")
    args = parser.parse_args()
    run_exp(agent_count=args.agent_count, policy=args.policy, model_name=args.model_name)
