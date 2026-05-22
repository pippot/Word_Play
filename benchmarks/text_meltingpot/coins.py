from __future__ import annotations

import argparse
import random

from word_play.core import (
    Agent_Policy,
    Component,
    Entity,
    Environment,
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
from word_play.presets.systems.do_nothing import Do_Nothing
from word_play.presets.systems.preferences import Preference
from word_play.presets.systems.reward import award_reward
from benchmarks.text_meltingpot.common import BENCHMARK_STEPS, normalized_probability, normalized_steps
from word_play.utils import tilemap_to_entities
from word_play.utils.tilemap import find_tile_positions


MANDATED_NUM_PLAYERS = 2
MAX_EPISODE_LENGTH = BENCHMARK_STEPS
MIN_EPISODE_LENGTH = normalized_steps(300)
INTERVAL_LENGTH = normalized_steps(100)
TERMINATION_PROBABILITY = 0.05
COIN_REGROW_TICK_INTERVAL = 1
COIN_REGROW_PROBABILITY = normalized_probability(0.0005)


class CoinPatch(Component):
    def __init__(self):
        super().__init__(tags=["coin_patch"])
        self.consumed = False

    def pre_actions_step(self, env: Environment) -> None:
        if not self.consumed:
            return
        if env.cur_step % COIN_REGROW_TICK_INTERVAL == 0 and random.random() < COIN_REGROW_PROBABILITY:
            self.consumed = False
            renderable = self.entity.get_component(Renderable)
            if renderable is not None:
                renderable.visible = True

    def post_actions_step(self, env: Environment) -> None:
        if self.consumed:
            return

        for agent in env.agents:
            if agent.position == self.entity.position:
                self.consumed = True
                renderable = self.entity.get_component(Renderable)
                if renderable is not None:
                    renderable.visible = False
                preference = agent.get_component(Preference)
                reward = preference.reward_for(self.entity) if preference is not None else 1.0
                award_reward(env, agent, reward)
                if preference is not None and not preference.is_preferred(self.entity):
                    for idx, other in enumerate(env.agents):
                        if other is not agent:
                            env.last_step_rewards[idx] -= abs(preference.mismatch_penalty)
                break


def run_exp(agent_count: int = MANDATED_NUM_PLAYERS, policy: str = "random", model_name: str = "openai/gpt-4o-mini"):
    assert agent_count == MANDATED_NUM_PLAYERS, "Coins requires exactly 2 players."
    exp_steps = MAX_EPISODE_LENGTH

    if policy == "llm":
        LLM_MODEL_REGISTRY.unload("coins")
        LLM_MODEL_REGISTRY.register(
            "coins",
            OpenRouter_Model,
            model_name=model_name,
            generation_config={"temperature": 0.3},
        )

    entity_tilemap = """
    WWWWWWWWWWWWWWWWW
    W..r...b...r...PW
    W...............W
    W...b.....r.....W
    W...............W
    WP..r...b...r...W
    W...............W
    W...b.....r.....W
    W...............W
    WWWWWWWWWWWWWWWWW
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
        "r": {
            "name": "Red Coin",
            "tags": ["coin", "red_coin"],
            "components": [
                CoinPatch(),
                Renderable(
                    sprite_path="src/items/currency/coin_red.png",
                    z_index=4,
                ),
            ],
        },
        "b": {
            "name": "Blue Coin",
            "tags": ["coin", "blue_coin"],
            "components": [
                CoinPatch(),
                Renderable(
                    sprite_path="src/items/currency/coin_blue.png",
                    z_index=4,
                ),
            ],
        },
    }

    entities = tilemap_to_entities(entity_tilemap.replace("P", "."), entity_tileset)
    spawn_positions = find_tile_positions(entity_tilemap, "P")

    coin_preferences = [
        ("red_coin", "red"),
        ("blue_coin", "blue"),
    ]

    for agent_id, (x, y) in enumerate(spawn_positions[:agent_count], start=1):
        preferred_coin_tag, preferred_coin_name = coin_preferences[agent_id - 1]
        agent_policy = (
            LLM_Action_And_Communication_Policy(
                model_key="coins",
                system_prompt=(
                    f"You are Player {agent_id} in Coins. "
                    f"Your own coin color is {preferred_coin_name}. "
                    "Move around and collect coins. Your own coin is good for you, "
                    "but taking the other player's coin hurts them."
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
                tags=["agent", "player", "default", preferred_coin_tag],
                actions=[
                    Do_Nothing(),
                    Move_Up(),
                    Move_Down(),
                    Move_Left(),
                    Move_Right(),
                ],
                components=[
                    agent_policy,
                    Preference(
                        reward_map={preferred_coin_tag: 1.0},
                        default_reward=1.0,
                        mismatch_penalty=-2.0,
                    ),
                    Collidable(collidable_tags=["wall"]),
                    Renderable(
                        sprite_path="src/characters/humanoids/human/farmer_man.png",
                        z_index=10,
                    ),
                ],
            )
        )

    env = Simple_2D_Grid_World(
        description="Coins, adapted from MeltingPot into a single-file Word Play environment.",
        entities=entities,
        entity_order=randomize_agent_order,
        observation_radius=4,
    )
    env.reward_func = lambda action_selections, current_env: list(current_env.last_step_rewards)

    for step in range(exp_steps):
        env.tick = env.cur_step
        if policy != "human" and not render_step(env, step_delay=0.0):
            break

        env.last_step_rewards = [0.0] * len(env.agents)

        cur_step_actions = []
        for agent_id, agent in enumerate(env.agents):
            observation = env.observe(agent_id)
            action, info = agent.get_component(Agent_Policy).select_action(observation)
            print(f"[step {step}] {agent.name} -> {action}")
            cur_step_actions.append(action)

        env.step(cur_step_actions)

        next_step = env.cur_step
        if next_step >= MAX_EPISODE_LENGTH:
            env.truncations = [True] * len(env.agents)
        elif next_step >= MIN_EPISODE_LENGTH and next_step % INTERVAL_LENGTH == 0:
            if random.random() < TERMINATION_PROBABILITY:
                env.truncations = [True] * len(env.agents)

        if all(env.truncations):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-count", type=int, default=MANDATED_NUM_PLAYERS)
    parser.add_argument("--policy", choices=["random", "llm", "human"], default="random")
    parser.add_argument("--model-name", default="openai/gpt-4o-mini")
    args = parser.parse_args()
    run_exp(agent_count=args.agent_count, policy=args.policy, model_name=args.model_name)
