from __future__ import annotations

import argparse
import random

from benchmarks.text_meltingpot.fruit_market import (
    CancelTradeOffer,
    DEFAULT_NUM_PLAYERS,
    EatFruit,
    FruitInventory,
    FruitMarketAvatar,
    FruitMarketManager,
    FruitTree,
    MAX_EPISODE_LENGTH,
    MAX_OFFER_QUANTITY,
    SetTradeOffer,
    ShovePlayer,
    WaterCost,
)
from word_play.core import Agent_Policy, Entity
from word_play.presets.action_policies.human import Human_Takes_Action
from word_play.presets.action_policies.llm_action_and_communication import LLM_Action_And_Communication_Policy
from word_play.presets.action_policies.random_policy import Random_Policy
from word_play.presets.entity_orderings import randomize_agent_order
from word_play.presets.environments.simple_2d_grid_world import Simple_2D_Grid_World
from word_play.presets.models import LLM_MODEL_REGISTRY, OpenRouter_Model
from word_play.presets.movement.simple_2d_grid import Collidable, Move_Down, Move_Left, Move_Right, Move_Up, Position_2D
from word_play.presets.renderers import Renderable, render_step
from word_play.presets.systems.do_nothing import Do_Nothing
from word_play.utils import tilemap_to_entities
from word_play.utils.tilemap import find_tile_positions


MODEL_KEY = "fruit_market__concentric_rivers"


def run_exp(agent_count: int = DEFAULT_NUM_PLAYERS, policy: str = "random", model_name: str = "openai/gpt-4o-mini"):
    entity_tilemap = """
    WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
    W.............................W
    WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
    W.tttttttttttttttttttttttttttW
    W.tttttttttttttttttttttttttttW
    W.tttRRRRRRRRRRRRRRRRRRRRRtttW
    W.tttRtttttttttttttttttttRtttW
    W.tttRtttttttttttttttttttRtttW
    W.tttRttRRRRRRRRRRRRRRRttRtttW
    W.tttRttRtttttttttttttRttRtttW
    W.tttRttRtttttttttttttRttRtttW
    W.tttRttRttRRRRRRRRRttRttRtttW
    W.tttRttRttRPtPtPtPRttRttRtttW
    W.tttRttRttRtPtPtPtRttRttRtttW
    W.tttRttRttRttPtPttRttRttRtttW
    W.tttRttRttRtPtPtPtRttRttRtttW
    W.tttRttRttRttPtPttRttRttRtttW
    W.tttRttRttRtPtPtPtRttRttRtttW
    W.tttRttRttRPtPtPtPRttRttRtttW
    W.tttRttRttRRRRRRRRRttRttRtttW
    W.tttRttRtttttttttttttRttRtttW
    W.tttRttRtttttttttttttRttRtttW
    W.tttRttRRRRRRRRRRRRRRRttRtttW
    W.tttRtttttttttttttttttttRtttW
    W.tttRtttttttttttttttttttRtttW
    W.tttRRRRRRRRRRRRRRRRRRRRRtttW
    W.tttttttttttttttttttttttttttW
    W.tttttttttttttttttttttttttttW
    W.tttttttttttttttttttttttttttW
    W.............................W
    WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
    """

    if policy == "llm":
        LLM_MODEL_REGISTRY.unload(MODEL_KEY)
        LLM_MODEL_REGISTRY.register(
            MODEL_KEY,
            OpenRouter_Model,
            model_name=model_name,
            generation_config={"temperature": 0.3},
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
        "t": {
            "name": "Potential Fruit Tree",
            "tags": ["tree_site"],
            "components": [
                FruitTree(),
                Renderable(
                    sprite_path="src/world_tiles/outdoors/nature/ripe_fruit_tree.png",
                    z_index=3,
                ),
            ],
        },
        "R": {
            "name": "River",
            "components": [
                WaterCost(),
                Renderable(
                    sprite_path="src/world_tiles/indoors/floors/shallow_water_tile.png",
                    z_index=1,
                ),
            ],
        },
    }

    entities = tilemap_to_entities(entity_tilemap.replace("P", "."), entity_tileset)
    spawn_positions = find_tile_positions(entity_tilemap, "P")
    random.shuffle(spawn_positions)

    entities.append(
        Entity(
            name="Fruit Market Manager",
            position=Position_2D(-1, -1),
            components=[FruitMarketManager()],
        )
    )

    for agent_id, (x, y) in enumerate(spawn_positions[:agent_count], start=1):
        specialty = "apple" if agent_id % 2 == 0 else "banana"
        agent_policy = (
            LLM_Action_And_Communication_Policy(
                model_key=MODEL_KEY,
                system_prompt=(
                    f"You are Player {agent_id} in Fruit Market: Concentric Rivers. "
                    f"You are a {specialty} farmer: you harvest {specialty}s reliably, "
                    f"but your favorite food is {'banana' if specialty == 'apple' else 'apple'} worth 8 reward. "
                    "Harvest fruit by standing on trees, eat to reset hunger, and barter with nearby players."
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
                    EatFruit("apple"),
                    EatFruit("banana"),
                    CancelTradeOffer(),
                    ShovePlayer(1),
                    ShovePlayer(-1),
                    *[
                        SetTradeOffer(give_fruit, give_amount, want_amount)
                        for give_fruit in ("apple", "banana")
                        for give_amount in range(1, MAX_OFFER_QUANTITY + 1)
                        for want_amount in range(1, MAX_OFFER_QUANTITY + 1)
                    ],
                ],
                components=[
                    agent_policy,
                    FruitInventory(),
                    FruitMarketAvatar(specialty),
                    Collidable(collidable_tags=["wall"]),
                    Renderable(
                        sprite_path="src/characters/humanoids/human/farmer_man.png",
                        z_index=10,
                    ),
                ],
            )
        )

    env = Simple_2D_Grid_World(
        description=(
            "Fruit Market: Concentric Rivers, adapted from MeltingPot into "
            "a single-file Word Play environment."
        ),
        entities=entities,
        entity_order=randomize_agent_order,
        observation_radius=5,
    )
    env.reward_func = lambda action_selections, current_env: list(current_env.last_step_rewards)

    for step in range(MAX_EPISODE_LENGTH):
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
        for event in getattr(env, "fruit_market_events", []):
            print(f"[step {step}] fruit_market_concentric_rivers -> {event}")

        if all(env.truncations):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-count", type=int, default=DEFAULT_NUM_PLAYERS)
    parser.add_argument("--policy", choices=["random", "llm", "human"], default="random")
    parser.add_argument("--model-name", default="openai/gpt-4o-mini")
    args = parser.parse_args()
    run_exp(agent_count=args.agent_count, policy=args.policy, model_name=args.model_name)
