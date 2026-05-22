from __future__ import annotations

import argparse
import random

from benchmarks.text_meltingpot.common import BENCHMARK_STEPS, BoatFood, BoatRaceManager, BoatRower, Flail, Paddle
from word_play.core import Agent_Policy, Entity
from word_play.presets.action_policies.human import Human_Takes_Action
from word_play.presets.action_policies.random_policy import Random_Policy
from word_play.presets.entity_orderings import randomize_agent_order
from word_play.presets.environments.simple_2d_grid_world import Simple_2D_Grid_World
from word_play.presets.movement.simple_2d_grid import Collidable, Move_Down, Move_Left, Move_Right, Move_Up, Position_2D
from word_play.presets.renderers import Renderable, render_step
from word_play.presets.systems.do_nothing import Do_Nothing
from word_play.presets.systems.inventory import Inventory
from word_play.utils import tilemap_to_entities
from word_play.utils.tilemap import find_tile_positions


DEFAULT_NUM_PLAYERS = 6
MODEL_KEY = "boat_race__eight_races"


def run_exp(agent_count: int = DEFAULT_NUM_PLAYERS, policy: str = "random", model_name: str = "openai/gpt-4o-mini"):
    exp_steps = BENCHMARK_STEPS

    if policy == "llm":
        from word_play.presets.models import LLM_MODEL_REGISTRY, OpenRouter_Model

        LLM_MODEL_REGISTRY.unload(MODEL_KEY)
        LLM_MODEL_REGISTRY.register(
            MODEL_KEY,
            OpenRouter_Model,
            model_name=model_name,
            generation_config={"temperature": 0.25},
        )

    entity_tilemap = r"""
    WWWWWWWWWWWWWWWWWWWWWWWWWW
    W........................W
    W........................W
    W........................W
    W......AAAAAAAAAAAA......W
    W......AAAAAAAAAAAA......W
    W......AAAAAAAAAAAA......W
    W......AAAAAAAAAAAA......W
    W........................W
    W......S..SS..SS..S......W
    W......SBBSSBBSSBBS......W
    W......S..SS..SS..S......W
    ~~~~~~~~gg~~gg~~gg~~~~~~~~
    ~~~~~~~~{{~~{{~~{{~~~~~~~~
    ~~~~~~~~AA~~AA~~AA~~~~~~~~
    ~~~~~~~~{{~~{{~~{{~~~~~~~~
    ~~~~~~~~{{~~{{~~{{~~~~~~~~
    ~~~~~~~~AA~~AA~~AA~~~~~~~~
    ~~~~~~~~{{~~{{~~{{~~~~~~~~
    ~~~~~~~~{{~~{{~~{{~~~~~~~~
    ~~~~~~~~AA~~AA~~AA~~~~~~~~
    ~~~~~~~~{{~~{{~~{{~~~~~~~~
    ~~~~~~~~{{~~{{~~{{~~~~~~~~
    ~~~~~~~~AA~~AA~~AA~~~~~~~~
    ~~~~~~~~bb~~bb~~bb~~~~~~~~
    ~~~~~~~ossoossoosso~~~~~~~
    W......SbbSSbbSSbbS......W
    W......SBBSSBBSSBBS......W
    W......S..SS..SS..S......W
    W........................W
    W......AAAAAAAAAAAA......W
    W......AAAAAAAAAAAA......W
    W......AAAAAAAAAAAA......W
    W......AAAAAAAAAAAA......W
    W........................W
    W....________________....W
    W....________________....W
    WWWWWWWWWWWWWWWWWWWWWWWWWW
    """
    entity_tileset = {
        "W": {
            "name": "Wall",
            "tags": ["wall", "blocker"],
            "components": [
                Collidable(collidable_tags=["wall", "blocker"]),
                Renderable(sprite_path="src/world_tiles/outdoors/terrain/wall_stone.png", z_index=3),
            ],
        },
        "~": {
            "name": "Deep Water",
            "tags": ["water", "blocker"],
            "components": [
                Collidable(collidable_tags=["blocker"]),
                Renderable(sprite_path="src/world_tiles/outdoors/terrain/water.png", z_index=1),
            ],
        },
        "{": {
            "name": "Water",
            "tags": ["water"],
            "components": [Renderable(sprite_path="src/world_tiles/outdoors/terrain/water.png", z_index=1)],
        },
        "g": {
            "name": "Goal Water",
            "tags": ["water", "goal"],
            "components": [Renderable(sprite_path="src/world_tiles/outdoors/terrain/water.png", z_index=1)],
        },
        "S": {
            "name": "Semaphore Tile",
            "tags": ["floor"],
            "components": [Renderable(sprite_path="src/world_tiles/indoors/floors/checker_floor.png", z_index=1)],
        },
        "B": {
            "name": "Barrier",
            "tags": ["barrier", "blocker"],
            "components": [
                Collidable(collidable_tags=["blocker"]),
                Renderable(sprite_path="src/world_tiles/indoors/walls/fence.png", z_index=3),
            ],
        },
        "A": {
            "name": "Apple",
            "tags": ["apple", "food"],
            "components": [
                BoatFood(reward=1.0),
                Renderable(sprite_path="src/items/consumables/lpc_food/apple.png", z_index=4),
            ],
        },
        "b": {
            "name": "Boat",
            "tags": ["boat"],
            "components": [Renderable(sprite_path="src/world_tiles/outdoors/objects/boat.png", z_index=2)],
        },
        "o": {
            "name": "Oar",
            "tags": ["oar"],
            "components": [Renderable(sprite_path="src/items/weapons/polearm_spear/basic_spear.png", z_index=3)],
        },
        "s": {
            "name": "Boat Seat",
            "tags": ["seat", "goal"],
            "components": [Renderable(sprite_path="src/world_tiles/indoors/floors/wood_floor.png", z_index=1)],
        },
        "_": {
            "name": "Spawn Dock",
            "tags": ["spawn", "floor"],
            "components": [Renderable(sprite_path="src/world_tiles/indoors/floors/wood_floor.png", z_index=1)],
        },
    }

    entities = tilemap_to_entities(entity_tilemap, entity_tileset)
    spawn_positions = find_tile_positions(entity_tilemap, "_")
    random.shuffle(spawn_positions)
    if agent_count > len(spawn_positions):
        raise ValueError(f"Map only has {len(spawn_positions)} spawn positions.")

    for agent_id, (x, y) in enumerate(spawn_positions[:agent_count], start=1):
        if policy == "llm":
            from word_play.presets.action_policies.llm_action_and_communication import (
                LLM_Action_And_Communication_Policy,
            )

            agent_policy = LLM_Action_And_Communication_Policy(
                model_key=MODEL_KEY,
                system_prompt=(
                    f"You are Player {agent_id} in Boat Race: Eight Races. "
                    "Coordinate Paddle actions with another player to advance the race; flail only when coordination fails."
                ),
                use_chain_of_thought=True,
                observation_memory_window=4,
                conversation_memory_window=4,
            )
        elif policy == "human":
            agent_policy = Human_Takes_Action()
        else:
            agent_policy = Random_Policy()

        entities.append(
            Entity(
                name=f"Player {agent_id}",
                position=Position_2D(x, y),
                tags=["agent", "player", "default"],
                actions=[Do_Nothing(), Move_Up(), Move_Down(), Move_Left(), Move_Right(), Paddle(), Flail()],
                components=[
                    agent_policy,
                    BoatRower(),
                    Inventory(max_size=0),
                    Collidable(collidable_tags=["wall", "blocker"]),
                    Renderable(sprite_path="src/characters/humanoids/human/sailor.png", z_index=10),
                ],
            )
        )

    entities.append(Entity(name="Boat Race Manager", position=Position_2D(0, 0), components=[BoatRaceManager(8)]))

    env = Simple_2D_Grid_World(
        description="Boat Race: Eight Races, adapted from Melting Pot.",
        entities=entities,
        entity_order=randomize_agent_order,
        observation_radius=4,
    )
    env.boat_events = []
    env.reward_func = lambda action_selections, current_env: list(current_env.last_step_rewards)

    for step in range(exp_steps):
        if policy != "human" and not render_step(env, step_delay=0.0):
            break
        env.last_step_rewards = [0.0] * len(env.agents)
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
        for event in env.boat_events:
            print(f"[step {step}] boat -> {event}")
        if any(env.last_step_rewards):
            print(f"[step {step}] rewards -> {env.last_step_rewards}")
        if all(env.terminations) or all(env.truncations):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-count", type=int, default=DEFAULT_NUM_PLAYERS)
    parser.add_argument("--policy", choices=["random", "llm", "human"], default="random")
    parser.add_argument("--model-name", default="openai/gpt-4o-mini")
    args = parser.parse_args()
    run_exp(agent_count=args.agent_count, policy=args.policy, model_name=args.model_name)
