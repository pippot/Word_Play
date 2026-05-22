from __future__ import annotations

import argparse
import random

from benchmarks.text_meltingpot.common import BENCHMARK_STEPS, Flag, GrabFlag, PaintballManager, PaintballZap
from word_play.core import Agent_Policy, Entity
from word_play.presets.action_policies.human import Human_Takes_Action
from word_play.presets.action_policies.random_policy import Random_Policy
from word_play.presets.entity_orderings import randomize_agent_order
from word_play.presets.environments.simple_2d_grid_world import Simple_2D_Grid_World
from word_play.presets.movement.simple_2d_grid import Collidable, Move_Down, Move_Left, Move_Right, Move_Up, Position_2D
from word_play.presets.renderers import Renderable, render_step
from word_play.presets.systems.cooldown import Cooldown
from word_play.presets.systems.do_nothing import Do_Nothing
from word_play.presets.systems.freezable import Freezable
from word_play.presets.systems.health import Health
from word_play.presets.systems.inventory import Inventory
from word_play.utils import tilemap_to_entities
from word_play.utils.tilemap import find_tile_positions


DEFAULT_NUM_PLAYERS = 8
MODEL_KEY = "paintball__capture_the_flag"


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

    entity_tilemap = """
    IIIIIIIIIIIIIIIIIIIIIII
    IWWWWWWWWWWWWWWWWWWWWWI
    IWPPP,PPPP,F,PPPP,PPPWI
    IWPPP,,PP,,,,,PP,,PPPWI
    IWPPP,,,,,,,,,,,,,PPPWI
    IWP,,WW,,,,,,,,,WW,,PWI
    IWXXWWW,WWWWWWW,WWWXXWI
    IWXXW,X,,,,,,,,,X,WXXWI
    IWXX,,W,,,WWW,,,W,,XXWI
    IW,,,,W,,,,,,,,,W,,,,WI
    IW,,,,WWW,,,,,WWW,,,,WI
    IW,,,,,,,,,I,,,,,,,,,WI
    IW,,,,WWW,,,,,WWW,,,,WI
    IW,,,,W,,,,,,,,,W,,,,WI
    IWXX,,W,,,WWW,,,W,,XXWI
    IWXXW,X,,,,,,,,,X,WXXWI
    IWXXWWW,WWWWWWW,WWWXXWI
    IWQ,,WW,,,,,,,,,WW,,QWI
    IWQQQ,,,,,,,,,,,,,QQQWI
    IWQQQ,,QQ,,,,,QQ,,QQQWI
    IWQQQ,QQQQ,G,QQQQ,QQQWI
    IWWWWWWWWWWWWWWWWWWWWWI
    IIIIIIIIIIIIIIIIIIIIIII
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
        "X": {
            "name": "Wall",
            "tags": ["wall", "blocker", "destructible"],
            "components": [
                Health(max_health=1),
                Collidable(collidable_tags=["wall", "blocker"]),
                Renderable(sprite_path="src/world_tiles/outdoors/terrain/wall_stone.png", z_index=3),
            ],
        },
        "F": {
            "name": "Red Flag",
            "tags": [],
            "components": [
                Flag("red"),
                Renderable(sprite_path="src/items/misc/flag.png", z_index=4),
            ],
        },
        "G": {
            "name": "Blue Flag",
            "tags": [],
            "components": [
                Flag("blue"),
                Renderable(sprite_path="src/items/misc/flag.png", z_index=4),
            ],
        },
    }

    entity_map = entity_tilemap
    for tile in "PQ,I":
        entity_map = entity_map.replace(tile, ".")
    entities = tilemap_to_entities(entity_map, entity_tileset)
    red_spawns = find_tile_positions(entity_tilemap, "P")
    blue_spawns = find_tile_positions(entity_tilemap, "Q")
    random.shuffle(red_spawns)
    random.shuffle(blue_spawns)
    if agent_count > len(red_spawns) + len(blue_spawns):
        raise ValueError(f"Map only has {len(red_spawns) + len(blue_spawns)} spawn positions.")
    red_count = min(len(red_spawns), (agent_count + 1) // 2)
    blue_count = agent_count - red_count
    spawn_assignments = [("red", pos) for pos in red_spawns[:red_count]] + [
        ("blue", pos) for pos in blue_spawns[:blue_count]
    ]
    red_base = find_tile_positions(entity_tilemap, "F")[0]
    blue_base = find_tile_positions(entity_tilemap, "G")[0]

    for agent_id, (team, (x, y)) in enumerate(spawn_assignments, start=1):
        if policy == "llm":
            from word_play.presets.action_policies.llm_action_and_communication import (
                LLM_Action_And_Communication_Policy,
            )

            agent_policy = LLM_Action_And_Communication_Policy(
                model_key=MODEL_KEY,
                system_prompt=(
                    f"You are Player {agent_id} on the {team} team in Paintball Capture the Flag. "
                    "Shoot opposing players, grab the enemy flag, and carry it back to your own flag base."
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
                name=f"{team.title()} Player {agent_id}",
                position=Position_2D(x, y),
                tags=["agent", "player", "default", team],
                actions=[
                    Do_Nothing(),
                    Move_Up(),
                    Move_Down(),
                    Move_Left(),
                    Move_Right(),
                    PaintballZap(),
                    GrabFlag(),
                ],
                components=[
                    agent_policy,
                    Inventory(max_size=1, accepted_tags=["flag"]),
                    Cooldown(cooldowns={"paintball": 2}),
                    Freezable(default_duration=2),
                    Health(max_health=3, starting_health=2),
                    Collidable(collidable_tags=["wall", "blocker"]),
                    Renderable(sprite_path="src/characters/humanoids/human/farmer_man.png", z_index=10),
                ],
            )
        )

    entities.append(
        Entity(
            name="Paintball Manager",
            position=Position_2D(0, 0),
            components=[PaintballManager(mode="ctf", red_base=red_base, blue_base=blue_base)],
        )
    )

    env = Simple_2D_Grid_World(
        description="Paintball: Capture the Flag, adapted from Melting Pot.",
        entities=entities,
        entity_order=randomize_agent_order,
        observation_radius=5,
    )
    env.paintball_events = []
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
        for event in env.paintball_events:
            print(f"[step {step}] paintball -> {event}")
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
