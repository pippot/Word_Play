from __future__ import annotations

import argparse
import random

from benchmarks.text_meltingpot.externality_mushrooms import (
    Externality_Mushroom,
    Externality_Zap,
    Externality_Zap_State,
    MAX_EPISODE_LENGTH,
    MUSHROOM_DIGESTION_STEPS,
    configure_externality_mushrooms,
)
from benchmarks.text_meltingpot.common import normalized_steps
from word_play.core import Agent_Policy, Entity
from word_play.presets.action_policies.human import Human_Takes_Action
from word_play.presets.action_policies.llm_action_and_communication import LLM_Action_And_Communication_Policy
from word_play.presets.action_policies.random_policy import Random_Policy
from word_play.presets.entity_orderings import randomize_agent_order
from word_play.presets.environments.simple_2d_grid_world import Simple_2D_Grid_World
from word_play.presets.models import LLM_MODEL_REGISTRY, OpenRouter_Model
from word_play.presets.movement.simple_2d_grid import Collidable, Move_Down, Move_Left, Move_Right, Move_Up, Position_2D
from word_play.presets.renderers import Renderable, render_step
from word_play.presets.systems.cooldown import Cooldown
from word_play.presets.systems.do_nothing import Do_Nothing
from word_play.presets.systems.freezable import Freezable
from word_play.utils import tilemap_to_entities
from word_play.utils.tilemap import find_tile_positions


DEFAULT_NUM_PLAYERS = 5


def run_exp(
    agent_count: int = DEFAULT_NUM_PLAYERS,
    policy: str = "random",
    model_name: str = "openai/gpt-4o-mini",
):
    entity_tilemap = """
    WWWWWWWWWWWWWWWWWWWWWWW
    W.....................W
    W.R............G......W
    W........R............W
    W.....................W
    W............G........W
    W...B......O..........W
    W..................B..W
    W........R............W
    W.....................W
    W....B........G.......W
    W.....................W
    WWWWWWWWWWWWWWWWWWWWWWW
    """

    if policy == "llm":
        LLM_MODEL_REGISTRY.unload("externality_mushrooms__dense")
        LLM_MODEL_REGISTRY.register(
            "externality_mushrooms__dense",
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
                model_key="externality_mushrooms__dense",
                system_prompt=(
                    f"You are Player {agent_id} in Externality Mushrooms: Dense. "
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
        description="Externality Mushrooms: Dense",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-count", type=int, default=DEFAULT_NUM_PLAYERS)
    parser.add_argument("--policy", choices=["random", "llm", "human"], default="random")
    parser.add_argument("--model-name", default="openai/gpt-4o-mini")
    args = parser.parse_args()
    run_exp(agent_count=args.agent_count, policy=args.policy, model_name=args.model_name)
