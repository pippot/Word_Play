from __future__ import annotations

import argparse
import random

from benchmarks.text_meltingpot.common import (
    BENCHMARK_STEPS,
    DepositFactoryCube,
    FactoryCube,
    FactoryHopper,
    FactoryManager,
    FactoryMoveDown,
    FactoryMoveLeft,
    FactoryMoveRight,
    FactoryMoveUp,
    FactoryStamina,
    nearby_within,
)
from word_play.core import Agent_Policy, Entity
from word_play.presets.action_policies.human import Human_Takes_Action
from word_play.presets.action_policies.random_policy import Random_Policy
from word_play.presets.entity_orderings import randomize_agent_order
from word_play.presets.environments.simple_2d_grid_world import Simple_2D_Grid_World
from word_play.presets.movement.simple_2d_grid import Collidable, Position_2D
from word_play.presets.renderers import Renderable, render_step
from word_play.presets.systems.do_nothing import Do_Nothing
from word_play.presets.systems.inventory import Inventory, Pick_Up_Item
from word_play.presets.systems.zap import Zap_Change
from word_play.utils import tilemap_to_entities
from word_play.utils.tilemap import find_tile_positions


DEFAULT_NUM_PLAYERS = 3
MODEL_KEY = "factory_commons__either_or"


def run_exp(agent_count: int = DEFAULT_NUM_PLAYERS, policy: str = "random", model_name: str = "openai/gpt-4o-mini"):
    exp_steps = BENCHMARK_STEPS

    if policy == "llm":
        from word_play.presets.models import LLM_MODEL_REGISTRY, OpenRouter_Model

        LLM_MODEL_REGISTRY.unload(MODEL_KEY)
        LLM_MODEL_REGISTRY.register(
            MODEL_KEY,
            OpenRouter_Model,
            model_name=model_name,
            generation_config={"temperature": 0.2},
        )

    entity_tilemap = """
    WWWWWWWWWWWWWWWWWWWWWWW
    W..........c..........W
    W.........cCc.........W
    W..h...h...H...h...h..W
    W..O...O.......O...O..W
    W.........cCc.........W
    W..h...h.......h...h..W
    W..O...O.......O...O..W
    W.........cCc.........W
    W..........c..........W
    WWWWWWWWWWWWWWWWWWWWWWW
    """
    entity_tileset = {
        "W": {
            "name": "Wall",
            "tags": ["wall", "blocker"],
            "components": [
                Collidable(collidable_tags=["wall", "blocker"]),
                Renderable(sprite_path="", wall_set="src/world_tiles/indoors/wall_sets/bright_blue_wall", z_index=3),
            ],
        },
        "c": {
            "name": "Waiting Cube",
            "tags": [],
            "components": [
                FactoryCube(live=False),
                Renderable(sprite_path="src/items/gems/blue_cube.png", z_index=4),
            ],
        },
        "C": {
            "name": "Cube",
            "tags": [],
            "components": [
                FactoryCube(live=True),
                Renderable(sprite_path="src/items/gems/blue_cube.png", z_index=4),
            ],
        },
        "H": {
            "name": "Double Hopper",
            "tags": ["blocker"],
            "components": [
                FactoryHopper(output_count=2),
                Collidable(collidable_tags=["blocker"]),
                Renderable(sprite_path="src/world_tiles/indoors/machines/hopper.png", z_index=3),
            ],
        },
        "h": {
            "name": "Single Hopper",
            "tags": ["blocker"],
            "components": [
                FactoryHopper(output_count=1),
                Collidable(collidable_tags=["blocker"]),
                Renderable(sprite_path="src/world_tiles/indoors/machines/hopper.png", z_index=3),
            ],
        },
        "O": {
            "name": "Output Belt",
            "tags": ["belt"],
            "components": [Renderable(sprite_path="src/world_tiles/indoors/floors/white_grid_floor.png", z_index=1)],
        },
    }

    entities = tilemap_to_entities(entity_tilemap, entity_tileset)
    spawn_positions = [
        (12, 5),
        (11, 5),
        (13, 5),
        *[position for position in find_tile_positions(entity_tilemap, ".") if position not in {(12, 5), (11, 5), (13, 5)}],
    ]
    if agent_count > len(spawn_positions):
        raise ValueError(f"Map only has {len(spawn_positions)} spawn positions.")

    random.shuffle(spawn_positions)
    for agent_id, (x, y) in enumerate(spawn_positions[:agent_count], start=1):
        if policy == "llm":
            from word_play.presets.action_policies.llm_action_and_communication import (
                LLM_Action_And_Communication_Policy,
            )

            agent_policy = LLM_Action_And_Communication_Policy(
                model_key=MODEL_KEY,
                system_prompt=(
                    f"You are Player {agent_id} in Factory Commons: Either Or. Pick up cubes, deposit them into hoppers, "
                    "then eat the apples that spawn. If stamina is 0, choose Do nothing to rest. If Eat is available, "
                    "choose Eat. If carrying a cube and Deposit is available, choose Deposit. Only pick up live Cube, "
                    "not Waiting Cube."
                ),
                use_chain_of_thought=True,
                observation_memory_window=1,
                conversation_memory_window=1,
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
                actions=[
                    Do_Nothing(),
                    FactoryMoveUp(),
                    FactoryMoveDown(),
                    FactoryMoveLeft(),
                    FactoryMoveRight(),
                    Pick_Up_Item(["cube"]),
                    DepositFactoryCube(),
                    Zap_Change(
                        allowed_tag="apple",
                        target_is_nearby=nearby_within(2),
                        reward=1.0,
                        action_name="Eat",
                    ),
                ],
                components=[
                    agent_policy,
                    Inventory(max_size=1, accepted_tags=["cube"]),
                    FactoryStamina(maximum=10),
                    Collidable(collidable_tags=["wall", "blocker"]),
                    Renderable(sprite_path="src/characters/humanoids/human/factory_worker.png", z_index=10),
                ],
            )
        )

    entities.append(Entity(name="Factory Manager", position=Position_2D(0, 0), components=[FactoryManager()]))

    env = Simple_2D_Grid_World(
        description="Factory Commons: Either Or, adapted from Melting Pot.",
        entities=entities,
        entity_order=randomize_agent_order,
        observation_radius=4,
    )
    env.factory_events = []
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
            stamina = agent.get_component(FactoryStamina).current
            print(f"[step {step}] {agent.name} stamina={stamina} -> {action}")
            cur_step_actions.append(action)
        env.step(cur_step_actions)
        for event in env.factory_events:
            print(f"[step {step}] factory -> {event}")
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
