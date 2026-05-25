from __future__ import annotations

import argparse
import random

from benchmarks.text_meltingpot.common import BENCHMARK_STEPS, Commons_Apple_Patch, Commons_Zap, normalized_steps

from word_play.core import Agent_Policy, Entity
from word_play.presets.action_policies.llm_action_and_communication import (
    LLM_Action_And_Communication_Policy,
)
from word_play.presets.action_policies.human import Human_Takes_Action
from word_play.presets.action_policies.random_policy import Random_Policy
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
from word_play.presets.systems.cooldown import Cooldown
from word_play.presets.systems.do_nothing import Do_Nothing
from word_play.presets.systems.freezable import Freezable
from word_play.presets.systems.reward import Rewardable
from word_play.utils import tilemap_to_entities
from word_play.utils.tilemap import find_tile_positions


def run_exp(agent_count: int = 7, policy: str = "random", model_name: str = "openai/gpt-4o-mini"):
    exp_steps = BENCHMARK_STEPS

    if policy == "llm":
        LLM_MODEL_REGISTRY.unload("commons_harvest__open")
        LLM_MODEL_REGISTRY.register(
            "commons_harvest__open",
            OpenRouter_Model,
            model_name=model_name,
            generation_config={"temperature": 0.2},
        )

    entity_tilemap = """
    WWWWWWWWWWWWWWWWWWWWWWWW
    WAAA....A......A....AAAW
    WAA....AAA....AAA....AAW
    WA....AAAAA..AAAAA....AW
    W......AAA....AAA......W
    W.......A......A.......W
    W..A................A..W
    W.AAA..Q........Q..AAA.W
    WAAAAA............AAAAAW
    W.AAA..............AAA.W
    W..A................A..W
    W......................W
    W......................W
    W......................W
    W..PPPPPPPPPPPPPPPPPP..W
    W.PPPPPPPPPPPPPPPPPPPP.W
    WPPPPPPPPPPPPPPPPPPPPPPW
    WWWWWWWWWWWWWWWWWWWWWWWW
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
        "A": {
            "name": "Apple",
            "components": [
                Commons_Apple_Patch(apple_regrowth_probs=[0.0, 0.0025, 0.005, 0.025]),
                Rewardable(amount=1.0),
                Renderable(sprite_path="src/items/consumables/lpc_food/apple.png", z_index=4),
            ],
        },
    }

    entities = tilemap_to_entities(entity_tilemap.replace("P", ".").replace("Q", "."), entity_tileset)
    inside_spawns = find_tile_positions(entity_tilemap, "Q")
    outside_spawns = find_tile_positions(entity_tilemap, "P")
    random.shuffle(inside_spawns)
    random.shuffle(outside_spawns)
    spawn_positions = inside_spawns[:2] + outside_spawns
    if agent_count > len(spawn_positions):
        raise ValueError(f"Map only has {len(spawn_positions)} spawn positions.")

    for agent_id, (x, y) in enumerate(spawn_positions[:agent_count], start=1):
        agent_policy = (
            LLM_Action_And_Communication_Policy(
                model_key="commons_harvest__open",
                system_prompt=(
                    f"You are Player {agent_id} in Commons Harvest: Open. "
                    "Eat apples for +1, but leave enough nearby apples so patches keep regrowing. "
                    "Use zap sparingly to deter competitors."
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
                    Commons_Zap(),
                ],
                components=[
                    agent_policy,
                    Cooldown(cooldowns={"zap": normalized_steps(2)}),
                    Freezable(default_duration=normalized_steps(4)),
                    Collidable(collidable_tags=["wall"]),
                    Renderable(sprite_path="src/characters/humanoids/human/farmer_man.png", z_index=10),
                ],
            )
        )

    env = Simple_2D_Grid_World(
        description="Commons Harvest: Open, adapted from MeltingPot.",
        entities=entities,
        entity_order=randomize_agent_order,
        observation_radius=4,
    )

    for step in range(exp_steps):
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
        for event in getattr(env, "commons_events", []):
            print(f"[step {step}] commons -> {event}")

        if all(env.terminations) or all(env.truncations):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-count", type=int, default=7)
    parser.add_argument("--policy", choices=["random", "llm", "human"], default="random")
    parser.add_argument("--model-name", default="openai/gpt-4o-mini")
    args = parser.parse_args()
    run_exp(agent_count=args.agent_count, policy=args.policy, model_name=args.model_name)
