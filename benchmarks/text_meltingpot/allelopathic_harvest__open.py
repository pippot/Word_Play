import argparse
import random

from benchmarks.text_meltingpot.common import (
    Allelopathic_Berry_Patch,
    Allelopathic_Plant_Berry,
    Allelopathic_Zap_State,
    BENCHMARK_STEPS,
    normalized_steps,
)
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
from word_play.presets.systems.preferences import Preference
from word_play.presets.systems.zap import Zap_Player
from word_play.utils import tilemap_to_entities
from word_play.utils.tilemap import ascii_to_tilemap_array, find_tile_positions


DEFAULT_NUM_PLAYERS = 16
MAX_EPISODE_LENGTH = BENCHMARK_STEPS
MODEL_KEY = "allelopathic_harvest__open"

OPEN_ASCII_MAP = """
333PPPP12PPP322P32PPP1P13P3P3
1PPPP2PP122PPP3P232121P2PP2P1
P1P3P11PPP13PPP31PPPP23PPPPPP
PPPPP2P2P1P2P3P33P23PP2P2PPPP
P1PPPPPPP2PPP12311PP3321PPPPP
133P2PP2PPP3PPP1PPP2213P112P1
3PPPPPPPPPPPPP31PPPPPP1P3112P
PP2P21P21P33PPPPPPP3PP2PPPP1P
PPPPP1P1P32P3PPP22PP1P2PPPP2P
PPP3PP3122211PPP2113P3PPP1332
PP12132PP1PP1P321PP1PPPPPP1P3
PPP222P12PPPP1PPPP1PPP321P11P
PPP2PPPP3P2P1PPP1P23322PP1P13
23PPP2PPPP2P3PPPP3PP3PPP3PPP2
2PPPP3P3P3PP3PP3P1P3PP11P21P1
21PPP2PP331PP3PPP2PPPPP2PP3PP
P32P2PP2P1PPPPPPP12P2PPP1PPPP
P3PP3P2P21P3PP2PP11PP1323P312
2P1PPPPP1PPP1P2PPP3P32P2P331P
PPPPP1312P3P2PPPP3P32PPPP2P11
P3PPPP221PPP2PPPPPPPP1PPP311P
32P3PPPPPPPPPP31PPPP3PPP13PPP
PPP3PPPPP3PPPPPP232P13PPPPP1P
P1PP1PPP2PP3PPPPP33321PP2P3PP
P13PPPP1P333PPPP2PP213PP2P3PP
1PPPPP3PP2P1PP21P3PPPP231P2PP
1331P2P12P2PPPP2PPP3P23P21PPP
P3P131P3PPP13P1PPP222PPPP11PP
2P3PPPPPPPP2P323PPP2PPP1PPP2P
21PPPPPPP12P23P1PPPPPP13P3P11
"""


def preferred_color_for_agent(agent_id: int) -> str:
    return "red" if agent_id <= DEFAULT_NUM_PLAYERS // 2 else "green"


def run_exp(agent_count: int, policy: str, model_name: str):
    if policy == "llm":
        LLM_MODEL_REGISTRY.unload(MODEL_KEY)
        LLM_MODEL_REGISTRY.register(
            MODEL_KEY,
            OpenRouter_Model,
            model_name=model_name,
            generation_config={"temperature": 0.3},
        )

    tilemap = ascii_to_tilemap_array(OPEN_ASCII_MAP)
    entity_tileset = {
        "P": {
            "name": "Empty Berry Patch",
            "components": [
                Allelopathic_Berry_Patch(auto_consume=True, grows=True),
                Renderable(
                    sprite_path="src/items/consumables/vegetables/mushroom_red.png",
                    z_index=2,
                    visible=False,
                ),
            ],
        },
        "1": {
            "name": "Red Berry Patch",
            "components": [
                Allelopathic_Berry_Patch("red", auto_consume=True, grows=True),
                Renderable(
                    sprite_path="src/items/consumables/vegetables/mushroom_red.png",
                    z_index=2,
                ),
            ],
        },
        "2": {
            "name": "Green Berry Patch",
            "components": [
                Allelopathic_Berry_Patch("green", auto_consume=True, grows=True),
                Renderable(
                    sprite_path="src/items/consumables/vegetables/mushroom_green.png",
                    z_index=2,
                ),
            ],
        },
        "3": {
            "name": "Blue Berry Patch",
            "components": [
                Allelopathic_Berry_Patch("blue", auto_consume=True, grows=True),
                Renderable(
                    sprite_path="src/items/consumables/vegetables/mushroom_blue.png",
                    z_index=2,
                ),
            ],
        },
    }

    entities = tilemap_to_entities(tilemap, entity_tileset)
    spawn_positions = find_tile_positions(tilemap, "P")
    if agent_count > len(spawn_positions):
        raise ValueError(f"Only {len(spawn_positions)} empty patches are available as spawn points.")

    random.shuffle(spawn_positions)
    for agent_id, (x, y) in enumerate(spawn_positions[:agent_count], start=1):
        preferred_color = preferred_color_for_agent(agent_id)
        entities.append(
            Entity(
                name=f"Player {agent_id}",
                position=Position_2D(x, y),
                tags=[
                    "agent",
                    "player",
                    "default",
                    f"player_who_likes_{preferred_color}",
                ],
                actions=[
                    Do_Nothing(),
                    Move_Up(),
                    Move_Down(),
                    Move_Left(),
                    Move_Right(),
                    Zap_Player(),
                    Allelopathic_Plant_Berry("red"),
                    Allelopathic_Plant_Berry("green"),
                    Allelopathic_Plant_Berry("blue"),
                ],
                components=[
                    (
                        LLM_Action_And_Communication_Policy(
                            model_key=MODEL_KEY,
                            system_prompt=(
                                f"You are Player {agent_id} in Allelopathic Harvest: Open. "
                                f"You get +2 from {preferred_color} berries and +1 from other berries. "
                                "Eat ripe berries, plant new berries on empty patches, and use zaps when it helps your score."
                            ),
                            use_chain_of_thought=False,
                            observation_memory_window=0,
                            conversation_memory_window=0,
                        )
                        if policy == "llm"
                        else Human_Takes_Action() if policy == "human" else Random_Policy()
                    ),
                    Preference(reward_map={preferred_color: 2.0}, default_reward=1.0),
                    Cooldown(cooldowns={"zap": normalized_steps(4), "plant": normalized_steps(2)}),
                    Freezable(default_duration=normalized_steps(25)),
                    Allelopathic_Zap_State(),
                    Renderable(
                        sprite_path="src/characters/humanoids/human/farmer_man.png",
                        z_index=10,
                    ),
                ],
            )
        )

    env = Simple_2D_Grid_World(
        description=(
            "Allelopathic Harvest: Open, adapted from MeltingPot into a Word Play "
            "grid world."
        ),
        entities=entities,
        entity_order=randomize_agent_order,
        observation_radius=4,
    )

    for step in range(MAX_EPISODE_LENGTH):
        if policy != "human" and not render_step(env, step_delay=0.0):
            break

        env.last_step_rewards = [0.0] * len(env.agents)
        env.reward_func = lambda action_selections, current_env: list(current_env.last_step_rewards)

        cur_step_actions = []
        for agent_id, agent in enumerate(env.agents):
            observation = env.observe(agent_id)
            action, info = agent.get_component(Agent_Policy).select_action(observation)
            print(f"[step {step}] {agent.name} -> {action}")
            cur_step_actions.append(action)

        env.step(cur_step_actions)
        for agent, info in zip(env.agents, env.infos):
            action_info = info.get("action_info")
            if action_info:
                print(f"[step {step}] {agent.name} action info: {action_info}")
        for event in getattr(env, "allelopathic_events", []) or ["No allelopathic harvest events."]:
            print(f"[step {step}] {event}")
        if any(env.last_rewards):
            rewards = ", ".join(f"{agent.name}: {reward:g}" for agent, reward in zip(env.agents, env.last_rewards))
            print(f"[step {step}] rewards: {rewards}")
        if all(env.truncations):
            break


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
