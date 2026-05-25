from __future__ import annotations

import argparse
import random

from benchmarks.text_meltingpot.common import (
    ActorCanMove,
    BENCHMARK_STEPS,
    CatchPrey,
    PredatorPreyFood,
    PredatorPreyRole,
)
from word_play.core import Agent_Policy, Entity
from word_play.presets.action_policies.human import Human_Takes_Action
from word_play.presets.action_policies.random_policy import Random_Policy
from word_play.presets.entity_orderings import randomize_agent_order
from word_play.presets.environments.simple_2d_grid_world import Simple_2D_Grid_World
from word_play.presets.movement.simple_2d_grid import Collidable, Move_Down, Move_Left, Move_Right, Move_Up, Position_2D
from word_play.presets.renderers import Renderable, render_step
from word_play.presets.systems.do_nothing import Do_Nothing
from word_play.presets.systems.respawnable import Respawnable
from word_play.utils import tilemap_to_entities
from word_play.utils.tilemap import find_tile_positions


DEFAULT_NUM_PLAYERS = 13
DEFAULT_NUM_PREDATORS = 5
MODEL_KEY = "predator_prey__orchard"


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
    WWWWWWWWWWWWWWWWWWWWWWW
    WWAA.P.PP..AWWA....ACWW
    WA..AAAAAA.PWW..AAq..CW
    WP.AACAAAAA....AAAAA..W
    W.q.AAAAAA..CA.AAAAAA.W
    WA...P....P...A......AW
    WAA..AAA............AAW
    WWW..AAA..WWWWPPPACWWWW
    WWW...A.P.WWWWWWWWWWWWW
    WPP...A.P...WWWWWWWWW.W
    W.....A......PP.......W
    W.GGGGGGGG...P.C...C..W
    W.GGGGGGGGGG.....C....W
    W...GGGGGGGG...C...C..W
    W..GGGGGGGG......C...PW
    W..GGGGGGGGGG..C...C..W
    W....GGGGGGGG....C.q..W
    WW...................WW
    WWWWWWWWWWWWWWWWWWWWWWW
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
        "G": {
            "name": "Tall Grass",
            "tags": ["grass"],
            "components": [
                Collidable(collidable_tags=["predator"]),
                Renderable(sprite_path="src/world_tiles/outdoors/terrain/grass_1.png", z_index=1),
            ],
        },
        "A": {
            "name": "Apple",
            "tags": [],
            "components": [
                PredatorPreyFood("apple", reward=1.0),
                Renderable(sprite_path="src/items/consumables/lpc_food/apple.png", z_index=4),
            ],
        },
        "C": {
            "name": "Acorn",
            "tags": [],
            "components": [
                PredatorPreyFood("acorn", reward=18.0),
                Renderable(sprite_path="src/items/materials/misc/brown_pebbles.png", z_index=4),
            ],
        },
    }

    entities = tilemap_to_entities(entity_tilemap.replace("P", ".").replace("q", "."), entity_tileset)
    predator_spawns = find_tile_positions(entity_tilemap, "P")
    prey_spawns = find_tile_positions(entity_tilemap, "q")
    open_spawns = find_tile_positions(entity_tilemap, ".")
    random.shuffle(predator_spawns)
    random.shuffle(prey_spawns)
    random.shuffle(open_spawns)
    predator_count = min(DEFAULT_NUM_PREDATORS, agent_count, len(predator_spawns))
    prey_count = agent_count - predator_count
    if prey_count > len(prey_spawns) + len(open_spawns):
        raise ValueError("Map does not have enough spawn positions.")
    spawn_assignments = [("predator", pos) for pos in predator_spawns[:predator_count]]
    spawn_assignments.extend(("prey", pos) for pos in (prey_spawns + open_spawns)[:prey_count])

    for agent_id, (role, (x, y)) in enumerate(spawn_assignments, start=1):
        if policy == "llm":
            from word_play.presets.action_policies.llm_action_and_communication import (
                LLM_Action_And_Communication_Policy,
            )

            agent_policy = LLM_Action_And_Communication_Policy(
                model_key=MODEL_KEY,
                system_prompt=(
                    f"You are Player {agent_id}, a {role}, in Predator Prey Orchard. "
                    "Predators hunt prey. Prey collect apples and acorns while using the orchard and grass for safety."
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
                name=f"{role.title()} {agent_id}",
                position=Position_2D(x, y),
                tags=["agent", "player", "default"],
                actions=[
                    Do_Nothing(),
                    Move_Up(ActorCanMove()),
                    Move_Down(ActorCanMove()),
                    Move_Left(ActorCanMove()),
                    Move_Right(ActorCanMove()),
                    CatchPrey(),
                ],
                components=[
                    agent_policy,
                    Respawnable(respawn_position=Position_2D(x, y), inactive_position=Position_2D(-1000, -1000)),
                    PredatorPreyRole(role),
                    Collidable(collidable_tags=["wall", "blocker"]),
                    Renderable(
                        sprite_path=(
                            "src/characters/animals/wolf.png"
                            if role == "predator"
                            else "src/characters/humanoids/human/farmer_man.png"
                        ),
                        z_index=10,
                    ),
                ],
            )
        )

    env = Simple_2D_Grid_World(
        description="Predator Prey: Orchard, adapted from Melting Pot.",
        entities=entities,
        entity_order=randomize_agent_order,
        observation_radius=5,
    )
    env.predator_prey_events = []
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
        for event in env.predator_prey_events:
            print(f"[step {step}] predator_prey -> {event}")
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
