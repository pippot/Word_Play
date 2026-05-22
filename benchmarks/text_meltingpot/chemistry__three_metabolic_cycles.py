from __future__ import annotations

import argparse

from benchmarks.text_meltingpot.common import BENCHMARK_STEPS, CHEMISTRY_START_POSITIONS, Cycle_Site, Molecule
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
from word_play.presets.systems.do_nothing import Do_Nothing
from word_play.presets.systems.inventory import Inventory, Pick_Up_Item
from word_play.utils import tilemap_to_entities
from word_play.utils.tilemap import find_tile_positions


DEFAULT_NUM_PLAYERS = 8
MAX_EPISODE_LENGTH = BENCHMARK_STEPS
MODEL_KEY = "chemistry__three_metabolic_cycles"


def run_exp(agent_count: int, policy: str, model_name: str):
    if policy == "llm":
        LLM_MODEL_REGISTRY.unload(MODEL_KEY)
        LLM_MODEL_REGISTRY.register(
            MODEL_KEY,
            OpenRouter_Model,
            model_name=model_name,
            generation_config={"temperature": 0.2},
        )

    entity_tilemap = """
    WWWWWWWWWWWWWWWWWWWWWWWWWWW
    W...........a.............W
    W........c................W
    W...........b.............W
    W.........................W
    W.....................1...W
    W.........................W
    W1..3....hhhhhhh.....3..2.W
    W.........................W
    W.2.......................W
    W.........................W
    W.......c.................W
    W.........a..........4...6W
    W.......b.................W
    W......................5..W
    WWWWWWWWWWWWWWWWWWWWWWWWWWW
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
        "a": {
            "name": "Blue Compound AX",
            "components": [
                Cycle_Site("blue"),
                Renderable(sprite_path="src/items/consumables/magic/blue_scroll.png", z_index=2),
            ],
        },
        "b": {
            "name": "Blue Compound BX",
            "components": [
                Cycle_Site("blue"),
                Renderable(sprite_path="src/items/consumables/magic/blue_scroll.png", z_index=2),
            ],
        },
        "c": {
            "name": "Blue Compound CX",
            "components": [
                Cycle_Site("blue"),
                Renderable(sprite_path="src/items/consumables/magic/blue_scroll.png", z_index=2),
            ],
        },
        "h": {
            "name": "Energy",
            "components": [
                Molecule("energy"),
                Renderable(sprite_path="src/items/consumables/magic/golden_potion.png", z_index=4),
            ],
        },
        "1": {
            "name": "Green Compound AY",
            "components": [
                Cycle_Site("green"),
                Renderable(sprite_path="src/items/consumables/magic/green_scroll.png", z_index=2),
            ],
        },
        "2": {
            "name": "Green Compound BY",
            "components": [
                Cycle_Site("green"),
                Renderable(sprite_path="src/items/consumables/magic/green_scroll.png", z_index=2),
            ],
        },
        "3": {
            "name": "Green Compound CY",
            "components": [
                Cycle_Site("green"),
                Renderable(sprite_path="src/items/consumables/magic/green_scroll.png", z_index=2),
            ],
        },
        "4": {
            "name": "Red Compound AZ",
            "components": [
                Cycle_Site("red"),
                Renderable(sprite_path="src/items/consumables/magic/red_scroll.png", z_index=2),
            ],
        },
        "5": {
            "name": "Red Compound BZ",
            "components": [
                Cycle_Site("red"),
                Renderable(sprite_path="src/items/consumables/magic/red_scroll.png", z_index=2),
            ],
        },
        "6": {
            "name": "Red Compound CZ",
            "components": [
                Cycle_Site("red"),
                Renderable(sprite_path="src/items/consumables/magic/red_scroll.png", z_index=2),
            ],
        },
    }

    entities = tilemap_to_entities(entity_tilemap, entity_tileset)
    floor_positions = find_tile_positions(entity_tilemap, ".")
    spawn_positions = [
        *[position for position in CHEMISTRY_START_POSITIONS if position in floor_positions],
        *[position for position in floor_positions if position not in CHEMISTRY_START_POSITIONS],
    ]
    if agent_count > len(spawn_positions):
        raise ValueError(f"Map only has {len(spawn_positions)} spawn positions.")

    for agent_id, (x, y) in enumerate(spawn_positions[:agent_count], start=1):
        if agent_id <= 3:
            role_prompt = "Your job is blue/X: carry energy to Blue Compounds, make Molecule X, then bring X toward Y. "
        elif agent_id <= 6:
            role_prompt = "Your job is green/Y: carry energy to Green Compounds, make Molecule Y, then bring Y toward X. "
        else:
            role_prompt = "Your job is red/food3: carry energy to Red Compounds and pick up Food 3 when it appears. "
        agent_policy = (
            LLM_Action_And_Communication_Policy(
                model_key=MODEL_KEY,
                system_prompt=(
                    "You are playing Chemistry: Three Metabolic Cycles. "
                    f"You are Player {agent_id}. {role_prompt}"
                    "Do not all go to the same cycle. "
                    "Use the positions in the observation. Move right increases x, left decreases x, "
                    "up increases y, and down decreases y. If Pick up is available, take it. "
                    "If your inventory is empty, move onto a useful loose molecule: energy, food, x, or y. "
                    "If holding energy, move onto a Blue, Green, or Red Compound; those compounds are cycle sites. "
                    "If holding x, move onto y; if holding y, move onto x. "
                    "Avoid inhibitors IX/IY. Do not drop energy, x, y, or food unless no useful movement is available. "
                    "Food1 and food2 pay +1 automatically while held; food3 pays +10 automatically while held. "
                    "Do not choose Do nothing unless no useful move is available."
                ),
                use_chain_of_thought=True,
                observation_memory_window=2,
                conversation_memory_window=0,
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
                    Pick_Up_Item(["molecule"]),
                    Move_Up(),
                    Move_Down(),
                    Move_Left(),
                    Move_Right(),
                ],
                components=[
                    agent_policy,
                    Inventory(accepted_tags=[], max_size=1),
                    Collidable(collidable_tags=["wall"]),
                    Renderable(sprite_path="src/characters/humanoids/human/farmer_man.png", z_index=10),
                ],
            )
        )

    env = Simple_2D_Grid_World(
        description="Chemistry: Three Metabolic Cycles, adapted from MeltingPot.",
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
            info = info or {}
            if info.get("raw_response"):
                print(f"[step {step}] {agent.name} raw -> {info['raw_response'].strip()}")
            print(f"[step {step}] {agent.name} -> {action}")
            cur_step_actions.append(action)

        env.step(cur_step_actions)
        for event in getattr(env, "chemistry_events", []) or ["No chemical reactions fired."]:
            print(f"[step {step}] chemistry -> {event}")

        if all(env.terminations) or all(env.truncations):
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
