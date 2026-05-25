from __future__ import annotations

import argparse

from benchmarks.text_meltingpot.common import (
    BENCHMARK_STEPS,
    COOKING_DELIVERY_REWARD,
    Plate_Ready_Soup_With_Dish,
    normalized_steps,
)
from word_play.core import Agent_Policy, Entity
from word_play.presets.action_policies.human import Human_Takes_Action
from word_play.presets.action_policies.random_policy import Random_Policy
from word_play.presets.entity_orderings import randomize_agent_order
from word_play.presets.environments.simple_2d_grid_world import Simple_2D_Grid_World
from word_play.presets.movement.simple_2d_grid import (
    Collidable,
    Move_Down,
    Move_Left,
    Move_Right,
    Move_Up,
    Position_2D,
)
from word_play.presets.renderers import Renderable, render_step
from word_play.presets.systems.containers.presets import Regrowable_Item_Source, Take_From_Infinite_Source
from word_play.presets.systems.crafter import Crafter, Crafter_Recipe, Load_First_Into_Crafter
from word_play.presets.systems.do_nothing import Do_Nothing
from word_play.presets.systems.inventory import Drop_Item, Inventory, Put_In_Container, Take_First_From_Container
from word_play.presets.systems.reward import Rewardable
from word_play.utils import tilemap_to_entities
from word_play.utils.tilemap import find_tile_positions


DEFAULT_NUM_PLAYERS = 2
MODEL_KEY = "collaborative_cooking__asymmetric"


def run_exp(agent_count: int = DEFAULT_NUM_PLAYERS, policy: str = "random", model_name: str = "openai/gpt-4o-mini"):
    exp_steps = BENCHMARK_STEPS

    if policy == "llm":
        from word_play.presets.models import LLM_MODEL_REGISTRY, OpenRouter_Model

        LLM_MODEL_REGISTRY.unload(MODEL_KEY)
        LLM_MODEL_REGISTRY.register(
            MODEL_KEY,
            OpenRouter_Model,
            model_name=model_name,
            generation_config={"temperature": 0.3},
        )

    entity_tilemap = """
    #########
    O.#T#O#.T
    #.P.C.P.#
    #...C...#
    ###D#D###
    """
    entity_tileset = {
        "#": {
            "name": "Counter",
            "tags": ["counter", "blocker"],
            "components": [
                Inventory(max_size=1),
                Collidable(collidable_tags=["blocker"]),
                Renderable(sprite_path="src/world_tiles/indoors/stations/kitchen_counter.png", z_index=4),
            ],
        },
        "O": {
            "name": "Tomato Source",
            "tags": ["source", "tomato_source", "blocker"],
            "components": [
                Regrowable_Item_Source(
                    item_factory=Entity(
                        name="Tomato",
                        position=Position_2D(0, 0),
                        tags=["tomato", "ingredient"],
                        components=[Renderable(sprite_path="src/items/consumables/vegetables/tomato_2.png", z_index=5)],
                    ),
                    regen_rate=1,
                ),
                Collidable(collidable_tags=["blocker"]),
                Renderable(
                    sprite_path="src/world_tiles/indoors/stations/crate.png",
                    overlay_sprite="src/items/consumables/vegetables/tomato_2.png",
                    overlay_mode="center",
                    overlay_scale=0.7,
                    z_index=4,
                ),
            ],
        },
        "D": {
            "name": "Dish Source",
            "tags": ["source", "dish_source", "blocker"],
            "components": [
                Regrowable_Item_Source(
                    item_factory=Entity(
                        name="Dish",
                        position=Position_2D(0, 0),
                        tags=["dish"],
                        components=[Renderable(sprite_path="src/items/consumables/misc_food/plate.png", z_index=5)],
                    ),
                    regen_rate=1,
                ),
                Collidable(collidable_tags=["blocker"]),
                Renderable(
                    sprite_path="src/world_tiles/indoors/stations/crate.png",
                    overlay_sprite="src/items/consumables/misc_food/plate.png",
                    overlay_mode="center",
                    overlay_scale=0.7,
                    z_index=4,
                ),
            ],
        },
        "C": {
            "name": "Cooking Pot",
            "tags": ["pot", "crafter", "blocker"],
            "components": [
                Crafter(
                    recipes=[
                        Crafter_Recipe(
                            input_names=("Tomato", "Tomato", "Tomato"),
                            output=Entity(
                                name="Soup",
                                position=Position_2D(0, 0),
                                tags=["soup"],
                                components=[
                                    Renderable(sprite_path="src/items/consumables/misc_food/soup.png", z_index=5)
                                ],
                            ),
                            duration=normalized_steps(20),
                        )
                    ]
                ),
                Collidable(collidable_tags=["blocker"]),
                Renderable(sprite_path="src/world_tiles/indoors/stations/pot.png", z_index=4),
            ],
        },
        "T": {
            "name": "Delivery Window",
            "tags": ["delivery", "blocker"],
            "components": [
                Rewardable(amount=COOKING_DELIVERY_REWARD, recipients="all", counter_attr="completed_orders"),
                Collidable(collidable_tags=["blocker"]),
                Renderable(sprite_path="src/world_tiles/indoors/stations/delivery.png", z_index=4),
            ],
        },
    }

    entities = tilemap_to_entities(entity_tilemap.replace("P", "."), entity_tileset)
    spawn_positions = find_tile_positions(entity_tilemap, "P")
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
                    f"You are Player {agent_id} in Collaborative Cooking: Asymmetric. "
                    "Goal: make and deliver soup for shared reward. Recipe: load exactly 3 Tomatoes into one "
                    "Cooking Pot to cook 1 Soup. Use Take Tomato, then Load your held Tomato into the same "
                    "Cooking Pot until it shows 3/3. Wait for the pot to say ready, then take a Dish and use it "
                    "to collect Soup from the ready Cooking Pot. Inventory holds one item; use Counters or Drop to "
                    "make room. Turn in held Soup at the Delivery Window."
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
                actions=[
                    Do_Nothing(),
                    Move_Up(),
                    Move_Down(),
                    Move_Left(),
                    Move_Right(),
                    Drop_Item(),
                    Take_From_Infinite_Source(),
                    Load_First_Into_Crafter(),
                    Plate_Ready_Soup_With_Dish(),
                    Put_In_Container(["delivery"], ["soup"], destroy_item=True),
                    Take_First_From_Container(target_tags=["counter"]),
                    Put_In_Container(target_tags=["counter"], destroy_item=False),
                ],
                components=[
                    agent_policy,
                    Inventory(max_size=1, accepted_tags=["tomato", "dish", "soup", "ingredient"]),
                    Collidable(collidable_tags=["wall", "blocker"]),
                    Renderable(sprite_path="src/characters/humanoids/human/farmer_man.png", z_index=10),
                ],
            )
        )

    env = Simple_2D_Grid_World(
        description="Collaborative Cooking: Asymmetric, adapted from Melting Pot.",
        entities=entities,
        entity_order=randomize_agent_order,
        observation_radius=4,
    )
    env.completed_orders = 0
    env.reward_func = lambda action_selections, current_env: list(current_env.last_step_rewards)

    for step in range(exp_steps):
        if policy != "human" and not render_step(env, step_delay=0.0):
            break

        env.last_step_rewards = [0.0] * len(env.agents)
        cur_step_actions = []
        for agent_id, agent in enumerate(env.agents):
            observation = env.observe(agent_id)
            action, info = agent.get_component(Agent_Policy).select_action(observation)
            print(f"[step {step}] {agent.name} -> {action} | completed_orders={env.completed_orders}")
            cur_step_actions.append(action)

        env.step(cur_step_actions)
        if all(env.terminations) or all(env.truncations):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-count", type=int, default=DEFAULT_NUM_PLAYERS)
    parser.add_argument("--policy", choices=["random", "llm", "human"], default="random")
    parser.add_argument("--model-name", default="openai/gpt-4o-mini")
    args = parser.parse_args()
    run_exp(agent_count=args.agent_count, policy=args.policy, model_name=args.model_name)
