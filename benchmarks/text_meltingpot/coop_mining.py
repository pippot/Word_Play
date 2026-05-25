from __future__ import annotations

import argparse
import random

from word_play.core import (
    Action,
    Agent_Policy,
    Component,
    Entity,
    Environment,
    Target_Is_Nearby,
    Target_Not_Self,
)
from word_play.presets.action_policies.llm_action_and_communication import (
    LLM_Action_And_Communication_Policy,
)
from word_play.presets.action_policies.human import Human_Takes_Action
from word_play.presets.action_policies.random_policy import Random_Policy
from word_play.presets.action_validations import Target_Has_Component
from word_play.presets.entity_orderings import randomize_agent_order
from word_play.presets.environments.simple_2d_grid_world import Simple_2D_Grid_World
from word_play.presets.models import (
    LLM_MODEL_REGISTRY,
    OpenRouter_Model,
)
from word_play.presets.movement.simple_2d_grid import (
    Collidable,
    Move_Down,
    Move_Left,
    Move_Right,
    Move_Up,
    Position_2D,
)
from word_play.presets.renderers import (
    Renderable,
    render_step,
)
from word_play.presets.systems.cooldown import (
    Action_On_Cooldown,
    Cooldown,
)
from word_play.presets.systems.coordinated_action import Coordinated_Action
from word_play.presets.systems.do_nothing import Do_Nothing
from word_play.presets.systems.reward import award_reward
from benchmarks.text_meltingpot.common import BENCHMARK_STEPS, normalized_probability, normalized_steps
from word_play.utils import tilemap_to_entities
from word_play.utils.tilemap import find_tile_positions


MANDATED_NUM_PLAYERS = 2
MAX_EPISODE_LENGTH = BENCHMARK_STEPS
MIN_EPISODE_LENGTH = normalized_steps(1000)
INTERVAL_LENGTH = normalized_steps(100)
TERMINATION_PROBABILITY = 0.2
MINE_COOLDOWN = normalized_steps(3)
IRON_REWARD = 1.0
GOLD_REWARD = 8.0
IRON_REGROW_PROB = normalized_probability(0.0002)
GOLD_REGROW_PROB = normalized_probability(0.00008)
GOLD_MINERS_REQUIRED = 2


ASCII_MAP = """
WWWWWWWWWWWWWWWWWWWWWWWWWWW
WOOOOOOOOOOOOOOOOOOOOOOOOOW
WOPOOOOOOOOOPOOOOOPOOOOOPOW
WOOOOOOOOWOOOOOOOOOOOOOOOOW
WOOOOOOOOWOOOOOOOOOOWOOOOOW
WOOOOOOOOWOOOOOOOOOOWOOOOOW
WOOOOOOOOWWWWWWWOOOOWOOOPOW
WOPOWWOOOOWOOOOOOOOOWOOOOOW
WOOOOOOOOOWOOPOOOOOOOOOOOOW
WOOOOOOOOOWOOOOOWWWOOOOOOOW
WOOOOOOOOOWOOOOOOOOOOOOOOOW
WOOOOOOOOOOOOOOOOOOOOOOOPOW
WOPOOOWWWOOOOOOWWWWWWWWOOOW
WOOWWWWOOOOOOOOOOOOOOOOOOOW
WOOOOOWOOOOWOOOOOPOOOOOOOOW
WOOOOOWOOOOWOOOOOOOOOOOOPOW
WOOOOOWOOOOOWOOOOOOOOWOOOOW
WOOOOOOWOOOOOWWWWOOOOWOOOOW
WOPOOOOOWOOOOOOOOOOOOWOOOOW
WOOOOOOOOWOOOPOOOOOOOOOOPOW
WOOOOOOOOOWOOOOOOOOWOOOOOOW
WOOOOWOOOOOOOOOOOOOWOOOOOOW
WOOOOWOOOOOOOOOWWWWWWWWOOOW
WOOOOWOOOOOOOOOOOOWOOOOOOOW
WOPOOOOOOPOOOOOOOPOOOOOOPOW
WOOOOOOOOOOOOOOOOOOOOOOOOOW
WWWWWWWWWWWWWWWWWWWWWWWWWWW
"""


def sample_ore_type() -> str:
    return "gold" if random.random() < 0.2 else "iron"


class OreNode(Component):
    def __init__(self, ore_type: str | None = None):
        super().__init__(tags=["ore_node"])
        self.ore_type = ore_type
        self.depleted = False

    def post_initialization(self) -> None:
        if self.ore_type is None:
            self.ore_type = sample_ore_type()

    def mine(self) -> None:
        self.depleted = True

    def pre_actions_step(self, env: Environment) -> None:
        if self.depleted:
            if random.random() < GOLD_REGROW_PROB:
                self.depleted = False
                self.ore_type = "gold"
                return
            if random.random() < IRON_REGROW_PROB:
                self.depleted = False
                self.ore_type = "iron"
                return


class Mine_Iron_Ore(Action):
    def __init__(self):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(),
                Target_Has_Component(OreNode),
                Action_On_Cooldown("mine"),
            ]
        )

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False
        ore = target.get_component(OreNode)
        return ore is not None and ore.ore_type == "iron" and not ore.depleted

    def exec_action(self, actor, target, env, kwargs=None):
        cooldown = actor.get_component(Cooldown)
        if cooldown is not None:
            cooldown.start("mine")
        ore = target.get_component(OreNode)
        assert ore is not None
        ore.mine()
        award_reward(env, actor, IRON_REWARD)
        env.iron_mined = getattr(env, "iron_mined", 0) + 1
        return {"success": True, "ore_type": "iron", "reward": IRON_REWARD}

    def action_description_text(self, actor, target, env) -> str:
        return "Mine Iron Ore."


class Mine_Gold_Ore(Coordinated_Action):
    def __init__(self):
        super().__init__(
            "Mine Gold Ore together.",
            GOLD_MINERS_REQUIRED,
            coordination_key="mine_gold_ore",
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(),
                Target_Has_Component(OreNode),
                Action_On_Cooldown("mine"),
            ],
        )

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if self._cached_result(actor, target, env) is not None:
            return True
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False
        ore = target.get_component(OreNode)
        return ore is not None and ore.ore_type == "gold" and not ore.depleted

    def exec_coordinated_action(self, actor, target, env, participants, kwargs=None):
        ore = target.get_component(OreNode)
        assert ore is not None
        ore.mine()
        rewarded = []
        for participant in participants:
            cooldown = participant.get_component(Cooldown)
            if cooldown is not None:
                cooldown.start("mine")
            award_reward(env, participant, GOLD_REWARD)
            rewarded.append(participant.name)
        env.gold_mined = getattr(env, "gold_mined", 0) + 1
        return {"ore_type": "gold", "miners": rewarded, "reward_each": GOLD_REWARD}

    def action_description_text(self, actor, target, env) -> str:
        return "Mine Gold Ore together."


def run_exp(agent_count: int = MANDATED_NUM_PLAYERS, policy: str = "random", model_name: str = "openai/gpt-4o-mini"):
    assert agent_count == MANDATED_NUM_PLAYERS, "Coop Mining requires exactly 2 players."
    exp_steps = MAX_EPISODE_LENGTH

    entity_tilemap = ASCII_MAP
    entity_tileset = {
        "W": {
            "name": "Rock Wall",
            "tags": ["wall"],
            "components": [
                Collidable(collidable_tags=["wall"]),
                Renderable(
                    sprite_path="",
                    wall_set="src/world_tiles/indoors/wall_sets/bright_rock_wall",
                    z_index=3,
                ),
            ],
        },
        "O": {
            "name": "Ore Vein",
            "tags": ["ore_site"],
            "components": [
                OreNode(),
                Renderable(
                    sprite_path="src/items/materials/misc/iron_ore.png",
                    z_index=2,
                ),
            ],
        },
    }

    entities = tilemap_to_entities(entity_tilemap.replace("P", "."), entity_tileset)
    spawn_positions = find_tile_positions(entity_tilemap, "P")
    for agent_id, (x, y) in enumerate(spawn_positions[:agent_count], start=1):
        agent_policy = (
            LLM_Action_And_Communication_Policy(
                model_key="coop_mining",
                system_prompt=(
                    f"You are Player {agent_id} in Coop Mining. "
                    "Mining iron gives safe solo reward. Mining gold gives much bigger reward, "
                    "but only if another player helps mine the same gold vein within a short window. "
                    "Coordinate when gold is nearby and fall back to iron when coordination is unlikely."
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
                    Mine_Iron_Ore(),
                    Mine_Gold_Ore(),
                ],
                components=[
                    agent_policy,
                    Cooldown(cooldowns={"mine": MINE_COOLDOWN}),
                    Collidable(collidable_tags=["wall"]),
                    Renderable(
                        sprite_path="src/characters/humanoids/human/farmer_man.png",
                        z_index=10,
                    ),
                ],
            )
        )

    env = Simple_2D_Grid_World(
        description="Coop Mining, adapted from MeltingPot into a single-file Word Play environment.",
        entities=entities,
        entity_order=randomize_agent_order,
        observation_radius=3,
    )
    env.reward_func = lambda action_selections, current_env: list(current_env.last_step_rewards)
    env.iron_mined = 0
    env.gold_mined = 0

    for step in range(exp_steps):
        if policy != "human" and not render_step(env, step_delay=0.0):
            break

        env.tick = env.cur_step
        env.last_step_rewards = [0.0] * len(env.agents)

        cur_step_actions = []
        for agent_id, agent in enumerate(env.agents):
            observation = env.observe(agent_id)
            action, info = agent.get_component(Agent_Policy).select_action(observation)
            print(
                f"[step {step}] {agent.name} -> {action} "
                f"| iron_mined={env.iron_mined} gold_mined={env.gold_mined}"
            )
            cur_step_actions.append(action)

        env.step(cur_step_actions)

        if env.cur_step >= MIN_EPISODE_LENGTH and (env.cur_step - MIN_EPISODE_LENGTH) % INTERVAL_LENGTH == 0:
            if random.random() < TERMINATION_PROBABILITY:
                env.truncations = [True] * len(env.agents)
                break

        if env.cur_step >= MAX_EPISODE_LENGTH:
            env.truncations = [True] * len(env.agents)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-count", type=int, default=MANDATED_NUM_PLAYERS)
    parser.add_argument("--policy", choices=["random", "llm", "human"], default="random")
    parser.add_argument("--model-name", default="openai/gpt-4o-mini")
    args = parser.parse_args()
    if args.policy == "llm":
        LLM_MODEL_REGISTRY.unload("coop_mining")
        LLM_MODEL_REGISTRY.register(
            "coop_mining",
            OpenRouter_Model,
            model_name=args.model_name,
            generation_config={"temperature": 0.3},
        )
    run_exp(
        agent_count=args.agent_count,
        policy=args.policy,
        model_name=args.model_name,
    )
