from __future__ import annotations

import argparse
import random

from word_play.core import (
    Action,
    Action_Validation,
    Agent_Policy,
    Component,
    Entity,
    Environment,
    Target_Is_Nearby,
    Target_Is_Self,
    Target_Not_Self,
)
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
from word_play.presets.systems.reward import award_reward
from benchmarks.text_meltingpot.common import BENCHMARK_STEPS, normalized_probability, normalized_steps
from word_play.utils import tilemap_to_entities
from word_play.utils.tilemap import find_tile_positions


DEFAULT_NUM_PLAYERS = 16
MAX_EPISODE_LENGTH = BENCHMARK_STEPS
MODEL_KEY = "fruit_market"
MAX_OFFER_QUANTITY = 3
TRADING_RADIUS = 4
HUNGER_DELAY = normalized_steps(50, minimum=2)
TREE_REGROW_TIME = normalized_steps(50)
WATER_STAMINA_COST = 1
MAX_STAMINA = 18
FRUIT_SPAWN_PROBABILITY = normalized_probability(0.1)


def grid_distance(first: Entity, second: Entity) -> int:
    return abs(first.position.x - second.position.x) + abs(first.position.y - second.position.y)


def position_is_blocked(position: Position_2D, actor: Entity, env: Environment) -> bool:
    for entity in env.state.entities:
        if entity is actor or entity.position != position:
            continue
        collidable = entity.get_component(Collidable)
        if collidable is not None and any(tag in actor.get_component(Collidable).collidable_tags for tag in entity.tags):
            return True
    return False


class FruitInventory(Component):
    def __init__(self):
        super().__init__()
        self.apples = 0
        self.bananas = 0

    def count(self, fruit: str) -> int:
        return self.apples if fruit == "apple" else self.bananas

    def add(self, fruit: str, amount: int = 1) -> None:
        if fruit == "apple":
            self.apples += amount
        else:
            self.bananas += amount

    def remove(self, fruit: str, amount: int = 1) -> bool:
        if self.count(fruit) < amount:
            return False
        if fruit == "apple":
            self.apples -= amount
        else:
            self.bananas -= amount
        return True


class FruitMarketAvatar(Component):
    def __init__(self, specialty: str):
        super().__init__(tags=[f"{specialty}_farmer"])
        self.specialty = specialty
        self.favorite = "banana" if specialty == "apple" else "apple"
        self.hunger = 0
        self.stamina = MAX_STAMINA
        self.current_offer: dict[str, int | str] | None = None

    def harvest_probability(self, fruit: str) -> float:
        return 1.0 if fruit == self.specialty else 0.04

    def reward_for_eating(self, fruit: str) -> float:
        return 8.0 if fruit == self.favorite else 1.0

    def post_actions_step(self, env: Environment) -> None:
        self.hunger += 1
        if self.hunger >= HUNGER_DELAY:
            self.stamina = max(0, self.stamina - 1)
            if self.stamina == 0:
                award_reward(env, self.entity, -1.0)


class FruitTree(Component):
    def __init__(self, fruit: str | None = None):
        super().__init__(tags=["fruit_tree"])
        self.fruit = fruit
        self.available = fruit is not None
        self.regrow_timer = 0

    def post_initialization(self) -> None:
        if self.fruit is None:
            if random.random() < FRUIT_SPAWN_PROBABILITY:
                self.fruit = random.choice(["apple", "banana"])
                self.available = True
        self._sync()

    def pre_actions_step(self, env: Environment) -> None:
        if self.fruit is None or self.available:
            return
        self.regrow_timer -= 1
        if self.regrow_timer <= 0:
            self.available = True
            self._sync()

    def post_actions_step(self, env: Environment) -> None:
        if self.fruit is None or not self.available:
            return
        for agent in env.agents:
            if agent.position != self.entity.position:
                continue
            avatar = agent.get_component(FruitMarketAvatar)
            inventory = agent.get_component(FruitInventory)
            if avatar is None or inventory is None:
                continue
            if random.random() <= avatar.harvest_probability(self.fruit):
                inventory.add(self.fruit)
                self.available = False
                self.regrow_timer = TREE_REGROW_TIME
                self._sync()
                env.fruit_market_events.append(f"{agent.name} harvested one {self.fruit}.")
            else:
                env.fruit_market_events.append(f"{agent.name} failed to harvest {self.fruit}.")
            return

    def _sync(self) -> None:
        if self.fruit is None:
            self.entity.name = "Empty Tree Site"
        else:
            self.entity.name = f"{self.fruit.title()} Tree" + (" with fruit" if self.available else " regrowing")


class WaterCost(Component):
    def __init__(self):
        super().__init__(tags=["water"])

    def post_actions_step(self, env: Environment) -> None:
        for agent in env.agents:
            if agent.position != self.entity.position:
                continue
            avatar = agent.get_component(FruitMarketAvatar)
            if avatar is None:
                continue
            avatar.stamina = max(0, avatar.stamina - WATER_STAMINA_COST)
            env.fruit_market_events.append(f"{agent.name} spent stamina crossing water.")
            if avatar.stamina == 0:
                award_reward(env, agent, -1.0)


class HasFruit(Action_Validation):
    def __init__(self, fruit: str):
        self.fruit = fruit

    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        inventory = actor.get_component(FruitInventory)
        return inventory is not None and inventory.count(self.fruit) > 0


class EatFruit(Action):
    def __init__(self, fruit: str):
        super().__init__(validation_rules=[Target_Is_Self(), HasFruit(fruit)])
        self.fruit = fruit

    def exec_action(self, actor: Entity, target_entity: Entity, env: Environment, kwargs=None):
        inventory = actor.get_component(FruitInventory)
        avatar = actor.get_component(FruitMarketAvatar)
        if inventory is None or avatar is None or not inventory.remove(self.fruit):
            return {"success": False}
        reward = avatar.reward_for_eating(self.fruit)
        avatar.hunger = 0
        avatar.stamina = min(MAX_STAMINA, avatar.stamina + 3)
        award_reward(env, actor, reward)
        return {"ate": self.fruit, "reward": reward, "stamina": avatar.stamina}

    def action_description_text(self, actor: Entity, target_entity: Entity, env: Environment) -> str:
        return f"Eat one {self.fruit}."


class SetTradeOffer(Action):
    def __init__(self, give_fruit: str, give_amount: int, want_amount: int):
        super().__init__(validation_rules=[Target_Is_Self()])
        self.give_fruit = give_fruit
        self.want_fruit = "banana" if give_fruit == "apple" else "apple"
        self.give_amount = give_amount
        self.want_amount = want_amount

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        if not super().is_valid(actor, target, env, kwargs=kwargs):
            return False
        inventory = actor.get_component(FruitInventory)
        return inventory is not None and inventory.count(self.give_fruit) >= self.give_amount

    def exec_action(self, actor, target, env, kwargs=None):
        avatar = actor.get_component(FruitMarketAvatar)
        if avatar is None:
            return {"success": False}
        avatar.current_offer = {
            "give_fruit": self.give_fruit,
            "give_amount": self.give_amount,
            "want_fruit": self.want_fruit,
            "want_amount": self.want_amount,
        }
        return {"offer": dict(avatar.current_offer)}

    def action_description_text(self, actor, target, env) -> str:
        return (
            f"Offer {self.give_amount} {self.give_fruit}"
            f" for {self.want_amount} {self.want_fruit}."
        )


class CancelTradeOffer(Action):
    def __init__(self):
        super().__init__(validation_rules=[Target_Is_Self()])

    def exec_action(self, actor, target, env, kwargs=None):
        avatar = actor.get_component(FruitMarketAvatar)
        if avatar is not None:
            avatar.current_offer = None
        return {"cancelled_trade_offer": True}

    def action_description_text(self, actor, target, env) -> str:
        return "Cancel current trade offer."


class ShovePlayer(Action):
    def __init__(self, direction: int):
        super().__init__(validation_rules=[Target_Not_Self(), Target_Is_Nearby()])
        self.direction = direction

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        return super().is_valid(actor, target, env, kwargs=kwargs) and target.get_component(FruitMarketAvatar) is not None

    def exec_action(self, actor, target, env, kwargs=None):
        dx = target.position.x - actor.position.x
        dy = target.position.y - actor.position.y
        if abs(dx) + abs(dy) == 0:
            dx = self.direction
        if abs(dx) > abs(dy):
            step_x, step_y = (1 if dx > 0 else -1) * self.direction, 0
        else:
            step_x, step_y = 0, (1 if dy > 0 else -1) * self.direction
        destination = Position_2D(target.position.x + step_x, target.position.y + step_y)
        if target.has_component(Collidable) and position_is_blocked(destination, target, env):
            return {"shoved": target.name, "blocked": True}
        target.position = destination
        return {"shoved": target.name, "to": str(destination)}

    def action_description_text(self, actor, target, env) -> str:
        verb = "Shove" if self.direction == 1 else "Pull"
        return f"{verb} {target.name}."


class FruitMarketManager(Component):
    def pre_actions_step(self, env: Environment) -> None:
        env.fruit_market_events = []

    def post_actions_step(self, env: Environment) -> None:
        self._resolve_trades(env)
        next_step = env.cur_step + 1
        if next_step >= MAX_EPISODE_LENGTH:
            env.truncations = [True] * len(env.agents)

    def _resolve_trades(self, env: Environment) -> None:
        agents = env.agents.copy()
        traded: set[Entity] = set()
        for actor in agents:
            if actor in traded:
                continue
            actor_state = actor.get_component(FruitMarketAvatar)
            actor_inventory = actor.get_component(FruitInventory)
            if actor_state is None or actor_inventory is None or not actor_state.current_offer:
                continue
            for partner in agents:
                if partner is actor or partner in traded or grid_distance(actor, partner) > TRADING_RADIUS:
                    continue
                partner_state = partner.get_component(FruitMarketAvatar)
                partner_inventory = partner.get_component(FruitInventory)
                if partner_state is None or partner_inventory is None or not partner_state.current_offer:
                    continue
                if self._offers_match(actor_state.current_offer, partner_state.current_offer):
                    if not actor_inventory.remove(actor_state.current_offer["give_fruit"], actor_state.current_offer["give_amount"]):
                        continue
                    if not partner_inventory.remove(partner_state.current_offer["give_fruit"], partner_state.current_offer["give_amount"]):
                        actor_inventory.add(actor_state.current_offer["give_fruit"], actor_state.current_offer["give_amount"])
                        continue
                    actor_inventory.add(partner_state.current_offer["give_fruit"], partner_state.current_offer["give_amount"])
                    partner_inventory.add(actor_state.current_offer["give_fruit"], actor_state.current_offer["give_amount"])
                    env.fruit_market_events.append(f"{actor.name} traded with {partner.name}.")
                    actor_state.current_offer = None
                    partner_state.current_offer = None
                    traded.update({actor, partner})
                    break

    def _offers_match(self, first: dict, second: dict) -> bool:
        return (
            first["give_fruit"] == second["want_fruit"]
            and first["want_fruit"] == second["give_fruit"]
            and first["give_amount"] == second["want_amount"]
            and first["want_amount"] == second["give_amount"]
        )


def run_exp(agent_count: int = DEFAULT_NUM_PLAYERS, policy: str = "random", model_name: str = "openai/gpt-4o-mini"):
    entity_tilemap = """
    WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
    WtttttttttttttttttttttttttttttW
    WtP..t..a.P..t.P..b..t..P..tttW
    Wtt..P...t.....P.....t...P.tttW
    Wt....RRRRRRRRRRRRRRRRR....tttW
    WtP.tRtttttttttttttttttRtP.tttW
    Wt...RtP...a.P.t.P.b..PR...tttW
    WtP.tRtttttttttttttttttRtP.tttW
    Wt....RRRRRRRRRRRRRRRRR....tttW
    Wt..P..t.....P..t.....P..tttttW
    Wt..P....b.P..t..P.a..t..P.tttW
    WtttttttttttttttttttttttttttttW
    WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
    """

    if policy == "llm":
        LLM_MODEL_REGISTRY.unload(MODEL_KEY)
        LLM_MODEL_REGISTRY.register(
            MODEL_KEY,
            OpenRouter_Model,
            model_name=model_name,
            generation_config={"temperature": 0.3},
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
        "t": {
            "name": "Potential Fruit Tree",
            "tags": ["tree_site"],
            "components": [
                FruitTree(),
                Renderable(
                    sprite_path="src/world_tiles/outdoors/nature/ripe_fruit_tree.png",
                    z_index=3,
                ),
            ],
        },
        "a": {
            "name": "Apple Tree with fruit",
            "tags": ["tree_site"],
            "components": [
                FruitTree("apple"),
                Renderable(
                    sprite_path="src/world_tiles/outdoors/nature/ripe_fruit_tree.png",
                    z_index=3,
                ),
            ],
        },
        "b": {
            "name": "Banana Tree with fruit",
            "tags": ["tree_site"],
            "components": [
                FruitTree("banana"),
                Renderable(
                    sprite_path="src/world_tiles/outdoors/nature/ripe_fruit_tree.png",
                    z_index=3,
                ),
            ],
        },
        "R": {
            "name": "River",
            "components": [
                WaterCost(),
                Renderable(
                    sprite_path="src/world_tiles/indoors/floors/shallow_water_tile.png",
                    z_index=1,
                ),
            ],
        },
    }

    entities = tilemap_to_entities(entity_tilemap.replace("P", "."), entity_tileset)
    spawn_positions = find_tile_positions(entity_tilemap, "P")
    random.shuffle(spawn_positions)

    entities.append(
        Entity(
            name="Fruit Market Manager",
            position=Position_2D(-1, -1),
            components=[FruitMarketManager()],
        )
    )

    for agent_id, (x, y) in enumerate(spawn_positions[:agent_count], start=1):
        specialty = "apple" if agent_id % 2 == 0 else "banana"
        agent_policy = (
            LLM_Action_And_Communication_Policy(
                model_key=MODEL_KEY,
                system_prompt=(
                    f"You are Player {agent_id} in Fruit Market. "
                    f"You are a {specialty} farmer: you harvest {specialty}s reliably, "
                    f"but your favorite food is {'banana' if specialty == 'apple' else 'apple'} worth 8 reward. "
                    "Harvest fruit by standing on trees, eat to reset hunger, and barter with nearby players."
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
                    EatFruit("apple"),
                    EatFruit("banana"),
                    CancelTradeOffer(),
                    ShovePlayer(1),
                    ShovePlayer(-1),
                    *[
                        SetTradeOffer(give_fruit, give_amount, want_amount)
                        for give_fruit in ("apple", "banana")
                        for give_amount in range(1, MAX_OFFER_QUANTITY + 1)
                        for want_amount in range(1, MAX_OFFER_QUANTITY + 1)
                    ],
                ],
                components=[
                    agent_policy,
                    FruitInventory(),
                    FruitMarketAvatar(specialty),
                    Collidable(collidable_tags=["wall"]),
                    Renderable(
                        sprite_path="src/characters/humanoids/human/farmer_man.png",
                        z_index=10,
                    ),
                ],
            )
        )

    env = Simple_2D_Grid_World(
        description="Fruit Market, adapted from MeltingPot into a single-file Word Play environment.",
        entities=entities,
        entity_order=randomize_agent_order,
        observation_radius=5,
    )
    env.reward_func = lambda action_selections, current_env: list(current_env.last_step_rewards)

    for step in range(MAX_EPISODE_LENGTH):
        if policy != "human" and not render_step(env, step_delay=0.0):
            break

        env.last_step_rewards = [0.0] * len(env.agents)

        cur_step_actions = []
        for agent_id, agent in enumerate(env.agents):
            observation = env.observe(agent_id)
            action, info = agent.get_component(Agent_Policy).select_action(observation)
            print(f"[step {step}] {agent.name} -> {action}")
            cur_step_actions.append(action)

        env.step(cur_step_actions)
        for event in getattr(env, "fruit_market_events", []):
            print(f"[step {step}] fruit_market -> {event}")

        if all(env.truncations):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-count", type=int, default=DEFAULT_NUM_PLAYERS)
    parser.add_argument("--policy", choices=["random", "llm", "human"], default="random")
    parser.add_argument("--model-name", default="openai/gpt-4o-mini")
    args = parser.parse_args()
    run_exp(agent_count=args.agent_count, policy=args.policy, model_name=args.model_name)
