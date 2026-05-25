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
    Target_Is_Self,
    Target_Not_Self,
)
from word_play.presets.action_policies.llm_action_and_communication import (
    LLM_Action_And_Communication_Policy,
)
from word_play.presets.action_policies.human import Human_Takes_Action
from word_play.presets.action_policies.random_policy import Random_Policy
from word_play.presets.action_validations import Target_Within_Range
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
from word_play.presets.observation.simple_observation import Simple_Observation
from word_play.presets.renderers import Renderable, render_step
from word_play.presets.systems.do_nothing import Do_Nothing
from word_play.presets.systems.reward import award_reward
from benchmarks.text_meltingpot.common import BENCHMARK_STEPS, normalized_probability, normalized_steps
from word_play.utils import tilemap_to_entities
from word_play.utils.tilemap import find_tile_positions


DEFAULT_NUM_PLAYERS = 6
MAX_EPISODE_LENGTH = BENCHMARK_STEPS
MODEL_KEY = "gift_refinements"
NUM_TOKEN_TYPES = 3
MAX_TOKENS_PER_TYPE = 15
TOKEN_REGROW_PROBABILITY = normalized_probability(0.0002)
MIN_EPISODE_LENGTH = normalized_steps(1000)
INTERVAL_LENGTH = normalized_steps(100)
TERMINATION_PROBABILITY = 0.2
GIFT_RANGE = 5
GIFT_MULTIPLIER = 5
SUCCESSFUL_GIFT_REWARD = 10.0


class TokenInventory(Component):
    def __init__(self):
        super().__init__()
        self.tokens = [0] * NUM_TOKEN_TYPES

    @property
    def raw(self) -> int:
        return self.tokens[0]

    @property
    def refined(self) -> int:
        return self.tokens[1]

    @property
    def finest(self) -> int:
        return self.tokens[2]

    def total(self) -> int:
        return sum(self.tokens)

    def add(self, token_type: int, amount: int) -> int:
        space = MAX_TOKENS_PER_TYPE - self.tokens[token_type]
        added = max(0, min(space, amount))
        self.tokens[token_type] += added
        return added

    def remove(self, token_type: int, amount: int = 1) -> bool:
        if self.tokens[token_type] < amount:
            return False
        self.tokens[token_type] -= amount
        return True

    def rawest_available_type(self) -> int | None:
        for token_type, count in enumerate(self.tokens):
            if count > 0:
                return token_type
        return None

    def can_receive(self, token_type: int, amount: int) -> bool:
        return self.tokens[token_type] + amount <= MAX_TOKENS_PER_TYPE

    def consume_all(self) -> int:
        total = self.total()
        self.tokens = [0] * NUM_TOKEN_TYPES
        return total


class TokenPatch(Component):
    def __init__(self):
        super().__init__(tags=["token_patch"])
        self.available = False

    def post_initialization(self) -> None:
        self.available = random.random() < TOKEN_REGROW_PROBABILITY * 10
        self._sync()

    def pre_actions_step(self, env: Environment) -> None:
        if not self.available and random.random() < TOKEN_REGROW_PROBABILITY:
            self.available = True
            self._sync()

    def post_actions_step(self, env: Environment) -> None:
        if not self.available:
            return
        for agent in env.agents:
            if agent.position != self.entity.position:
                continue
            inventory = agent.get_component(TokenInventory)
            if inventory is not None and inventory.add(0, 1):
                self.available = False
                self._sync()
                env.gift_refinement_events.append(f"{agent.name} picked up a raw token.")
            return

    def _sync(self) -> None:
        self.entity.name = "Raw Token" if self.available else "Empty Token Site"
        for tag in ("token", "raw_token", "empty_token_site"):
            while tag in self.entity.tags:
                self.entity.tags.remove(tag)
        self.entity.tags.append("raw_token" if self.available else "empty_token_site")
        if self.available:
            self.entity.tags.append("token")
        renderable = self.entity.get_component(Renderable)
        if renderable is not None:
            renderable.visible = self.available


class HasAnyToken(Action_Validation):
    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        inventory = actor.get_component(TokenInventory)
        return inventory is not None and inventory.total() > 0


class TargetCanReceiveGift(Action_Validation):
    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        actor_inventory = actor.get_component(TokenInventory)
        target_inventory = target_entity.get_component(TokenInventory)
        if actor_inventory is None or target_inventory is None:
            return False
        token_type = actor_inventory.rawest_available_type()
        if token_type is None:
            return False
        receive_type = min(token_type + 1, NUM_TOKEN_TYPES - 1)
        amount = GIFT_MULTIPLIER if token_type < NUM_TOKEN_TYPES - 1 else 1
        return target_inventory.can_receive(receive_type, amount)


class RefineAndGift(Action):
    def __init__(self):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Within_Range(GIFT_RANGE),
                HasAnyToken(),
                TargetCanReceiveGift(),
            ]
        )

    def exec_action(self, actor: Entity, target_entity: Entity, env: Environment, kwargs=None):
        actor_inventory = actor.get_component(TokenInventory)
        target_inventory = target_entity.get_component(TokenInventory)
        token_type = actor_inventory.rawest_available_type()
        receive_type = min(token_type + 1, NUM_TOKEN_TYPES - 1)
        amount = GIFT_MULTIPLIER if token_type < NUM_TOKEN_TYPES - 1 else 1
        actor_inventory.remove(token_type)
        added = target_inventory.add(receive_type, amount)
        if SUCCESSFUL_GIFT_REWARD:
            award_reward(env, actor, SUCCESSFUL_GIFT_REWARD)
        env.gift_refinement_events.append(
            f"{actor.name} gifted one type-{token_type} token to {target_entity.name}; "
            f"{target_entity.name} received {added} type-{receive_type} token(s)."
        )
        return {
            "gifted_token_type": token_type,
            "received_token_type": receive_type,
            "received_amount": added,
            "reward": SUCCESSFUL_GIFT_REWARD,
        }

    def action_description_text(self, actor: Entity, target_entity: Entity, env: Environment) -> str:
        return f"Refine and gift your rawest token to {target_entity.name}."


class ConsumeTokens(Action):
    def __init__(self):
        super().__init__(validation_rules=[Target_Is_Self(), HasAnyToken()])

    def exec_action(self, actor: Entity, target_entity: Entity, env: Environment, kwargs=None):
        inventory = actor.get_component(TokenInventory)
        reward = inventory.consume_all()
        award_reward(env, actor, reward)
        return {"consumed_tokens": reward, "reward": reward}

    def action_description_text(self, actor: Entity, target_entity: Entity, env: Environment) -> str:
        return "Consume all tokens in your inventory for reward."


class GiftRefinementManager(Component):
    def pre_actions_step(self, env: Environment) -> None:
        env.gift_refinement_events = []

    def post_actions_step(self, env: Environment) -> None:
        next_step = env.cur_step + 1
        if next_step >= MAX_EPISODE_LENGTH:
            env.truncations = [True] * len(env.agents)
        elif next_step >= MIN_EPISODE_LENGTH and next_step % INTERVAL_LENGTH == 0:
            if random.random() < TERMINATION_PROBABILITY:
                env.truncations = [True] * len(env.agents)


def _relative_text(actor: Entity, target: Entity) -> str:
    dx = target.position.x - actor.position.x
    dy = target.position.y - actor.position.y
    return f"relative=({dx:+d},{dy:+d}), distance={abs(dx) + abs(dy)}"


def _direction_text(actor: Entity, target: Entity) -> str:
    dx = target.position.x - actor.position.x
    dy = target.position.y - actor.position.y
    moves = []
    if dx > 0:
        moves.append("Move right")
    elif dx < 0:
        moves.append("Move left")
    if dy > 0:
        moves.append("Move up")
    elif dy < 0:
        moves.append("Move down")
    return " or ".join(moves) if moves else "already here"


def _token_text(entity: Entity) -> str:
    inventory = entity.get_component(TokenInventory)
    if inventory is None:
        return "raw=0, refined=0, finest=0"
    return f"raw={inventory.raw}, refined={inventory.refined}, finest={inventory.finest}"


def _important_actions_text(possible_actions: list) -> str:
    matches = [
        f"  [{idx}] {selection}"
        for idx, selection in enumerate(possible_actions)
        if "Refine and gift" in str(selection) or "Consume all tokens" in str(selection)
    ]
    if not matches:
        return "IMPORTANT AVAILABLE ACTIONS: no gift or consume action is valid this turn."
    return "IMPORTANT AVAILABLE ACTIONS:\n" + "\n".join(matches)


def _format_gift_entities(nearby_entities: list[Entity], agent: Entity) -> str:
    lines = ["VISIBLE GIFT ENTITIES:"]
    ordered = sorted(
        nearby_entities,
        key=lambda entity: (
            abs(entity.position.x - agent.position.x) + abs(entity.position.y - agent.position.y),
            entity.name,
        ),
    )
    for entity in ordered:
        if entity is agent:
            continue
        if entity.is_agent:
            lines.append(f"- {entity.name}: {_relative_text(agent, entity)}, tokens=({_token_text(entity)})")
        elif "raw_token" in entity.tags:
            lines.append(f"- Raw Token: {_relative_text(agent, entity)}; direction={_direction_text(agent, entity)}")
    if len(lines) == 1:
        return "VISIBLE GIFT ENTITIES: none"
    return "\n".join(lines)


class GiftRefinementsWorld(Simple_2D_Grid_World):
    def observe(self, agent_id: int):
        agent = self.agents[agent_id]
        nearby_entities = self.entities_in_observation_square(agent.position)
        possible_actions = self.possible_actions(agent)
        inventory = agent.get_component(TokenInventory)
        total = inventory.total() if inventory is not None else 0
        raw_tokens = [entity for entity in nearby_entities if "raw_token" in entity.tags]
        lines = [
            "GIFT REFINEMENT STATUS:",
            f"  your_tokens: {_token_text(agent)}",
            f"  consume_reward_if_used_now: {total}",
            "  mechanics: step onto Raw Token to pick it up; gift your rawest token to a nearby player for +10; consume tokens for their stored value.",
        ]
        if raw_tokens:
            nearest = min(
                raw_tokens,
                key=lambda entity: abs(entity.position.x - agent.position.x)
                + abs(entity.position.y - agent.position.y),
            )
            lines.append(f"  nearest_raw_token: {_relative_text(agent, nearest)}; direction={_direction_text(agent, nearest)}")
        return Simple_Observation(
            possible_actions=possible_actions,
            nearby_entities=nearby_entities,
            agent=agent,
            last_reward=self.last_rewards[agent_id],
            info=self.infos[agent_id],
            observation_radius=self.observation_radius,
            extra_sections=("\n".join(lines), _important_actions_text(possible_actions)),
            nearby_entities_formatter=lambda entities, current_agent: _format_gift_entities(entities, current_agent),
        )


def run_exp(agent_count: int = DEFAULT_NUM_PLAYERS, policy: str = "random", model_name: str = "openai/gpt-4o-mini"):
    if policy == "llm":
        LLM_MODEL_REGISTRY.unload(MODEL_KEY)
        LLM_MODEL_REGISTRY.register(
            MODEL_KEY,
            OpenRouter_Model,
            model_name=model_name,
            generation_config={"temperature": 0.3},
        )

    entity_tilemap = """
    WWWWWWWWWWWWWWWWWWWWWWWWWWW
    WTTTTTTTTTTTTTTTTTTTTTTTTTW
    WTPTTTTTTTTTPTTTTTPTTTTTPTW
    WTTTTTTTTWTTTTTTTTTTTTTTTTW
    WTTTTTTTTWTTTTTTTTTTWTTTTTW
    WTTTTTTTTWTTTTTTTTTTWTTTTTW
    WTTTTTTTTWWWWWWWTTTTWTTTPTW
    WTPTWWTTTTWTTTTTTTTTWTTTTTW
    WTTTTTTTTTWTTPTTTTTTTTTTTTW
    WTTTTTTTTTWTTTTTWWWTTTTTTTW
    WTTTTTTTTTWTTTTTTTTTTTTTTTW
    WTTTTTTTTTTTTTTTTTTTTTTTPTW
    WTPTTTWWWTTTTTTWWWWWWWWTTTW
    WTTWWWWTTTTTTTTTTTTTTTTTTTW
    WTTTTTWTTTTWTTTTTPTTTTTTTTW
    WTTTTTWTTTTWTTTTTTTTTTTTPTW
    WTTTTTWTTTTTWTTTTTTTTWTTTTW
    WTTTTTTWTTTTTWWWWTTTTWTTTTW
    WTPTTTTTWTTTTTTTTTTTTWTTTTW
    WTTTTTTTTWTTTPTTTTTTTTTTPTW
    WTTTTTTTTTWTTTTTTTTWTTTTTTW
    WTTTTWTTTTTTTTTTTTTWTTTTTTW
    WTTTTWTTTTTTTTTWWWWWWWWTTTW
    WTTTTWTTTTTTTTTTTTWTTTTTTTW
    WTPTTTTTTPTTTTTTTPTTTTTTPTW
    WTTTTTTTTTTTTTTTTTTTTTTTTTW
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
        "T": {
            "name": "Empty Token Site",
            "tags": ["empty_token_site"],
            "components": [
                TokenPatch(),
                Renderable(
                    sprite_path="src/items/materials/valuables/gold_coin.png",
                    z_index=4,
                    visible=False,
                ),
            ],
        },
    }

    entities = tilemap_to_entities(entity_tilemap.replace("P", "."), entity_tileset)
    spawn_positions = find_tile_positions(entity_tilemap, "P")
    random.shuffle(spawn_positions)

    entities.append(
        Entity(
            name="Gift Refinement Manager",
            position=Position_2D(-1, -1),
            components=[GiftRefinementManager()],
        )
    )

    for agent_id, (x, y) in enumerate(spawn_positions[:agent_count], start=1):
        agent_policy = (
            LLM_Action_And_Communication_Policy(
                model_key=MODEL_KEY,
                system_prompt=(
                    f"You are Player {agent_id} in Gift Refinements. "
                    "Pick up raw tokens, refine value by gifting tokens to others, "
                    "and consume tokens for reward when you have accumulated value."
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
                    RefineAndGift(),
                    ConsumeTokens(),
                ],
                components=[
                    agent_policy,
                    TokenInventory(),
                    Collidable(collidable_tags=["wall"]),
                    Renderable(
                        sprite_path="src/characters/humanoids/human/farmer_man.png",
                        z_index=10,
                    ),
                ],
            )
        )

    env = GiftRefinementsWorld(
        description="Gift Refinements, adapted from MeltingPot into a single-file Word Play environment.",
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
        for event in getattr(env, "gift_refinement_events", []):
            print(f"[step {step}] gift_refinements -> {event}")

        if all(env.truncations):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-count", type=int, default=DEFAULT_NUM_PLAYERS)
    parser.add_argument("--policy", choices=["random", "llm", "human"], default="random")
    parser.add_argument("--model-name", default="openai/gpt-4o-mini")
    args = parser.parse_args()
    run_exp(agent_count=args.agent_count, policy=args.policy, model_name=args.model_name)
