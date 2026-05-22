from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model_params import SUPPORTED_MODEL_MODES, make_policy, register_model

from word_play.core import (
    Action,
    Action_Selection,
    Action_Validation,
    Agent_Policy,
    Component,
    Entity,
    Environment,
    Observation,
    Target_Is_Nearby,
    Target_Not_Self,
)
from word_play.presets.action_validations import Target_Has_Component
from word_play.presets.entity_orderings import randomize_agent_order
from word_play.presets.environments.simple_env_reset_mixin import Simple_Env_Reset_Mixin
from word_play.presets.movement.simple_2d_grid import (
    ADJACENT_2D_MOVEMENT_SYSTEM,
    Collidable,
    Move_Down,
    Move_Left,
    Move_Right,
    Move_Up,
    Position_2D,
)
from word_play.presets.observation.simple_observation import Simple_Observation
from word_play.presets.observation.utils import entity_state_to_str
from word_play.presets.systems.do_nothing import Do_Nothing


GAME_NAME = "Territory"
DEFAULT_MAX_STEPS = 30
WIDTH = 5
HEIGHT = 5
OBSERVATION_RADIUS = 2
FULL_OBSERVATION_MODES = {"human_action", "random_action"}

OBJECTIVE_TEXT = "\n".join(
    [
        "OBJECTIVE: Territory",
        "  Move around the grid and control territory tiles.",
        "  Claim neutral tiles to start scoring from them.",
        "  Break rival tiles until their control reaches 0 before claiming them.",
        "  Repair damaged tiles you own.",
        "  Zap adjacent rival agents for a small immediate reward swing.",
        "  Each controlled tile gives its owner +1 reward at the end of each step.",
    ]
)

LLM_GAME_RULES = (
    "Maximize cumulative reward by controlling territory tiles. Each owned tile gives +1 reward per step. "
    "Claim neutral tiles, break rival tiles before claiming them, repair damaged tiles you own, and zap "
    "adjacent rivals when it improves your position."
)


class Territory_Tile(Component):
    def __init__(
        self,
        owner_id: str | None = None,
        *,
        control: int | None = None,
        max_control: int = 2,
        reward_per_step: float = 1.0,
    ):
        super().__init__(tags=["territory_tile"])
        self.owner_id = owner_id
        self.max_control = max_control
        self.control = max_control if owner_id is not None and control is None else (control or 0)
        self.reward_per_step = reward_per_step

    def claim(self, actor: Entity) -> dict:
        previous_owner = self.owner_id
        self.owner_id = actor.name
        self.control = 1
        return {
            "event": "claimed",
            "previous_owner": previous_owner,
            "owner": self.owner_id,
            "control": self.control,
            "max_control": self.max_control,
        }

    def repair(self, actor: Entity) -> dict:
        previous_control = self.control
        self.control = min(self.max_control, self.control + 1)
        return {
            "event": "repaired",
            "owner": self.owner_id,
            "previous_control": previous_control,
            "control": self.control,
            "max_control": self.max_control,
        }

    def break_control(self, actor: Entity) -> dict:
        previous_owner = self.owner_id
        previous_control = self.control
        self.control = max(0, self.control - 1)
        event = "damaged"
        if self.control == 0:
            self.owner_id = None
            event = "neutralized"

        return {
            "event": event,
            "previous_owner": previous_owner,
            "owner": self.owner_id,
            "previous_control": previous_control,
            "control": self.control,
            "max_control": self.max_control,
        }


class Territory_Player(Component):
    def __init__(self, zap_reward: float = 0.5, zapped_penalty: float = -0.5):
        super().__init__(
            tags=["territory_player"],
            actions=[
                Claim_Territory(),
                Repair_Territory(),
                Break_Territory(),
                Zap_Territory_Player(),
            ],
        )
        self.zap_reward = zap_reward
        self.zapped_penalty = zapped_penalty
        self.pending_reward = 0.0
        self.zaps_landed = 0
        self.times_zapped = 0

    def pre_actions_step(self, env: Environment) -> None:
        self.pending_reward = 0.0


class Target_Can_Be_Claimed(Action_Validation):
    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        tile = target_entity.get_component(Territory_Tile)
        return tile is not None and tile.owner_id is None


class Target_Is_Damaged_Own_Territory(Action_Validation):
    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        tile = target_entity.get_component(Territory_Tile)
        return tile is not None and tile.owner_id == actor.name and tile.control < tile.max_control


class Target_Is_Enemy_Territory(Action_Validation):
    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        tile = target_entity.get_component(Territory_Tile)
        return (
            tile is not None
            and tile.owner_id is not None
            and tile.owner_id != actor.name
            and tile.control > 0
        )


class Claim_Territory(Action):
    def __init__(self):
        super().__init__(validation_rules=[Target_Is_Nearby(), Target_Can_Be_Claimed()])

    def exec_action(
        self, actor: Entity, target_entity: Entity, env: Environment, kwargs: dict | None
    ) -> dict | None:
        return target_entity.get_component(Territory_Tile).claim(actor)

    def action_description_text(
        self, actor: Entity, target_entity: Entity, env: Environment
    ) -> str:
        return f"Claim {target_entity.name} at {target_entity.position}."


class Repair_Territory(Action):
    def __init__(self):
        super().__init__(validation_rules=[Target_Is_Nearby(), Target_Is_Damaged_Own_Territory()])

    def exec_action(
        self, actor: Entity, target_entity: Entity, env: Environment, kwargs: dict | None
    ) -> dict | None:
        return target_entity.get_component(Territory_Tile).repair(actor)

    def action_description_text(
        self, actor: Entity, target_entity: Entity, env: Environment
    ) -> str:
        tile = target_entity.get_component(Territory_Tile)
        return (
            f"Repair {target_entity.name} at {target_entity.position} "
            f"({tile.control}/{tile.max_control})."
        )


class Break_Territory(Action):
    def __init__(self):
        super().__init__(validation_rules=[Target_Is_Nearby(), Target_Is_Enemy_Territory()])

    def exec_action(
        self, actor: Entity, target_entity: Entity, env: Environment, kwargs: dict | None
    ) -> dict | None:
        return target_entity.get_component(Territory_Tile).break_control(actor)

    def action_description_text(
        self, actor: Entity, target_entity: Entity, env: Environment
    ) -> str:
        tile = target_entity.get_component(Territory_Tile)
        return (
            f"Break {target_entity.name} at {target_entity.position} "
            f"owned by {tile.owner_id} ({tile.control}/{tile.max_control})."
        )


class Zap_Territory_Player(Action):
    def __init__(self):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(),
                Target_Has_Component(Territory_Player),
            ]
        )

    def exec_action(
        self, actor: Entity, target_entity: Entity, env: Environment, kwargs: dict | None
    ) -> dict | None:
        actor_state = actor.get_component(Territory_Player)
        target_state = target_entity.get_component(Territory_Player)
        actor_state.pending_reward += actor_state.zap_reward
        target_state.pending_reward += actor_state.zapped_penalty
        actor_state.zaps_landed += 1
        target_state.times_zapped += 1
        return {
            "event": "zapped",
            "target": target_entity.name,
            "zap_reward": actor_state.zap_reward,
            "target_penalty": actor_state.zapped_penalty,
        }

    def action_description_text(
        self, actor: Entity, target_entity: Entity, env: Environment
    ) -> str:
        return f"Zap {target_entity.name}."


def territory_tiles(env: Environment) -> list[Entity]:
    return [
        entity
        for entity in env.state.entities
        if entity.get_component(Territory_Tile) is not None
    ]


def territory_scores(env: Environment) -> dict[str, float]:
    scores = {agent.name: 0.0 for agent in env.agents}
    for entity in territory_tiles(env):
        tile = entity.get_component(Territory_Tile)
        if tile.owner_id in scores:
            scores[tile.owner_id] += tile.reward_per_step
    return scores


def territory_owned_counts(env: Environment) -> dict[str, int]:
    counts = {agent.name: 0 for agent in env.agents}
    for entity in territory_tiles(env):
        tile = entity.get_component(Territory_Tile)
        if tile.owner_id in counts:
            counts[tile.owner_id] += 1
    return counts


def territory_control_reward(
    agent_actions: list[Action_Selection],
    env: Environment,
) -> list[float]:
    scores = territory_scores(env)
    rewards = [scores[agent.name] for agent in env.agents]
    for idx, agent in enumerate(env.agents):
        player = agent.get_component(Territory_Player)
        if player is not None:
            rewards[idx] += player.pending_reward
    return rewards


@dataclass(slots=True)
class Territory_LLM_Observation(Observation):
    agent: Entity
    env: Environment
    last_reward: float
    info: dict
    step_count: int
    max_steps: int

    def __str__(self) -> str:
        return "\n\n".join(
            [
                f"REWARD THIS TURN: {self.last_reward} | STEP: {self.step_count}/{self.max_steps}",
                f"YOUR STATE: {format_territory_llm_agent(self.agent, self.env)}",
                format_territory_action_range_entities(self.agent, self.env),
            ]
        )


def format_territory_llm_agent(agent: Entity, env: Environment) -> str:
    player = agent.get_component(Territory_Player)
    controlled_counts = territory_owned_counts(env)
    scores = territory_scores(env)
    return (
        f"position={agent.position}, "
        f"tiles_owned={controlled_counts[agent.name]}, "
        f"step_score={scores[agent.name]}, "
        f"zaps_landed={player.zaps_landed}, "
        f"times_zapped={player.times_zapped}"
    )


def format_territory_action_range_entities(agent: Entity, env: Environment) -> str:
    entities = [
        entity
        for entity in env.entities_near_position(agent.position)
        if entity is not agent and "wall" not in entity.tags
    ]
    entities.sort(key=lambda entity: (_position_sort_key(agent.position, entity.position), entity.name))

    lines = []
    for entity in entities:
        relation = _relative_position_label(agent.position, entity.position)
        tile = entity.get_component(Territory_Tile)
        if tile is not None:
            lines.append(
                f"  - {relation} {entity.name} at {entity.position}: "
                f"{_territory_tile_status(tile, agent)}"
            )
            continue

        if entity.get_component(Territory_Player) is not None:
            lines.append(f"  - {relation} {entity.name} at {entity.position}: rival agent, zappable")

    if not lines:
        return "ACTION-RANGE ENTITIES:\n  None"
    return "ACTION-RANGE ENTITIES:\n" + "\n".join(lines)


def _territory_tile_status(tile: Territory_Tile, agent: Entity) -> str:
    control = f"control={tile.control}/{tile.max_control}"
    if tile.owner_id is None:
        return f"neutral, claimable, {control}"
    if tile.owner_id == agent.name:
        if tile.control < tile.max_control:
            return f"yours, damaged, repairable, {control}"
        return f"yours, {control}"
    return f"owned by {tile.owner_id}, breakable, {control}"


def _relative_position_label(agent_position: Position_2D, target_position: Position_2D) -> str:
    dx = target_position.x - agent_position.x
    dy = target_position.y - agent_position.y
    if dx == 0 and dy == 0:
        return "current"
    if dx == 0 and dy == 1:
        return "up"
    if dx == 0 and dy == -1:
        return "down"
    if dx == -1 and dy == 0:
        return "left"
    if dx == 1 and dy == 0:
        return "right"
    return "nearby"


def _position_sort_key(agent_position: Position_2D, target_position: Position_2D) -> tuple[int, int]:
    order = {
        "current": 0,
        "up": 1,
        "right": 2,
        "down": 3,
        "left": 4,
    }
    return (
        order.get(_relative_position_label(agent_position, target_position), 99),
        abs(target_position.x - agent_position.x) + abs(target_position.y - agent_position.y),
    )


def use_compact_llm_observation(model_mode: str) -> bool:
    return model_mode not in FULL_OBSERVATION_MODES


class Territory_Environment(Simple_Env_Reset_Mixin, Environment):
    def __init__(
        self,
        description: str,
        entities: list[Entity],
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        width: int = WIDTH,
        height: int = HEIGHT,
        observation_radius: int = OBSERVATION_RADIUS,
        llm_observation: bool = False,
    ):
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        self.max_steps = max_steps
        self.width = width
        self.height = height
        self.observation_radius = observation_radius
        self.llm_observation = llm_observation

        super().__init__(
            description=description,
            entities=entities,
            movement_system=ADJACENT_2D_MOVEMENT_SYSTEM,
            reward_func=territory_control_reward,
            entity_order=randomize_agent_order,
        )

    def observe(self, agent_id: int) -> Observation:
        agent = self.agents[agent_id]
        last_reward = self.last_rewards[agent_id]
        normalized_last_reward = 0.0 if last_reward is None else last_reward
        if self.llm_observation:
            return Territory_LLM_Observation(
                possible_actions=self.possible_actions(agent),
                agent=agent,
                env=self,
                last_reward=normalized_last_reward,
                info=self.infos[agent_id],
                step_count=self.cur_step,
                max_steps=self.max_steps,
            )

        return Simple_Observation(
            possible_actions=self.possible_actions(agent),
            nearby_entities=self.entities_in_observation_square(agent.position),
            agent=agent,
            last_reward=normalized_last_reward,
            info=self.infos[agent_id],
            observation_radius=self.observation_radius,
            extra_sections=(
                OBJECTIVE_TEXT,
                self.scoreboard_text(),
                self.map_text(),
            ),
            agent_formatter=lambda current_agent: format_territory_agent(current_agent, self),
            nearby_entities_formatter=format_territory_nearby_entities,
        )

    def environment_start_of_step(self, action_selections: list[Action_Selection]) -> None:
        pass

    def environment_end_of_step(self, action_selections: list[Action_Selection]) -> None:
        scores = territory_scores(self)
        for agent_idx, agent in enumerate(self.agents):
            self.infos[agent_idx]["territory_score"] = scores[agent.name]

        if self.cur_step + 1 >= self.max_steps:
            self.truncations = [True] * len(self.agents)

    def entities_in_observation_square(self, position: Position_2D) -> list[Entity]:
        return [
            entity
            for entity in self.state.entities
            if isinstance(entity.position, Position_2D)
            and abs(entity.position.x - position.x) <= self.observation_radius
            and abs(entity.position.y - position.y) <= self.observation_radius
        ]

    def tile_at(self, x: int, y: int) -> Entity | None:
        for entity in territory_tiles(self):
            if entity.position == Position_2D(x, y):
                return entity
        return None

    def map_text(self) -> str:
        agent_symbols = {
            agent.name: chr(ord("A") + idx)
            for idx, agent in enumerate(self.agents)
        }
        owner_symbols = {
            agent.name: chr(ord("a") + idx)
            for idx, agent in enumerate(self.agents)
        }
        lines = ["MAP: agents=A/B/C, owned tiles=a/b/c, neutral=."]
        for y in reversed(range(self.height)):
            row = []
            for x in range(self.width):
                agents_here = [
                    agent for agent in self.agents if agent.position == Position_2D(x, y)
                ]
                if agents_here:
                    row.append(agent_symbols[agents_here[0].name])
                    continue

                tile_entity = self.tile_at(x, y)
                tile = (
                    tile_entity.get_component(Territory_Tile)
                    if tile_entity is not None
                    else None
                )
                if tile is None or tile.owner_id is None:
                    row.append(".")
                else:
                    row.append(owner_symbols.get(tile.owner_id, "?"))
            lines.append(f"  y={y}  " + " ".join(row))
        lines.append("       " + " ".join(str(x) for x in range(self.width)))
        return "\n".join(lines)

    def scoreboard_text(self) -> str:
        scores = territory_scores(self)
        controlled_counts = territory_owned_counts(self)

        lines = ["SCOREBOARD:"]
        for agent in self.agents:
            player = agent.get_component(Territory_Player)
            lines.append(
                f"  {agent.name}: tiles={controlled_counts[agent.name]}, "
                f"step_score={scores[agent.name]}, "
                f"zaps_landed={player.zaps_landed}, zapped={player.times_zapped}"
            )
        return "\n".join(lines)


def format_territory_nearby_entities(nearby_entities: list[Entity], agent: Entity) -> str:
    lines = []
    for entity in nearby_entities:
        if entity is agent:
            continue

        tile = entity.get_component(Territory_Tile)
        if tile is not None:
            if tile.owner_id is None:
                owner = "neutral"
            elif tile.owner_id == agent.name:
                owner = "you"
            else:
                owner = tile.owner_id
            distance = abs(entity.position.x - agent.position.x) + abs(entity.position.y - agent.position.y)
            range_note = "action range" if distance <= 1 else f"visible, {distance} steps away"
            lines.append(
                f"- {entity.name} at {entity.position}: "
                f"owner={owner}, control={tile.control}/{tile.max_control}, {range_note}"
            )
            continue

        if entity.get_component(Territory_Player) is not None:
            lines.append(f"- Rival agent: {entity.name} at {entity.position}")
            continue

        if "wall" in entity.tags:
            continue

        lines.append(f"- {entity_state_to_str(entity).replace(chr(10), chr(10) + '  ')}")

    return "Nearby Entities:\n" + "\n".join(lines) if lines else "Nearby Entities: None"


def format_territory_agent(agent: Entity, env: Environment) -> str:
    player = agent.get_component(Territory_Player)
    controlled_counts = territory_owned_counts(env)
    scores = territory_scores(env)
    return "\n".join(
        [
            f"name: {agent.name}",
            f"position: {agent.position}",
            f"controlled_tiles: {controlled_counts[agent.name]}",
            f"tile_reward_per_step: {scores[agent.name]}",
            f"zaps_landed: {player.zaps_landed}",
            f"times_zapped: {player.times_zapped}",
        ]
    )


def build_agent(
    *,
    name: str,
    position: Position_2D,
    model_mode: str,
    model_key: str | None,
) -> Entity:
    return Entity(
        name=name,
        position=position,
        tags=["agent"],
        actions=[
            Do_Nothing(),
            Move_Up(),
            Move_Down(),
            Move_Left(),
            Move_Right(),
        ],
        components=[
            make_policy(model_mode, model_key),
            Territory_Player(),
            Collidable(collidable_tags=["wall", "agent"]),
        ],
    )


def build_territory_tile(
    *,
    x: int,
    y: int,
    owner_id: str | None = None,
) -> Entity:
    return Entity(
        name=f"Tile ({x}, {y})",
        position=Position_2D(x, y),
        components=[
            Territory_Tile(owner_id=owner_id, control=2 if owner_id else 0),
        ],
    )


def build_boundary_walls(width: int, height: int) -> list[Entity]:
    walls = []
    for x in range(-1, width + 1):
        walls.append(build_wall(x, -1))
        walls.append(build_wall(x, height))
    for y in range(height):
        walls.append(build_wall(-1, y))
        walls.append(build_wall(width, y))
    return walls


def build_wall(x: int, y: int) -> Entity:
    return Entity(
        name=f"Boundary Wall ({x}, {y})",
        position=Position_2D(x, y),
        tags=["wall"],
        components=[Collidable()],
    )


def build_entities(model_mode: str) -> list[Entity]:
    model_key = (
        register_model(model_mode, GAME_NAME, game_rules=LLM_GAME_RULES)
        if model_mode != "human_action"
        else None
    )

    agents = [
        build_agent(
            name="Agent A",
            position=Position_2D(0, 0),
            model_mode=model_mode,
            model_key=model_key,
        ),
        build_agent(
            name="Agent B",
            position=Position_2D(WIDTH - 1, HEIGHT - 1),
            model_mode=model_mode,
            model_key=model_key,
        ),
        build_agent(
            name="Agent C",
            position=Position_2D(0, HEIGHT - 1),
            model_mode=model_mode,
            model_key=model_key,
        ),
    ]
    starting_owners = {
        (0, 0): "Agent A",
        (1, 0): "Agent A",
        (0, 1): "Agent A",
        (WIDTH - 1, HEIGHT - 1): "Agent B",
        (WIDTH - 2, HEIGHT - 1): "Agent B",
        (WIDTH - 1, HEIGHT - 2): "Agent B",
        (0, HEIGHT - 1): "Agent C",
        (1, HEIGHT - 1): "Agent C",
        (0, HEIGHT - 2): "Agent C",
    }

    tiles = [
        build_territory_tile(
            x=x,
            y=y,
            owner_id=starting_owners.get((x, y)),
        )
        for y in range(HEIGHT)
        for x in range(WIDTH)
    ]

    return [*agents, *tiles, *build_boundary_walls(WIDTH, HEIGHT)]


@contextlib.contextmanager
def csv_log(log_path: str) -> Iterator[csv.DictWriter]:
    parent_dir = os.path.dirname(log_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "game_name",
                "model_mode",
                "observations",
                "actions",
                "selection_info",
                "rewards",
                "cumulative_rewards",
                "scores",
                "map",
                "infos",
                "terminations",
                "truncations",
            ],
        )
        writer.writeheader()
        yield writer


def run_exp(
    model_mode: str = "human_action",
    max_steps: int = DEFAULT_MAX_STEPS,
    log_path: str | None = None,
) -> None:
    env = Territory_Environment(
        description=GAME_NAME,
        entities=build_entities(model_mode),
        max_steps=max_steps,
        llm_observation=use_compact_llm_observation(model_mode),
    )
    cumulative_rewards = {agent.name: 0.0 for agent in env.agents}

    print(f"\n{'=' * 60}")
    print(f"  Game: {GAME_NAME}  |  Mode: {model_mode}  |  Steps: {max_steps}")
    print(f"  Agents: {[agent.name for agent in env.agents]}")
    print(f"{'=' * 60}")
    print(env.map_text())
    print()

    log_ctx = csv_log(log_path) if log_path else contextlib.nullcontext(None)
    with log_ctx as log_writer:
        for step_idx in range(max_steps):
            cur_step_actions = []
            observations, actions_str, selection_infos = {}, {}, {}

            for agent_id, agent in enumerate(env.agents):
                observation = env.observe(agent_id)
                action, info = agent.get_component(Agent_Policy).select_action(observation)
                cur_step_actions.append(action)
                observations[agent.name] = str(observation)
                actions_str[agent.name] = str(action)
                selection_infos[agent.name] = info or {}
                print(f"[step {step_idx}] {agent.name} -> {action}")

            env.step(cur_step_actions)

            rewards = {
                agent.name: float(env.last_rewards[idx])
                for idx, agent in enumerate(env.agents)
            }
            for agent_name, reward in rewards.items():
                cumulative_rewards[agent_name] += reward
            scores = territory_scores(env)

            print(f"           rewards: {rewards}")
            print(f"           cumulative: {cumulative_rewards}")
            print(f"           territory scores: {scores}")
            print(env.map_text())
            print()

            if log_writer is not None:
                row = {
                    "step": step_idx,
                    "game_name": GAME_NAME,
                    "model_mode": model_mode,
                    "observations": observations,
                    "actions": actions_str,
                    "selection_info": selection_infos,
                    "rewards": rewards,
                    "cumulative_rewards": cumulative_rewards.copy(),
                    "scores": scores,
                    "map": env.map_text(),
                    "infos": {a.name: env.infos[i] for i, a in enumerate(env.agents)},
                    "terminations": env.terminations,
                    "truncations": env.truncations,
                }
                log_writer.writerow(
                    {
                        k: json.dumps(v, default=str) if isinstance(v, (dict, list)) else v
                        for k, v in row.items()
                    }
                )

            if any(env.terminations):
                print(f"Episode terminated at step {step_idx + 1}.")
                break
            if any(env.truncations):
                print(f"Reached max steps ({max_steps}).")
                break

    if log_path:
        print(f"Saved log to {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Territory melting-pot-style example.")
    parser.add_argument(
        "model_mode",
        nargs="?",
        default="human_action",
        choices=list(SUPPORTED_MODEL_MODES),
    )
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max_rounds", type=int, dest="max_steps_legacy", default=None)
    parser.add_argument("--log", dest="log_path", default=None)
    parser.add_argument("--reward_log", dest="log_path", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_exp(
        model_mode=args.model_mode,
        max_steps=args.max_steps if args.max_steps_legacy is None else args.max_steps_legacy,
        log_path=args.log_path,
    )
