from __future__ import annotations

import argparse
import random

from benchmarks.text_meltingpot.common import BENCHMARK_STEPS, HillZone, PaintballManager, PaintballZap
from word_play.core import Agent_Policy, Entity
from word_play.presets.action_policies.human import Human_Takes_Action
from word_play.presets.action_policies.random_policy import Random_Policy
from word_play.presets.entity_orderings import randomize_agent_order
from word_play.presets.environments.simple_2d_grid_world import Simple_2D_Grid_World
from word_play.presets.movement.simple_2d_grid import Collidable, Move_Down, Move_Left, Move_Right, Move_Up, Position_2D
from word_play.presets.observation.simple_observation import Simple_Observation
from word_play.presets.renderers import Renderable, render_step
from word_play.presets.systems.cooldown import Cooldown
from word_play.presets.systems.do_nothing import Do_Nothing
from word_play.presets.systems.freezable import Freezable
from word_play.presets.systems.health import Health
from word_play.utils import tilemap_to_entities
from word_play.utils.tilemap import find_tile_positions


DEFAULT_NUM_PLAYERS = 8
MODEL_KEY = "paintball__king_of_the_hill"


def _relative_text(actor: Entity, target: Entity) -> str:
    dx = target.position.x - actor.position.x
    dy = target.position.y - actor.position.y
    return f"relative=({dx:+d},{dy:+d}), distance={abs(dx) + abs(dy)}"


def _direction_to_xy(agent: Entity, xy: tuple[int, int]) -> str:
    dx = xy[0] - agent.position.x
    dy = xy[1] - agent.position.y
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


def _team(entity: Entity) -> str:
    if "red" in entity.tags:
        return "red"
    if "blue" in entity.tags:
        return "blue"
    return "unknown"


def _paintball_manager(env) -> PaintballManager:
    for entity in env.state.entities:
        manager = entity.get_component(PaintballManager)
        if manager is not None:
            return manager
    raise RuntimeError("Paintball manager missing.")


def _health_text(entity: Entity) -> str:
    health = entity.get_component(Health)
    if health is None:
        return "health=unknown"
    return f"health={health.health}/{health.max_health}"


def _important_actions_text(possible_actions: list) -> str:
    matches = [
        f"  [{idx}] {selection}"
        for idx, selection in enumerate(possible_actions)
        if "Shoot paintball" in str(selection) and "Wall" not in str(selection)
    ]
    if not matches:
        return (
            "IMPORTANT AVAILABLE ACTIONS: no enemy shot is valid this turn; "
            "move toward the hill unless already on it."
        )
    return "IMPORTANT AVAILABLE ACTIONS:\n" + "\n".join(matches)


def _hill_map_symbol(entity: Entity, agent: Entity) -> tuple[int, str] | None:
    if entity is agent:
        return 100, "A"
    if entity.is_agent:
        return 90, "R" if _team(entity) == "red" else "B"
    if entity.get_component(HillZone) is not None:
        return 80, "H"
    if "destructible" in entity.tags:
        return 70, "X"
    if "wall" in entity.tags:
        return 60, "#"
    return None


def _hill_local_map(env, agent: Entity, nearby_entities: list[Entity]) -> str:
    radius = env.observation_radius
    cells: dict[tuple[int, int], tuple[int, str]] = {
        (x, y): (0, ".")
        for y in range(agent.position.y - radius, agent.position.y + radius + 1)
        for x in range(agent.position.x - radius, agent.position.x + radius + 1)
    }
    for entity in nearby_entities:
        symbol = _hill_map_symbol(entity, agent)
        if symbol is None:
            continue
        xy = (entity.position.x, entity.position.y)
        if xy in cells and symbol[0] >= cells[xy][0]:
            cells[xy] = symbol

    rows = []
    for y in range(agent.position.y + radius, agent.position.y - radius - 1, -1):
        rows.append("".join(cells[(x, y)][1] for x in range(agent.position.x - radius, agent.position.x + radius + 1)))
    return "LOCAL MAP (visible only; A you, R/B players, H hill, X breakable wall, # wall):\n" + "\n".join(rows)


def _format_hill_entities(nearby_entities: list[Entity], agent: Entity) -> str:
    lines = ["VISIBLE PAINTBALL ENTITIES:"]
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
            relation = "teammate" if _team(entity) == _team(agent) else "enemy"
            lines.append(f"- {entity.name} ({relation}): {_relative_text(agent, entity)}, {_health_text(entity)}")
        elif entity.get_component(HillZone) is not None:
            direction = _direction_to_xy(agent, (entity.position.x, entity.position.y))
            lines.append(f"- Hill tile: {_relative_text(agent, entity)}; direction={direction}")
        elif "destructible" in entity.tags:
            lines.append(f"- Destructible wall: {_relative_text(agent, entity)}, {_health_text(entity)}")
    if len(lines) == 1:
        return "VISIBLE PAINTBALL ENTITIES: none"
    return "\n".join(lines)


class PaintballKingOfTheHillWorld(Simple_2D_Grid_World):
    def observe(self, agent_id: int):
        agent = self.agents[agent_id]
        nearby_entities = self.entities_in_observation_square(agent.position)
        possible_actions = self.possible_actions(agent)
        manager = _paintball_manager(self)
        visible_hill_positions = [
            (entity.position.x, entity.position.y)
            for entity in nearby_entities
            if entity.get_component(HillZone) is not None
        ]
        on_hill = (agent.position.x, agent.position.y) in visible_hill_positions
        lines = [
            "KING OF THE HILL STATUS:",
            f"  your_team: {_team(agent)}",
            f"  your_health: {_health_text(agent)}",
            f"  score: red={manager.red_score:g}, blue={manager.blue_score:g}",
            f"  on_hill: {on_hill}",
            "  goal_now: stand on Hill tiles with teammates and shoot enemies off the hill.",
        ]
        if visible_hill_positions:
            nearest = min(
                visible_hill_positions,
                key=lambda xy: abs(xy[0] - agent.position.x) + abs(xy[1] - agent.position.y),
            )
            lines.append(f"  nearest_visible_hill_direction: {_direction_to_xy(agent, nearest)}")
        if on_hill:
            lines.append("  immediate_priority: stay on the hill; shoot visible enemies, otherwise hold position.")
        elif visible_hill_positions:
            lines.append("  immediate_priority: move toward nearest_visible_hill_direction.")
        else:
            lines.append("  immediate_priority: explore until a Hill tile is visible; shooting walls is only useful if blocked.")
        return Simple_Observation(
            possible_actions=possible_actions,
            nearby_entities=nearby_entities,
            agent=agent,
            last_reward=self.last_rewards[agent_id],
            info=self.infos[agent_id],
            observation_radius=self.observation_radius,
            extra_sections=(
                "\n".join(lines),
                _hill_local_map(self, agent, nearby_entities),
                _important_actions_text(possible_actions),
            ),
            nearby_entities_formatter=lambda entities, current_agent: _format_hill_entities(entities, current_agent),
        )


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
    IIIIIIIIIIIIIIIIIIIIIII
    IWWWWWWWWWWWWWWWWWWWWWI
    IWPPP,PPPP,P,PPPP,PPPWI
    IWPPP,,PP,,,,,PP,,PPPWI
    IWPPP,,,,,,,,,,,,,PPPWI
    IWP,,WW,,,,,,,,,WW,,PWI
    IW,,,WWXWWWXWWW,WW,,,WI
    IW,,,,,,uuuuuuu,X,,,,WI
    IW,,,,WlGGGGGGGrW,,,,WI
    IWXWWXWlGGGGGGGrWXWWXWI
    IWXWWXWlGGGGGGGrWXWWXWI
    IW,,,,XlGGGIGGGrX,,,,WI
    IWXWWXWlGGGGGGGrWXWWXWI
    IWXWWXWlGGGGGGGrWXWWXWI
    IW,,,,WlGGGGGGGrW,,,,WI
    IW,,,,X,ddddddd,,,,,,WI
    IW,,,WW,WWWXWWWXWW,,,WI
    IWQ,,WW,,,,,,,,,WW,,QWI
    IWQQQ,,,,,,,,,,,,,QQQWI
    IWQQQ,,QQ,,,,,QQ,,QQQWI
    IWQQQ,QQQQ,Q,QQQQ,QQQWI
    IWWWWWWWWWWWWWWWWWWWWWI
    IIIIIIIIIIIIIIIIIIIIIII
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
        "X": {
            "name": "Wall",
            "tags": ["wall", "blocker", "destructible"],
            "components": [
                Health(max_health=1),
                Collidable(collidable_tags=["wall", "blocker"]),
                Renderable(sprite_path="src/world_tiles/outdoors/terrain/wall_stone.png", z_index=3),
            ],
        },
        "G": {
            "name": "Hill",
            "tags": [],
            "components": [
                HillZone(),
                Renderable(sprite_path="src/world_tiles/outdoors/terrain/grass_1.png", z_index=1),
            ],
        },
    }

    entity_map = entity_tilemap
    for tile in "PQ,Iurdl":
        entity_map = entity_map.replace(tile, ".")
    entities = tilemap_to_entities(entity_map, entity_tileset)
    red_spawns = find_tile_positions(entity_tilemap, "P")
    blue_spawns = find_tile_positions(entity_tilemap, "Q")
    random.shuffle(red_spawns)
    random.shuffle(blue_spawns)
    if agent_count > len(red_spawns) + len(blue_spawns):
        raise ValueError(f"Map only has {len(red_spawns) + len(blue_spawns)} spawn positions.")
    red_count = min(len(red_spawns), (agent_count + 1) // 2)
    blue_count = agent_count - red_count
    spawn_assignments = [("red", pos) for pos in red_spawns[:red_count]] + [
        ("blue", pos) for pos in blue_spawns[:blue_count]
    ]

    for agent_id, (team, (x, y)) in enumerate(spawn_assignments, start=1):
        if policy == "llm":
            from word_play.presets.action_policies.llm_action_and_communication import (
                LLM_Action_And_Communication_Policy,
            )

            agent_policy = LLM_Action_And_Communication_Policy(
                model_key=MODEL_KEY,
                system_prompt=(
                    f"You are Player {agent_id} on the {team} team in Paintball King of the Hill. "
                    "Move onto Hill tiles first and hold them with teammates to score. Shoot opposing players when "
                    "they contest the hill. Do not shoot walls unless movement toward the hill is actually blocked."
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
                name=f"{team.title()} Player {agent_id}",
                position=Position_2D(x, y),
                tags=["agent", "player", "default", team],
                actions=[
                    Do_Nothing(),
                    Move_Up(),
                    Move_Down(),
                    Move_Left(),
                    Move_Right(),
                    PaintballZap(),
                ],
                components=[
                    agent_policy,
                    Cooldown(cooldowns={"paintball": 2}),
                    Freezable(default_duration=2),
                    Health(max_health=3, starting_health=2),
                    Collidable(collidable_tags=["wall", "blocker"]),
                    Renderable(sprite_path="src/characters/humanoids/human/farmer_man.png", z_index=10),
                ],
            )
        )

    entities.append(Entity(name="Paintball Manager", position=Position_2D(0, 0), components=[PaintballManager("hill")]))

    env = PaintballKingOfTheHillWorld(
        description="Paintball: King of the Hill, adapted from Melting Pot.",
        entities=entities,
        entity_order=randomize_agent_order,
        observation_radius=5,
    )
    env.paintball_events = []
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
        for event in env.paintball_events:
            print(f"[step {step}] paintball -> {event}")
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
