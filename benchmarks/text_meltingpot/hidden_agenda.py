from __future__ import annotations

import argparse
from collections import Counter

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
from benchmarks.text_meltingpot.common import BENCHMARK_STEPS, normalized_probability, normalized_steps
from word_play.utils import tilemap_to_entities
from word_play.utils.tilemap import find_tile_positions


MANDATED_NUM_PLAYERS = 5
MAX_EPISODE_LENGTH = BENCHMARK_STEPS
MODEL_KEY = "hidden_agenda"
GEM_GOAL = max(3, normalized_steps(32))
GEM_REGROW_PROBABILITY = normalized_probability(0.001)
VOTING_FRAME_FREQUENCY = normalized_steps(200, minimum=3)
DELIBERATION_DURATION = normalized_steps(25, minimum=2)
FREEZE_RANGE = 2
FREEZE_COOLDOWN = normalized_steps(50)


def grid_distance(first: Entity, second: Entity) -> int:
    return abs(first.position.x - second.position.x) + abs(first.position.y - second.position.y)


def freeze_range(actor: Entity, target: Entity, env: Environment) -> bool:
    return grid_distance(actor, target) <= FREEZE_RANGE


def hidden_agenda_manager(env: Environment) -> "HiddenAgendaManager":
    for entity in env.state.entities:
        manager = entity.get_component(HiddenAgendaManager)
        if manager is not None:
            return manager
    raise RuntimeError("Hidden Agenda manager missing.")


class HiddenAgendaState(Component):
    def __init__(self, role: str, spawn_position: Position_2D):
        super().__init__()
        self._role = role
        self.spawn_position = Position_2D(spawn_position.x, spawn_position.y)
        self.held_gems = 0
        self.status = "active"
        self.frozen = False
        self.voted_out = False
        self.vote: str | None = None
        self.freeze_cooldown = 0
        self._base_actions: list[Action] = []

    def post_initialization(self) -> None:
        self._base_actions = list(self.entity.actions)

    def pre_actions_step(self, env: Environment) -> None:
        if self.freeze_cooldown > 0:
            self.freeze_cooldown -= 1

    @property
    def is_crewmate(self) -> bool:
        return self._role == "crewmate"

    @property
    def is_impostor(self) -> bool:
        return self._role == "impostor"

    @property
    def active(self) -> bool:
        return not self.frozen and not self.voted_out

    def set_base_actions(self) -> None:
        if self.active:
            self.entity.actions = list(self._base_actions)
            self.status = "active"
        else:
            self.entity.actions = [Do_Nothing()]

    def set_voting_actions(self, num_players: int) -> None:
        if self.active:
            self.entity.actions = [Do_Nothing(), AbstainVote(), *[VoteForPlayer(i) for i in range(1, num_players + 1)]]
            self.status = "deliberating"
            self.vote = "abstain"
        else:
            self.entity.actions = [Do_Nothing()]
            self.vote = "no-vote"

    def freeze_permanently(self) -> None:
        self.frozen = True
        self.status = "frozen"
        self.vote = "no-vote"
        self.entity.actions = [Do_Nothing()]

    def vote_out(self) -> None:
        self.voted_out = True
        self.status = "voted_out"
        self.vote = "no-vote"
        self.entity.actions = [Do_Nothing()]


class GemPatch(Component):
    def __init__(self):
        super().__init__(tags=["gem"])
        self.available = True

    def pre_actions_step(self, env: Environment) -> None:
        if not self.available and random_available(GEM_REGROW_PROBABILITY):
            self.available = True
            self._sync()

    def collect(self) -> bool:
        if not self.available:
            return False
        self.available = False
        self._sync()
        return True

    def _sync(self) -> None:
        self.entity.name = "Gem" if self.available else "Empty Gem Site"
        renderable = self.entity.get_component(Renderable)
        if renderable is not None:
            renderable.visible = self.available


class GemDeposit(Component):
    def __init__(self):
        super().__init__(tags=["gem_deposit"])


class VotingSpawn(Component):
    def __init__(self):
        super().__init__(tags=["voting_spawn"])


def random_available(probability: float) -> bool:
    import random

    return random.random() < probability


class ActorIsImpostor(Action_Validation):
    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        state = actor.get_component(HiddenAgendaState)
        manager = hidden_agenda_manager(env)
        return state is not None and state.is_impostor and state.active and not manager.voting_active


class FreezeReady(Action_Validation):
    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        state = actor.get_component(HiddenAgendaState)
        return state is not None and state.freeze_cooldown <= 0


class TargetIsActiveCrewmate(Action_Validation):
    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        state = target_entity.get_component(HiddenAgendaState)
        return state is not None and state.is_crewmate and state.active


class VotingIsActive(Action_Validation):
    def is_valid(self, actor: Entity, target_entity: Entity, env: Environment) -> bool:
        state = actor.get_component(HiddenAgendaState)
        return state is not None and state.active and hidden_agenda_manager(env).voting_active


class FreezeCrewmate(Action):
    def __init__(self):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(freeze_range),
                ActorIsImpostor(),
                FreezeReady(),
                TargetIsActiveCrewmate(),
            ]
        )

    def exec_action(self, actor: Entity, target_entity: Entity, env: Environment, kwargs=None):
        actor_state = actor.get_component(HiddenAgendaState)
        target_state = target_entity.get_component(HiddenAgendaState)
        target_state.freeze_permanently()
        actor_state.freeze_cooldown = FREEZE_COOLDOWN
        manager = hidden_agenda_manager(env)
        manager.pending_deliberation = True
        env.hidden_agenda_events.append(f"{actor.name} froze {target_entity.name}; deliberation will begin.")
        return {"froze": target_entity.name, "triggered_deliberation": True}

    def action_description_text(self, actor: Entity, target_entity: Entity, env: Environment) -> str:
        return f"Freeze {target_entity.name}."


class VoteForPlayer(Action):
    def __init__(self, player_number: int):
        super().__init__(validation_rules=[Target_Is_Self(), VotingIsActive()])
        self.player_number = player_number

    def exec_action(self, actor: Entity, target_entity: Entity, env: Environment, kwargs=None):
        state = actor.get_component(HiddenAgendaState)
        state.vote = f"Player {self.player_number}"
        return {"vote": state.vote}

    def action_description_text(self, actor: Entity, target_entity: Entity, env: Environment) -> str:
        return f"Vote for Player {self.player_number}."


class AbstainVote(Action):
    def __init__(self):
        super().__init__(validation_rules=[Target_Is_Self(), VotingIsActive()])

    def exec_action(self, actor: Entity, target_entity: Entity, env: Environment, kwargs=None):
        state = actor.get_component(HiddenAgendaState)
        state.vote = "abstain"
        return {"vote": "abstain"}

    def action_description_text(self, actor: Entity, target_entity: Entity, env: Environment) -> str:
        return "Abstain from the vote."


class HiddenAgendaManager(Component):
    def __init__(self, voting_positions: list[tuple[int, int]]):
        super().__init__()
        self.voting_positions = voting_positions
        self.voting_active = False
        self.voting_timer = 0
        self.pending_deliberation = False
        self.gems_deposited = 0
        self.outcome = "ongoing"

    def pre_actions_step(self, env: Environment) -> None:
        env.hidden_agenda_events = []

    def post_actions_step(self, env: Environment) -> None:
        if self.outcome != "ongoing":
            return
        self._collect_gems(env)
        self._deposit_gems(env)
        if self.outcome != "ongoing":
            return
        if self.voting_active:
            self.voting_timer -= 1
            if self.voting_timer <= 0:
                self._resolve_vote(env)
            return
        if self._active_crewmates(env) <= 1:
            self._finish(env, winner="impostor", reason="Only one crewmate remains active.")
            return
        if self.pending_deliberation or (env.cur_step + 1) % VOTING_FRAME_FREQUENCY == 0:
            self._start_deliberation(env)
        if env.cur_step + 1 >= MAX_EPISODE_LENGTH and self.outcome == "ongoing":
            env.truncations = [True] * len(env.agents)

    def _collect_gems(self, env: Environment) -> None:
        gems = [entity for entity in env.state.entities if entity.get_component(GemPatch) is not None]
        for agent in env.agents:
            state = agent.get_component(HiddenAgendaState)
            if state is None or not state.active or not state.is_crewmate or state.held_gems >= 1:
                continue
            for gem in gems:
                patch = gem.get_component(GemPatch)
                if gem.position == agent.position and patch.collect():
                    state.held_gems += 1
                    env.hidden_agenda_events.append(f"{agent.name} picked up a gem.")
                    break

    def _deposit_gems(self, env: Environment) -> None:
        deposit_positions = {
            (entity.position.x, entity.position.y)
            for entity in env.state.entities
            if entity.get_component(GemDeposit) is not None
        }
        for agent in env.agents:
            state = agent.get_component(HiddenAgendaState)
            if state is None or not state.active or not state.is_crewmate or state.held_gems <= 0:
                continue
            if (agent.position.x, agent.position.y) not in deposit_positions:
                continue
            self.gems_deposited += state.held_gems
            env.hidden_agenda_events.append(
                f"{agent.name} deposited {state.held_gems} gem(s); progress {self.gems_deposited}/{GEM_GOAL}."
            )
            state.held_gems = 0
            if self.gems_deposited >= GEM_GOAL:
                self._finish(env, winner="crewmates", reason="Crewmates deposited enough gems.")
                return

    def _start_deliberation(self, env: Environment) -> None:
        self.pending_deliberation = False
        self.voting_active = True
        self.voting_timer = DELIBERATION_DURATION
        positions = self.voting_positions or [(agent.position.x, agent.position.y) for agent in env.agents]
        for idx, agent in enumerate(env.agents):
            state = agent.get_component(HiddenAgendaState)
            if state is None:
                continue
            if state.active:
                x, y = positions[idx % len(positions)]
                agent.position = Position_2D(x, y)
            state.set_voting_actions(len(env.agents))
        env.hidden_agenda_events.append("Deliberation started.")

    def _resolve_vote(self, env: Environment) -> None:
        self.voting_active = False
        active_states = [agent.get_component(HiddenAgendaState) for agent in env.agents]
        active_voters = [state for state in active_states if state is not None and state.active]
        votes = Counter(state.vote for state in active_voters if state.vote and state.vote != "abstain")
        voted_player = None
        if votes:
            candidate, count = votes.most_common(1)[0]
            if count > len(active_voters) / 2:
                voted_player = candidate

        if voted_player is not None:
            player_number = int(voted_player.split()[-1])
            if 1 <= player_number <= len(env.agents):
                target = env.agents[player_number - 1]
                target_state = target.get_component(HiddenAgendaState)
                if target_state is not None and target_state.active:
                    target_state.vote_out()
                    env.hidden_agenda_events.append(f"{target.name} was voted out.")
                    if target_state.is_impostor:
                        self._finish(env, winner="crewmates", reason="The impostor was voted out.")
                        return

        for agent in env.agents:
            state = agent.get_component(HiddenAgendaState)
            if state is None:
                continue
            if state.active:
                agent.position = Position_2D(state.spawn_position.x, state.spawn_position.y)
            state.vote = None
            state.set_base_actions()

        if self._active_crewmates(env) <= 1:
            self._finish(env, winner="impostor", reason="Only one crewmate remains active.")

    def _active_crewmates(self, env: Environment) -> int:
        return sum(
            1
            for agent in env.agents
            if (state := agent.get_component(HiddenAgendaState)) is not None and state.is_crewmate and state.active
        )

    def _finish(self, env: Environment, *, winner: str, reason: str) -> None:
        self.outcome = f"{winner}_win"
        env.hidden_agenda_events.append(f"{winner} win: {reason}")
        for idx, agent in enumerate(env.agents):
            state = agent.get_component(HiddenAgendaState)
            if state is None:
                continue
            if winner == "crewmates":
                env.last_step_rewards[idx] += 1.0 if state.is_crewmate else -1.0
            else:
                env.last_step_rewards[idx] += 1.0 if state.is_impostor else -1.0
        env.terminations = [True] * len(env.agents)


def run_exp(agent_count: int = MANDATED_NUM_PLAYERS, policy: str = "random", model_name: str = "openai/gpt-4o-mini"):
    assert agent_count == MANDATED_NUM_PLAYERS, "Hidden Agenda requires exactly 5 players."

    if policy == "llm":
        LLM_MODEL_REGISTRY.unload(MODEL_KEY)
        LLM_MODEL_REGISTRY.register(
            MODEL_KEY,
            OpenRouter_Model,
            model_name=model_name,
            generation_config={"temperature": 0.3},
        )

    entity_tilemap = """
    WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
    WG.........WW.......WW..........W
    W......G...WWWWWWWWWWW..G...G...W
    W.G....G...W..V.V.V..W.....G...GW
    W....G..G..W.V.....V.W.....G....W
    W.G...G....W..V...V..W..G.....G.W
    W..G.G..G..W...V.V...W..G..G....W
    WW........WWWWWWWWWWWWW........WW
    W...............................W
    W..........P..OOOOO..P..........W
    W..........PP.OOOOO.PP..........W
    W..........PP.OOOOO.PP..........W
    W..........P..OOOOO..P..........W
    W...............................W
    WW........WWWWWWWWWWWWW........WW
    W....G..G..WWWWWWWWWWW..G....G..W
    W......G...WWWWWWWWWWW..G.G.....W
    W.G......G.WWWWWWWWWWWG.....G...W
    W.....G....WWWWWWWWWWW..G......GW
    W.G....G..GWWWWWWWWWWW..G....G..W
    W...G..G...WWWWWWWWWWWG.........W
    WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
    """
    entity_tileset = {
        "W": {
            "name": "Ship Wall",
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
        "G": {
            "name": "Gem",
            "tags": ["gem"],
            "components": [
                GemPatch(),
                Renderable(
                    sprite_path="src/items/gems/blue_cube.png",
                    z_index=4,
                ),
            ],
        },
        "O": {
            "name": "Gem Deposit",
            "tags": ["gem_deposit"],
            "components": [
                GemDeposit(),
                Renderable(
                    sprite_path="src/world_tiles/indoors/floors/trap_door_tile.png",
                    z_index=2,
                ),
            ],
        },
        "V": {
            "name": "Voting Spawn",
            "tags": ["voting_spawn"],
            "components": [
                VotingSpawn(),
                Renderable(
                    sprite_path="src/world_tiles/indoors/floors/white_grid_floor.png",
                    z_index=1,
                ),
            ],
        },
    }

    entities = tilemap_to_entities(entity_tilemap.replace("P", "."), entity_tileset)
    spawn_positions = find_tile_positions(entity_tilemap, "P")
    voting_positions = find_tile_positions(entity_tilemap, "V")

    entities.append(
        Entity(
            name="Hidden Agenda Manager",
            position=Position_2D(-1, -1),
            components=[HiddenAgendaManager(voting_positions=voting_positions)],
        )
    )

    roles = ["crewmate", "crewmate", "crewmate", "crewmate", "impostor"]
    for agent_id, ((x, y), role) in enumerate(zip(spawn_positions[:agent_count], roles), start=1):
        if role == "impostor":
            prompt = (
                f"You are Player {agent_id} in Hidden Agenda. You are the impostor. "
                "Freeze crewmates when it helps, survive deliberations, and prevent gem deposits. "
                "Players 1-4 are crewmates; Player 5 is the impostor."
            )
        else:
            prompt = (
                f"You are Player {agent_id} in Hidden Agenda. You are a crewmate. "
                "Collect gems, deposit them in the central grate, watch for suspicious freezing, "
                "and vote out the impostor during deliberations."
            )
        agent_policy = (
            LLM_Action_And_Communication_Policy(
                model_key=MODEL_KEY,
                system_prompt=prompt,
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
                    FreezeCrewmate(),
                ],
                components=[
                    agent_policy,
                    HiddenAgendaState(role, Position_2D(x, y)),
                    Collidable(collidable_tags=["wall"]),
                    Renderable(
                        sprite_path="src/characters/humanoids/human/farmer_man.png",
                        z_index=10,
                    ),
                ],
            )
        )

    env = Simple_2D_Grid_World(
        description="Hidden Agenda, adapted from MeltingPot into a single-file Word Play environment.",
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
        for event in getattr(env, "hidden_agenda_events", []):
            print(f"[step {step}] hidden_agenda -> {event}")

        if all(env.terminations) or all(env.truncations):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-count", type=int, default=MANDATED_NUM_PLAYERS)
    parser.add_argument("--policy", choices=["random", "llm", "human"], default="random")
    parser.add_argument("--model-name", default="openai/gpt-4o-mini")
    args = parser.parse_args()
    run_exp(agent_count=args.agent_count, policy=args.policy, model_name=args.model_name)
