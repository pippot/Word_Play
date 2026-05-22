from __future__ import annotations

import contextlib
import csv
import json
import os
from dataclasses import dataclass
from typing import Iterator

from word_play.core import (
    Action,
    Action_Selection,
    Agent_Policy,
    Component,
    Entity,
    Environment,
    Observation,
    Target_Is_Nearby,
)
from word_play.presets.action_validations import Target_Has_Component
from word_play.presets.entity_orderings import entity_definition_order
from word_play.presets.environments.simple_env_reset_mixin import Simple_Env_Reset_Mixin
from word_play.presets.movement.single_point import (
    SINGLE_POINT_MOVEMENT_SYSTEM,
    Single_Point_Position,
)
from word_play.presets.systems.communication import (
    Communication_Policy,
    Human_Communication_Policy,
)

try:
    from .payoff_matrix_utils import normalize_payoff_matrix
    from .model_params import SUPPORTED_MODEL_MODES, make_policy, register_model
except ImportError:
    from payoff_matrix_utils import normalize_payoff_matrix
    from model_params import SUPPORTED_MODEL_MODES, make_policy, register_model


class Payoff_Matrix(Component):
    """
    Manages payoff resolution for a simultaneous-move stage game.

    Manages payoff resolution for a simultaneous-move stage game.

    The payoff matrix can be specified as either a symmetric single-player
    payoff row or as a fully expanded payoff matrix with tuple leaves.

    Action names are provided as a flat list whose length must match the
    number of rows in the payoff matrix.  Each name maps 1-to-1 to an action
    index, which is what agents submit when voting.

    Parameters
    ----------
    game_name:
        Human-readable name shown in observations.
    num_players:
        Number of players participating in the game. Must be >= 2.
    action_names:
        Ordered list of strategy labels.  Length must equal the number of
        actions (top-level entries) in `payoff_matrix`.
    payoff_matrix:
        Either a symmetric single-player payoff row, or a full payoff matrix
        whose leaves contain one reward per player.

        Example (2-player Prisoner's Dilemma, actions [Cooperate, Defect]):
            [[3, 0],   # Cooperate vs [C, D]
             [5, 1]]   # Defect    vs [C, D]

    objective_text:
        Multi-line string with the game rules and payoff structure. This is
        passed to model-backed agents as system-prompt context instead of
        being repeated in every observation.
    auto_add_payoff_matrix_actions_to_all_agents:
        If True (default), voting actions are automatically added to every
        agent in the environment on instantiation.  Set to False if you want
        to control which agents have voting actions manually.
    """

    def __init__(
        self,
        *,
        game_name: str,
        num_players: int = 2,
        action_names: list[str],
        payoff_matrix: list,
        objective_text: str,
        auto_add_payoff_matrix_actions_to_all_agents: bool = True,
    ):
        super().__init__()

        if num_players < 2:
            raise ValueError("Payoff_Matrix requires at least two players.")
        if len(action_names) < 2:
            raise ValueError("Payoff_Matrix requires at least two action names.")
        if len(action_names) != len(payoff_matrix):
            raise ValueError(
                f"action_names has {len(action_names)} entries but payoff_matrix has "
                f"{len(payoff_matrix)} rows. They must match."
            )
        if not isinstance(auto_add_payoff_matrix_actions_to_all_agents, bool):
            raise TypeError("auto_add_payoff_matrix_actions_to_all_agents must be a bool.")

        self.game_name = game_name
        self.num_players = num_players
        self.action_names = list(action_names)
        self.objective_text = objective_text
        self.auto_add_payoff_matrix_actions_to_all_agents = auto_add_payoff_matrix_actions_to_all_agents
        self.payoff_matrix = normalize_payoff_matrix(payoff_matrix, num_players=num_players)

        self.pending_rewards: list[float] = []
        self.current_round_votes: dict[Entity, int] = {}
        self.last_joint_actions: dict[str, str] | None = None
        self.last_payoffs: dict[str, float] | None = None

    def on_instantiation(self, env: Environment, seed: int | None) -> None:
        if len(env.agents) < self.num_players:
            raise ValueError(
                f"Payoff_Matrix requires at least {self.num_players} agents "
                f"but only {len(env.agents)} are present."
            )
        if self.auto_add_payoff_matrix_actions_to_all_agents:
            self._ensure_actions(env)

    def pre_actions_step(self, env: Environment) -> None:
        self.pending_rewards = [0.0] * len(env.agents)

    def choose_action(self, actor: Entity, action_index: int) -> None:
        if not (0 <= action_index < len(self.action_names)):
            raise ValueError(f"Invalid action index: {action_index}")
        if actor in self.current_round_votes:
            return  # duplicate vote — no-op
        if len(self.current_round_votes) >= self.num_players:
            return  # already have enough votes — no-op
        self.current_round_votes[actor] = action_index

    def post_actions_step(self, env: Environment) -> None:
        step_info: dict[str, object] = {
            "step_count": env.step_count + 1,
            "votes_received": len(self.current_round_votes),
            "votes_needed": self.num_players,
            "current_votes": {
                agent.name: self.action_names[idx]
                for agent, idx in self.current_round_votes.items()
            },
        }

        if len(self.current_round_votes) == self.num_players:
            voting_agents = list(self.current_round_votes.keys())
            action_indices = list(self.current_round_votes.values())

            # Index into the N-dimensional matrix
            payoffs = self.payoff_matrix
            for idx in action_indices:
                payoffs = payoffs[idx]

            for agent, reward in zip(voting_agents, payoffs):
                self.pending_rewards[env.agent_to_idx[agent]] = reward

            self.last_joint_actions = {
                agent.name: self.action_names[idx]
                for agent, idx in zip(voting_agents, action_indices)
            }
            self.last_payoffs = {
                agent.name: float(reward)
                for agent, reward in zip(voting_agents, payoffs)
            }
            step_info["vote_status"] = "resolved"
            step_info["joint_actions"] = self.last_joint_actions.copy()
            step_info["payoffs"] = self.last_payoffs.copy()
            self.current_round_votes = {}
        else:
            step_info["vote_status"] = "collecting"

        for agent in env.agents:
            env.infos[env.agent_to_idx[agent]]["action_info"] = step_info

        env.step_count += 1
        if env.step_count >= env.max_steps:
            env.truncations = [True] * len(env.agents)

    def _ensure_actions(self, env: Environment) -> None:
        for agent in env.agents:
            existing = {
                action.action_index
                for action in agent.actions
                if isinstance(action, Payoff_Matrix_Action)
            }
            for idx, name in enumerate(self.action_names):
                if idx not in existing:
                    agent.actions.append(Payoff_Matrix_Action(action_index=idx, action_name=name))


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Payoff_Matrix_Action(Action):
    def __init__(self, action_index: int, action_name: str):
        self.action_index = action_index
        self.action_name = action_name
        super().__init__(
            validation_rules=[
                Target_Is_Nearby(),
                Target_Has_Component(Payoff_Matrix),
            ]
        )

    def exec_action(
        self, actor: Entity, target_entity: Entity, env: Environment, kwargs: dict | None
    ) -> dict | None:
        payoff_matrix = target_entity.get_component(Payoff_Matrix)
        if payoff_matrix is None:
            raise ValueError("Payoff_Matrix_Action requires a target with a Payoff_Matrix component.")
        payoff_matrix.choose_action(actor, self.action_index)
        return None

    def action_description_text(
        self, actor: Entity, target_entity: Entity, env: Environment
    ) -> str:
        return f"Choose action: {self.action_name}."


# ---------------------------------------------------------------------------
# Observation formatter
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Payoff_Matrix_Observation(Observation):
    agent_name: str
    last_reward: float
    info: dict
    step_count: int
    max_steps: int
    last_joint_actions: dict[str, str] | None
    last_payoffs: dict[str, float] | None

    def __str__(self) -> str:
        return "\n\n".join(
            filter(
                None,
                [
                    self._last_reward_block(),
                    self._state_block(),
                    self._actions_block(),
                ],
            )
        )

    def _last_reward_block(self) -> str:
        if self.last_joint_actions is None or self.last_payoffs is None:
            return f"REWARD THIS TURN: {self.last_reward:+.1f}"

        your_payoff = self.last_payoffs.get(self.agent_name, 0.0)
        other_payoffs = {
            agent_name: payoff
            for agent_name, payoff in self.last_payoffs.items()
            if agent_name != self.agent_name
        }
        action_summary = ", ".join(
            f"{agent_name}: {action_name}"
            for agent_name, action_name in self.last_joint_actions.items()
        )
        if len(other_payoffs) == 1:
            other_name, other_payoff = next(iter(other_payoffs.items()))
            payoff_summary = f"your payoff: {your_payoff:+.1f} | {other_name} payoff: {other_payoff:+.1f}"
        else:
            payoff_summary = f"your payoff: {your_payoff:+.1f} | others: {other_payoffs}"

        return "\n".join(
            [
                f"REWARD THIS TURN: {self.last_reward:+.1f}",
                "",
                "LAST ROUND RESULT:",
                f"  actions: {action_summary}",
                f"  {payoff_summary}",
            ]
        )

    def _state_block(self) -> str:
        return (
            f"STEP: {self.step_count}/{self.max_steps} "
            f"(steps remaining: {max(0, self.max_steps - self.step_count)})"
        )

    def _actions_block(self) -> str:
        if not self.possible_actions:
            return "AVAILABLE ACTIONS:\n  None"
        return "AVAILABLE ACTIONS:\n" + "\n".join(
            f"  [{idx}] {selection}"
            for idx, selection in enumerate(self.possible_actions)
        )


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def payoff_matrix_reward(
    agent_actions: list[Action_Selection], env: "Payoff_Matrix_Environment"
) -> list[float]:
    return env.get_payoff_matrix_component().pending_rewards.copy()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class Payoff_Matrix_Environment(Simple_Env_Reset_Mixin, Environment):
    def __init__(self, description: str, entities: list[Entity], *, max_steps: int = 20):
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")

        pm_entities = [e for e in entities if e.get_component(Payoff_Matrix) is not None]
        if len(pm_entities) != 1:
            raise ValueError("Payoff_Matrix_Environment requires exactly one entity with a Payoff_Matrix component.")

        self.max_steps = max_steps
        self.step_count = 0

        super().__init__(
            description=description,
            entities=entities,
            movement_system=SINGLE_POINT_MOVEMENT_SYSTEM,
            reward_func=payoff_matrix_reward,
            entity_order=entity_definition_order,
        )

    def _reset(self, seed=None) -> None:
        super()._reset(seed=seed)

    def observe(self, agent_id: int) -> Observation:
        agent = self.agents[agent_id]
        pm = self.get_payoff_matrix_component()
        last_reward = self.last_rewards[agent_id]
        return Payoff_Matrix_Observation(
            possible_actions=self.possible_actions(agent),
            agent_name=agent.name,
            last_reward=0.0 if last_reward is None else last_reward,
            info=self.infos[agent_id],
            step_count=self.step_count,
            max_steps=self.max_steps,
            last_joint_actions=pm.last_joint_actions,
            last_payoffs=pm.last_payoffs,
        )

    def environment_start_of_step(self, action_selections: list[Action_Selection]) -> None:
        pass

    def environment_end_of_step(self, action_selections: list[Action_Selection]) -> None:
        pass

    def get_payoff_matrix_component(self) -> Payoff_Matrix:
        for entity in self.state.entities:
            pm = entity.get_component(Payoff_Matrix)
            if pm is not None:
                return pm
        raise ValueError("No Payoff_Matrix component found in environment.")


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def build_payoff_matrix_voting_box(
    *,
    game_name: str,
    num_players: int = 2,
    action_names: list[str],
    payoff_matrix: list,
    objective_text: str,
    auto_add_payoff_matrix_actions_to_all_agents: bool = True,
    name: str = "Voting Box",
) -> Entity:
    return Entity(
        name=name,
        position=Single_Point_Position(),
        components=[
            Payoff_Matrix(
                game_name=game_name,
                num_players=num_players,
                action_names=action_names,
                payoff_matrix=payoff_matrix,
                objective_text=objective_text,
                auto_add_payoff_matrix_actions_to_all_agents=auto_add_payoff_matrix_actions_to_all_agents,
            )
        ],
    )


def build_entities(
    *,
    game_name: str,
    action_names: list[str],
    payoff_matrix: list,
    objective_text: str,
    model_mode: str,
    num_players: int = 2,
    num_agents: int | None = None,
    communication_enabled: bool = False,
    auto_add_payoff_matrix_actions_to_all_agents: bool = True,
) -> list[Entity]:
    total_agents = num_players if num_agents is None else num_agents
    if total_agents < num_players:
        raise ValueError("num_agents must be at least num_players.")

    model_key = (
        register_model(model_mode, game_name, game_rules=objective_text)
        if model_mode != "human_action"
        else None
    )

    agent_entities = []
    for idx in range(total_agents):
        components = [make_policy(model_mode, model_key)]
        if communication_enabled and model_mode == "human_action":
            components.append(Human_Communication_Policy())
        agent_entities.append(Entity(
            name=f"Agent {chr(ord('A') + idx)}",
            position=Single_Point_Position(),
            actions=[],
            components=components,
        ))

    voting_box = build_payoff_matrix_voting_box(
        game_name=game_name,
        num_players=num_players,
        action_names=action_names,
        payoff_matrix=payoff_matrix,
        objective_text=objective_text,
        auto_add_payoff_matrix_actions_to_all_agents=auto_add_payoff_matrix_actions_to_all_agents,
    )
    return [*agent_entities, voting_box]


# ---------------------------------------------------------------------------
# Communication phase
# ---------------------------------------------------------------------------

def run_communication_phase(
    agents: list[Entity], env: Payoff_Matrix_Environment, step_count: int
) -> None:
    for agent in agents:
        if agent.get_component(Communication_Policy) is None:
            return

    for speaker in agents:
        comm = speaker.get_component(Communication_Policy)
        recipients = [a for a in agents if a is not speaker]
        comm.start_conversation(agents, env, info=f"Pre-action communication, step {step_count}.")
        message = comm.send_message(recipients, env)
        for recipient in recipients:
            recipient.get_component(Communication_Policy).receive_message(message, speaker, env)
        comm.end_conversation(agents, env)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _csv_log(log_path: str) -> Iterator[csv.DictWriter]:
    parent_dir = os.path.dirname(log_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "step", "game_name", "model_mode", "observations", "actions",
            "selection_info", "joint_actions", "payoffs", "rewards",
            "infos", "terminations", "truncations",
        ])
        writer.writeheader()
        yield writer


def run_payoff_matrix_game(
    *,
    game_name: str,
    action_names: list[str],
    payoff_matrix: list,
    objective_text: str,
    model_mode: str = "human_action",
    max_steps: int = 20,
    log_path: str | None = None,
    num_players: int = 2,
    num_agents: int | None = None,
    communication_enabled: bool = False,
    auto_add_payoff_matrix_actions_to_all_agents: bool = True,
) -> None:
    entities = build_entities(
        game_name=game_name,
        action_names=action_names,
        payoff_matrix=payoff_matrix,
        objective_text=objective_text,
        model_mode=model_mode,
        num_players=num_players,
        num_agents=num_agents,
        communication_enabled=communication_enabled,
        auto_add_payoff_matrix_actions_to_all_agents=auto_add_payoff_matrix_actions_to_all_agents,
    )
    env = Payoff_Matrix_Environment(description=game_name, entities=entities, max_steps=max_steps)
    pm = env.get_payoff_matrix_component()

    print(f"\n{'=' * 60}")
    print(f"  Game: {game_name}  |  Mode: {model_mode}  |  Steps: {max_steps}")
    print(f"  Players: {[a.name for a in env.agents]}")
    print(f"{'=' * 60}\n")

    log_ctx = _csv_log(log_path) if log_path else contextlib.nullcontext(None)

    with log_ctx as log_writer:
        for step_idx in range(max_steps):
            if communication_enabled:
                run_communication_phase(env.agents, env, step_idx)

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
                if info:
                    if info.get("reasoning"):
                        print(f"[step {step_idx}] reasoning:\n{info['reasoning']}")
                    if info.get("raw_response"):
                        print(f"[step {step_idx}] raw response:\n{info['raw_response']}")

            env.step(cur_step_actions)

            print(f"           joint: {pm.last_joint_actions} | payoffs: {pm.last_payoffs}")

            if log_writer is not None:
                row = {
                    "step": step_idx,
                    "game_name": game_name,
                    "model_mode": model_mode,
                    "observations": observations,
                    "actions": actions_str,
                    "selection_info": selection_infos,
                    "joint_actions": pm.last_joint_actions,
                    "payoffs": pm.last_payoffs,
                    "rewards": {a.name: env.last_rewards[i] for i, a in enumerate(env.agents)},
                    "infos": {a.name: env.infos[i] for i, a in enumerate(env.agents)},
                    "terminations": env.terminations,
                    "truncations": env.truncations,
                }
                log_writer.writerow({
                    k: json.dumps(v, default=str) if isinstance(v, (dict, list)) else v
                    for k, v in row.items()
                })

            if any(env.terminations):
                print(f"\nEpisode terminated at step {step_idx + 1}.")
                break
            if any(env.truncations):
                print(f"\nReached max steps ({max_steps}).")
                break

    if log_path:
        print(f"Saved log to {log_path}")


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_common_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run a payoff-matrix game.")
    parser.add_argument(
        "model_mode",
        nargs="?",
        default="human_action",
        choices=list(SUPPORTED_MODEL_MODES),
    )
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--max_rounds", type=int, dest="max_steps_legacy", default=None)
    parser.add_argument("--log", dest="log_path", default=None)
    parser.add_argument("--reward_log", dest="log_path", default=None)
    return parser.parse_args()
