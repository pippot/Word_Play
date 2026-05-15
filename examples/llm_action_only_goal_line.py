from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from word_play.core import Action_Selection, Agent_Policy, Entity, Environment, Observation  # noqa: E402
from word_play.presets.action_policies.llm_action_and_communication import (  # noqa: E402
    LLM_Action_And_Communication_Policy,
)
from word_play.presets.entity_orderings import entity_definition_order  # noqa: E402
from word_play.presets.models import LLM_MODEL_REGISTRY, OpenRouter_Model, register_model  # noqa: E402
from word_play.presets.movement.simple_1d_grid import (  # noqa: E402
    INFINITE_1D_MOVEMENT_SYSTEM,
    Position_1D,
)
from word_play.presets.movement.simple_2d_grid import Move_Left, Move_Right  # noqa: E402
from word_play.presets.observation.simple_observation import Simple_Observation  # noqa: E402
from word_play.presets.systems.do_nothing import Do_Nothing  # noqa: E402


def goal_line_reward(agent_actions: list[Action_Selection], env: "Goal_Line_Env") -> list[float]:
    if env.agents[0].position.x == env.goal_x:
        return [1.0]
    return [-0.05]


class Goal_Line_Env(Environment):
    def __init__(
        self,
        model_key: str,
        start_x: int = 0,
        goal_x: int = 3,
        max_steps: int = 6,
        action_generation_config: dict | None = None,
    ):
        self.start_x = start_x
        self.goal_x = goal_x
        self.max_steps = max_steps

        agent = Entity(
            name="Explorer",
            position=Position_1D(start_x),
            actions=[
                Do_Nothing(),
                Move_Left(),
                Move_Right(),
            ],
            components=[
                LLM_Action_And_Communication_Policy(
                    model_key=model_key,
                    system_prompt=(
                        "You control Explorer in a tiny one-dimensional world. "
                        "Select actions that move current_x toward goal_x. "
                        "Return only the requested JSON action choice."
                    ),
                    action_generation_config=action_generation_config,
                    action_max_new_tokens=512,
                )
            ],
        )

        super().__init__(
            description="One-agent action-only LLM policy demo.",
            entities=[agent],
            movement_system=INFINITE_1D_MOVEMENT_SYSTEM,
            reward_func=goal_line_reward,
            entity_order=entity_definition_order,
        )

    def observe(self, agent_id: int) -> Observation:
        agent = self.agents[agent_id]
        return Simple_Observation(
            possible_actions=self.possible_actions(agent),
            nearby_entities=self.entities_near_position(agent.position),
            agent=agent,
            last_reward=self.last_rewards[agent_id],
            info=self.infos[agent_id],
            extra_sections=(
                "\n".join(
                    [
                        "OBJECTIVE:",
                        "  Move along the number line until current_x equals goal_x.",
                        "  Choose one available action each step.",
                    ]
                ),
                "\n".join(
                    [
                        "GOAL LINE STATE:",
                        f"  current_x: {agent.position.x}",
                        f"  goal_x: {self.goal_x}",
                        f"  step: {self.cur_step}/{self.max_steps}",
                    ]
                ),
            ),
        )

    def environment_start_of_step(self, action_selections: list[Action_Selection]):
        pass

    def environment_end_of_step(self, action_selections: list[Action_Selection]):
        if self.agents[0].position.x == self.goal_x:
            self.terminations[0] = True
        if self.cur_step + 1 >= self.max_steps:
            self.truncations[0] = True

    def _reset(self, seed=None) -> None:
        if hasattr(self, "state"):
            self.state.entities[0].position = Position_1D(self.start_x)


def run_exp():
    model_key = "goal_line_openrouter"
    model_name = "openai/gpt-5-mini"
    api_key_env = "OPENROUTER_API_KEY"
    openrouter_config = {
        "temperature": 0.0,
        "reasoning": {"effort": "minimal", "exclude": True},
    }

    if not os.getenv(api_key_env):
        raise EnvironmentError(f"Missing environment variable: {api_key_env}")

    if model_key not in LLM_MODEL_REGISTRY:
        register_model(
            model_key,
            OpenRouter_Model,
            model_name=model_name,
            generation_config=openrouter_config,
            api_key_env=api_key_env,
            base_url="https://openrouter.ai/api/v1",
            app_name="Word Play",
        )

    env = Goal_Line_Env(
        model_key=model_key,
        action_generation_config={
            **openrouter_config,
            "response_format": {"type": "json_object"},
        },
    )

    print("LLM_Action_And_Communication_Policy action-only demo")
    print(f"OpenRouter model: {model_name}")
    print("Agent actions: Do nothing, Move left, Move right")
    print("No communication actions are added.\n")

    action_history = []
    cumulative_reward = 0.0

    while not env.terminations[0] and not env.truncations[0]:
        cur_step_actions = []
        for agent_id, agent in enumerate(env.agents):
            observation = env.observe(agent_id)
            action, info = agent.get_component(Agent_Policy).select_action(observation)
            position_before = agent.position.x
            cur_step_actions.append(action)

        env.step(cur_step_actions)

        reward = env.last_rewards[0]
        cumulative_reward += reward
        action_history.append(
            {
                "step": len(action_history),
                "position_before": position_before,
                "action": str(action),
                "position_after": env.agents[0].position.x,
                "reward": reward,
                "raw_response": info.get("raw_response"),
            }
        )

    print("Action history:")
    for row in action_history:
        print(
            f"  step={row['step']} "
            f"x:{row['position_before']} -> {row['position_after']} "
            f"action={row['action']} "
            f"reward={row['reward']} "
            f"raw={row['raw_response']}"
        )

    print("\nSummary:")
    print(f"  final_position: {env.agents[0].position.x}")
    print(f"  goal_position: {env.goal_x}")
    print(f"  reached_goal: {env.terminations[0]}")
    print(f"  cumulative_reward: {cumulative_reward:.2f}")


if __name__ == "__main__":
    run_exp()
