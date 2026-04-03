from word_play.core import Agent_Policy
from word_play.presets.action_policies.human import Human_Takes_Action
from word_play.presets.action_policies.llm_action_and_communication import LLM_Action_And_Communication_Policy
from word_play.presets.environments.line_tracking_1d import Line_Tracking_1D, build_line_tracking_agent
from word_play.presets.models import Human_Model, Lazy_Model_Handle, LLM_MODEL_REGISTRY, OpenRouter_Model

import sys


def register_model(model_mode: str) -> str:
    model_key = "line_tracking_model"
    if model_mode == "human_llm":
        LLM_MODEL_REGISTRY[model_key] = Lazy_Model_Handle(lambda: Human_Model())
        return model_key
    if model_mode == "openrouter_small":
        LLM_MODEL_REGISTRY[model_key] = Lazy_Model_Handle(
            lambda: OpenRouter_Model(
                model_name="meta-llama/llama-3.2-1b-instruct",
                system_prompt=(
                    'Your goal is to keep the agent aligned with the moving target. '
                    'Use the target side and drift direction to decide whether to move left, stay still, or move right. '
                    'Respond only with JSON: {"action_choice_idx": <integer>}.'
                ),
                generation_params={"temperature": 0.0},
                app_name="Word Play",
            )
        )
        return model_key
    if model_mode == "openrouter_mid":
        LLM_MODEL_REGISTRY[model_key] = Lazy_Model_Handle(
            lambda: OpenRouter_Model(
                model_name="meta-llama/llama-3-8b-instruct",
                system_prompt="You are controlling a line tracking system. Choose the action that keeps the agent aligned with the moving target and follow the output format exactly.",
                generation_params={"temperature": 0.0},
                app_name="Word Play",
            )
        )
        return model_key
    if model_mode == "openrouter_large":
        LLM_MODEL_REGISTRY[model_key] = Lazy_Model_Handle(
            lambda: OpenRouter_Model(
                model_name="openai/gpt-5.4",
                system_prompt="You are controlling a line tracking system. Choose the action that keeps the agent aligned with the moving target and follow the output format exactly.",
                generation_params={"temperature": 0.0},
                app_name="Word Play",
            )
        )
        return model_key
    raise ValueError(f"Unsupported model_mode: {model_mode}")


def run_exp(model_mode: str = "human_action", exp_steps: int = 30):
    model_key = register_model(model_mode) if model_mode != "human_action" else None
    agent = build_line_tracking_agent(
        "Tracker",
        LLM_Action_And_Communication_Policy(model_key=model_key)
        if model_mode != "human_action"
        else Human_Takes_Action(),
    )
    env = Line_Tracking_1D(
        description="1D line tracking benchmark.",
        entities=[agent],
        max_steps=exp_steps,
    )

    for step in range(exp_steps):
        cur_step_actions = []
        for agent_id, controlled_agent in enumerate(env.agents):
            observation = env.observe(agent_id)
            action, info = controlled_agent.get_component(Agent_Policy).select_action(observation)
            print(f"[step {step}] {controlled_agent.name} -> {action}")
            if info:
                if info.get("reasoning"):
                    print(f"[step {step}] reasoning:\n{info['reasoning']}")
                if info.get("raw_response"):
                    print(f"[step {step}] raw response:\n{info['raw_response']}")
            cur_step_actions.append(action)

        env.step(cur_step_actions)
        if env.terminations[0]:
            print(f"Episode ended at step {step + 1}.")
            break
        if env.truncations[0]:
            print(f"Episode reached the step limit at step {step + 1}.")
            break


if __name__ == "__main__":
    run_exp(sys.argv[1] if len(sys.argv) > 1 else "human_action")
