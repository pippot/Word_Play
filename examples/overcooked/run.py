"""Overcooked kitchen environment - Run the preview demo or LLM policy.

Preview mode: hardcoded action sequences - no API key needed.
LLM mode: uses OpenRouter API to power agent decision making.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
sys.path.insert(0, "/Users/iamsogoodlo/Documents/Projects/Word_Play-dev copy/src")
from pathlib import Path
from word_play.core.components import Non_Agent_Policy

try:
    from examples.overcooked.environment import OvercookedKitchenEnv
except ModuleNotFoundError:
    from environment import OvercookedKitchenEnv

from word_play.presets.action_policies.follow_action_sequence import Follow_Action_Sequence
from word_play.presets.action_policies.llm_action_and_communication import LLM_Action_And_Communication_Policy
from word_play.presets.action_policies.overcooked_preview_policy import (
    EXPEDITER_SEQUENCE,
    PREP_COOK_SEQUENCE,
)
from word_play.presets.models import Lazy_Model_Handle, LLM_MODEL_REGISTRY, OpenRouter_Model, Human_Model
from word_play.presets.renderers import run_exp

DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "experiments" / "logs" / "overcooked"
DEFAULT_MODEL_NAME = "openai/gpt-4o-mini"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

AGENT_MODEL_KEYS = ["prep_cook_model", "expediter_model"]


def build_system_prompt(agent_name: str) -> str:
    """Generate system prompt for the kitchen agent."""
    if agent_name == "Prep Cook":
        return """You are a Prep Cook in a kitchen. Your job is to prepare ingredients.
                Look at the kitchen layout, identify what ingredients are needed (tomato, protein),
                and chop them at the Chopping Board. Then pass them to the Expediter.
                Reply with ONLY a JSON object: {"action_choice_idx": N} where N is the action index."""
    return """You are an Expediter in a kitchen. Your job is to assemble and serve dishes.
            Collect prepared ingredients from the Pass Counter, cook them on the Stove,
            plate them at the Plate Stack, and serve at the Delivery Hatch.
            Reply with ONLY a JSON object: {"action_choice_idx": N} where N is the action index."""


def register_models(model_name: str, use_human: bool = False) -> list[str]:
    """Register models for both agents."""
    if use_human:
        for key in AGENT_MODEL_KEYS:
            LLM_MODEL_REGISTRY[key] = Lazy_Model_Handle(lambda: Human_Model())
        return AGENT_MODEL_KEYS

    # Register OpenRouter models for each agent with their own system prompts
    for i, key in enumerate(AGENT_MODEL_KEYS):
        agent_name = "Prep Cook" if i == 0 else "Expediter"
        # Use closure to capture agent_name by value
        def make_loader(name=agent_name):
            return OpenRouter_Model(
                model_name=model_name,
                system_prompt=build_system_prompt(name),
                generation_params={"temperature": 0.3},
                app_name="Word Play Overcooked",
            )
        LLM_MODEL_REGISTRY[key] = Lazy_Model_Handle(make_loader)

    return AGENT_MODEL_KEYS


def build_llm_policy(model_key: str) -> LLM_Action_And_Communication_Policy:
    """Build an LLM policy for an agent."""
    return LLM_Action_And_Communication_Policy(
        model_key=model_key,
        action_generation_config={"temperature": 0.3},
        use_chain_of_thought=True,
        observation_memory_window=3,
    )

def attach_preview_policies(env: OvercookedKitchenEnv) -> None:
    """Attach hardcoded preview policies to both agents."""
    for entity in env.state.entities:
        if entity.name == "Prep Cook":
            policy = Follow_Action_Sequence(PREP_COOK_SEQUENCE, skip_invalid_actions=True)
            entity.components[Non_Agent_Policy] = policy
            policy.entity = entity
            entity.is_agent = False
        elif entity.name == "Expediter":
            policy = Follow_Action_Sequence(EXPEDITER_SEQUENCE, skip_invalid_actions=True)
            entity.components[Non_Agent_Policy] = policy
            policy.entity = entity
            entity.is_agent = False

    env._init_agent_list()
    env._init_agent_idx_dict()
    import copy as copy_module
    env.initial_state = copy_module.deepcopy(env.state)


def llm_policy_builder(env, agent, model_key: str):
    """Build LLM policy for agent - matches run_exp signature."""
    # Map agent name to the specific model key registered for that agent
    if agent.name == "Prep Cook":
        return build_llm_policy(AGENT_MODEL_KEYS[0])
    else:
        return build_llm_policy(AGENT_MODEL_KEYS[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the overcooked kitchen demo.")
    parser.add_argument("--policy", choices=["preview", "llm"], default="preview",
                        help="Policy mode: preview (hardcoded) or llm (requires API key).")
    parser.add_argument("--tile-size", type=int, default=56, help="Tile size for rendering.")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between auto-stepped frames.")
    parser.add_argument("--no-logs", action="store_true", help="Do not save a replay log.")
    parser.add_argument("--log-path", default=None, help="Optional explicit path for the replay log.")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Directory for auto-named replay logs.")
    parser.add_argument("--paused", action="store_true", help="Start paused instead of autoplaying.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Model name for LLM mode.")
    parser.add_argument("--human", action="store_true", help="Use human input instead of LLM API.")
    args = parser.parse_args()

    env = OvercookedKitchenEnv()

    if args.policy == "preview":
        attach_preview_policies(env)
        run_exp(
            env=env,
            tile_size=args.tile_size,
            sidebar_width=380,
            keep_logs=not args.no_logs,
            log_path=args.log_path,
            log_root=args.log_dir,
            autoplay=not args.paused,
            step_delay=args.delay,
            record_title="Overcooked Kitchen Preview Demo",
            initial_notes=["Overcooked preview demo - hardcoded action sequences (no API key needed)."],
        )
    else:  # llm mode
        # Check if all model keys have models registered
        if not all(k in LLM_MODEL_REGISTRY for k in AGENT_MODEL_KEYS):
            register_models(args.model_name, use_human=args.human)

        run_exp(
            env=env,
            policy="llm",
            tile_size=args.tile_size,
            sidebar_width=420,
            keep_logs=not args.no_logs,
            log_path=args.log_path,
            log_root=args.log_dir,
            autoplay=not args.paused,
            step_delay=args.delay,
            model_name=args.model_name,
            model_key="dummy",  # Not used since we register models ourselves
            base_url=DEFAULT_BASE_URL,
            llm_policy_builder=llm_policy_builder,
            record_title="Overcooked Kitchen LLM Demo",
            initial_notes=["Overcooked with LLM agents - both agents making decisions."],
        )


if __name__ == "__main__":
    main()
