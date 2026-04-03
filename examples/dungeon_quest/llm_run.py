from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from examples.dungeon_quest.environment import DungeonRaidEnv  # noqa: E402
from word_play.core import Agent_Policy  # noqa: E402
from word_play.presets.action_policies.llm_action_and_communication import (  # noqa: E402
    LLM_Action_And_Communication_Policy,
)
from word_play.presets.models import OpenAIChatModel, register_model  # noqa: E402
from word_play.presets.renderers import EnvironmentLayoutAdapter, PygameRenderer  # noqa: E402

DEFAULT_LOG_DIR = PROJECT_ROOT / "experiments" / "logs" / "dungeon_quest"


def _compact_lines(text: str, limit: int = 4) -> list[str]:
    """Return a few non-empty lines from an observation string."""
    return [line.strip() for line in text.splitlines() if line.strip()][:limit]


def _compact_action_lines(observation) -> list[str]:
    """Return the full numbered list of currently available actions."""
    return [f"[{idx}] {selection}" for idx, selection in enumerate(observation.possible_actions)]


def _build_hud(env: DungeonRaidEnv, observation_summary: dict | None = None, action_info: dict | None = None) -> None:
    """Update the renderer HUD with the LLM's current observation and selected action."""
    sidebar_lines = []
    sidebar_lines.append("Agent Thought Process:")
    if action_info is not None and action_info.get("reasoning"):
        sidebar_lines.extend([line for line in str(action_info["reasoning"]).splitlines() if line.strip()])
    else:
        sidebar_lines.append("(no explicit reasoning returned)")

    if observation_summary is not None:
        sidebar_lines.append("Possible Actions:")
        sidebar_lines.extend(observation_summary["actions"])

    env.hud_sidebar_header = "Agent"
    env.hud_sidebar_lines = sidebar_lines[:32]


def build_llm_dungeon_env(*, model_key: str, renderer: PygameRenderer | None = None) -> DungeonRaidEnv:
    """Create the dungeon env and replace the preview policy with the shared LLM agent policy."""
    env = DungeonRaidEnv(renderer=renderer)
    env.hud_sidebar_width = 620

    raider = env.agents[0]
    components = {
        ctype: component
        for ctype, component in raider.components.items()
        if not isinstance(component, Agent_Policy)
    }
    policy = LLM_Action_And_Communication_Policy(
        model_key=model_key,
        system_prompt=(
            "You are controlling a dungeon raider in a sparse-reward dungeon crawl. "
            "Read the observation literally. Use positions, walls, enemy locations, exit state, and the coordinate rules to decide what to do. "
            "Trust the coordinate system and dx/dy offsets, not the rendered image. "
            "Treat the bottom-left wall tile of each room as the origin (0, 0). "
            "Move right means x+1. Move left means x-1. Move up means y+1. Move down means y-1. "
            "Positive dy means the target is above you. Negative dy means the target is below you. "
            "Your main goal is to defeat the final boss, the Ash Warden. "
            "To reach the boss, you must progress through the dungeon one room at a time. "
            "To leave a room, you must kill every monster in that room. "
            "The Moon Crypt contains a chest reward. If the room is clear and the chest is available, open it and choose one reward before leaving. "
            "When all monsters are dead, the exit opens. "
            "If enemies remain, prioritize reaching and killing them. "
            "If no enemies remain, move to the exit tile. "
            "If you are already standing on the open exit tile, choose Enter the exit immediately. "
            "Choose exactly one valid action from the provided action list."
        ),
        action_generation_config={"temperature": 0.2},
        message_generation_config={"temperature": 0.3},
        reasoning_generation_config={"temperature": 0.2},
        use_chain_of_thought=True,
    )
    policy.entity = raider
    components[type(policy)] = policy
    raider.components = components
    raider.is_agent = True

    original_end_of_step = env.environment_end_of_step

    def wrapped_end_of_step(action_selections):
        original_end_of_step(action_selections)
        if env.agents:
            observation = env.observe(0)
            observation_summary = {
                "summary": " ".join(_compact_lines(str(observation), limit=3)),
                "lines": _compact_lines(str(observation), limit=10),
                "full_text": str(observation),
                "actions": _compact_action_lines(observation),
            }
            latest_action_info = getattr(env, "_latest_llm_action_info", None)
            _build_hud(env, observation_summary, latest_action_info)

    env.environment_end_of_step = wrapped_end_of_step

    initial_observation = env.observe(0)
    _build_hud(
        env,
        {
            "summary": " ".join(_compact_lines(str(initial_observation), limit=3)),
            "lines": _compact_lines(str(initial_observation), limit=10),
            "full_text": str(initial_observation),
            "actions": _compact_action_lines(initial_observation),
        },
    )
    return env


def run_llm_dungeon_example(
    *,
    model_name: str,
    model_key: str,
    base_url: str,
    tile_size: int,
    autoplay: bool,
    render_delay: float,
    log_path: str | None,
    log_dir: str | None,
) -> str | None:
    """Run the live LLM dungeon simulation and return the saved replay pickle path."""
    register_model(
        model_key,
        lambda: OpenAIChatModel(
            model_name=model_name,
            api_key_env="OPENROUTER_API_KEY",
            base_url=base_url,
            default_generation_config={"temperature": 0.2},
        ),
    )

    renderer = PygameRenderer(layout=EnvironmentLayoutAdapter(), tile_size=tile_size)

    def reset_factory() -> DungeonRaidEnv:
        return build_llm_dungeon_env(model_key=model_key, renderer=renderer)

    def step_builder(env: DungeonRaidEnv):
        selections = []
        for agent_id, agent in enumerate(env.agents):
            observation = env.observe(agent_id)
            observation_summary = {
                "summary": " ".join(_compact_lines(str(observation), limit=3)),
                "lines": _compact_lines(str(observation), limit=10),
                "full_text": str(observation),
                "actions": _compact_action_lines(observation),
            }
            policy = agent.get_component(Agent_Policy)
            selection, info = policy.select_action(observation)
            setattr(env, "_latest_llm_observation_summary", observation_summary)
            action_info = {
                "selection": str(selection),
                "raw_response": info.get("raw_response"),
                "reasoning": info.get("reasoning"),
                "attempt": info.get("attempt"),
            }
            setattr(env, "_latest_llm_action_info", action_info)
            _build_hud(env, observation_summary, action_info)
            selections.append(selection)
        return selections

    initial_env = reset_factory()
    renderer.run_live_view(
        initial_env,
        step_builder=step_builder,
        keep_logs=True,
        log_path=log_path,
        log_root=log_dir or str(DEFAULT_LOG_DIR),
        record_title="Dungeon Quest LLM Example",
        autoplay=autoplay,
        step_delay=render_delay,
        reset_factory=reset_factory,
        initial_notes=["Dungeon quest LLM run booted.", "Replay auto-saves to a pickle log."],
    )
    return renderer.last_record_path


def main() -> None:
    """Parse CLI arguments and run the live dungeon LLM example."""
    parser = argparse.ArgumentParser(description="Run the dungeon quest example with an LLM-controlled agent.")
    parser.add_argument("--model-name", default="openai/gpt-5-mini")
    parser.add_argument("--model-key", default="openrouter_dungeon_quest")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--tile-size", type=int, default=72)
    parser.add_argument("--paused", action="store_true", help="Start paused instead of autoplaying.")
    parser.add_argument("--render-delay", type=float, default=0.25, help="Seconds between auto-stepped frames.")
    parser.add_argument("--log-path", default=None, help="Optional explicit path for the replay log pickle.")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Directory for auto-named replay logs.")
    args = parser.parse_args()

    saved_log_path = run_llm_dungeon_example(
        model_name=args.model_name,
        model_key=args.model_key,
        base_url=args.base_url,
        tile_size=args.tile_size,
        autoplay=not args.paused,
        render_delay=args.render_delay,
        log_path=args.log_path,
        log_dir=args.log_dir,
    )
    if saved_log_path is not None:
        print(f"Saved replay log: {saved_log_path}")


if __name__ == "__main__":
    main()
