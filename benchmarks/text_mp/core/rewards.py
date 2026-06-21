from __future__ import annotations

from word_play.core import Action_Selection, Environment


def buffered_reward_func(_: list[Action_Selection], env: Environment) -> list[float]:
    return list(getattr(env, "last_step_rewards", [0.0] * len(env.agents)))


def install_reward_buffer(env: Environment) -> None:
    env.last_step_rewards = [0.0] * len(env.agents)
    env.reward_func = buffered_reward_func


def reset_step_rewards(env: Environment) -> None:
    env.last_step_rewards = [0.0] * len(env.agents)

