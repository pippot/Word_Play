from __future__ import annotations

from word_play.core import Action_Selection, Environment


class TimeLimit:
    def __init__(self, env: Environment, max_episode_steps: int) -> None:
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be a positive integer.")
        self.env = env
        self.max_episode_steps = max_episode_steps
        self.elapsed_steps = 0

    def __getattr__(self, item):
        return getattr(self.env, item)

    def reset(self, seed=None):
        self.elapsed_steps = 0
        return self.env.reset(seed=seed)

    def step(self, action_selections: list[Action_Selection]) -> None:
        self.env.step(action_selections)
        self.elapsed_steps += 1

        if self.elapsed_steps >= self.max_episode_steps and not any(self.env.terminations):
            self.env.truncations = [True for _ in self.env.truncations]
