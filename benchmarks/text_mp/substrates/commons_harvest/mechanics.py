from __future__ import annotations

import random

from word_play.core import Action, Component, Entity, Environment, Target_Is_Nearby, Target_Not_Self
from word_play.presets.action_validations import Target_Has_Tag
from word_play.presets.systems.cooldown import Action_On_Cooldown, Cooldown
from word_play.presets.systems.do_nothing import Do_Nothing
from word_play.presets.systems.reward import Rewardable, award_reward

from benchmarks.text_mp.core.timing import BENCHMARK_STEPS, normalized_probability, normalized_steps


def commons_events(env: Environment) -> list[str]:
    if getattr(env, "_commons_events_step", None) != env.cur_step:
        env.tick = env.cur_step
        env.commons_events = []
        env._commons_events_step = env.cur_step
    return env.commons_events


class Commons_Apple_Patch(Component):
    def __init__(self, apple_regrowth_probs: list[float] | None = None):
        self.active_tags = ["apple"]
        self.inactive_tags = ["depleted_apple_patch"]
        super().__init__(tags=[*self.active_tags, "apple_patch"])
        self.density_radius = 2
        self.density_regrow_probs = [
            normalized_probability(prob)
            for prob in (apple_regrowth_probs or [0.0, 0.0025, 0.005, 0.025])
        ]
        self.consumed = False

    def pre_actions_step(self, env: Environment) -> None:
        commons_events(env)
        if self.consumed and self._should_regrow(env):
            self.regrow()
            env.commons_events.append(f"{self.entity.name} regrew.")

    def post_actions_step(self, env: Environment) -> None:
        self._collect(env)
        self._maybe_end_episode(env)

    def _collect(self, env: Environment) -> None:
        if self.consumed:
            return
        for agent in env.agents:
            if agent.position == self.entity.position:
                self.consume()
                rewardable = self.entity.get_component(Rewardable)
                reward = rewardable.reward_for(agent, env) if rewardable is not None else 1.0
                if rewardable is None:
                    award_reward(env, agent, reward)
                env.commons_events.append(f"{agent.name} ate {self.entity.name} for +{reward:g}.")
                break

    def consume(self) -> None:
        self.consumed = True
        self._sync_tags()

    def regrow(self) -> None:
        self.consumed = False
        self._sync_tags()

    def _should_regrow(self, env: Environment) -> bool:
        prob = self._regrow_prob_this_tick(env)
        return prob >= 1.0 or (prob > 0 and random.random() < prob)

    def _regrow_prob_this_tick(self, env: Environment) -> float:
        nearby = 0
        for entity in env.state.entities:
            patch = entity.get_component(Commons_Apple_Patch)
            if patch is None or patch.consumed or entity is self.entity:
                continue
            distance = abs(self.entity.position.x - entity.position.x) + abs(self.entity.position.y - entity.position.y)
            if distance <= self.density_radius:
                nearby += 1
        return self.density_regrow_probs[min(nearby, len(self.density_regrow_probs) - 1)]

    def _sync_tags(self) -> None:
        removed_tags = self.active_tags if self.consumed else self.inactive_tags
        added_tags = self.inactive_tags if self.consumed else self.active_tags
        for tag in removed_tags:
            while tag in self.entity.tags:
                self.entity.tags.remove(tag)
        for tag in added_tags:
            if tag not in self.entity.tags:
                self.entity.tags.append(tag)

    def _maybe_end_episode(self, env: Environment) -> None:
        if getattr(env, "_commons_end_step", None) == env.cur_step:
            return
        env._commons_end_step = env.cur_step
        next_step = env.cur_step + 1
        max_steps = getattr(env, "max_episode_steps", BENCHMARK_STEPS)
        if next_step >= max_steps:
            env.truncations = [True for _ in env.agents]
            return
        if (
            next_step >= normalized_steps(1000)
            and next_step % normalized_steps(100) == 0
            and random.random() < 0.15
        ):
            env.truncations = [True for _ in env.agents]


class Role_Based_Punishment_Tile(Component):
    def __init__(self, punished_tag: str = "putative_cooperator", penalty: float = -10.0):
        super().__init__(tags=["punishment_tile"])
        self.punished_tag = punished_tag
        self.penalty = penalty

    def pre_actions_step(self, env: Environment) -> None:
        commons_events(env)

    def post_actions_step(self, env: Environment) -> None:
        for agent in env.agents:
            if agent.position == self.entity.position and self.punished_tag in agent.tags:
                award_reward(env, agent, self.penalty)
                env.commons_events.append(f"{agent.name} crossed a punishment tile for {self.penalty:g}.")


class Commons_Freezable(Component):
    def __init__(self, default_duration: int):
        super().__init__()
        self.default_duration = default_duration
        self.frozen_ticks = 0
        self._unfrozen_actions: list[Action] | None = None

    def post_initialization(self) -> None:
        self._unfrozen_actions = list(self.entity.actions)

    def freeze(self) -> None:
        self.frozen_ticks = max(self.frozen_ticks, self.default_duration)
        if self._unfrozen_actions is None:
            self._unfrozen_actions = list(self.entity.actions)
        self.entity.actions = [Do_Nothing()]

    def post_actions_step(self, env: Environment) -> None:
        if self.frozen_ticks <= 0:
            return
        self.frozen_ticks -= 1
        if self.frozen_ticks == 0 and self._unfrozen_actions is not None:
            self.entity.actions = list(self._unfrozen_actions)


class Commons_Zap(Action):
    def __init__(self):
        super().__init__(
            validation_rules=[
                Action_On_Cooldown("zap"),
                Target_Is_Nearby(),
                Target_Not_Self(),
                Target_Has_Tag(["player"]),
            ],
        )

    def exec_action(self, actor: Entity, target: Entity, env: Environment, kwargs=None):
        cooldown = actor.get_component(Cooldown)
        if cooldown is not None:
            cooldown.start("zap")

        freezable = target.get_component(Commons_Freezable)
        if freezable is None:
            return {"success": False, "reason": "not_freezable"}
        freezable.freeze()
        return {"success": True, "zapped": target.name}

    def action_description_text(self, actor: Entity, target: Entity, env: Environment) -> str:
        return f"Zap {target.name}."
