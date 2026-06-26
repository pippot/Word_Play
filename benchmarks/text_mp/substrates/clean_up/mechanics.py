from __future__ import annotations

import random

from word_play.core import Action, Component, Entity, Environment, Target_Is_Nearby, Target_Not_Self
from word_play.presets.action_validations import Target_Has_Tag
from word_play.presets.movement.simple_2d_grid import Collidable, Position_2D
from word_play.presets.systems.cooldown import Action_On_Cooldown, Cooldown
from word_play.presets.systems.freezable import Freezable
from word_play.presets.systems.reward import award_reward

from benchmarks.text_mp.core.timing import BENCHMARK_STEPS, normalized_probability, normalized_steps


DIRT_SPAWN_PROBABILITY = 0.5
DIRT_SPAWN_DELAY = normalized_steps(50)
MIN_EPISODE_LENGTH = normalized_steps(1000)
INTERVAL_LENGTH = normalized_steps(100)
TERMINATION_PROBABILITY = 0.2
MAX_APPLE_GROWTH_RATE = normalized_probability(0.05)
THRESHOLD_DEPLETION = 0.4
THRESHOLD_RESTORATION = 0.0
ZAP_DURATION = normalized_steps(50)


def clean_up_events(env: Environment) -> list[str]:
    if getattr(env, "_clean_up_events_step", None) != env.cur_step:
        env.tick = env.cur_step
        env.clean_up_events = []
        env._clean_up_events_step = env.cur_step
    return env.clean_up_events


class ApplePatch(Component):
    def __init__(self):
        super().__init__(tags=["apple", "apple_patch"])
        self.consumed = False
        self.regrow_prob = MAX_APPLE_GROWTH_RATE

    def consume(self) -> bool:
        if self.consumed:
            return False
        self.consumed = True
        self._sync_tags()
        return True

    def pre_actions_step(self, env: Environment) -> None:
        clean_up_events(env)
        if self.consumed and random.random() < self.regrow_prob:
            self.consumed = False
            self._sync_tags()
            clean_up_events(env).append(f"{self.entity.name} regrew.")

    def _sync_tags(self) -> None:
        for tag in ("apple", "depleted_apple_patch"):
            while tag in self.entity.tags:
                self.entity.tags.remove(tag)
        if self.consumed:
            self.entity.name = "Depleted Apple Patch"
            self.entity.tags.append("depleted_apple_patch")
        else:
            self.entity.name = "Apple"
            self.entity.tags.append("apple")


class CleanDirt(Action):
    def __init__(self):
        super().__init__(
            validation_rules=[
                Target_Not_Self(),
                Target_Is_Nearby(),
            ]
        )

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        return super().is_valid(actor, target, env, kwargs=kwargs) and "dirt_patch" in target.tags

    def exec_action(self, actor, target, env, kwargs=None):
        while "dirt_patch" in target.tags:
            target.tags.remove("dirt_patch")
        target.name = "River Tile"
        clean_up_events(env).append(f"{actor.name} cleaned a dirty river tile.")
        return {"cleaned": target.name}

    def action_description_text(self, actor, target, env) -> str:
        return f"Clean {target.name}."


class CleanUpZap(Action):
    def __init__(self):
        super().__init__(
            validation_rules=[
                Action_On_Cooldown("zap"),
                Target_Is_Nearby(),
                Target_Not_Self(),
                Target_Has_Tag(["player"]),
            ],
        )

    def exec_action(self, actor, target, env, kwargs=None):
        cooldown = actor.get_component(Cooldown)
        if cooldown is not None:
            cooldown.start("zap")

        freezable = target.get_component(Freezable)
        if freezable is None:
            return {"success": False, "reason": "not_freezable"}
        freezable.freeze()
        clean_up_events(env).append(f"{actor.name} zapped {target.name}.")
        return {"success": True, "zapped": target.name}

    def action_description_text(self, actor, target, env):
        return f"Zap {target.name}."


class PollutionManager(Component):
    def __init__(self, river_positions: list[tuple[int, int]]):
        super().__init__()
        self.river_positions = river_positions

    def pre_actions_step(self, env: Environment) -> None:
        clean_up_events(env)

    def post_actions_step(self, env: Environment) -> None:
        self._collect_apples(env)
        self._update_apple_regrowth(env)
        self._spawn_dirt(env)
        self._maybe_end_episode(env)

    def _collect_apples(self, env: Environment) -> None:
        apples = [entity for entity in env.state.entities if entity.get_component(ApplePatch) is not None]
        for agent in env.agents:
            for apple in apples:
                if apple.position != agent.position:
                    continue
                patch = apple.get_component(ApplePatch)
                if patch.consume():
                    award_reward(env, agent, 1.0)
                    clean_up_events(env).append(f"{agent.name} ate an apple for +1.")

    def _update_apple_regrowth(self, env: Environment) -> None:
        pollution = self.current_pollution_fraction(env)
        if pollution <= THRESHOLD_RESTORATION:
            regrow_prob = MAX_APPLE_GROWTH_RATE
        elif pollution >= THRESHOLD_DEPLETION:
            regrow_prob = 0.0
        else:
            remaining = 1.0 - (pollution - THRESHOLD_RESTORATION) / (THRESHOLD_DEPLETION - THRESHOLD_RESTORATION)
            regrow_prob = MAX_APPLE_GROWTH_RATE * max(0.0, remaining)

        for entity in env.state.entities:
            patch = entity.get_component(ApplePatch)
            if patch is not None:
                patch.regrow_prob = regrow_prob

    def _spawn_dirt(self, env: Environment) -> None:
        if env.cur_step < DIRT_SPAWN_DELAY or random.random() >= DIRT_SPAWN_PROBABILITY:
            return

        occupied = {
            (entity.position.x, entity.position.y)
            for entity in env.state.entities
            if "dirt_patch" in entity.tags
        }
        available = [position for position in self.river_positions if position not in occupied]
        if not available:
            return

        x, y = random.choice(available)
        env.instantiate_entity(
            Entity(
                name="Dirty River Tile",
                position=Position_2D(x, y),
                tags=["wall", "water", "dirt_patch"],
                components=[
                    Collidable(collidable_tags=["wall"]),
                ],
            )
        )
        clean_up_events(env).append("A dirty river tile appeared.")

    def _maybe_end_episode(self, env: Environment) -> None:
        if getattr(env, "_clean_up_end_checked_step", None) == env.cur_step:
            return
        env._clean_up_end_checked_step = env.cur_step
        next_step = env.cur_step + 1
        max_steps = getattr(env, "max_episode_steps", BENCHMARK_STEPS)
        if next_step >= max_steps:
            env.truncations = [True for _ in env.agents]
            return
        if next_step < MIN_EPISODE_LENGTH:
            return
        if next_step % INTERVAL_LENGTH == 0 and random.random() < TERMINATION_PROBABILITY:
            env.truncations = [True for _ in env.agents]

    def current_pollution_fraction(self, env: Environment) -> float:
        dirt_count = sum(1 for entity in env.state.entities if "dirt_patch" in entity.tags)
        if not self.river_positions:
            return 0.0
        return dirt_count / len(self.river_positions)
