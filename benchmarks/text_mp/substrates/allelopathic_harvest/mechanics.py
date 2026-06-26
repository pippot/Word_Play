from __future__ import annotations

import random

from word_play.core import Action, Component, Entity, Environment, Target_Is_Nearby, Target_Not_Self
from word_play.presets.action_validations import Target_Has_Component
from word_play.presets.movement.simple_2d_grid import Position_2D
from word_play.presets.systems.containers.presets import Regrowable_Item_Source
from word_play.presets.systems.cooldown import Action_On_Cooldown, Cooldown
from word_play.presets.systems.freezable import Freezable
from word_play.presets.systems.preferences import Preference
from word_play.presets.systems.reward import award_reward
from word_play.presets.systems.zap import Zap_Change, ZapMarking

from benchmarks.text_mp.core.timing import BENCHMARK_STEPS, normalized_probability, normalized_steps


ALLELOPATHIC_COLORS = ("red", "green", "blue")


def allelopathic_events(env: Environment) -> list[str]:
    if getattr(env, "_allelopathic_events_step", None) != env.cur_step:
        env.tick = env.cur_step
        env.allelopathic_events = []
        env._allelopathic_events_step = env.cur_step
    return env.allelopathic_events


def berry_item(color: str) -> Entity:
    return Entity(
        name=f"{color.title()} Berry",
        position=Position_2D(0, 0),
        tags=["berry", color],
    )


def standard_berry_source(color: str) -> Regrowable_Item_Source:
    return Regrowable_Item_Source(
        item_factory=berry_item(color),
        max_capacity=1,
        regen_rate=1,
    )


def standard_berry_patch(color: str) -> Entity:
    return Entity(
        name=f"{color.title()} Berry Patch",
        position=Position_2D(0, 0),
        tags=["berry_patch", "berry", color],
        components=[standard_berry_source(color)],
    )


class PaintBerry(Action):
    def __init__(self, color: str):
        super().__init__(validation_rules=[Action_On_Cooldown("color")])
        self.color = color
        self.zap_change = Zap_Change(
            allowed_tag="berry_patch",
            output=lambda: standard_berry_patch(color),
        )

    def is_valid(self, actor, target_entity, env, kwargs="unconsidered") -> bool:
        return super().is_valid(actor, target_entity, env, kwargs=kwargs) and self.zap_change.is_valid(
            actor,
            target_entity,
            env,
            kwargs,
        )

    def exec_action(self, actor, target_entity, env, kwargs=None):
        cooldown = actor.get_component(Cooldown)
        if cooldown is not None:
            cooldown.start("color")
        result = self.zap_change.exec_action(actor, target_entity, env, kwargs)
        if result.get("success"):
            allelopathic_events(env).append(f"{actor.name} painted {target_entity.name} {self.color}.")
        return result

    def action_description_text(self, actor, target_entity, env):
        return f"Paint {target_entity.name} {self.color}."


class AllelopathicBerryPatch(Component):
    minimum_time_to_ripen = normalized_steps(10)
    base_growth_rate = normalized_probability(5e-6)

    def __init__(
        self,
        color: str | None = None,
        *,
        ripe: bool = True,
        auto_consume: bool = False,
        grows: bool = False,
    ):
        super().__init__(tags=["berry_patch"])
        self.color = color
        self.ripe = ripe if color is not None else False
        self.age = self.minimum_time_to_ripen if self.ripe else 0
        self.harvested = False
        self.auto_consume = auto_consume
        self.grows = grows

    def post_initialization(self) -> None:
        self._sync()

    def pre_actions_step(self, env: Environment) -> None:
        allelopathic_events(env)
        if self.grows:
            self._grow(env)

    def post_actions_step(self, env: Environment) -> None:
        self._maybe_end_episode(env)
        if self.harvested:
            self.harvested = False
            self._sync()
            return
        if not self.auto_consume or not self.available:
            return
        agent = next(
            (
                agent for agent in env.agents
                if agent.position == self.entity.position and not self._is_removed(agent)
            ),
            None,
        )
        if agent is None:
            return
        preference = agent.get_component(Preference)
        reward = preference.reward_for(self.entity) if preference is not None else 1.0
        color = self.consume()
        award_reward(env, agent, reward)
        allelopathic_events(env).append(f"{agent.name} ate a {color} berry for +{reward:g}.")

    @property
    def available(self) -> bool:
        return self.color is not None and self.ripe and not self.harvested

    def plant(self, color: str) -> None:
        self.color = color
        self.ripe = False
        self.age = 0
        self.harvested = False
        self._sync()

    def ripen(self) -> None:
        if self.color is None:
            return
        self.ripe = True
        self._sync()

    def consume(self) -> str | None:
        if not self.available:
            return None
        color = self.color
        self.color = None
        self.ripe = False
        self.age = 0
        self.harvested = False
        self._sync()
        return color

    def _sync(self) -> None:
        for tag in (*ALLELOPATHIC_COLORS, "berry", "ripe_berry", "unripe_berry", "empty_patch"):
            while tag in self.entity.tags:
                self.entity.tags.remove(tag)

        if self.color is None:
            self.entity.name = "Empty Berry Patch"
            self.entity.tags.append("empty_patch")
        else:
            self.entity.name = f"{self.color.title()} Berry Patch"
            self.entity.tags.extend(["berry", self.color, "ripe_berry" if self.ripe else "unripe_berry"])

    def _grow(self, env: Environment) -> None:
        if self.color is not None:
            if not self.ripe:
                self.age += 1
                if self.age >= self.minimum_time_to_ripen:
                    self.ripen()
                    allelopathic_events(env).append(f"A {self.color} berry ripened.")
            return

        for color, weight in self._growth_weights(env).items():
            if random.random() < self.base_growth_rate * weight:
                self.plant(color)
                allelopathic_events(env).append(f"A new {color} berry sprouted.")
                return

    def _growth_weights(self, env: Environment) -> dict[str, float]:
        if getattr(env, "_allelopathic_growth_step", None) == env.cur_step:
            return env._allelopathic_growth_weights

        active_counts = {color: 0 for color in ALLELOPATHIC_COLORS}
        for entity in env.state.entities:
            patch = entity.get_component(AllelopathicBerryPatch)
            if patch is not None and patch.color is not None:
                active_counts[patch.color] += 1

        total_active = sum(active_counts.values())
        env._allelopathic_growth_weights = (
            {color: 1 / len(ALLELOPATHIC_COLORS) for color in ALLELOPATHIC_COLORS}
            if total_active == 0
            else {color: active_counts[color] / total_active for color in ALLELOPATHIC_COLORS}
        )
        env._allelopathic_growth_step = env.cur_step
        return env._allelopathic_growth_weights

    def _is_removed(self, agent: Entity) -> bool:
        zap_state = agent.get_component(AllelopathicZapState)
        return zap_state is not None and zap_state.removed

    def _maybe_end_episode(self, env: Environment) -> None:
        if getattr(env, "_allelopathic_end_checked_step", None) == env.cur_step:
            return
        env._allelopathic_end_checked_step = env.cur_step
        max_steps = getattr(env, "max_episode_steps", BENCHMARK_STEPS)
        if env.cur_step + 1 >= max_steps:
            env.truncations = [True for _ in env.agents]


class AllelopathicPlantBerry(Action):
    def __init__(self, color: str, freeze_duration: int = normalized_steps(2)):
        super().__init__(
            validation_rules=[
                Action_On_Cooldown("plant"),
                Target_Not_Self(),
                Target_Is_Nearby(),
                Target_Has_Component(AllelopathicBerryPatch),
            ]
        )
        self.color = color
        self.freeze_duration = freeze_duration

    def exec_action(self, actor, target, env, kwargs=None):
        target_name = target.name
        cooldown = actor.get_component(Cooldown)
        if cooldown is not None:
            cooldown.start("plant")
        freezable = actor.get_component(Freezable)
        if freezable is not None:
            freezable.freeze(self.freeze_duration)
        target.get_component(AllelopathicBerryPatch).plant(self.color)
        allelopathic_events(env).append(f"{actor.name} planted a {self.color} berry on {target_name}.")
        return {"planted": self.color, "target": target_name, "position": str(target.position)}

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        patch = target.get_component(AllelopathicBerryPatch)
        return super().is_valid(actor, target, env, kwargs=kwargs) and patch is not None and patch.color is None

    def action_description_text(self, actor, target, env):
        return f"Plant {self.color} berry on {target.name}."


class AllelopathicZapState(ZapMarking):
    def __init__(
        self,
        freeze_duration: int = normalized_steps(25),
        remove_duration: int = normalized_steps(50),
        recovery_time: int = normalized_steps(50),
    ):
        super().__init__(freeze_duration=freeze_duration, penalty=0.0, recovery_time=recovery_time)
        self.remove_duration = remove_duration
        self.removed = False

    def zap_hit(self, env: Environment) -> dict:
        if self.mark_level == 0:
            return super().zap_hit(env)

        self.mark_level = 0
        self.recovery_counter = 0
        self.removed = True
        freezable = self.entity.get_component(Freezable)
        if freezable is not None:
            freezable.freeze(self.remove_duration)
        return {"level": 2, "effect": "removed", "duration": self.remove_duration}

    def post_actions_step(self, env: Environment) -> None:
        super().post_actions_step(env)
        if not self.removed:
            return
        freezable = self.entity.get_component(Freezable)
        if freezable is not None and freezable.is_frozen:
            return
        self.removed = False
