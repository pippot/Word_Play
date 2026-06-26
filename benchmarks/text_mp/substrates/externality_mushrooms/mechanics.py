from __future__ import annotations

import random

from word_play.core import Action, Component, Entity, Environment, Target_Is_Nearby, Target_Not_Self
from word_play.presets.action_validations import Target_Has_Tag
from word_play.presets.movement.simple_2d_grid import Position_2D
from word_play.presets.systems.cooldown import Action_On_Cooldown, Cooldown
from word_play.presets.systems.freezable import Freezable
from word_play.presets.systems.respawnable import Respawnable
from word_play.presets.systems.reward import award_reward

from benchmarks.text_mp.core.timing import BENCHMARK_STEPS, normalized_steps


MIN_EPISODE_LENGTH = normalized_steps(1000)
INTERVAL_LENGTH = normalized_steps(100)
TERMINATION_PROBABILITY = 0.2
MUSHROOM_SPOIL_STEPS = {
    "red": normalized_steps(200),
    "green": normalized_steps(100),
    "blue": normalized_steps(75),
    "orange": normalized_steps(10_000_000),
}
MUSHROOM_DIGESTION_STEPS = {
    "green": normalized_steps(10),
    "blue": normalized_steps(15),
    "orange": normalized_steps(15),
}


def mushroom_events(env: Environment) -> list[str]:
    if getattr(env, "_mushroom_events_step", None) != env.cur_step:
        env.tick = env.cur_step
        env.mushroom_events = []
        env._mushroom_events_step = env.cur_step
    return env.mushroom_events


def maybe_end_externality_episode(env: Environment) -> None:
    if getattr(env, "_externality_end_checked_step", None) == env.cur_step:
        return
    env._externality_end_checked_step = env.cur_step
    next_step = env.cur_step + 1
    max_steps = getattr(env, "max_episode_steps", BENCHMARK_STEPS)
    if next_step >= max_steps or (
        next_step >= MIN_EPISODE_LENGTH
        and next_step % INTERVAL_LENGTH == 0
        and random.random() < TERMINATION_PROBABILITY
    ):
        env.truncations = [True for _ in env.agents]


class ExternalityZapState(Respawnable):
    def __init__(self, recovery_time: int = normalized_steps(50)):
        super().__init__(inactive_tag="removed")
        self.mark_level = 0
        self.recovery_time = recovery_time
        self.recovery_ticks = 0

    def pre_actions_step(self, env: Environment) -> None:
        mushroom_events(env)
        super().pre_actions_step(env)

    def hit(self, freeze_duration: int, remove_duration: int) -> dict:
        freezable = self.entity.get_component(Freezable)
        if self.mark_level == 0:
            self.mark_level = 1
            self.recovery_ticks = self.recovery_time
            if freezable is not None:
                freezable.freeze(freeze_duration)
            return {"level": 1, "effect": "freeze", "duration": freeze_duration}

        self.mark_level = 0
        self.recovery_ticks = 0
        self.remove_temporarily(remove_duration)
        if freezable is not None:
            freezable.freeze(remove_duration)
        return {"level": 2, "effect": "removed", "duration": remove_duration}

    def post_actions_step(self, env: Environment) -> None:
        maybe_end_externality_episode(env)
        if self.recovery_ticks > 0:
            self.recovery_ticks -= 1
            if self.recovery_ticks == 0:
                self.mark_level = 0


class ExternalityZap(Action):
    def __init__(
        self,
        freeze_duration: int = normalized_steps(25),
        remove_duration: int = normalized_steps(50),
    ):
        super().__init__(
            validation_rules=[
                Action_On_Cooldown("zap"),
                Target_Is_Nearby(),
                Target_Not_Self(),
                Target_Has_Tag(["player"]),
            ]
        )
        self.freeze_duration = freeze_duration
        self.remove_duration = remove_duration

    def is_valid(self, actor, target, env, kwargs="unconsidered") -> bool:
        return (
            super().is_valid(actor, target, env, kwargs=kwargs)
            and target.get_component(ExternalityZapState) is not None
        )

    def exec_action(self, actor, target, env, kwargs=None):
        cooldown = actor.get_component(Cooldown)
        if cooldown is not None:
            cooldown.start("zap")
        state = target.get_component(ExternalityZapState)
        result = {
            "success": True,
            "zapped": target.name,
            **state.hit(self.freeze_duration, self.remove_duration),
        }
        mushroom_events(env).append(f"{actor.name} zapped {target.name}: {result['effect']}.")
        return result

    def action_description_text(self, actor, target, env):
        return f"Zap {target.name}."


class ExternalityMushroom(Component):
    def __init__(self, kind: str):
        super().__init__(tags=["mushroom", kind])
        self.kind = kind
        self.age = 0

    def pre_actions_step(self, env: Environment) -> None:
        mushroom_events(env)

    def post_actions_step(self, env: Environment) -> None:
        maybe_end_externality_episode(env)
        if self.entity not in env.state.entities:
            return

        eater = next(
            (
                agent
                for agent in env.agents
                if agent.position == self.entity.position and not self._is_removed(agent)
            ),
            None,
        )
        if eater is not None:
            self._eat(env, eater)
            return

        self.age += 1
        if self.age >= MUSHROOM_SPOIL_STEPS[self.kind]:
            kind = self.kind
            env.destroy_entity(self.entity)
            mushroom_events(env).append(f"A {kind} mushroom spoiled.")

    def _eat(self, env: Environment, eater: Entity) -> None:
        kind = self.kind
        if kind == "red":
            award_reward(env, eater, 1.0)
            mushroom_events(env).append(f"{eater.name} ate a red mushroom for +1.")
        elif kind == "green":
            reward = 2.0 / len(env.agents)
            for agent in env.agents:
                award_reward(env, agent, reward)
            self._freeze(eater, MUSHROOM_DIGESTION_STEPS["green"])
            mushroom_events(env).append(f"{eater.name} ate a green mushroom; everyone got +{reward:g}.")
        elif kind == "blue":
            others = [agent for agent in env.agents if agent is not eater]
            reward = 3.0 / len(others) if others else 0.0
            for agent in others:
                award_reward(env, agent, reward)
            self._freeze(eater, MUSHROOM_DIGESTION_STEPS["blue"])
            mushroom_events(env).append(f"{eater.name} ate a blue mushroom; others got +{reward:g}.")
        elif kind == "orange":
            award_reward(env, eater, -1.0)
            self._freeze(eater, MUSHROOM_DIGESTION_STEPS["orange"])
            destroyed = self._destroy_red_mushrooms(env)
            mushroom_events(env).append(
                f"{eater.name} ate an orange mushroom for -1; destroyed {destroyed} red mushrooms."
            )

        env.destroy_entity(self.entity)
        self._regrow_after_eating(env, kind)

    def _regrow_after_eating(self, env: Environment, eaten_kind: str) -> None:
        if eaten_kind in {"red", "green", "blue"} and random.random() < 0.25:
            self._spawn_mushroom(env, "red")
        if eaten_kind in {"green", "blue"} and random.random() < 0.40:
            self._spawn_mushroom(env, "green")
        if eaten_kind == "blue" and random.random() < 0.60:
            self._spawn_mushroom(env, "blue")
        if eaten_kind == "orange":
            self._spawn_mushroom(env, "orange")

    def _spawn_mushroom(self, env: Environment, kind: str) -> None:
        occupied = {
            (entity.position.x, entity.position.y)
            for entity in env.state.entities
            if entity.get_component(ExternalityMushroom) is not None or entity in env.agents
        }
        open_positions = [
            position
            for position in getattr(env, "externality_mushroom_potential_positions", [])
            if position not in occupied
        ]
        if not open_positions:
            return

        x, y = random.choice(open_positions)
        env.instantiate_entity(mushroom_entity(kind, Position_2D(x, y)))
        mushroom_events(env).append(f"A {kind} mushroom regrew.")

    def _destroy_red_mushrooms(self, env: Environment) -> int:
        red_mushrooms = [
            entity
            for entity in env.state.entities
            if (mushroom := entity.get_component(ExternalityMushroom)) is not None and mushroom.kind == "red"
        ]
        destroy_count = max(1, int(len(red_mushrooms) * 0.25)) if red_mushrooms else 0
        for mushroom in random.sample(red_mushrooms, destroy_count):
            env.destroy_entity(mushroom)
        return destroy_count

    def _freeze(self, entity: Entity, duration: int) -> None:
        freezable = entity.get_component(Freezable)
        if freezable is not None:
            freezable.freeze(duration)

    def _is_removed(self, entity: Entity) -> bool:
        zap_state = entity.get_component(ExternalityZapState)
        return zap_state is not None and not zap_state.active


def mushroom_entity(kind: str, position: Position_2D) -> Entity:
    return Entity(
        name=f"{kind.title()} Mushroom",
        position=Position_2D(position.x, position.y),
        components=[ExternalityMushroom(kind)],
    )
