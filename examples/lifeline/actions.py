"""
Everything an agent can do: move supply, write to the board, talk.

Move_* and Do_Nothing come from the engine unchanged; only the
Lifeline-specific actions live here.
"""

from __future__ import annotations

from word_play.core import Action, Entity, Target_Is_Self
from word_play.presets.action_args import String_Arg
from word_play.presets.movement.simple_2d_grid import Position_2D
from word_play.presets.systems.communication.core import Communication_Policy

from .config import MAX_BOARD_TEXT_CHARS, NUM_CHAT_ROUNDS
from .validations import (
    A_Partner_Is_In_Talk_Range,
    At_A_Zone,
    Is_Adjacent_To,
    Is_Carrying_Supply,
    Near_The_Board,
    Not_Already_Carrying,
    Supply_Not_Carried,
    Talk_Not_On_Cooldown,
    Target_Is_Supply,
    conversation_partners_in_range,
)

class Pickup_Supply(Action):
    """Pick up an unclaimed, adjacent supply unit. Carries at most one at a time."""
    def __init__(self) -> None:
        super().__init__(
            validation_rules=[
                Target_Is_Supply(),
                Is_Adjacent_To(),
                Supply_Not_Carried(),
                Not_Already_Carrying(),
            ],
        )

    def exec_action(self, actor, target_entity, env, kwargs) -> dict | None:
        env.carrying[actor] = target_entity
        target_entity.position = Position_2D(actor.position.x, actor.position.y)
        return {"picked_up": target_entity.name}

    def action_description_text(self, actor, target_entity, env) -> str:
        return f"Pick up {target_entity.name}."


class Deliver_Supply(Action):
    """Deliver the carried supply unit while standing on a zone tile."""
    def __init__(self) -> None:
        super().__init__(
            validation_rules=[
                Target_Is_Self(),
                Is_Carrying_Supply(),
                At_A_Zone(),
            ],
        )

    def exec_action(self, actor, target_entity, env, kwargs) -> dict | None:
        supply = env.carrying[actor]
        zone_name = next(
            (
                name for name, z in env.zones.items()
                if z.position.x == actor.position.x and z.position.y == actor.position.y
            ),
            None,
        )
        if zone_name is None:
            return {"delivered": False, "reason": "not on a zone"}

        corrupted = supply in env.corrupted_supplies
        # A contaminated shipment does not help the zone: it is still logged
        # (so the pattern stays visible to everyone) and it still consumes the
        # unit, but it never advances the quota. Otherwise an agent that
        # ignores contamination could fill a zone's quota with garbage and
        # thereby earn the couriers their fairness bonus.
        within_quota = (
            not corrupted
            and env.zone_day_counts[zone_name] < env.zone_quotas[zone_name]
        )
        if not corrupted:
            env.zone_day_counts[zone_name] += 1
        env.zone_total_counts[zone_name] += 1

        record = {
            "generation": env.generation_index,
            "day": env.current_day,
            "step": env.cur_step + 1,
            "agent": actor.name,
            "zone": zone_name,
            "corrupted": corrupted,
            "within_quota": within_quota,
        }
        env.delivery_log.append(record)
        env._new_deliveries.append(record)

        del env.carrying[actor]
        env.corrupted_supplies.discard(supply)
        # Deferred: Environment.step() is iterating state.entities right now.
        env._supplies_awaiting_removal.append(supply)

        env.render_state.emit(
            "deliver", pod=supply.name, agent=actor.name, zone=zone_name,
            corrupted=corrupted, step=env.cur_step + 1,
        )
        return {"delivered": True, "zone": zone_name, "corrupted": corrupted}

    def action_description_text(self, actor, target_entity, env) -> str:
        return "Deliver your carried supply."


class Drop_Supply(Action):
    """Discard the carried supply unit without delivering it."""
    def __init__(self) -> None:
        super().__init__(
            validation_rules=[
                Target_Is_Self(),
                Is_Carrying_Supply(),
            ],
        )

    def exec_action(self, actor, target_entity, env, kwargs) -> dict | None:
        supply = env.carrying.pop(actor)
        env.corrupted_supplies.discard(supply)
        # Deferred: Environment.step() is iterating state.entities right now.
        env._supplies_awaiting_removal.append(supply)
        env.render_state.emit(
            "discard", pod=supply.name, agent=actor.name, step=env.cur_step + 1,
        )
        return {"discarded": supply.name}

    def action_description_text(self, actor, target_entity, env) -> str:
        return "Discard your carried supply (e.g. if you suspect it's contaminated)."


class Board_Text_Arg(String_Arg):
    def __init__(self) -> None:
        super().__init__(validators=[lambda arg, actor, target, env: bool(arg.strip())])

    def arg_description(self, actor, target_entity, env) -> str:
        return (
            f"a short note, <= {MAX_BOARD_TEXT_CHARS} chars, no semicolons ';' "
            "(they break parsing)"
        )


class Write_Board(Action):
    """Post a short note to the shared board. Requires standing near it."""
    def __init__(self) -> None:
        super().__init__(
            validation_rules=[Target_Is_Self(), Near_The_Board()],
            required_kwargs={"text": Board_Text_Arg()},
        )

    def exec_action(self, actor, target_entity, env, kwargs) -> dict | None:
        text = (kwargs or {}).get("text", "").strip()[:MAX_BOARD_TEXT_CHARS]
        entry = {
            "generation": env.generation_index,
            "day": env.current_day,
            "step": env.cur_step + 1,
            "author": actor.name,
            "text": text,
        }
        env.board_entries.append(entry)
        # Only the first post of each day earns anything, so the board can't be
        # farmed by spamming it.
        if actor not in env._board_posters_today:
            env._board_posters_today.add(actor)
            env._board_reward_pending.add(actor)
        env.render_state.emit(
            "board_post", agent=actor.name, text=text, step=env.cur_step + 1,
        )
        return {"posted": text}

    def action_description_text(self, actor, target_entity, env) -> str:
        return "Write a note on the shared board (persists across generations)."


def lifeline_chat_format(
    participants: list[Entity], env, info: str | None = None
) -> None:
    """Run NUM_CHAT_ROUNDS of public chat, logging every message."""
    for speaker in participants:
        speaker.get_component(Communication_Policy).start_conversation(
            participants, env, info=info
        )

    for turn in range(NUM_CHAT_ROUNDS):
        for speaker in participants:
            recipients = [e for e in participants if e is not speaker]
            message = speaker.get_component(Communication_Policy).send_message(
                recipients,
                env,
                info=info if turn == 0 else None,
            )
            env.message_log.append(
                {
                    "step": env.cur_step + 1,
                    "turn": turn,
                    "speaker": speaker.name,
                    "text": str(message),
                }
            )
            env.render_state.emit(
                "speech",
                entity=speaker,
                text=str(message),
                turn=turn,
                step=env.cur_step + 1,
            )
            for recipient in recipients:
                recipient.get_component(Communication_Policy).receive_message(
                    message, speaker, env
                )

    for speaker in participants:
        speaker.get_component(Communication_Policy).end_conversation(
            participants, env, info=info
        )


class Make_Public_Statement(Action):
    """
    Hold a multi-round public conversation with nearby agents.

    Unlike examples/waystation.py's version of this action, this one does NOT
    use a single global "one conversation per step" lock -- it only blocks an
    agent from joining a second conversation in the same step, so distant
    clusters of agents can talk concurrently. This matters here because
    populations are meant to scale up. It also uses a Manhattan talk radius
    rather than the engine's same-tile-only default (see
    conversation_partners_in_range).
    """
    def __init__(self) -> None:
        self.conversation_format = lifeline_chat_format
        super().__init__(
            validation_rules=[
                Target_Is_Self(),
                A_Partner_Is_In_Talk_Range(),
                Talk_Not_On_Cooldown(),
            ],
        )

    def exec_action(self, actor, target_entity, env, kwargs) -> dict | None:
        participants = conversation_partners_in_range(actor, env)
        participants.append(actor)
        if any(p in env._talked_this_step for p in participants):
            return None
        env._talked_this_step.update(participants)
        env.last_talk_step[actor.name] = env.cur_step
        info = (
            f"Day {env.current_day + 1}, step "
            f"{(env.cur_step % env.steps_per_day) + 1}/{env.steps_per_day}. "
            "Speak concisely. One short sentence."
        )
        self.conversation_format(participants, env, info=info)
        return None

    def action_description_text(self, actor, target_entity, env) -> str:
        return "Talk to nearby players."
