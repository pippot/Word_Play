"""
Everything an agent can do: move, pick up / deliver / discard supply, write to
the board.

Do_Nothing comes from the engine unchanged. The four moves are the engine's
Move_* actions with one change: their description names the tile they lead
to, so an agent weighing "is that tile contaminated?" doesn't have to do the
coordinate arithmetic itself.
"""

from __future__ import annotations

from word_play.core import Action, Action_Selection, Target_Is_Self
from word_play.presets.action_args import Int_Range_Arg, String_Arg
from word_play.presets.movement.simple_2d_grid import (
    Move_Down,
    Move_Left,
    Move_Right,
    Move_Up,
    Position_2D,
)

from .config import MAX_BOARD_SLOTS, MAX_BOARD_TEXT_CHARS
from .validations import (
    At_A_Zone,
    Is_Adjacent_To,
    Is_Carrying_Supply,
    Near_The_Board,
    Not_Already_Carrying,
    Supply_Not_Carried,
    Target_Is_Supply,
)


class Lifeline_Move_Up(Move_Up):
    def action_description_text(self, actor, target_entity, env) -> str:
        return f"Move up to ({actor.position.x}, {actor.position.y - 1})."


class Lifeline_Move_Down(Move_Down):
    def action_description_text(self, actor, target_entity, env) -> str:
        return f"Move down to ({actor.position.x}, {actor.position.y + 1})."


class Lifeline_Move_Left(Move_Left):
    def action_description_text(self, actor, target_entity, env) -> str:
        return f"Move left to ({actor.position.x - 1}, {actor.position.y})."


class Lifeline_Move_Right(Move_Right):
    def action_description_text(self, actor, target_entity, env) -> str:
        return f"Move right to ({actor.position.x + 1}, {actor.position.y})."


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
        # unit, but it never counts toward that zone's total. Otherwise an
        # agent that ignores contamination could inflate a zone's count with
        # garbage and thereby earn the couriers a false sense of balance.
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
        return "Deliver your carried supply to the zone you are standing on."


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
        return "Discard your carried supply."


class Board_Text_Arg(String_Arg):
    def __init__(self) -> None:
        super().__init__(validators=[lambda arg, actor, target, env: bool(arg.strip())])

    def arg_description(self, actor, target_entity, env) -> str:
        return f"your note, at most {MAX_BOARD_TEXT_CHARS} characters"


class Board_Slot_Arg(Int_Range_Arg):
    """Which of the board's fixed slots to write (1-indexed)."""
    def __init__(self) -> None:
        super().__init__(min=1, max=MAX_BOARD_SLOTS + 1)

    def arg_description(self, actor, target_entity, env) -> str:
        return f"int in [1, {MAX_BOARD_SLOTS}] -- which slot to write or overwrite"


class Write_Board(Action):
    """
    Post a short note to one of the board's fixed slots. Requires standing
    near the board. The board never grows past MAX_BOARD_SLOTS entries: an
    empty slot is a free write, an occupied slot is an overwrite that erases
    whatever was there before -- including something a future generation
    might have needed.
    """
    def __init__(self) -> None:
        super().__init__(
            validation_rules=[Target_Is_Self(), Near_The_Board()],
            required_kwargs={"slot": Board_Slot_Arg(), "text": Board_Text_Arg()},
        )

    def exec_action(self, actor, target_entity, env, kwargs) -> dict | None:
        slot = kwargs["slot"]
        full_text = kwargs["text"].strip()
        text = full_text[:MAX_BOARD_TEXT_CHARS]
        previous = env.board_slots[slot - 1]
        previous = dict(previous) if previous is not None else None
        env.board_slots[slot - 1] = {
            "generation": env.generation_index,
            "day": env.current_day,
            "step": env.cur_step + 1,
            "author": actor.name,
            "text": text,
        }
        env.board_version += 1
        env.render_state.emit(
            "board_post", agent=actor.name, text=text, slot=slot,
            overwritten=previous is not None, step=env.cur_step + 1,
        )
        return {
            "posted": text,
            "slot": slot,
            "overwritten": previous is not None,
            "previous": previous,
            "truncated": len(full_text) > MAX_BOARD_TEXT_CHARS,
        }

    def action_description_text(self, actor, target_entity, env) -> str:
        return (
            f"Write a note on the shared board (needs slot 1-{MAX_BOARD_SLOTS} "
            f"and text up to {MAX_BOARD_TEXT_CHARS} characters)."
        )


def describe_selection(selection: Action_Selection) -> str:
    """
    One-line, human-readable description of a chosen action, including its
    arguments. Used for the agent's own action memory, LAST ACTION feedback
    and the event log. Call it BEFORE the step executes: move descriptions
    are computed from the actor's current position.
    """
    if isinstance(selection.action, Write_Board) and selection.action_kwargs:
        text = str(selection.action_kwargs.get("text", "")).strip()
        if len(text) > 80:
            text = text[:77] + "..."
        return f'Write to board slot {selection.action_kwargs.get("slot")}: "{text}"'
    return str(selection).rstrip(".")
