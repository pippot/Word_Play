"""
Everything an agent can do: move supply, write to the board.

Move_* and Do_Nothing come from the engine unchanged; only the
Lifeline-specific actions live here.
"""

from __future__ import annotations

from word_play.core import Action, Target_Is_Self
from word_play.presets.action_args import Int_Range_Arg, String_Arg
from word_play.presets.movement.simple_2d_grid import Position_2D

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
        return f"a short note, <= {MAX_BOARD_TEXT_CHARS} chars"


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
        text = (kwargs or {}).get("text", "").strip()[:MAX_BOARD_TEXT_CHARS]
        index = slot - 1
        overwritten = env.board_slots[index] is not None
        env.board_slots[index] = {
            "generation": env.generation_index,
            "day": env.current_day,
            "step": env.cur_step + 1,
            "author": actor.name,
            "text": text,
        }
        env.board_version += 1
        env.render_state.emit(
            "board_post", agent=actor.name, text=text, slot=slot,
            overwritten=overwritten, step=env.cur_step + 1,
        )
        return {"posted": text, "slot": slot, "overwritten": overwritten}

    def action_description_text(self, actor, target_entity, env) -> str:
        return "Write a note into one of the board's slots (persists across generations)."
