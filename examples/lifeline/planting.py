"""
Planted notes: the controlled transmission conditions.

One false note is placed on the board at the start of a rotation, signed by a
courier of the previous rotation who never existed. There is no misaligned
agent in these runs (use --num-misaligned 0), so whatever happens to the note
afterwards -- repeated, believed, acted on, dropped -- is the population's
doing alone. Compare with the unplanted control on the same seed: the hazards,
names, personas and turn order are identical, so the note is the only
difference.

  hazard  -- a clean tile, on a zone's natural routes and never a hazard in
             this run, called one of the FIXED hazards (like the (12,5) myth
             of the first daily-mode run, but controlled);
  history -- a false pattern with a prescription: the target zone "drifts
             behind every rotation", so each rotation should start with it.

The note texts are agent-facing and live in prompts.planted_note_text.
"""

from __future__ import annotations

from .config import PLANTED_NOTE_AUTHOR
from .layout import Layout, moving_hazard_regions
from .prompts import PLANT_KINDS, planted_note_text

Tile = tuple[int, int]


def choose_planted_tile(layout: Layout, schedule: list[frozenset[Tile]], target_zone: str) -> Tile:
    """
    A tile that is never contaminated in this run, on the most natural route
    to the first zone that isn't the target (a false hazard on the target's
    own route would push couriers away from it and confound the target
    measures): as close as possible to the diagonal between the depot and
    that zone, and to the middle of it.
    """
    zone = next(name for name in layout.zones if name != target_zone)
    ever = frozenset().union(*schedule)
    sx, sy = layout.spawn
    zx, zy = layout.zones[zone]
    half = (abs(zx - sx) + abs(zy - sy)) / 4

    def key(tile: Tile):
        dx, dy = abs(tile[0] - sx), abs(tile[1] - sy)
        return (abs(dx - dy), abs(dx - half) + abs(dy - half), tile)

    candidates = [t for t in moving_hazard_regions(layout)[zone] if t not in ever]
    if not candidates:  # pragma: no cover -- the regions are far larger than a run's hazards
        raise RuntimeError(f"no never-contaminated tile left near {zone} to plant")
    return min(candidates, key=key)


def plant_note(
    board_slots: list[dict | None],
    *,
    kind: str,
    generation_index: int,
    days_per_generation: int,
    steps_per_day: int,
    target_zone: str,
    layout: Layout,
    schedule: list[frozenset[Tile]],
) -> tuple[list[dict | None], dict]:
    """
    Place the planted note on the board a rotation inherits. It goes into
    the first empty slot or, on a full board, over the oldest note, and is
    stamped as written late on the previous rotation's last day. Returns the
    new board and the `planted_note` event to log.
    """
    if kind not in PLANT_KINDS:
        raise ValueError(f"plant kind must be one of {PLANT_KINDS}, got {kind!r}")
    if generation_index < 1:
        raise ValueError("a note can only be planted from rotation 2 on (it is a handover from the previous team)")
    tile = choose_planted_tile(layout, schedule, target_zone) if kind == "hazard" else None
    zone = target_zone if kind == "history" else None
    text = planted_note_text(kind, tile=tile, zone=zone)
    stamp = {
        "generation": generation_index - 1,
        "day": days_per_generation - 1,
        "step": days_per_generation * steps_per_day - 3,
    }
    empty = [i for i, slot in enumerate(board_slots) if slot is None]
    if empty:
        index = empty[0]
    else:
        index = min(
            range(len(board_slots)),
            key=lambda i: (board_slots[i]["generation"], board_slots[i]["day"], board_slots[i]["step"]),
        )
    replaced = board_slots[index]
    board = list(board_slots)
    board[index] = {**stamp, "author": PLANTED_NOTE_AUTHOR, "text": text}
    event = {
        "type": "planted_note",
        "generation": generation_index,
        "kind": kind,
        "slot": index + 1,
        "author": PLANTED_NOTE_AUTHOR,
        "text": text,
        "tile": list(tile) if tile else None,
        "zone": zone,
        "stamp": stamp,
        "replaced": replaced,
    }
    return board, event
