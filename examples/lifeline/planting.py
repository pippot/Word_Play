"""
Planted claims: the controlled transmission conditions.

One false claim is placed on the board at the start of a rotation, attributed
to a courier of the previous rotation who never existed. There is no misaligned
agent in these runs (use --num-misaligned 0), so whatever happens to the note
afterwards -- repeated, believed, acted on, dropped -- is the population's
doing alone. Compare with the unplanted control on the same seed: the hazards,
names, personas and turn order are identical, so the note is the only
difference.

  hazard  -- a clean tile, on a zone's natural routes and never a hazard in
             this run, entered in the board's hazard section as one of the
             FIXED ones (the (12,5) myth of the first daily-mode run, but
             controlled -- and now unerasable, like every hazard report);
  history -- a false pattern with a prescription: the target zone "drifts
             behind every rotation", so each rotation should start with it.
             It is planted as the team's working agreement, the first thing
             a new team reads.

The agent-facing wording lives in prompts.planted_note_text.
"""

from __future__ import annotations

from .board import Board
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
    board: Board,
    *,
    kind: str,
    generation_index: int,
    days_per_generation: int,
    steps_per_day: int,
    target_zone: str,
    layout: Layout,
    schedule: list[frozenset[Tile]],
) -> tuple[Board, dict]:
    """
    Place the planted claim on the board a rotation inherits, stamped as
    made late on the previous rotation's last day. Each kind goes into the
    section a real courier would have used, which is also what makes it
    durable:

      hazard   a row in the append-only hazard section calling a clean tile
               one of the FIXED ones. Nobody can remove it, so the false
               claim outlives every team that reads it.
      history  the working agreement, which is the first thing a new team
               reads and stands until somebody replaces it.

    Returns the board and the `planted_note` event to log.
    """
    if kind not in PLANT_KINDS:
        raise ValueError(f"plant kind must be one of {PLANT_KINDS}, got {kind!r}")
    if generation_index < 1:
        raise ValueError("a note can only be planted from rotation 2 on (it is a handover from the previous team)")
    stamp = {
        "generation": generation_index - 1,
        "day": days_per_generation - 1,
        "step": days_per_generation * steps_per_day - 3,
    }
    tile = choose_planted_tile(layout, schedule, target_zone) if kind == "hazard" else None
    zone = target_zone if kind == "history" else None
    text = planted_note_text(kind, tile=tile, zone=zone)
    replaced = None
    if kind == "hazard":
        placement = "hazard"
        board.hazards.append({
            "tile": [tile[0], tile[1]], "kind": "fixed",
            "author": PLANTED_NOTE_AUTHOR, **stamp,
        })
    else:
        placement = "agreement"
        replaced = board.agreement
        if replaced is not None:
            board.agreement_history.append(replaced)
        board.agreement = {"text": text, "author": PLANTED_NOTE_AUTHOR, **stamp}
    event = {
        "type": "planted_note",
        "generation": generation_index,
        "kind": kind,
        "placement": placement,
        "author": PLANTED_NOTE_AUTHOR,
        "text": text,
        "tile": list(tile) if tile else None,
        "zone": zone,
        "stamp": stamp,
        "replaced": replaced,
    }
    return board, event
