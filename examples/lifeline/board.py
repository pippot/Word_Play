"""
The shared board: one artifact, four sections, each with its own rules about
who may change what.

  notes       free text in a fixed number of slots. Writing to a filled slot
              erases it. Everything that isn't a hazard, an agreement or a
              delivery count lives here: argument, news, accusation.
  hazards     append-only tile reports, one row per tile with its reporters.
              Nothing can be erased, so a warning can never be destroyed --
              and a false report can never be taken back either.
  agreement   one free-text block: how the team has agreed to run the relay.
              Anyone may replace it, but the replacement is visible: the
              section carries the last few changes with their authors.
  deliveries  one row per person, only that person may change it. It belongs
              to the rotation and is cleared when the rotation ends, so it
              lives on the environment (see Lifeline_Env.delivery_reports)
              rather than here.

The first three cross rotation boundaries -- they are the only thing a new
team inherits, which is why transmission is measured on them. The Board object
is handed from one generation's environment to the next (see
experiment.run_experiment), so it is a plain data holder with no environment
reference: every mutation takes the stamp (author, generation, day, step)
from its caller.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from .config import (
    AGREEMENT_HISTORY_SHOWN,
    MAX_BOARD_SLOTS,
    MAX_HAZARD_REPORTERS_PER_TILE,
    MAX_HAZARD_TILES,
)

Tile = tuple[int, int]

# What a reporter claims about a tile. "fixed" and "moving" are the two
# claims a courier cannot check within a rotation (hazards never move during
# one), so both are cheap to assert and expensive to falsify -- which is the
# point: it is the structured version of the (12,5) myth.
HAZARD_KINDS: tuple[str, ...] = ("fixed", "moving", "unsure")


@dataclass
class Board:
    """The board a rotation inherits. Mutated in place by the actions."""

    slots: list[dict | None] = field(default_factory=lambda: [None] * MAX_BOARD_SLOTS)
    hazards: list[dict] = field(default_factory=list)
    agreement: dict | None = None
    # Every agreement this board has had, oldest first, without the current
    # one. Kept whole for the metrics; only the last AGREEMENT_HISTORY_SHOWN
    # are ever shown to an agent.
    agreement_history: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------ construction

    @classmethod
    def empty(cls, slot_count: int = MAX_BOARD_SLOTS) -> "Board":
        return cls(slots=[None] * slot_count)

    def snapshot(self) -> dict:
        """A JSON-safe deep copy, for the event log and the checkpoint."""
        return deepcopy({
            "slots": self.slots,
            "hazards": self.hazards,
            "agreement": self.agreement,
            "agreement_history": self.agreement_history,
        })

    @classmethod
    def from_snapshot(cls, data) -> "Board":
        """
        Rebuild from a snapshot. A bare list is a board from before the
        sections existed (old checkpoints and logs), and loads as notes only.
        """
        if data is None:
            return cls.empty()
        if isinstance(data, list):
            return cls(slots=deepcopy(data))
        data = deepcopy(data)
        return cls(
            slots=data.get("slots") or [None] * MAX_BOARD_SLOTS,
            hazards=data.get("hazards") or [],
            agreement=data.get("agreement"),
            agreement_history=data.get("agreement_history") or [],
        )

    # ------------------------------------------------------------------ notes

    @property
    def filled(self) -> int:
        return sum(1 for slot in self.slots if slot is not None)

    def write_slot(self, index: int, entry: dict) -> dict | None:
        """Write slot `index` (0-based). Returns whatever it erased, or None."""
        previous = self.slots[index]
        previous = dict(previous) if previous is not None else None
        self.slots[index] = entry
        return previous

    # ------------------------------------------------------------------ hazards

    def hazard_rows(self) -> list[tuple[Tile, list[dict]]]:
        """One row per tile with its reports, tiles in first-reported order."""
        rows: dict[Tile, list[dict]] = {}
        for report in self.hazards:
            rows.setdefault(tuple(report["tile"]), []).append(report)
        return list(rows.items())

    def hazard_tiles(self) -> list[Tile]:
        return [tile for tile, _ in self.hazard_rows()]

    def reported_hazard(self, tile: Tile, author: str) -> bool:
        return any(tuple(r["tile"]) == tuple(tile) and r["author"] == author for r in self.hazards)

    def add_hazard(
        self, tile: Tile, *, kind: str, author: str, generation: int, day: int, step: int
    ) -> dict | None:
        """
        Append a hazard report. Nothing checks it: an agent may report a tile
        it has never stepped on, and may call a tile fixed or moving on no
        evidence -- that is the deception channel this section exists to make
        attributable and permanent.

        Returns the new report, or None if this author had already reported
        this tile (the section holds one report per tile per author, so
        re-reporting cannot be used to bury the list).
        """
        tile = (int(tile[0]), int(tile[1]))
        if kind not in HAZARD_KINDS:
            raise ValueError(f"hazard kind must be one of {HAZARD_KINDS}, got {kind!r}")
        if self.reported_hazard(tile, author):
            return None
        existing = [r for r in self.hazards if tuple(r["tile"]) == tile]
        if len(existing) >= MAX_HAZARD_REPORTERS_PER_TILE:
            return None
        report = {
            "tile": [tile[0], tile[1]], "kind": kind, "author": author,
            "generation": generation, "day": day, "step": step,
        }
        self.hazards.append(report)
        self._evict_oldest_hazard_tiles()
        return report

    def _evict_oldest_hazard_tiles(self) -> None:
        """
        The section is append-only to the agents, but it cannot grow without
        bound across a long run (three hazards move every rotation, so new
        tiles keep arriving). Past MAX_HAZARD_TILES distinct tiles the
        least recently reported tile drops off, with all of its reports --
        a board that has run out of room, not an agent erasing a warning.
        """
        rows = self.hazard_rows()
        while len(rows) > MAX_HAZARD_TILES:
            oldest = min(rows, key=lambda row: max(r["step"] for r in row[1]))
            self.hazards = [r for r in self.hazards if tuple(r["tile"]) != oldest[0]]
            rows = self.hazard_rows()

    # ------------------------------------------------------------------ agreement

    def set_agreement(
        self, text: str, *, author: str, generation: int, day: int, step: int
    ) -> dict | None:
        """Replace the working agreement. Returns the one it replaced, or None."""
        previous = self.agreement
        if previous is not None:
            self.agreement_history.append(previous)
        self.agreement = {
            "text": text, "author": author,
            "generation": generation, "day": day, "step": step,
        }
        return previous

    def recent_agreements(self, count: int = AGREEMENT_HISTORY_SHOWN) -> list[dict]:
        """The last few replaced agreements, most recently replaced first."""
        return list(reversed(self.agreement_history[-count:]))
