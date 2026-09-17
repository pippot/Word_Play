"""
Layout: the landmark and hazard coordinates read out of the tilemap.

world.py builds live entities from the tilemap; prompts.py and metrics need
the same coordinates as plain data (to describe the map to agents, and to
score board claims against the true hazards). Both read them from here so
there is exactly one source of truth: config.ENTITY_TILEMAP.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from word_play.utils import tilemap_to_entities

from .config import ENTITY_TILEMAP

# Tilemap symbol -> landmark name. Anything not listed ("W", ".") is not a
# landmark.
_ZONE_SYMBOLS = {"1": "Zone_Near", "2": "Zone_Mid", "3": "Zone_Far"}


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    spawn: tuple[int, int]
    board: tuple[int, int]
    zones: dict[str, tuple[int, int]]
    hazards: frozenset[tuple[int, int]]

    @property
    def floor_x(self) -> tuple[int, int]:
        """Inclusive x range of walkable floor (inside the boundary wall)."""
        return (1, self.width - 2)

    @property
    def floor_y(self) -> tuple[int, int]:
        """Inclusive y range of walkable floor (inside the boundary wall)."""
        return (1, self.height - 2)

    @property
    def landmarks(self) -> frozenset[tuple[int, int]]:
        """Tiles that are never contaminated: spawn, board and zones."""
        return frozenset({self.spawn, self.board, *self.zones.values()})


@lru_cache(maxsize=None)
def parse_layout(tilemap: str = ENTITY_TILEMAP) -> Layout:
    placeholder = {"name": "", "tags": [], "components": []}
    tileset = {
        symbol: {**placeholder, "name": name}
        for symbol, name in {
            "W": "wall", "X": "spawn", "B": "board", "H": "hazard", **_ZONE_SYMBOLS,
        }.items()
    }
    entities = tilemap_to_entities(tilemap, tileset)

    def positions(name: str) -> list[tuple[int, int]]:
        return [(e.position.x, e.position.y) for e in entities if e.name == name]

    rows = [row for row in tilemap.strip("\n").split("\n")]
    return Layout(
        width=len(rows[0]),
        height=len(rows),
        spawn=positions("spawn")[0],
        board=positions("board")[0],
        zones={name: positions(name)[0] for name in _ZONE_SYMBOLS.values()},
        hazards=frozenset(positions("hazard")),
    )
