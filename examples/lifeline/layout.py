"""
Layout: the landmark and hazard coordinates read out of the tilemap, and the
per-generation schedule of the hazards that move.

world.py builds live entities from the tilemap; prompts.py and metrics need
the same coordinates as plain data (to describe the map to agents, and to
score board claims against the true hazards). Both read them from here so
there is exactly one source of truth: config.ENTITY_TILEMAP.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from functools import lru_cache

from word_play.utils import tilemap_to_entities

from .config import ENTITY_TILEMAP, MOVING_HAZARD_REGION

# Tilemap symbol -> landmark name. Anything not listed ("W", ".") is not a
# landmark.
_ZONE_SYMBOLS = {"1": "Zone_Near", "2": "Zone_Mid", "3": "Zone_Far"}

Tile = tuple[int, int]


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    spawn: Tile
    board: Tile
    zones: dict[str, Tile]
    fixed_hazards: frozenset[Tile]
    moving_hazards: frozenset[Tile]  # generation-1 positions of the hazards that move

    @property
    def hazards(self) -> frozenset[Tile]:
        """Generation-1 hazards: the fixed ones plus the moving ones as drawn."""
        return self.fixed_hazards | self.moving_hazards

    @property
    def floor_x(self) -> tuple[int, int]:
        """Inclusive x range of walkable floor (inside the boundary wall)."""
        return (1, self.width - 2)

    @property
    def floor_y(self) -> tuple[int, int]:
        """Inclusive y range of walkable floor (inside the boundary wall)."""
        return (1, self.height - 2)

    @property
    def landmarks(self) -> frozenset[Tile]:
        """Tiles that are never contaminated: spawn, board and zones."""
        return frozenset({self.spawn, self.board, *self.zones.values()})

    def on_floor(self, tile: Tile) -> bool:
        (x0, x1), (y0, y1) = self.floor_x, self.floor_y
        return x0 <= tile[0] <= x1 and y0 <= tile[1] <= y1


@lru_cache(maxsize=None)
def parse_layout(tilemap: str = ENTITY_TILEMAP) -> Layout:
    placeholder = {"name": "", "tags": [], "components": []}
    tileset = {
        symbol: {**placeholder, "name": name}
        for symbol, name in {
            "W": "wall", "X": "spawn", "B": "board", "H": "fixed_hazard",
            "M": "moving_hazard", **_ZONE_SYMBOLS,
        }.items()
    }
    entities = tilemap_to_entities(tilemap, tileset)

    def positions(name: str) -> list[Tile]:
        return [(e.position.x, e.position.y) for e in entities if e.name == name]

    rows = [row for row in tilemap.strip("\n").split("\n")]
    return Layout(
        width=len(rows[0]),
        height=len(rows),
        spawn=positions("spawn")[0],
        board=positions("board")[0],
        zones={name: positions(name)[0] for name in _ZONE_SYMBOLS.values()},
        fixed_hazards=frozenset(positions("fixed_hazard")),
        moving_hazards=frozenset(positions("moving_hazard")),
    )


# ============================================================================
# ROUTES
# ============================================================================

def clean_distance(layout: Layout, start: Tile, goal: Tile, hazards: frozenset[Tile] | set[Tile]) -> int | None:
    """Length of the shortest walk from start to goal that never steps on a
    hazard, or None if there is none."""
    seen, queue = {start}, deque([(start, 0)])
    while queue:
        (x, y), dist = queue.popleft()
        if (x, y) == goal:
            return dist
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nxt in seen or nxt in hazards or not layout.on_floor(nxt):
                continue
            seen.add(nxt)
            queue.append((nxt, dist + 1))
    return None


def has_clean_shortest_path(start: Tile, goal: Tile, hazards: frozenset[Tile] | set[Tile]) -> bool:
    """Is there a monotone (Manhattan-shortest) route from start to goal that
    avoids every hazard?"""
    seen, queue = {start}, deque([start])
    while queue:
        position = queue.popleft()
        if position == goal:
            return True
        x, y = position
        steps = []
        if goal[0] != x:
            steps.append((x + (1 if goal[0] > x else -1), y))
        if goal[1] != y:
            steps.append((x, y + (1 if goal[1] > y else -1)))
        for nxt in steps:
            if nxt in seen or nxt in hazards:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return False


def keeps_pacing(layout: Layout, hazards: frozenset[Tile]) -> bool:
    """
    The constraints every generation's hazards must satisfy, so the pacing
    the map is tuned for holds in every generation:
      * Zone_Mid and Zone_Far keep a hazard-free shortest path (knowing the
        map lets a courier stay optimal, not merely safe);
      * Zone_Near's clean route stays the 2-step detour around the lazy
        route's hazard, never longer.
    """
    spawn = layout.spawn
    near = layout.zones["Zone_Near"]
    manhattan_near = abs(near[0] - spawn[0]) + abs(near[1] - spawn[1])
    return (
        has_clean_shortest_path(spawn, layout.zones["Zone_Mid"], hazards)
        and has_clean_shortest_path(spawn, layout.zones["Zone_Far"], hazards)
        and clean_distance(layout, spawn, near, hazards) == manhattan_near + 2
    )


# ============================================================================
# MOVING HAZARDS
# ============================================================================

def moving_hazard_candidates(layout: Layout) -> list[Tile]:
    """Tiles a moving hazard may be placed on: inside MOVING_HAZARD_REGION (the
    travel routes between spawn and the zones), never a landmark or a fixed
    hazard, and never next to the spawn point or the board, which everyone
    passes on every trip."""
    (x0, x1), (y0, y1) = MOVING_HAZARD_REGION
    crowded = {layout.spawn, layout.board}
    return [
        (x, y)
        for x in range(x0, x1 + 1)
        for y in range(y0, y1 + 1)
        if (x, y) not in layout.landmarks
        and (x, y) not in layout.fixed_hazards
        and all(abs(x - cx) + abs(y - cy) > 1 for cx, cy in crowded)
    ]


def hazard_schedule(seed: int, num_generations: int, layout: Layout | None = None) -> list[frozenset[Tile]]:
    """
    The hazard set for every generation of a run. Generation 1 is the map as
    drawn; from generation 2 on, every moving hazard jumps to a new candidate
    tile (never the tile it was on the generation before), redrawn until the
    pacing constraints hold. Depends only on the seed, so a control and a
    treatment run with the same seed face identical hazards.
    """
    layout = layout or parse_layout()
    rng = random.Random(f"hazards-{seed}")
    candidates = moving_hazard_candidates(layout)
    schedule = [layout.hazards]
    previous = layout.moving_hazards
    for _ in range(1, num_generations):
        pool = [tile for tile in candidates if tile not in previous]
        for _attempt in range(1000):
            moved = frozenset(rng.sample(pool, len(layout.moving_hazards)))
            if keeps_pacing(layout, layout.fixed_hazards | moved):
                break
        else:  # pragma: no cover -- the region is far too large for this to happen
            raise RuntimeError("could not place the moving hazards without breaking the map's pacing")
        schedule.append(layout.fixed_hazards | moved)
        previous = moved
    return schedule
