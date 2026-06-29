"""SinglePointLayout — pick a mode, it just works."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from word_play.presets.movement.single_point import Single_Point_Position
from word_play.presets.renderers.layout import Position_Layout_Adapter

if TYPE_CHECKING:
    from word_play.core import Environment, Render_Context


@dataclass
class _State:
    cached_background: list[dict[str, Any]] | None = None
    offsets_by_entity: dict[Any, tuple[float, float]] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# Layout mode presets
# ═══════════════════════════════════════════════════════════════════

_MODES = {
    "ring":      {"radius": 3.5, "room": True,  "table": True},
    "compass":   {"radius": 3.5, "room": True,  "table": True},
    "circle":    {"radius": 3.5, "room": True,  "table": True},
    "boardroom": {"radius": 2.5, "room": True,  "table": False},
    "horseshoe": {"radius": 2.8, "room": True,  "table": False},
    "auditorium":{"radius": 3.0, "room": True,  "table": False},
    "debate":    {"radius": 2.5, "room": True,  "table": False},
}


# ═══════════════════════════════════════════════════════════════════
# Offset generators — each returns exactly n (x, y) tuples
# ═══════════════════════════════════════════════════════════════════

def _ring_offsets(n: int, r: float) -> list[tuple[float, float]]:
    """Evenly spaced circle.  Y compressed 0.7× for isometric feel."""
    return [(r * math.cos(a), r * math.sin(a) * 0.7)
            for a in (-math.pi / 2 + 2 * math.pi * i / n for i in range(n))]


def _compass_offsets(n: int, r: float) -> list[tuple[float, float]]:
    """N/E/S/W for 1–4, ring beyond."""
    if n == 1: return [(0.0, 0.0)]
    if n == 2: return [(-r, 0.0), (r, 0.0)]
    if n == 3: return [(0.0, -r), (r * 0.866, r * 0.5), (-r * 0.866, r * 0.5)]
    if n == 4: return [(0.0, -r), (r, 0.0), (0.0, r), (-r, 0.0)]
    return _ring_offsets(n, r)


def _boardroom_offsets(n: int, r: float) -> list[tuple[float, float]]:
    """Rectangular table — agents on both long sides and ends.

    Layout (top-down view):
        ┌─────────────────┐
        │ ○  ○  ○  ○  ○  │ ← top row  (max per_side agents)
        │○                 ○│ ← left end (1-2)
        │                  │
        │○                 ○│ ← right end (1-2)
        │ ○  ○  ○  ○  ○  │ ← bottom row
        └─────────────────┘
    """
    w, h = r * 1.2, r * 0.8
    # How many seats per long side?
    # For n agents: roughly 40% on top, 40% on bottom, 20% on ends
    per_side = max(1, int(n * 0.4))
    if per_side * 2 >= n:
        per_side = (n + 1) // 2

    def _interp(i, count):
        """Spread i across [-w, +w] for count items."""
        if count <= 1:
            return 0.0
        return ((i / (count - 1)) * 2 - 1) * w

    offs = []
    remaining = n

    # Top row
    top_n = min(per_side, remaining)
    offs.extend((_interp(i, top_n), -h) for i in range(top_n))
    remaining -= top_n

    # Bottom row
    bot_n = min(per_side, remaining)
    offs.extend((_interp(i, bot_n), h) for i in range(bot_n))
    remaining -= bot_n

    # Left side (1-2 agents)
    left_n = min(2, remaining)
    if left_n > 0:
        offs.extend((-w, y) for y in (-h * 0.5, h * 0.5)[:left_n])
        remaining -= left_n

    # Right side
    right_n = min(2, remaining)
    if right_n > 0:
        offs.extend((w, y) for y in (-h * 0.5, h * 0.5)[:right_n])
        remaining -= right_n

    # Overflow → extra ring
    if remaining > 0:
        offs.extend(_ring_offsets(remaining, r * 1.5))
    return offs


def _horseshoe_offsets(n: int, r: float) -> list[tuple[float, float]]:
    """U-shaped table, open at the bottom.

    Layout:
        ┌─────────────────┐
        │ ○  ○   ○   ○  ○│ ← top row
        │○                 │ ← left column
        │○                 │
        │                  │ ← *open* (audience/presenter faces this way)
        └─────────────────┘
    """
    w, h = r * 1.2, r * 0.7
    per_side = max(2, int(n * 0.45))
    if per_side * 2 > n:
        per_side = max(2, (n + 1) // 2)

    def _interp(i, count):
        if count <= 1:
            return 0.0
        return ((i / (count - 1)) * 2 - 1) * w

    offs = []
    remaining = n

    # Top row
    top_n = min(per_side, remaining)
    offs.extend((_interp(i, top_n), -h) for i in range(top_n))
    remaining -= top_n

    # Left column (from top toward bottom, but stopping above the gap)
    left_n = min(per_side - 1, remaining)
    if left_n > 0:
        for i in range(left_n):
            frac = (i + 1) / (left_n + 1)  # 0.33 .. 0.75
            offs.append((-w, -h + frac * 3 * h * 0.5))
        remaining -= left_n

    # Right column
    right_n = min(per_side - 1, remaining)
    if right_n > 0:
        for i in range(right_n):
            frac = (i + 1) / (right_n + 1)
            offs.append((w, -h + frac * 3 * h * 0.5))
        remaining -= right_n

    if remaining > 0:
        offs.extend(_ring_offsets(remaining, r * 1.5))
    return offs


def _auditorium_offsets(n: int, r: float) -> list[tuple[float, float]]:
    """Rows facing a stage.  Back rows higher (farther from stage).

    Layout:
        ┌──────────────────┐
        │    [ S T A G E ] │ ← front (y ≈ -r)
        │ ○ ○ ○ ○ ○ ○ ○ ○ │ ← row 1 (front)
        │ ○ ○ ○ ○ ○ ○ ○   │ ← row 2
        │   ○ ○ ○ ○ ○     │ ← row 3 (back, wider view but fewer seats)
        └──────────────────┘
    """
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0)]
    if n == 2:
        return [(-r * 0.3, 0.0), (r * 0.3, 0.0)]

    cols = min(8, int(math.sqrt(n) * 1.6))
    offs = []
    remaining = n
    row_spacing = r * 0.55
    col_spacing = r * 0.35

    for row_idx in range(10):  # safety cap
        if remaining <= 0:
            break
        row_cols = min(cols, remaining)
        y = r + row_idx * row_spacing  # stage at -r, rows go up
        for c in range(row_cols):
            x = -(row_cols - 1) * col_spacing / 2 + c * col_spacing
            offs.append((x, y))
        remaining -= row_cols
    return offs


def _debate_offsets(n: int, r: float) -> list[tuple[float, float]]:
    """Two sides facing each other across a gap.

    Layout:
        ○ ○ ○              ← pro side (left)
            ○ ○ ○          ← con side (right)
    Evenly split into left/right groups, stacked vertically.
    """
    w, h = r * 0.6, r * 0.8

    def _row_offsets(count, x_sign):
        if count == 0:
            return []
        if count == 1:
            return [(x_sign * w, 0.0)]
        return [(x_sign * w, ((i / (count - 1)) * 2 - 1) * h) for i in range(count)]

    left_n = (n + 1) // 2
    right_n = n - left_n
    return _row_offsets(left_n, -1) + _row_offsets(right_n, +1)


# ═══════════════════════════════════════════════════════════════════
# Dispatch
# ═══════════════════════════════════════════════════════════════════

_FN = {
    "ring": _ring_offsets, "circle": _ring_offsets,
    "compass": _compass_offsets,
    "boardroom": _boardroom_offsets,
    "horseshoe": _horseshoe_offsets,
    "auditorium": _auditorium_offsets,
    "debate": _debate_offsets,
}


# ═══════════════════════════════════════════════════════════════════
# Room background generation
# ═══════════════════════════════════════════════════════════════════

def _room_bounds(mode: str, n: int) -> tuple[int, int]:
    """Background room dimensions (width, height) for each mode."""
    if mode == "boardroom":
        return (max(14, int(n * 0.8 + 2)), max(7, int(n * 0.4 + 2)))
    if mode == "horseshoe":
        return (max(5, int(n * 0.7 + 2)), max(5, int(n * 0.6 + 2)))
    if mode == "auditorium":
        return (max(7, int(n * 0.45 + 2)), max(6, int(n * 0.55 + 2)))
    if mode == "debate":
        return (max(5, int(n * 0.4 + 2)), max(5, int(n * 0.35 + 2)))
    # ring / compass / circle
    if n <= 4:  return (5, 5)
    if n <= 8:  return (7, 7)
    if n <= 12: return (9, 7)
    return (11, 9)


def _generate_room_tiles(mode: str, has_table: bool, assets: dict, n_agents: int, base_x: int, base_y: int) -> list[dict]:
    """Generate floor + wall tiles for a single-point room."""
    rw, rh = _room_bounds(mode, n_agents)
    hw, hh = rw // 2, rh // 2
    floor_sprite = assets["floor"]
    table_sprite = assets["table"]
    wall_set = assets["wall"]
    tiles = []

    # Floor
    for y in range(-hh, hh + 1):
        for x in range(-hw, hw + 1):
            # Center table for ring/compass
            is_table = (x == 0 and y == 0 and has_table)
            tiles.append({
                "x": base_x + x, "y": base_y + y,
                "kind": "floor",
                "sprite": table_sprite if is_table else floor_sprite,
            })

    # Room-type-specific floor decorations
    if mode == "boardroom":
        # Draw a long table in the centre
        table_w = min(hw - 1, max(2, int(n_agents * 0.35)))
        for tx in range(-table_w, table_w + 1):
            tiles.append({"x": base_x + tx, "y": base_y, "kind": "floor", "sprite": table_sprite})
    elif mode == "horseshoe":
        # Draw a presenter podium at the open end
        tiles.append({"x": base_x, "y": base_y + hh - 1, "kind": "floor", "sprite": table_sprite})
    elif mode == "auditorium":
        # Draw a stage strip at the front (y = -hh)
        for sx in range(-hw, hw + 1):
            tiles.append({"x": base_x + sx, "y": base_y - hh, "kind": "floor", "sprite": table_sprite})
    elif mode == "debate":
        # Draw a dividing line down the centre
        for dy in range(-hh, hh + 1):
            tiles.append({"x": base_x, "y": base_y + dy, "kind": "floor", "sprite": table_sprite})

    # Walls
    for y in range(-hh - 1, hh + 2):
        for x in range(-hw - 1, hw + 2):
            if y == -hh - 1 or y == hh + 1 or x == -hw - 1 or x == hw + 1:
                tiles.append({
                    "x": base_x + x, "y": base_y + y,
                    "kind": "wall", "wall_set": wall_set,
                })
    return tiles


# ═══════════════════════════════════════════════════════════════════
# Assets
# ═══════════════════════════════════════════════════════════════════

_MODE_ASSETS = {
    "boardroom": {
        "wall": "sprite_library/src/world_tiles/indoors/wall_sets/bright_brick_wall",
        "floor": "sprite_library/src/world_tiles/floors_custom/wood_floor.png",
        "table": "sprite_library/src/world_tiles/indoors/furniture/table_wood.png",
    },
    "horseshoe": {
        "wall": "sprite_library/src/world_tiles/indoors/wall_sets/bright_fort_wall",
        "floor": "sprite_library/src/world_tiles/indoors/floors/day_brick_floor_c.png",
        "table": "sprite_library/src/world_tiles/indoors/furniture/table_wood.png",
    },
    "auditorium": {
        "wall": "sprite_library/src/world_tiles/indoors/wall_sets/dim_brick_wall",
        "floor": "sprite_library/src/world_tiles/indoors/floors/day_brick_floor_c.png",
        "table": "sprite_library/src/world_tiles/indoors/stations/prep_table.png",
    },
    "debate": {
        "wall": "sprite_library/src/world_tiles/indoors/wall_sets/dim_fort_wall",
        "floor": "sprite_library/src/world_tiles/indoors/floors/day_brick_floor_c.png",
        "table": "sprite_library/src/world_tiles/indoors/stations/prep_table.png",
    },
}


def _assets(mode: str) -> dict:
    a = _MODE_ASSETS.get(mode, {})
    return {
        "wall": a.get("wall", "sprite_library/src/world_tiles/indoors/wall_sets/overcooked_kitchen_wall"),
        "floor": a.get("floor", "sprite_library/src/world_tiles/indoors/floors/day_brick_floor_c.png"),
        "table": a.get("table", "sprite_library/src/world_tiles/indoors/stations/prep_table.png"),
    }


# ═══════════════════════════════════════════════════════════════════
# Layout adapter
# ═══════════════════════════════════════════════════════════════════


class SinglePointLayout(Position_Layout_Adapter):
    """Pick a mode — everything else auto-configures.

    Modes and how they arrange agents::

        ring        Evenly spaced circle.  Works for 1–∞.  Default.

        compass     1–4: N/E/S/W slots.  5+: same as ring.  Best ≤4.

        boardroom   Rectangular table.  Works 2–20.  Top+bottom rows
                    (≈80 % of agents) + left/right ends (1–2 each).

        horseshoe   U-shaped table, open at bottom.  Works 3–15.
                    Top row + left/right columns.  Presenter faces
                    the open end.

        auditorium  Grid rows facing stage.  Works 3–40.  Columns
                    scale with √n.  Stage rendered at front.

        debate      Two facing columns even split.  Works 2–12.
                    Dividing line down the centre.
    """

    def __init__(self, layout_mode: str = "ring"):
        cfg = _MODES.get(layout_mode, _MODES["ring"])
        self.base_x = 0
        self.base_y = 0
        self.radius = cfg["radius"]
        self.layout_mode = layout_mode
        self.include_room = cfg["room"]
        self.has_center_table = cfg["table"]
        self.only_agents = False
        asst = _assets(layout_mode)
        self.WALL_SET = asst["wall"]
        self.DEFAULT_FLOOR = asst["floor"]
        self.TABLE_SPRITE = asst["table"]
        self.groups: list[list[str]] = []

    def set_groups(self, *groups: list[str]) -> SinglePointLayout:
        """Assign agents to sides by group.  Group 0 fills the first
        position slots (left/top), group 1 fills the second set
        (right/bottom), and so on.

        Example::

            layout = SinglePointLayout("debate")
            layout.set_groups(["Alice", "Bob"], ["Charlie", "Dana"])
            # → Alice, Bob on left (pro); Charlie, Dana on right (con)
        """
        self.groups = [list(g) for g in groups]
        return self

    def _state(self, ctx):
        if ctx is None:
            raise ValueError("SinglePointLayout requires a Render_Context.")
        return ctx.value_for(self, _State)

    def prepare_env(self, env: "Environment", context: "Render_Context | None" = None) -> None:
        if env is None:
            return
        st = self._state(context)
        ents = list(getattr(env.state, "entities", []))
        pos = ([e for e in ents if getattr(e, "is_agent", False)] if self.only_agents
               else [e for e in ents if isinstance(getattr(e, "position", None), Single_Point_Position)])
        n = len(pos)
        st.offsets_by_entity.clear()
        if n == 0:
            return
        pos.sort(key=lambda e: (0 if getattr(e, "is_agent", False) else 1, e.name))
        # Honour agent groups if set
        if self.groups:
            group_map: dict[str, int] = {}
            for gi, g in enumerate(self.groups):
                for name in g:
                    group_map[name] = gi
            pos.sort(key=lambda e: (group_map.get(e.name, len(self.groups)), e.name))
        if self.include_room:
            st.cached_background = None
        fn = _FN.get(self.layout_mode, _ring_offsets)
        offsets = fn(n, self.radius)
        assert len(offsets) == n, f"{self.layout_mode}: got {len(offsets)} offsets for {n} agents"
        for e, (ox, oy) in zip(pos, offsets):
            st.offsets_by_entity[e] = (ox, oy)

    def background(self, env, positioned_entities, context=None):
        if not self.include_room:
            return []
        st = self._state(context)
        if st.cached_background is not None:
            return st.cached_background
        if env is None:
            return []
        n = len([e for e in getattr(env.state, "entities", []) if getattr(e, "is_agent", False)])
        if n == 0:
            return []
        st.cached_background = _generate_room_tiles(
            self.layout_mode, self.has_center_table, _assets(self.layout_mode),
            n, self.base_x, self.base_y,
        )
        return st.cached_background

    def screen_position(self, entity, env, context=None):
        pos = getattr(entity, "position", entity)
        if isinstance(pos, Single_Point_Position):
            ox, oy = self._state(context).offsets_by_entity.get(entity, (0.0, 0.0))
            return (self.base_x + ox, self.base_y + oy)
        if pos is not None:
            x, y = getattr(pos, "x", None), getattr(pos, "y", None)
            if x is not None and y is not None:
                return (float(x), float(y))
        return (self.base_x, self.base_y)
