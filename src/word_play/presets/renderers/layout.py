from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from word_play.presets.movement.single_point import Single_Point_Position

if TYPE_CHECKING:
    from word_play.core import Environment, Render_Context


_DEFAULT_FLOOR = "src/world_tiles/indoors/floors/day_brick_floor_c.png"


@dataclass(slots=True)
class _SinglePointLayoutState:
    cached_background: list[dict[str, Any]] | None = None
    offsets_by_entity: dict[Any, tuple[float, float]] = field(default_factory=dict)


def _render_frame(env: "Environment" | None) -> dict[str, Any]:
    if env is None:
        return {}
    render_state = getattr(env, "render_state", None)
    if render_state is None:
        return {}
    return getattr(render_state, "frame", {})


def _entity_grid_position(entity: Any) -> tuple[int, int] | None:
    position = getattr(entity, "position", None)
    x = getattr(position, "x", None)
    y = getattr(position, "y", None)
    if x is None or y is None:
        return None
    return int(x), int(y)


def inferred_floor_tiles(
    positioned_entities: list[tuple[Any, float, float]],
    floor_sprite: str,
) -> list[dict[str, Any]]:
    positions = [(int(x), int(y)) for _, x, y in positioned_entities]
    if not positions:
        return []

    xs = [position[0] for position in positions]
    ys = [position[1] for position in positions]
    return [
        {"x": x, "y": y, "kind": "floor", "sprite": floor_sprite}
        for x in range(min(xs), max(xs) + 1)
        for y in range(min(ys), max(ys) + 1)
    ]


def _floor_tiles_from_bounds(bounds: Any, floor_sprite: str) -> list[dict[str, Any]]:
    if isinstance(bounds, dict):
        min_x = int(bounds.get("min_x", 0))
        max_x = int(bounds.get("max_x", min_x))
        min_y = int(bounds.get("min_y", 0))
        max_y = int(bounds.get("max_y", min_y))
    elif isinstance(bounds, (tuple, list)) and len(bounds) == 4:
        min_x, max_x, min_y, max_y = (int(value) for value in bounds)
    else:
        return []

    return [
        {"x": x, "y": y, "kind": "floor", "sprite": floor_sprite}
        for x in range(min_x, max_x + 1)
        for y in range(min_y, max_y + 1)
    ]


def _install_communication_render_hooks(env) -> None:
    """Mirror chat messages into Renderable.last_message without changing core systems."""
    try:
        from word_play.presets.systems.communication import Communication_Policy
        from .renderer import Renderable
    except Exception:
        return

    for entity in getattr(getattr(env, "state", None), "entities", []):
        if not hasattr(entity, "get_component"):
            continue
        renderable = entity.get_component(Renderable)
        policy = entity.get_component(Communication_Policy)
        if renderable is None or policy is None:
            continue
        if getattr(policy, "_renderer_message_hook_installed", False):
            continue

        original_start = policy.start_conversation
        original_send = policy.send_message
        original_end = policy.end_conversation

        def wrapped_start(self, participants, env, info=None, *, _original=original_start):
            depth = int(getattr(env, "_renderer_conversation_depth", 0))
            if depth == 0:
                env._renderer_conversation_log = []
                env._renderer_conversation_participants = [participant.name for participant in participants]
                starter_name = getattr(getattr(self, "entity", None), "name", "Unknown")
                env._renderer_conversation_starter = starter_name
                env._renderer_conversation_log.append(f"Conversation started by {starter_name}")
            env._renderer_conversation_depth = depth + 1
            return _original(participants, env, info)

        def wrapped_send(self, recipients, env, info=None, *, _original=original_send, _renderable=renderable):
            message = _original(recipients, env, info)
            if message is not None:
                _renderable.last_message = str(message)
                _renderable._last_message_step = getattr(env, "cur_step", 0)
                conversation_log = list(getattr(env, "_renderer_conversation_log", []))
                speaker_name = getattr(getattr(self, "entity", None), "name", "Unknown")
                participant_count = max(1, len(getattr(env, "_renderer_conversation_participants", [])))
                prior_messages = max(
                    0,
                    len([entry for entry in conversation_log if entry.startswith("Round ") and ": " in entry])
                )
                round_number = prior_messages // participant_count + 1
                conversation_log.append(f"Round {round_number} - {speaker_name}: {message}")
                env._renderer_conversation_log = conversation_log[-8:]
            return message

        def wrapped_end(self, participants, env, info=None, *, _original=original_end):
            try:
                return _original(participants, env, info)
            finally:
                depth = max(0, int(getattr(env, "_renderer_conversation_depth", 0)) - 1)
                env._renderer_conversation_depth = depth
                if depth == 0:
                    env._renderer_conversation_participants = []
                    env._renderer_conversation_log = []
                    env._renderer_conversation_starter = None

        policy.start_conversation = MethodType(wrapped_start, policy)
        policy.send_message = MethodType(wrapped_send, policy)
        policy.end_conversation = MethodType(wrapped_end, policy)
        policy._renderer_message_hook_installed = True


def _expire_render_messages(env) -> None:
    """Clear transient speech bubbles after they have been visible for one step."""
    try:
        from .renderer import Renderable
    except Exception:
        return

    current_step = getattr(env, "cur_step", 0)
    for entity in getattr(getattr(env, "state", None), "entities", []):
        if not hasattr(entity, "get_component"):
            continue
        renderable = entity.get_component(Renderable)
        if renderable is None:
            continue
        message_step = getattr(renderable, "_last_message_step", None)
        if message_step is None:
            continue
        if current_step > message_step + 1:
            renderable.last_message = None
            renderable._last_message_step = None


def _install_human_prompt_hooks(env) -> None:
    """Patch human action/chat components at runtime when a renderer is active."""
    renderer = getattr(env, "renderer_impl", None)
    if renderer is None:
        return

    try:
        from word_play.presets.action_policies.human import Human_Takes_Action
    except Exception:
        Human_Takes_Action = None
    try:
        from word_play.presets.systems.communication.chat_room_action_communication.presets.policies import (
            Human_Communication_Policy,
        )
    except Exception:
        Human_Communication_Policy = None
    try:
        from .runtime import prompt_human_action, prompt_human_multi_select, prompt_human_text
    except Exception:
        return

    for entity in getattr(getattr(env, "state", None), "entities", []):
        if not hasattr(entity, "get_component"):
            continue

        if Human_Takes_Action is not None:
            action_policy = entity.get_component(Human_Takes_Action)
            if action_policy is not None and not getattr(action_policy, "_renderer_prompt_hook_installed", False):
                original_choose = action_policy._choose_action
                original_kwargs = action_policy._get_action_kwargs

                def wrapped_choose(self, observation, *, _original=original_choose):
                    active_renderer = getattr(observation.possible_actions[0].env, "renderer_impl", None) if observation.possible_actions else None
                    if active_renderer is None:
                        return _original(observation)
                    return prompt_human_action(active_renderer, observation.possible_actions[0].env, observation)

                def wrapped_kwargs(self, action_selection, *, _original=original_kwargs):
                    active_renderer = getattr(action_selection.env, "renderer_impl", None)
                    if active_renderer is None:
                        return _original(action_selection)
                    error_message = None
                    while True:
                        if (
                            action_selection.required_kwargs
                            and len(action_selection.required_kwargs) == 1
                        ):
                            arg_name, arg = next(iter(action_selection.required_kwargs.items()))
                            arg_description = arg.arg_description(
                                action_selection.actor,
                                action_selection.target_entity,
                                action_selection.env,
                            )
                            matches = re.findall(r"(\d+)\s*\(([^)]+)\)", arg_description)
                            if matches:
                                text = prompt_human_multi_select(
                                    active_renderer,
                                    action_selection.env,
                                    entity_name=action_selection.actor.name,
                                    position_label=str(action_selection.actor.position),
                                    header="Action Arguments",
                                    instructions=[
                                        str(action_selection),
                                        f"Choose values for: {arg_name}",
                                        "Use Up/Down to move, Space to toggle, Enter to submit.",
                                        *( [f"Error: {error_message}"] if error_message else [] ),
                                    ],
                                    options=[(int(idx), label) for idx, label in matches],
                                )
                                try:
                                    return action_selection.parse_and_validate_kwarg_list(text)
                                except Exception as exc:
                                    error_message = str(exc)
                                    continue

                        instructions = [
                            str(action_selection),
                            "Type the required values separated by ';'.",
                            "Press Enter to submit.",
                        ]
                        if action_selection.required_kwargs:
                            for name, arg in action_selection.required_kwargs.items():
                                instructions.append(
                                    f'{name}: {arg.arg_description(action_selection.actor, action_selection.target_entity, action_selection.env)}'
                                )
                        if error_message:
                            instructions.append(f"Error: {error_message}")

                        text = prompt_human_text(
                            active_renderer,
                            action_selection.env,
                            entity_name=action_selection.actor.name,
                            position_label=str(action_selection.actor.position),
                            header="Action Arguments",
                            instructions=instructions,
                        )
                        try:
                            return action_selection.parse_and_validate_kwarg_list(text)
                        except Exception as exc:
                            error_message = str(exc)

                action_policy._choose_action = MethodType(wrapped_choose, action_policy)
                action_policy._get_action_kwargs = MethodType(wrapped_kwargs, action_policy)
                action_policy._renderer_prompt_hook_installed = True

        if Human_Communication_Policy is not None:
            comm_policy = entity.get_component(Human_Communication_Policy)
            if comm_policy is not None and not getattr(comm_policy, "_renderer_prompt_hook_installed", False):
                original_send = comm_policy.send_message

                def wrapped_send(self, recipients, env, info=None, *, _original=original_send):
                    active_renderer = getattr(env, "renderer_impl", None)
                    if active_renderer is None or self.entity is None:
                        return _original(recipients, env, info)
                    recipient_names = ", ".join(recipient.name for recipient in recipients) or "nobody"
                    transcript = list(getattr(env, "_renderer_conversation_log", []))
                    instructions = [
                        f"Recipients: {recipient_names}",
                        *([str(info)] if info else []),
                    ]
                    if transcript:
                        instructions.append("Conversation so far:")
                        instructions.extend(transcript[-6:])
                    instructions.extend([
                        f"Now speaking: {self.entity.name}",
                        "Type your message.",
                        "Press Enter to send.",
                    ])
                    return prompt_human_text(
                        active_renderer,
                        env,
                        entity_name=self.entity.name,
                        position_label=str(self.entity.position),
                        header="Chat",
                        instructions=instructions,
                    )

                comm_policy.send_message = MethodType(wrapped_send, comm_policy)
                comm_policy._renderer_prompt_hook_installed = True


class Position_Layout_Adapter(ABC):
    """Map environment positions and optional backgrounds into render space."""

    @abstractmethod
    def screen_position(
        self,
        entity: Any,
        env: "Environment",
        context: "Render_Context | None" = None,
    ) -> tuple[float, float]:
        """Convert a world position into renderer grid coordinates."""

    def background(
        self,
        env: "Environment",
        positioned_entities: list[tuple[Any, float, float]],
        _context: "Render_Context | None" = None,
    ) -> list[dict[str, Any]]:
        """Return background tiles to draw behind entities."""
        return []

    def prepare_env(self, env: "Environment", _context: "Render_Context | None" = None) -> None:
        """Update any renderer-facing derived state before drawing."""
        return None


class Grid_Layout_Adapter(Position_Layout_Adapter):
    """Use entity x/y values directly as grid coordinates."""

    def background(
        self,
        env: "Environment",
        positioned_entities: list[tuple[Any, float, float]],
        _context: "Render_Context | None" = None,
    ) -> list[dict[str, Any]]:
        """Fetch background tiles from renderer data or auto-generate a floor."""
        render_frame = _render_frame(env)
        env_tiles = render_frame.get("world.background_tiles")
        if env_tiles:
            return list(env_tiles)

        floor_sprite = str(render_frame.get("world.floor_sprite", _DEFAULT_FLOOR))
        bounds_tiles = _floor_tiles_from_bounds(render_frame.get("world.bounds"), floor_sprite)
        if bounds_tiles:
            return bounds_tiles

        inferred_tiles = inferred_floor_tiles(positioned_entities, floor_sprite)
        if inferred_tiles:
            return inferred_tiles

        world_size = render_frame.get("world.size")
        if isinstance(world_size, dict):
            width = int(world_size.get("width", 0))
            height = int(world_size.get("height", 0))
            origin_x = int(world_size.get("origin_x", 0))
            origin_y = int(world_size.get("origin_y", 0))
            return [
                {"x": origin_x + x, "y": origin_y + y, "kind": "floor", "sprite": floor_sprite}
                for x in range(max(0, width))
                for y in range(max(0, height))
            ]
        return []

    def screen_position(
        self,
        entity: Any,
        _env: "Environment",
        _context: "Render_Context | None" = None,
    ) -> tuple[float, float]:
        """Project the position without changing its coordinates."""
        position = getattr(entity, "position", entity)
        x = getattr(position, 'x', None)
        y = getattr(position, 'y', None)
        if x is not None and y is not None:
            return float(x), float(y)
        # Fallback for plain tuple positions: (x, y) or (x, y, z)
        if hasattr(position, '__getitem__') and len(position) >= 2:
            return float(position[0]), float(position[1])
        return 0.0, 0.0

    def prepare_env(self, _env: "Environment", _context: "Render_Context | None" = None) -> None:
        """Grid rendering derives transient visuals at draw time."""
        return None


def _get_slot_offsets(n: int, radius: float) -> list[tuple[float, float]]:
    """Return predefined visual offsets for n entities at a single point.

    Layouts:
    - 1 entity: center
    - 2 entities: left/right
    - 3 entities: triangle (top + bottom corners)
    - 4 entities: compass layout (N, E, S, W)
    - 5-8: evenly distributed ring
    - 9+: concentric rings
    """
    if n == 1:
        return [(0.0, 0.0)]
    elif n == 2:
        # Left and right
        d = radius
        return [(-d, 0.0), (d, 0.0)]
    elif n == 3:
        # Triangle pointing up
        r = radius
        return [
            (0.0, -r),                    # Top
            (r * 0.866, r * 0.5),       # Bottom-right
            (-r * 0.866, r * 0.5),      # Bottom-left
        ]
    elif n == 4:
        # Compass layout (N, E, S, W) - classic "sharing a tile" look
        r = radius
        return [
            (0.0, -r),    # North (top)
            (r, 0.0),     # East (right)
            (0.0, r),     # South (bottom)
            (-r, 0.0),    # West (left)
        ]
    elif n <= 8:
        # Ring layout - evenly spaced around circle
        offsets = []
        for i in range(n):
            # Start at top, go clockwise
            angle = -math.pi / 2 + (2 * math.pi * i / n)
            offsets.append((
                radius * math.cos(angle),
                radius * math.sin(angle) * 0.7,  # Flatten Y for perspective
            ))
        return offsets
    else:
        # Concentric rings for large groups
        offsets = []
        ring_size = 8  # entities per outer ring
        for i in range(n):
            ring = i // ring_size
            index_in_ring = i % ring_size

            # Inner entities are closer, outer are further
            r = radius * (1 + ring * 0.5)
            angle = -math.pi / 2 + (2 * math.pi * index_in_ring / ring_size)
            offsets.append((
                r * math.cos(angle),
                r * math.sin(angle) * 0.7,
            ))
        return offsets


class SinglePointLayout(Position_Layout_Adapter):
    """Unified layout for entities at a single point position.

    Entities can be arranged in:
    - "compass" mode: N/E/S/W for up to 4, expanding to rings for more
    - "circle" mode: Circular arrangement with configurable radius

    Optional room background can be generated when include_room=True.
    """

    # Room background settings
    WALL_SET = "src/world_tiles/indoors/wall_sets/overcooked_kitchen_wall"
    TABLE_SPRITE = "src/world_tiles/indoors/stations/prep_table.png"
    DEFAULT_FLOOR = "src/world_tiles/indoors/floors/day_brick_floor_c.png"

    def __init__(
        self,
        center_x: float = 0,
        center_y: float = 0,
        radius: float = 0.35,
        layout_mode: str = "compass",  # "compass" or "circle"
        include_room: bool = False,
        only_agents: bool = False,
    ):
        """
        Args:
            center_x, center_y: The grid coordinates of the center point
            radius: Base radius for positioning in tile units
            layout_mode: "compass" (N/E/S/W slots) or "circle" (pure circular)
            include_room: If True, generate room walls and floor tiles
            only_agents: If True, only position agents (ignore non-agent entities)
        """
        self.base_x = center_x
        self.base_y = center_y
        self.radius = radius
        self.layout_mode = layout_mode
        self.include_room = include_room
        self.only_agents = only_agents

    def _runtime_state(self, context: "Render_Context | None") -> _SinglePointLayoutState:
        if context is None:
            raise ValueError("SinglePointLayout requires a Render_Context for renderer-private layout state.")
        return context.value_for(self, _SinglePointLayoutState)

    def _calculate_room_size(self, n_agents: int) -> tuple[int, int]:
        """Calculate room size based on agent count."""
        if n_agents <= 4:
            return 5, 5
        elif n_agents <= 8:
            return 7, 7
        elif n_agents <= 12:
            return 9, 7
        else:
            return 11, 9

    @staticmethod
    def _get_floor_sprite() -> str:
        """Get a nice floor sprite."""
        return SinglePointLayout.DEFAULT_FLOOR

    def prepare_env(self, env: "Environment", context: "Render_Context | None" = None) -> None:
        """Calculate visual positions for entities at this point."""
        if env is None:
            return

        runtime = self._runtime_state(context)
        entities = list(getattr(env.state, "entities", []))

        # Filter entities to position
        if self.only_agents:
            positioned_entities = [e for e in entities if getattr(e, "is_agent", False)]
        else:
            positioned_entities = [
                e for e in entities
                if isinstance(getattr(e, "position", None), Single_Point_Position)
            ]

        n = len(positioned_entities)
        runtime.offsets_by_entity.clear()
        if n == 0:
            return

        # Sort for stable ordering: agents first, then by name
        positioned_entities.sort(key=lambda e: (
            0 if getattr(e, "is_agent", False) else 1,
            e.name,
        ))

        # Calculate room size if needed
        agent_count = len([e for e in positioned_entities if getattr(e, "is_agent", False)])
        if self.include_room and agent_count > 0:
            runtime.cached_background = None

        # Calculate offsets based on layout mode
        if self.layout_mode == "compass":
            offsets = _get_slot_offsets(n, self.radius)
        else:  # circle
            offsets = []
            for i in range(n):
                angle = -math.pi / 2 + (2 * math.pi * i / n)
                offsets.append((
                    self.radius * math.cos(angle),
                    self.radius * math.sin(angle) * 0.7,
                ))

        # Cache transient layout offsets in renderer-owned private runtime state.
        for entity, (offset_x, offset_y) in zip(positioned_entities, offsets):
            runtime.offsets_by_entity[entity] = (offset_x, offset_y)

    def background(
        self,
        env: "Environment" | None,
        positioned_entities: list[tuple[Any, float, float]],
        context: "Render_Context | None" = None,
    ) -> list[dict[str, Any]]:
        """Generate optional room background with walls and flooring."""
        del positioned_entities
        if not self.include_room:
            return []

        runtime = self._runtime_state(context)
        if runtime.cached_background is not None:
            return runtime.cached_background

        if env is None:
            return []

        entities = list(getattr(env.state, "entities", []))
        agent_count = len([e for e in entities if getattr(e, "is_agent", False)])
        if agent_count == 0:
            return []

        room_w, room_h = self._calculate_room_size(agent_count)
        tiles = []

        half_w = room_w // 2
        half_h = room_h // 2

        # Generate interior floor tiles
        for y in range(-half_h, half_h + 1):
            for x in range(-half_w, half_w + 1):
                abs_x = self.base_x + x
                abs_y = self.base_y + y

                if x == 0 and y == 0:
                    # Meeting table at center
                    tiles.append({
                        "x": abs_x,
                        "y": abs_y,
                        "kind": "floor",
                        "sprite": self.TABLE_SPRITE,
                    })
                else:
                    tiles.append({
                        "x": abs_x,
                        "y": abs_y,
                        "kind": "floor",
                        "sprite": self._get_floor_sprite(),
                    })

        # Generate walls OUTSIDE the room
        wall_w = half_w + 1
        wall_h = half_h + 1

        for y in range(-wall_h, wall_h + 1):
            for x in range(-wall_w, wall_w + 1):
                is_outer_ring = (y == -wall_h or y == wall_h or x == -wall_w or x == wall_w)
                if not is_outer_ring:
                    continue

                abs_x = self.base_x + x
                abs_y = self.base_y + y

                tiles.append({
                    "x": abs_x,
                    "y": abs_y,
                    "kind": "wall",
                    "wall_set": self.WALL_SET,
                })

        runtime.cached_background = tiles
        return tiles

    def screen_position(
        self,
        entity: Any,
        env: "Environment",
        context: "Render_Context | None" = None,
    ) -> tuple[float, float]:
        """Convert position to screen coordinates."""
        position = getattr(entity, "position", entity)
        if isinstance(position, Single_Point_Position):
            offset_x, offset_y = self._runtime_state(context).offsets_by_entity.get(entity, (0.0, 0.0))
            return (self.base_x + offset_x, self.base_y + offset_y)

        # Fallback: try to get x/y attributes, default to center
        if position is not None:
            x = getattr(position, "x", None)
            y = getattr(position, "y", None)
            if x is not None and y is not None:
                return (float(x), float(y))

        return (self.base_x, self.base_y)
