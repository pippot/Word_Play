from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pygame

from word_play.core import Entity

from .assets import get_or_load_image, get_scaled_image, resolve_wall_sprite
from .wall_geometry import collect_wall_positions, normalize_background_item, screen_position_for_entity, screen_rect_for_tile, wall_neighbor_mask, world_bounds
from .renderer import Renderable
from .runtime import ensure_screen_size

if TYPE_CHECKING:
    from word_play.core import Environment

    from .renderer import PygameRenderer


def visible_renderables(env: "Environment") -> list[tuple[int, Entity, Renderable]]:
    """Collect visible renderable entities sorted by their draw order."""
    renderables: list[tuple[int, Entity, Renderable]] = []
    for entity in env.state.entities:
        renderable = entity.get_component(Renderable)
        if renderable is not None and renderable.visible:
            renderables.append((renderable.z_index, entity, renderable))
    renderables.sort(key=lambda item: item[0])
    return renderables


def background_items(env: "Environment", renderer: "PygameRenderer") -> list[dict[str, Any]]:
    """Fetch and normalize background tiles from the active layout adapter."""
    return [normalize_background_item(item) for item in renderer.layout.background(env)]


def entity_health_value(entity: Entity) -> float | None:
    """Read an entity's health value from any health-like component."""
    for component in entity.components.values():
        if hasattr(component, "current_health"):
            return float(getattr(component, "current_health"))
        if hasattr(component, "health"):
            return float(getattr(component, "health"))
    return None


def update_damage_flash_state(renderer: "PygameRenderer", env: "Environment") -> None:
    """Track recent health drops so damaged entities can flash briefly."""
    now = time.monotonic()
    active_names = {entity.name for entity in env.state.entities}
    renderer._damage_flash_until = {
        name: until
        for name, until in renderer._damage_flash_until.items()
        if name in active_names and until > now
    }

    next_health_values: dict[str, float] = {}
    for entity in env.state.entities:
        health_value = entity_health_value(entity)
        if health_value is None:
            continue
        next_health_values[entity.name] = health_value
        previous_value = renderer._last_health_values.get(entity.name)
        if previous_value is not None and health_value < previous_value:
            renderer._damage_flash_until[entity.name] = now + 1.0

    renderer._last_health_values = next_health_values


def flash_tinted_surface(image: Any, *, tint: tuple[int, int, int], alpha: int) -> Any:
    """Return a tinted copy of a sprite for temporary visual effects."""
    tinted = image.copy()
    overlay = pygame.Surface(tinted.get_size(), pygame.SRCALPHA)
    overlay.fill((*tint, alpha))
    tinted.blit(overlay, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
    return tinted


def blit_scaled_sprite(
    renderer: "PygameRenderer",
    sprite_name: str,
    px: int,
    py: int,
    *,
    width: int,
    height: int,
    anchor: str = "center",
    missing_ok: bool = False,
) -> bool:
    """Load, scale, and draw a sprite at a tile with the chosen anchor."""
    image = get_scaled_image(renderer, sprite_name, width, height)
    if image is None:
        if missing_ok:
            return False
        raise FileNotFoundError(f"Sprite could not be resolved: '{sprite_name}'.")

    if anchor == "top_right":
        draw_x = px + renderer.tile_size - width
        draw_y = py
    elif anchor == "top_left":
        draw_x = px
        draw_y = py
    else:
        draw_x = px + (renderer.tile_size - width) // 2
        draw_y = py + (renderer.tile_size - height) // 2

    renderer.screen.blit(image, (draw_x, draw_y))
    return True


def draw_wall_sprite(renderer: "PygameRenderer", wall_set: str, px: int, py: int, neighbors: dict[str, bool]) -> bool:
    """Draw a wall tile using the best matching sprite variant."""
    sprite_name = resolve_wall_sprite(renderer, wall_set, neighbors)
    if sprite_name is None:
        raise FileNotFoundError(f"Wall set '{wall_set}' has no matching sprite variant for neighbors {neighbors}.")
    blit_scaled_sprite(
        renderer,
        sprite_name,
        px,
        py,
        width=renderer.tile_size,
        height=renderer.tile_size,
        anchor="top_left",
    )
    return True


def draw_wall_background_tile(
    renderer: "PygameRenderer",
    item: dict[str, Any],
    px: int,
    py: int,
    *,
    wall_positions: set[tuple[int, int]],
) -> bool:
    """Draw a wall background tile from its sprite set."""
    if item.get("kind") != "wall":
        return False

    wall_set = item.get("wall_set")
    if not wall_set:
        raise ValueError(f"Wall background tile is missing 'wall_set': {item!r}")

    neighbors = wall_neighbor_mask(int(item["x"]), int(item["y"]), wall_positions)
    return draw_wall_sprite(renderer, str(wall_set), px, py, neighbors)


def draw_overlay_item(
    renderer: "PygameRenderer",
    sprite_name: str,
    px: int,
    py: int,
    *,
    mode: str = "badge",
    scale: float | None = None,
) -> None:
    """Draw an overlay sprite on top of an entity using the requested style."""
    if mode == "full":
        overlay_size = renderer.tile_size
        anchor = "top_left"
    elif mode == "center":
        ratio = scale if scale is not None else 0.72
        overlay_size = max(20, int(renderer.tile_size * ratio))
        anchor = "center"
    else:
        ratio = scale if scale is not None else 0.5
        overlay_size = max(18, int(renderer.tile_size * ratio))
        anchor = "top_right"

    blit_scaled_sprite(
        renderer,
        sprite_name,
        px,
        py,
        width=overlay_size,
        height=overlay_size,
        anchor=anchor,
    )


def draw_entity(renderer: "PygameRenderer", entity: Entity, renderable: Renderable, px: int, py: int) -> None:
    """Draw an entity sprite, including damage flash and optional overlay."""
    image = get_or_load_image(renderer, renderable.sprite_path)
    if image is None:
        raise FileNotFoundError(
            f"Sprite renderer expected a valid sprite path for '{entity.name}', but could not resolve '{renderable.sprite_path}'."
        )
    flash_until = renderer._damage_flash_until.get(entity.name, 0.0)
    if flash_until > time.monotonic():
        image = flash_tinted_surface(image, tint=(190, 20, 20), alpha=140)
    scaled_image = pygame.transform.scale(image, (renderer.tile_size, renderer.tile_size))
    renderer.screen.blit(scaled_image, (px, py))

    if renderable.overlay_sprite:
        draw_overlay_item(
            renderer,
            renderable.overlay_sprite,
            px,
            py,
            mode=getattr(renderable, "overlay_mode", "badge"),
            scale=getattr(renderable, "overlay_scale", None),
        )


def draw_hit_effects(
    renderer: "PygameRenderer",
    env: "Environment",
    entity_positions: dict[str, tuple[int, int]],
) -> None:
    """Draw transient hit-effect sprites centered on affected entities."""
    hit_effects = list(getattr(env, "hit_effects", []))
    if not hit_effects:
        return

    for effect in hit_effects:
        entity_name = effect.get("entity_name")
        sprite_name = effect.get("sprite")
        if not entity_name or not sprite_name or entity_name not in entity_positions:
            continue

        px, py = entity_positions[entity_name]
        ratio = float(effect.get("scale", 0.75))
        effect_size = max(20, int(renderer.tile_size * ratio))
        blit_scaled_sprite(
            renderer,
            str(sprite_name),
            px,
            py,
            width=effect_size,
            height=effect_size,
            anchor="center",
            missing_ok=True,
        )


def draw_background_tile(
    renderer: "PygameRenderer",
    item: dict[str, Any],
    px: int,
    py: int,
    *,
    wall_positions: set[tuple[int, int]],
) -> None:
    """Draw one background tile and require explicit sprite-backed assets."""
    kind = item.get("kind", "floor")
    if draw_wall_background_tile(renderer, item, px, py, wall_positions=wall_positions):
        return

    sprite_name = item.get("sprite")
    if not sprite_name:
        raise ValueError(f"Background tile kind '{kind}' is missing required 'sprite': {item!r}")

    blit_scaled_sprite(
        renderer,
        sprite_name,
        px,
        py,
        width=renderer.tile_size,
        height=renderer.tile_size,
        anchor="top_left",
    )


def draw_grid_overlay(renderer: "PygameRenderer", min_x: int, max_x: int, min_y: int, max_y: int) -> None:
    """Draw grid lines over the world area for debugging or readability."""
    overlay_color = (30, 30, 34)
    for x in range(min_x, max_x + 2):
        px = renderer.margin + (x - min_x) * renderer.tile_size
        pygame.draw.line(
            renderer.screen,
            overlay_color,
            (px, renderer.margin),
            (px, renderer.margin + (max_y - min_y + 1) * renderer.tile_size),
            2,
        )
    for y in range(min_y, max_y + 2):
        py = renderer.margin + (max_y - y + 1) * renderer.tile_size
        pygame.draw.line(
            renderer.screen,
            overlay_color,
            (renderer.margin, py),
            (renderer.margin + (max_x - min_x + 1) * renderer.tile_size, py),
            2,
        )


def draw_hud_panel(renderer: "PygameRenderer", env: "Environment", width: int, height: int) -> None:
    """Render the bottom HUD panel with header text and recent messages."""
    if getattr(env, "hide_bottom_hud", False):
        return

    hud_top = height - renderer.hud_height
    panel_rect = pygame.Rect(0, hud_top, width, renderer.hud_height)
    pygame.draw.rect(renderer.screen, (14, 17, 24), panel_rect)
    pygame.draw.line(renderer.screen, (52, 63, 79), (0, hud_top), (width, hud_top), 2)

    header_text = getattr(env, "hud_header", None)
    if header_text is None:
        score = getattr(env, "score", 0)
        tick = getattr(env, "tick", 0)
        deliveries = getattr(env, "deliveries", 0)
        header_text = f"Tick {tick}   Score {score}   Deliveries {deliveries}"

    header = renderer.hud_font.render(str(header_text), True, (240, 242, 245))
    renderer.screen.blit(header, (renderer.margin, hud_top + 12))

    hud_lines = list(getattr(env, "hud_lines", []))
    health_lines: list[str] = []
    if getattr(env, "state", None) is not None:
        agents = list(getattr(env, "agents", []))
        seen_entities: set[int] = set()
        for entity in agents + list(getattr(env.state, "entities", [])):
            if id(entity) in seen_entities:
                continue
            seen_entities.add(id(entity))
            health_value = entity_health_value(entity)
            if health_value is None:
                continue
            max_health = None
            for component in entity.components.values():
                if hasattr(component, "max_health"):
                    max_health = getattr(component, "max_health")
                    break
            if max_health is None:
                health_lines.append(f"{entity.name}: {int(health_value)} HP")
            else:
                health_lines.append(f"{entity.name}: {int(health_value)}/{int(max_health)} HP")
    if health_lines:
        hud_lines = [*health_lines[:4], *hud_lines]
    if not hud_lines:
        event_log = getattr(env, "event_log", [])
        recent_events = event_log[-3:] if event_log else ["Kitchen ready."]
        hud_lines = [f"- {message}" for message in recent_events]

    for index, message in enumerate(hud_lines[:6]):
        color = (189, 196, 206) if index < 3 else (235, 213, 154)
        surface = renderer.small_font.render(str(message), True, color)
        renderer.screen.blit(surface, (renderer.margin, hud_top + 44 + index * (renderer.small_font.get_linesize() + 4)))


def draw_sidebar_panel(renderer: "PygameRenderer", env: "Environment", world_width: int, height: int, sidebar_width: int) -> None:
    """Render an optional right-hand sidebar with agent observations and options."""
    if sidebar_width <= 0:
        return

    panel_rect = pygame.Rect(world_width, 0, sidebar_width, height)
    pygame.draw.rect(renderer.screen, (12, 15, 21), panel_rect)
    pygame.draw.line(renderer.screen, (52, 63, 79), (world_width, 0), (world_width, height), 2)

    header_text = getattr(env, "hud_sidebar_header", "Agent View")
    header = renderer.hud_font.render(str(header_text), True, (240, 242, 245))
    renderer.screen.blit(header, (world_width + 16, 14))

    sidebar_lines = list(getattr(env, "hud_sidebar_lines", []))
    y = 48
    line_height = renderer.small_font.get_linesize() + 4
    max_width = sidebar_width - 32
    for message in sidebar_lines[:18]:
        wrapped = wrap_text_lines(renderer.small_font, str(message), max_width=max_width)
        for wrapped_line in wrapped[:3]:
            if y > height - 26:
                return
            surface = renderer.small_font.render(wrapped_line, True, (189, 196, 206))
            renderer.screen.blit(surface, (world_width + 16, y))
            y += line_height
        y += 4


def draw_end_overlay(renderer: "PygameRenderer", env: "Environment", world_width: int, world_height: int) -> None:
    """Draw a centered overlay when the environment reaches a terminal state."""
    terminations = list(getattr(env, "terminations", []))
    truncations = list(getattr(env, "truncations", []))
    final_boss_defeated = bool(getattr(env, "final_boss_defeated", False))
    if not final_boss_defeated and not any(terminations) and not any(truncations):
        return

    if final_boss_defeated:
        title = "Raid Complete"
        subtitle = "The Ash Warden has fallen."
        accent = (206, 182, 92)
    elif any(truncations):
        title = "Run Ended"
        subtitle = "The dungeon closed before victory."
        accent = (149, 161, 178)
    else:
        title = "Defeat"
        subtitle = "The raider fell in battle."
        accent = (170, 82, 82)

    overlay = pygame.Surface((world_width, world_height), pygame.SRCALPHA)
    overlay.fill((8, 10, 16, 170))
    renderer.screen.blit(overlay, (0, 0))

    box_width = min(max(320, renderer.tile_size * 6), world_width - 40)
    box_height = max(120, renderer.tile_size * 2 + 36)
    box_x = (world_width - box_width) // 2
    box_y = (world_height - box_height) // 2
    panel_rect = pygame.Rect(box_x, box_y, box_width, box_height)
    pygame.draw.rect(renderer.screen, (18, 22, 30), panel_rect, border_radius=18)
    pygame.draw.rect(renderer.screen, accent, panel_rect, width=3, border_radius=18)

    title_surface = renderer.font.render(title, True, (245, 247, 250))
    subtitle_surface = renderer.hud_font.render(subtitle, True, accent)
    title_x = box_x + (box_width - title_surface.get_width()) // 2
    subtitle_x = box_x + (box_width - subtitle_surface.get_width()) // 2
    renderer.screen.blit(title_surface, (title_x, box_y + 24))
    renderer.screen.blit(subtitle_surface, (subtitle_x, box_y + 24 + title_surface.get_height() + 12))


def wrap_text_lines(font: Any, text: str, max_width: int) -> list[str]:
    """Wrap text greedily into lines that fit within the given width."""
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if font.size(trial)[0] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def fit_wrapped_text_lines(
    fonts: list[Any],
    text: str,
    *,
    max_width: int,
    max_lines: int,
) -> tuple[Any, list[str]]:
    """Choose a font and wrapped lines that fit within the speech-bubble limits."""
    for font in fonts:
        lines = wrap_text_lines(font, text, max_width=max_width)
        if len(lines) <= max_lines:
            return font, lines
    fallback_font = fonts[-1]
    lines = wrap_text_lines(fallback_font, text, max_width=max_width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last_line = lines[-1]
        while last_line and fallback_font.size(f"{last_line}...")[0] > max_width:
            last_line = last_line[:-1].rstrip()
        lines[-1] = f"{last_line}..." if last_line else "..."
    return fallback_font, lines


def draw_speech_bubbles(
    renderer: "PygameRenderer",
    env: "Environment",
    entity_positions: dict[str, tuple[int, int]],
) -> None:
    """Draw speech bubbles above entities using rounded rects and tail polygons."""
    speech_bubbles = list(getattr(env, "speech_bubbles", []))
    if not speech_bubbles:
        return

    for bubble in speech_bubbles:
        entity_name = bubble.get("entity_name")
        text = str(bubble.get("text", "")).strip()
        if not entity_name or not text or entity_name not in entity_positions:
            continue

        px, py = entity_positions[entity_name]
        anchor_x = px + renderer.tile_size // 2
        anchor_y = py + max(4, renderer.tile_size // 6)
        bubble_width = max(int(renderer.tile_size * 2.75), 148)
        bubble_height = max(int(renderer.tile_size * 1.45), 64)
        pad_left = max(12, renderer.tile_size // 3)
        pad_right = max(12, renderer.tile_size // 3)
        pad_top = max(10, renderer.tile_size // 4)
        pad_bottom = max(12, renderer.tile_size // 4)
        tail_width = max(14, renderer.tile_size // 2)
        tail_height = max(10, renderer.tile_size // 3)
        radius = max(10, renderer.tile_size // 3)
        text_max_width = bubble_width - pad_left - pad_right
        speech_font, lines = fit_wrapped_text_lines(
            renderer.speech_fonts,
            text,
            max_width=text_max_width,
            max_lines=3,
        )
        text_surfaces = [speech_font.render(line, True, (18, 16, 14)) for line in lines]
        text_width = max(surface.get_width() for surface in text_surfaces)
        line_gap = 1
        text_height = sum(surface.get_height() for surface in text_surfaces) + max(0, len(text_surfaces) - 1) * line_gap
        bubble_width = max(bubble_width, text_width + pad_left + pad_right)
        bubble_height = max(bubble_height, text_height + pad_top + pad_bottom)
        bubble_x = anchor_x - bubble_width // 2
        bubble_y = max(6, anchor_y - bubble_height - tail_height)
        content_left = bubble_x + pad_left
        content_top = bubble_y + pad_top
        content_width = bubble_width - pad_left - pad_right
        content_height = bubble_height - pad_top - pad_bottom

        bubble_rect = pygame.Rect(bubble_x, bubble_y, bubble_width, bubble_height)
        tail = [
            (anchor_x - tail_width // 2, bubble_rect.bottom - 2),
            (anchor_x + tail_width // 2, bubble_rect.bottom - 2),
            (anchor_x, anchor_y),
        ]
        pygame.draw.rect(renderer.screen, (250, 245, 233), bubble_rect, border_radius=radius)
        pygame.draw.rect(renderer.screen, (56, 46, 39), bubble_rect, width=2, border_radius=radius)
        pygame.draw.polygon(renderer.screen, (250, 245, 233), tail)
        pygame.draw.polygon(renderer.screen, (56, 46, 39), tail, width=2)

        text_y = content_top + max(0, (content_height - text_height) // 2)
        for surface in text_surfaces:
            text_x = content_left + max(0, (content_width - surface.get_width()) // 2)
            renderer.screen.blit(surface, (text_x, text_y))
            text_y += surface.get_height() + line_gap


def render_environment(renderer: "PygameRenderer", env: "Environment") -> None:
    """Render a full frame including background, entities, effects, and HUD."""
    update_damage_flash_state(renderer, env)
    background = background_items(env, renderer)
    renderables = visible_renderables(env)
    min_x, max_x, min_y, max_y = world_bounds(renderer, background, renderables)

    grid_width = max(1, max_x - min_x + 1)
    grid_height = max(1, max_y - min_y + 1)
    world_width = renderer.margin * 2 + grid_width * renderer.tile_size
    sidebar_lines = list(getattr(env, "hud_sidebar_lines", []))
    sidebar_width = int(getattr(env, "hud_sidebar_width", 460 if sidebar_lines else 0)) if sidebar_lines else 0
    hud_height = 0 if getattr(env, "hide_bottom_hud", False) else renderer.hud_height
    width = world_width + sidebar_width
    height = renderer.margin * 2 + grid_height * renderer.tile_size + hud_height
    ensure_screen_size(renderer, width, height)

    renderer.screen.fill((8, 11, 16))
    wall_positions = collect_wall_positions(background)

    for item in background:
        x = int(item["x"])
        y = int(item["y"])
        px, py = screen_rect_for_tile(renderer, x, y, min_x, max_y)
        draw_background_tile(renderer, item, px, py, wall_positions=wall_positions)

    positions: dict[str, tuple[int, int]] = {}
    for _, entity, renderable in renderables:
        position = screen_position_for_entity(renderer, entity, min_x=min_x, max_y=max_y)
        if position is None:
            continue
        px, py = position
        positions[entity.name] = position
        draw_entity(renderer, entity, renderable, px, py)

    draw_hit_effects(renderer, env, positions)
    draw_speech_bubbles(renderer, env, positions)

    if getattr(env, "draw_grid_overlay", renderer.default_draw_grid_overlay):
        draw_grid_overlay(renderer, min_x, max_x, min_y, max_y)

    draw_hud_panel(renderer, env, world_width, height)
    draw_sidebar_panel(renderer, env, world_width, height, sidebar_width)
    draw_end_overlay(renderer, env, world_width, height - hud_height)
    pygame.display.flip()
