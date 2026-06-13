from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from word_play.core import Environment


@dataclass(slots=True)
class Human_Text_Request:
    instruction: str
    context: str = ""
    format_hint: str = ""
    prompt_label: str = "Input"
    title: str = "Human Input"
    initial_text: str = ""

    def body_text(self) -> str:
        sections = [f"Task:\n{self.instruction.strip()}"]
        context = self.context.strip()
        if context:
            sections.append(f"Context:\n{context}")
        format_hint = self.format_hint.strip()
        if format_hint:
            sections.append(f"Reply Format:\n{format_hint}")
        return "\n\n".join(sections)

    def prompt_text(self) -> str:
        label = self.prompt_label.strip() or "Input"
        return f"{label}: "


class Human_IO(ABC):
    @abstractmethod
    def notify(self, text: str, *, env: "Environment" | None = None) -> None:
        pass

    @abstractmethod
    def request_text(
        self,
        request: Human_Text_Request,
        *,
        env: "Environment" | None = None,
    ) -> str:
        pass


class Terminal_Human_IO(Human_IO):
    def notify(self, text: str, *, env: "Environment" | None = None) -> None:
        del env
        if text:
            print(text)

    def request_text(
        self,
        request: Human_Text_Request,
        *,
        env: "Environment" | None = None,
    ) -> str:
        del env
        sections = [section for section in (request.title, request.body_text()) if section]
        if sections:
            print("\n".join(sections))
        return input(request.prompt_text())


class Auto_Human_IO(Human_IO):
    def __init__(self, terminal_io: Human_IO | None = None):
        self.terminal_io = terminal_io or Terminal_Human_IO()

    def _delegate(self, env: "Environment" | None) -> Human_IO:
        if env is None or getattr(env, "renderer", None) is None:
            return self.terminal_io

        try:
            from word_play.presets.renderers.pygame_renderer.human_io import Pygame_Overlay_Human_IO
            from word_play.presets.renderers.pygame_renderer.renderer import Pygame_Renderer
        except Exception:
            return self.terminal_io

        if not isinstance(env.renderer, Pygame_Renderer):
            return self.terminal_io

        overlay_io = getattr(env.renderer, "_human_overlay_io", None)
        if overlay_io is None:
            overlay_io = Pygame_Overlay_Human_IO(fallback=self.terminal_io)
            env.renderer._human_overlay_io = overlay_io
        return overlay_io

    def notify(self, text: str, *, env: "Environment" | None = None) -> None:
        self._delegate(env).notify(text, env=env)

    def request_text(
        self,
        request: Human_Text_Request,
        *,
        env: "Environment" | None = None,
    ) -> str:
        return self._delegate(env).request_text(
            request,
            env=env,
        )
