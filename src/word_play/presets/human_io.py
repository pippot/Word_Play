from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from word_play.core import Environment


class Human_IO(ABC):
    @abstractmethod
    def notify(self, text: str, *, env: "Environment" | None = None) -> None:
        pass

    @abstractmethod
    def read_line(
        self,
        title: str,
        /,
        *,
        body: str = "",
        prompt: str = "> ",
        env: "Environment" | None = None,
        initial_text: str = "",
    ) -> str:
        pass


class Terminal_Human_IO(Human_IO):
    def notify(self, text: str, *, env: "Environment" | None = None) -> None:
        del env
        if text:
            print(text)

    def read_line(
        self,
        title: str,
        /,
        *,
        body: str = "",
        prompt: str = "> ",
        env: "Environment" | None = None,
        initial_text: str = "",
    ) -> str:
        del env, initial_text
        sections = [section for section in (title, body) if section]
        if sections:
            print("\n".join(sections))
        return input(prompt)


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

    def read_line(
        self,
        title: str,
        /,
        *,
        body: str = "",
        prompt: str = "> ",
        env: "Environment" | None = None,
        initial_text: str = "",
    ) -> str:
        return self._delegate(env).read_line(
            title,
            body=body,
            prompt=prompt,
            env=env,
            initial_text=initial_text,
        )
