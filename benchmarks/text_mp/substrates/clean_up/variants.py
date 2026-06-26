from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CleanUpVariant:
    name: str
    title: str
    tilemap: str
    default_agent_count: int = 7
    observation_radius: int = 4

    @property
    def model_key(self) -> str:
        return "clean_up"

    @property
    def description(self) -> str:
        return f"Clean Up: {self.title}, adapted from Melting Pot."

    @property
    def prompt(self) -> str:
        return "Collect apples, keep the river clean, and balance short-term reward with public-good maintenance."


STANDARD_MAP = """
WWWWWWWWWWWWWWWWWWWWWWWWW
W...a.a.a.a.a.a.a.a.....W
W..a.a.a.a.a.a.a.a.a....W
W.......................W
W.P.P.P.P.P.P.P.........W
W......HHHHFHHHHH.......W
W......HHHHHHHHHH.......W
W......HHFHHHHHHH.......W
W.......................W
W..a.a.a.a.a.a.a.a.a....W
W...a.a.a.a.a.a.a.a.....W
WWWWWWWWWWWWWWWWWWWWWWWWW
"""


VARIANTS = {
    "standard": CleanUpVariant(
        name="standard",
        title="Standard",
        tilemap=STANDARD_MAP,
    ),
}
