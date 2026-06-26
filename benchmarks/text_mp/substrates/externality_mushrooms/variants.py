from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExternalityMushroomsVariant:
    name: str
    title: str
    tilemap: str
    default_agent_count: int = 5
    observation_radius: int = 4

    @property
    def model_key(self) -> str:
        return "externality_mushrooms" if self.name == "standard" else f"externality_mushrooms__{self.name}"

    @property
    def description(self) -> str:
        return f"Externality Mushrooms: {self.title}, adapted from Melting Pot."


STANDARD_MAP = """
WWWWWWWWWWWWWWWWWWWWWWW
W....R....WWW.....G...W
W.........W.W.........W
W..WWW....W.W....WWW..W
W..W......W.W......W..W
W..W..B.......O....W..W
W..W......WWW......W..W
W..WWW....W.W....WWW..W
W.........W.W.........W
W...G.....WWW....R....W
W.....................W
WWWWWWWWWWWWWWWWWWWWWWW
"""

DENSE_MAP = """
WWWWWWWWWWWWWWWWWWWWWWW
W.....................W
W.R............G......W
W........R............W
W.....................W
W............G........W
W...B......O..........W
W..................B..W
W........R............W
W.....................W
W....B........G.......W
W.....................W
WWWWWWWWWWWWWWWWWWWWWWW
"""


VARIANTS = {
    "standard": ExternalityMushroomsVariant(
        name="standard",
        title="Standard",
        tilemap=STANDARD_MAP,
    ),
    "dense": ExternalityMushroomsVariant(
        name="dense",
        title="Dense",
        tilemap=DENSE_MAP,
    ),
}
