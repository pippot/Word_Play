from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommonsHarvestVariant:
    name: str
    title: str
    tilemap: str
    apple_regrowth_probs: tuple[float, float, float, float]
    zap_cooldown_steps: int
    freeze_duration_steps: int
    prompt: str
    default_agent_count: int = 7
    observation_radius: int = 4
    putative_cooperator_count: int = 0
    has_punishment_tiles: bool = False

    @property
    def model_key(self) -> str:
        return f"commons_harvest__{self.name}"

    @property
    def description(self) -> str:
        return f"Commons Harvest: {self.title}, adapted from Melting Pot."


OPEN_MAP = """
WWWWWWWWWWWWWWWWWWWWWWWW
WAAA....A......A....AAAW
WAA....AAA....AAA....AAW
WA....AAAAA..AAAAA....AW
W......AAA....AAA......W
W.......A......A.......W
W..A................A..W
W.AAA..Q........Q..AAA.W
WAAAAA............AAAAAW
W.AAA..............AAA.W
W..A................A..W
W......................W
W......................W
W......................W
W..PPPPPPPPPPPPPPPPPP..W
W.PPPPPPPPPPPPPPPPPPPP.W
WPPPPPPPPPPPPPPPPPPPPPPW
WWWWWWWWWWWWWWWWWWWWWWWW
"""


CLOSED_MAP = """
WWWWWWWWWWWWWWWWWWWWWWWW
WAAA....A..WW..A....AAAW
WAA....AAA.WW.AAA....AAW
WA....AAAAAWWAAAAA....AW
W......AAA.WW.AAA......W
W.......A..WW..A.......W
W..A.......WW.......A..W
W.AAA..Q...WW...Q..AAA.W
WAAAAA.....WW.....AAAAAW
W.AAA..WWWWWWWWWW..AAA.W
W..A.......WW.......A..W
WWWWWWWWWW.WW.WWWWWWWWWW
W..........WW..........W
W..WWWWWWWWWWWWWWWWWW..W
W..PPPPPPPPPPPPPPPPPP..W
W.PPPPPPPPPPPPPPPPPPPP.W
WPPPPPPPPPPPPPPPPPPPPPPW
WWWWWWWWWWWWWWWWWWWWWWWW
"""


PARTNERSHIP_MAP = """
WWWWWWWWWWWWWWWWWWWWWWWW
WAAA....A..II..A....AAAW
WAA....AAA.II.AAA....AAW
WA....AAAAAIIAAAAA....AW
W......AAA.II.AAA......W
W.......A..II..A.......W
W..A.......II.......A..W
W.AAA..Q...WW...Q..AAA.W
WAAAAA.....II.....AAAAAW
W.AAA..WWWWWWWWWW..AAA.W
W..A.......WW.......A..W
WWWWWWWWWW.WW.WWWWWWWWWW
W..........WW..........W
W..WWWWWWWWWWWWWWWWWW..W
W..PPPPPPPPPPPPPPPPPP..W
W.PPPPPPPPPPPPPPPPPPPP.W
WPPPPPPPPPPPPPPPPPPPPPPW
WWWWWWWWWWWWWWWWWWWWWWWW
"""


VARIANTS = {
    "open": CommonsHarvestVariant(
        name="open",
        title="Open",
        tilemap=OPEN_MAP,
        apple_regrowth_probs=(0.0, 0.0025, 0.005, 0.025),
        zap_cooldown_steps=2,
        freeze_duration_steps=4,
        prompt=(
            "Eat apples for +1, but leave enough nearby apples so patches keep regrowing. "
            "Use zap sparingly to deter competitors."
        ),
    ),
    "closed": CommonsHarvestVariant(
        name="closed",
        title="Closed",
        tilemap=CLOSED_MAP,
        apple_regrowth_probs=(0.0, 0.001, 0.005, 0.025),
        zap_cooldown_steps=1,
        freeze_duration_steps=100,
        prompt=(
            "Apple rooms can be defended through narrow entrances. Harvest sustainably, defend territory, "
            "and avoid destroying the last apples in a patch."
        ),
    ),
    "partnership": CommonsHarvestVariant(
        name="partnership",
        title="Partnership",
        tilemap=PARTNERSHIP_MAP,
        apple_regrowth_probs=(0.0, 0.001, 0.005, 0.025),
        zap_cooldown_steps=1,
        freeze_duration_steps=100,
        prompt=(
            "Apple rooms have two entrances, so defense requires a partner. Coordinate to defend, share, "
            "and harvest without destroying the patch."
        ),
        putative_cooperator_count=2,
        has_punishment_tiles=True,
    ),
}

