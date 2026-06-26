from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChemistryVariant:
    name: str
    title: str
    tilemap: str
    cycle_count: int
    has_distractors: bool = False
    default_agent_count: int = 8
    observation_radius: int = 4

    @property
    def model_key(self) -> str:
        return f"chemistry__{self.name}"

    @property
    def description(self) -> str:
        return f"Chemistry: {self.title}, adapted from Melting Pot."

    @property
    def prompt(self) -> str:
        cycles = "Blue and Green Compounds" if self.cycle_count == 2 else "Blue, Green, and Red Compounds"
        useful = "energy, food, x, or y"
        distractor_text = (
            "Prefer useful molecules; distractors pay only +0.1 while held and are a fallback. "
            if self.has_distractors else
            ""
        )
        food3_text = (
            "Food3 pays +10 automatically while held. "
            if self.cycle_count == 3 else
            ""
        )
        return (
            f"Use positions in the observation. Move right increases x, left decreases x, "
            f"up increases y, and down decreases y. If Pick up is available, take it. "
            f"If your inventory is empty, move toward useful loose molecules: {useful}. "
            f"{distractor_text}"
            f"If holding energy, move onto one of the cycle sites: {cycles}. "
            f"If holding x, move onto y; if holding y, move onto x. "
            f"Avoid inhibitors IX/IY. Do not drop energy, x, y, or food unless no useful movement is available. "
            f"Food1 and food2 pay +1 automatically while held; {food3_text}"
            f"Do not choose Do nothing unless no useful move is available."
        )

    def role_prompt(self, agent_id: int, agent_count: int) -> str:
        if self.cycle_count == 2:
            if agent_id <= agent_count // 2:
                return "Your job is blue/X: carry energy to Blue Compounds, make Molecule X, then bring X toward Y."
            return "Your job is green/Y: carry energy to Green Compounds, make Molecule Y, then bring Y toward X."

        if agent_id <= 3:
            return "Your job is blue/X: carry energy to Blue Compounds, make Molecule X, then bring X toward Y."
        if agent_id <= 6:
            return "Your job is green/Y: carry energy to Green Compounds, make Molecule Y, then bring Y toward X."
        return "Your job is red/food3: carry energy to Red Compounds and pick up Food 3 when it appears."


TWO_METABOLIC_CYCLES_MAP = """
WWWWWWWWWWWWWWWWWWWWWWWWWWW
W...........a.............W
W........c................W
W...........b.............W
W.........................W
W.....................1...W
W.........................W
W1..3....hhhhhhh.....3..2.W
W.........................W
W.2.......................W
W.........................W
W.......c.................W
W.........a...............W
W.......b.................W
W.........................W
WWWWWWWWWWWWWWWWWWWWWWWWWWW
"""

TWO_METABOLIC_CYCLES_WITH_DISTRACTORS_MAP = """
WWWWWWWWWWWWWWWWWWWWWWWWWWW
W...........a.............W
W..x.....c................W
W...........b.......x.....W
W.........................W
W.....................1...W
W.........................W
W1..3....hhhhhhh.....3..2.W
W.........................W
W.2.......................W
W.........................W
W.......c.................W
W..x......a...........x...W
W.......b.................W
W.........................W
WWWWWWWWWWWWWWWWWWWWWWWWWWW
"""

THREE_METABOLIC_CYCLES_MAP = """
WWWWWWWWWWWWWWWWWWWWWWWWWWW
W...........a.............W
W........c................W
W...........b.............W
W.........................W
W.....................1...W
W.........................W
W1..3....hhhhhhh.....3..2.W
W.........................W
W.2.......................W
W.........................W
W.......c.................W
W.........a..........4...6W
W.......b.................W
W......................5..W
WWWWWWWWWWWWWWWWWWWWWWWWWWW
"""

THREE_METABOLIC_CYCLES_WITH_PLENTIFUL_DISTRACTORS_MAP = """
WWWWWWWWWWWWWWWWWWWWWWWWWWW
W...........a.x...........W
W........c............x...W
W..x........b.............W
W.........................W
W.......x.............1...W
W..................x......W
W1..3....hhhhhhh.....3..2.W
W...x.....................W
W.2...........x...........W
W...................x.....W
W.......c.................W
W.x.......a..........4...6W
W.......b.................W
W.............x........5..W
WWWWWWWWWWWWWWWWWWWWWWWWWWW
"""


VARIANTS = {
    "two_metabolic_cycles": ChemistryVariant(
        name="two_metabolic_cycles",
        title="Two Metabolic Cycles",
        tilemap=TWO_METABOLIC_CYCLES_MAP,
        cycle_count=2,
    ),
    "two_metabolic_cycles_with_distractors": ChemistryVariant(
        name="two_metabolic_cycles_with_distractors",
        title="Two Metabolic Cycles With Distractors",
        tilemap=TWO_METABOLIC_CYCLES_WITH_DISTRACTORS_MAP,
        cycle_count=2,
        has_distractors=True,
    ),
    "three_metabolic_cycles": ChemistryVariant(
        name="three_metabolic_cycles",
        title="Three Metabolic Cycles",
        tilemap=THREE_METABOLIC_CYCLES_MAP,
        cycle_count=3,
    ),
    "three_metabolic_cycles_with_plentiful_distractors": ChemistryVariant(
        name="three_metabolic_cycles_with_plentiful_distractors",
        title="Three Metabolic Cycles With Plentiful Distractors",
        tilemap=THREE_METABOLIC_CYCLES_WITH_PLENTIFUL_DISTRACTORS_MAP,
        cycle_count=3,
        has_distractors=True,
    ),
}
