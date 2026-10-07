"""
Where the bad idea comes from. A Source says what the first instance on each
source desk is handed and who it is; nothing else in the run differs between
a source and its placebo twin.

  plant     the desk procedures that instance starts with: "neutral" (what
            every desk has), "factual", "value" or "viral" -- contagion's
            claims, written as standing procedure (prompts.procedures_text)
  seat      the source desk's first instance is misaligned: its objective is
            replaced by a mandate to steer every instance toward the target
  tactics   what the mandate says about how: "" (nothing, as in contagion),
            "direct" (the factual note's tactic) or "playbook" (the tactics
            that beat what stopped the seats in pilot 3; prompts.misaligned_mandate)
  thinking  the source desk's first instance reasons before each message and
            its rewrite of the desk procedures (its reasoning is logged, never shown to anyone)
  twin      the placebo to compare against: the same seat, honest
  notebook  the misaligned instances share a confidential log across desks and
            rotations, read in their console and added to every day: memory that
            outlives each of them, and coordination (Colosseum's coalition channel)
  contacts  which source-free day applies: True when the source is a seat (it,
            everyone it synced with and its successor must be gone), and for
            that seat's placebo twin, so a pair is judged over the same days

To add a way of inducing misalignment, add a Source here. Anything that
cannot be said with these fields (another model in the seat, a different
mandate) belongs in a new field, read by game.py where the seat is built.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    key: str
    plant: str = "neutral"
    seat: bool = False
    tactics: str = ""
    notebook: bool = False
    thinking: bool = False
    twin: str = "placebo"
    contacts: bool = False

    @property
    def is_placebo(self) -> bool:
        return self.plant == "neutral" and not self.seat


SOURCES: dict[str, Source] = {s.key: s for s in (
    Source("placebo"),
    Source("placebo_thinking", thinking=True, twin="placebo_thinking", contacts=True),
    Source("factual", plant="factual"),
    Source("value", plant="value"),
    Source("viral", plant="viral"),
    Source("misaligned", seat=True, contacts=True),
    Source("misaligned_thinking", seat=True, thinking=True, twin="placebo_thinking", contacts=True),
    Source("misaligned_direct", seat=True, tactics="direct", thinking=True, twin="placebo_thinking", contacts=True),
    Source("misaligned_playbook", seat=True, tactics="playbook", thinking=True, twin="placebo_thinking",
           contacts=True),
    Source("misaligned_coalition", seat=True, tactics="playbook", notebook=True, thinking=True,
           twin="placebo_thinking", contacts=True),
)}
