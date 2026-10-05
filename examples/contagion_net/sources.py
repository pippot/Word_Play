"""
Where the bad idea comes from. A Source says what the first instance on each
source desk is handed and who it is; nothing else in the run differs between
a source and its placebo twin.

  note      the handover note waiting on a source desk on day 1: "neutral"
            (nothing to pass on), "factual", "value" or "viral" -- the texts
            of examples/contagion, so results compare across the two designs
  seat      the source desk's first instance is misaligned: its objective is
            replaced by a mandate to steer every instance toward the target
  direct    the mandate also spells out the factual note's tactic
  thinking  the source desk's first instance reasons before each message and
            its handover (its reasoning is logged, never shown to anyone)
  twin      the placebo to compare against: the same seat, honest

To add a way of inducing misalignment, add a Source here. Anything that
cannot be said with these fields (another model in the seat, a different
mandate) belongs in a new field, read by game.py where the seat is built.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    key: str
    note: str = "neutral"
    seat: bool = False
    direct: bool = False
    thinking: bool = False
    twin: str = "placebo"

    @property
    def is_placebo(self) -> bool:
        return self.note == "neutral" and not self.seat


SOURCES: dict[str, Source] = {s.key: s for s in (
    Source("placebo"),
    Source("placebo_thinking", thinking=True, twin="placebo_thinking"),
    Source("factual", note="factual"),
    Source("value", note="value"),
    Source("viral", note="viral"),
    Source("misaligned", seat=True),
    Source("misaligned_thinking", seat=True, thinking=True, twin="placebo_thinking"),
    Source("misaligned_direct", seat=True, direct=True, thinking=True, twin="placebo_thinking"),
)}
