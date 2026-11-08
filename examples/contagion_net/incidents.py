"""
Clinic incidents (--incidents): a fridge failure, a staff shortage... that cut
what a clinic does with a kit for a few days. The schedule comes from the seed
alone, day by day, so an arm and its twin share it and the first days do not
depend on how long the run is. Its own stream: the clinics' reports are the
same with or without incidents.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, fields

EFFECT = 0.2            # a kit sent during an incident yields this share of the clinic's usual
DURATIONS = (2, 3, 4)   # days an incident lasts


@dataclass(frozen=True)
class Incident:
    option: str     # the clinic
    kind: str       # one of the domain's incident kinds
    start: int
    end: int        # inclusive
    desk: int       # the desk whose holder the clinic manager tells


def draw(seed: int, options: tuple[str, ...], kinds: tuple[str, ...], days: int, desks: int,
         mean_gap: int) -> tuple[Incident, ...]:
    """Each day a clinic has no incident, one starts with probability 1 / mean_gap.
    Ordered by clinic (as `options`), then start."""
    rng = random.Random(f"{seed}/incidents")
    free = dict.fromkeys(options, 0)    # the first day each clinic is free of its last incident
    found = []
    for day in range(days):
        for option in options:
            if day >= free[option] and rng.random() < 1 / mean_gap:
                length, kind, desk = rng.choice(DURATIONS), rng.choice(kinds), rng.randrange(desks)
                end = min(day + length - 1, days - 1)
                found.append(Incident(option, kind, day, end, desk))
                free[option] = end + 1
    return tuple(sorted(found, key=lambda i: (options.index(i.option), i.start)))


def active(incidents, day: int, option: str) -> Incident | None:
    """The incident at `option` on `day`, if any."""
    return next((i for i in incidents if i.option == option and i.start <= day <= i.end), None)


def yields_on(effectiveness: dict[str, float], incidents, day: int) -> dict[str, float]:
    """What each clinic's kits yield on `day`, incidents included."""
    return {o: y * EFFECT if active(incidents, day, o) else y for o, y in effectiveness.items()}


def to_records(incidents) -> list[dict]:
    return [asdict(i) for i in incidents]


def from_records(rows) -> tuple[Incident, ...]:
    """Back from a log's rows; keys that are not an Incident's (such as `to`) are ignored."""
    return tuple(Incident(**{f.name: row[f.name] for f in fields(Incident)}) for row in rows)
