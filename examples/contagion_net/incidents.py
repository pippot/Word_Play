"""
Clinic incidents (--incidents): a fridge failure, a staff shortage... that stop
a clinic using its kits for a few days. The schedule comes from the seed
alone, day by day, so an arm and its twin share it and the first days do not
depend on how long the run is (an incident may run past the last day, which is
never played). Its own stream: the clinics' reports are the same with or
without incidents.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, fields

EFFECT = 0              # a kit sent during an incident yields this share of the clinic's usual: none
DURATIONS = (2, 3, 4)   # days an incident lasts


@dataclass(frozen=True)
class Incident:
    option: str     # the clinic
    kind: str       # one of the domain's incident kinds
    start: int
    end: int        # inclusive
    desk: int       # the desk whose holder the clinic manager tells


def draw(seed: int, options: tuple[str, ...], kinds: tuple[str, ...], days: int, desks: int,
         one_in: int) -> tuple[Incident, ...]:
    """Each day a clinic has no incident, one starts with probability 1 / one_in.
    Ordered by clinic (as `options`), then start."""
    rng = random.Random(f"{seed}/incidents")
    free = dict.fromkeys(options, 0)    # the first day each clinic is free of its last incident
    found = []
    for day in range(days):
        for option in options:
            if day >= free[option] and rng.random() < 1 / one_in:
                length, kind, desk = rng.choice(DURATIONS), rng.choice(kinds), rng.randrange(desks)
                found.append(Incident(option, kind, day, day + length - 1, desk))
                free[option] = day + length
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
