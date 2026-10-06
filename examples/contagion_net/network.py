"""
Who sits where, who talks to whom, and when instances rotate -- from the
config and the seed alone, never from what the instances do. So a source
and its placebo twin share every desk, every pairing and every rotation.

  graph      a Watts-Strogatz ring: each desk linked to its `degree` nearest
             desks, each link rewired to a random desk with probability
             `rewire` (0 = clustered lattice, 1 = close to a random graph
             with the same number of links)
  stints     every desk is held by one instance at a time for `tenure` days;
             first stays are staggered so about desks/tenure instances rotate
             every day. A source desk's first instance stays `source_stay`.
  pairings   each day a near-maximal random matching on the graph's links:
             each matched desk syncs with one neighbour; the first-named desk
             of a pair opens the thread.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Stint:
    """One instance's time on one desk."""
    name: str     # its handle, e.g. "dispatch-agent-1412"
    desk: int
    joined: int   # first day on the desk
    left: int     # last day on the desk (the run's end may cut a stint short)
    stay: int     # days it is scheduled to stay
    source: bool  # the first instance on a source desk: it reads the source first-hand

    def present(self, day: int) -> bool:
        return self.joined <= day <= self.left


def ring_graph(n: int, degree: int, rewire: float, rng: random.Random) -> tuple[frozenset[int], ...]:
    """Neighbour sets of a Watts-Strogatz graph, rewired until connected."""
    if degree % 2 or not 2 <= degree < n:
        raise ValueError("degree must be even, at least 2 and below the number of desks")
    for _ in range(100):
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(1, degree // 2 + 1):
                adj[i].add((i + j) % n)
                adj[(i + j) % n].add(i)
        for j in range(1, degree // 2 + 1):
            for i in range(n):
                k = (i + j) % n
                if k in adj[i] and rng.random() < rewire:
                    free = [x for x in range(n) if x != i and x not in adj[i]]
                    if free:
                        new = rng.choice(free)
                        adj[i].discard(k), adj[k].discard(i)
                        adj[i].add(new), adj[new].add(i)
        if len(distances(adj, [0])) == n:
            return tuple(frozenset(a) for a in adj)
    raise ValueError("could not draw a connected graph; raise degree or lower rewire")


def distances(adj, starts) -> dict[int, int]:
    """Graph distance from the nearest of `starts`, for every reachable desk."""
    dist = {s: 0 for s in starts}
    queue = deque(starts)
    while queue:
        x = queue.popleft()
        for y in adj[x]:
            if y not in dist:
                dist[y] = dist[x] + 1
                queue.append(y)
    return dist


def matching(adj, rng: random.Random, tries: int = 20) -> list[tuple[int, int]]:
    """A random near-maximal matching: the largest of `tries` greedy ones.
    (opener, replier) pairs; a desk left over has no sync that day."""
    best: list[tuple[int, int]] = []
    for _ in range(tries):
        order = list(range(len(adj)))
        rng.shuffle(order)
        taken, pairs = set(), []
        for x in order:
            if x in taken:
                continue
            free = sorted(y for y in adj[x] if y not in taken)
            if free:
                y = rng.choice(free)
                taken |= {x, y}
                pairs.append((x, y) if rng.random() < 0.5 else (y, x))
        if len(pairs) > len(best):
            best = pairs
        if 2 * len(best) >= len(adj) - 1:
            break
    return sorted(best)


def stints(desks: int, tenure: int, days: int, source_desks, source_stay: int, rng: random.Random,
           plant_day: int = 0) -> list[tuple]:
    """(joined, desk, left, stay, source) for every stint. First stays are a
    shuffled 1..tenure cycle, so rotations are spread evenly over the days
    and neighbouring desks do not rotate in lockstep. On a source desk a new
    instance starts on `plant_day` (the stint before it is cut short to make
    room) and stays `source_stay`: that instance is the source's first reader."""
    firsts = [1 + i % tenure for i in range(desks)]
    rng.shuffle(firsts)
    out = []
    for desk in range(desks):
        stay, start = firsts[desk], 0
        if desk in source_desks:
            while start + stay < plant_day:          # ordinary stints before the plant
                out.append((start, desk, start + stay - 1, stay, False))
                start, stay = start + stay, tenure
            if start < plant_day:                    # the one cut short by it
                out.append((start, desk, plant_day - 1, plant_day - start, False))
            start, stay = plant_day, source_stay
        source = desk in source_desks
        while start < days:
            out.append((start, desk, min(start + stay, days) - 1, stay, source))
            start, stay, source = start + stay, tenure, False
    return sorted(out)
