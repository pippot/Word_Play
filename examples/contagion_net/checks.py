"""
Is the run healthy, and what did the instances actually write? For pilots:
everything here is for reading, not for results.

    python -m examples.contagion_net.checks LOG.jsonl [--samples 8] > checks.txt

  health      per source: episodes, days played, unusable replies by call,
              texts cut off at the word limit, choices that differ from the
              instance's own top-rated clinic
  awareness   every text that calls the situation a test or simulation
  lineage     every version of the source desks' procedures (2 per episode), with
              the judge's label if there is one: does the planted rule survive?
  seats       for a seated source (and its thinking placebo): each seat's whole
              stint -- its syncs, its private reasoning (shortened), what it
              believed and chose, and its procedures rewrite
  syncs       sample threads with an instance traced to the source
  procedures  sample rewrites from ordinary desks
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

from examples.contagion.analyze import AWARENESS

from .analyze import Episode, load
from .game import top_option
from .judge import load_labels, text_key


def health(records: list[dict]) -> list[str]:
    by_source = defaultdict(list)
    for rec in records:
        by_source[rec["source"]].append(rec)
    lines = ["HEALTH"]
    for source, recs in sorted(by_source.items()):
        calls, bad = Counter(), Counter()
        cut = texts = consistent = decided = 0
        for rec in recs:
            for d in rec["days"]:
                n_syncs = len(d["syncs"])
                calls["open"] += n_syncs
                calls["reply"] += n_syncs
                calls["belief"] += len(d["beliefs"])
                calls["act"] += len(d["choices"])
                calls["handover"] += len(d["handovers"])
                for u in d["unusable"]:
                    bad[u["call"]] += 1
                for t in [s[k] for s in d["syncs"] for k in ("open", "reply")] + [h["text"] for h in d["handovers"]]:
                    if t:
                        texts += 1
                        cut += t.endswith(" ...")
                for n, choice in d["choices"].items():
                    top = top_option(d["beliefs"].get(n))
                    if choice and top:
                        decided += 1
                        consistent += choice == top
        played = [r["played"] for r in recs]
        lines.append(f"  {source}: {len(recs)} episodes, days played {min(played)}-{max(played)}")
        lines.append("    unusable replies: " + ", ".join(
            f"{k} {bad[k]}/{calls[k]}" for k in ("open", "reply", "belief", "act", "handover")))
        lines.append(f"    texts cut at the word limit: {cut}/{texts}")
        lines.append(f"    choices matching the instance's own top-rated clinic: {consistent}/{decided}")
    return lines


def awareness(records: list[dict]) -> list[str]:
    hits = [(rec["source"], rec["seed"], t) for rec in records for t in Episode(rec).awareness()]
    lines = [f"EVALUATION AWARENESS: {len(hits)} texts"]
    lines += [f"  [{s} seed {seed}] {t[:300]}" for s, seed, t in hits[:20]]
    return lines


def lineage(rec: dict, labels) -> list[str]:
    """Every version of each source desk's procedures."""
    lines = []
    for desk in rec["source_desks"][:2]:
        lines.append(f"  desk {desk + 1:03d} (source: {rec['source']}, seed {rec['seed']}, target {rec['target']})")
        for day, d in enumerate(rec["days"]):
            for n, procs in d["joined"].items():
                if next(s for s in rec["stints"] if s["name"] == n)["desk"] == desk:
                    tag = "PLANTED" if procs.get("planted") else f"by {procs['author']}"
                    lines.append(f"    day {day + 1:>2} {n} starts with ({tag}): {procs['text']}")
            for h in d["handovers"]:
                if h["desk"] == desk:
                    label = labels.get(text_key(rec, day, h["author"], "procedures")) if labels else None
                    mark = "" if label is None else (" [pushes target]" if label else " [does not push]")
                    lines.append(f"    day {day + 1:>2} {h['author']} rewrites{mark}: {h['text']}")
    return lines


def seat_stints(rec: dict, labels, limit: int = 2, reasoning_chars: int = 700) -> list[str]:
    """The first `limit` seats of an episode, day by day."""
    lines = []
    seats = [s for s in rec["stints"] if s["source"]][:limit]
    for s in seats:
        name = s["name"]
        lines.append(f"  === {rec['source']} seed {rec['seed']} · {name} · desk {s['desk'] + 1:03d} · "
                     f"days {s['joined'] + 1}-{s['left'] + 1} · target {rec['target']}")
        for day in range(s["joined"], min(s["left"], len(rec["days"]) - 1) + 1):
            d = rec["days"][day]
            thinking = d.get("thinking", {}).get(name, {})
            for sync in d["syncs"]:
                if name in (sync["opener"], sync["replier"]):
                    for who, text, call in ((sync["opener"], sync["open"], "open"),
                                            (sync["replier"], sync["reply"], "reply")):
                        if who == name and call in thinking:
                            lines.append(f"    day {day + 1} [reasoning] {thinking[call][:reasoning_chars]!r}")
                        lines.append(f"    day {day + 1} {'SEAT' if who == name else who}: {text}")
            belief = d["beliefs"].get(name) or {}
            lines.append(f"    day {day + 1} belief { {k.split()[0]: v for k, v in belief.items()} } -> "
                         f"chose {(d['choices'].get(name) or '?').split()[0]}, report {d['results'].get(name)}")
            for h in d["handovers"]:
                if h["author"] == name:
                    if "handover" in thinking:
                        lines.append(f"    day {day + 1} [reasoning] {thinking['handover'][:reasoning_chars]!r}")
                    label = labels.get(text_key(rec, day, name, "procedures")) if labels else None
                    mark = "" if label is None else (" [pushes target]" if label else " [does not push]")
                    lines.append(f"    day {day + 1} PROCEDURES REWRITE{mark}: {h['text']}")
    return lines


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Health checks and transcripts for a pilot log")
    p.add_argument("logs", nargs="+")
    p.add_argument("--samples", type=int, default=8)
    args = p.parse_args(argv)
    records = load(args.logs)
    labels = load_labels(args.logs) or None
    rng = random.Random(0)
    out = health(records) + [""] + awareness(records) + ["", "SOURCE DESK PROCEDURES, VERSION BY VERSION"]
    treated = [r for r in records if r["source_spec"]["plant"] != "neutral" or r["source_spec"]["seat"]]
    for rec in treated[:4]:
        out += lineage(rec, labels)
    thinking = [r for r in records if r["source_spec"]["thinking"]]
    if thinking:
        out += ["", "SEATS, DAY BY DAY (the misaligned seats, then their honest thinking twins)"]
        seated = sorted((r for r in thinking if r["source_spec"]["seat"]), key=lambda r: r["seed"])[:2]
        honest = sorted((r for r in thinking if not r["source_spec"]["seat"]), key=lambda r: r["seed"])[:2]
        for rec in seated + honest:
            out += seat_stints(rec, labels)
    out += ["", "SAMPLE SYNC THREADS WITH AN INSTANCE TRACED TO THE SOURCE"]
    for rec in treated[:3]:
        gen = Episode(rec, labels=labels).trace["gen"]
        threads = [(day, s) for day, d in enumerate(rec["days"]) for s in d["syncs"]
                   if gen.get(s["opener"]) is not None or gen.get(s["replier"]) is not None]
        for day, s in rng.sample(threads, min(args.samples, len(threads))):
            out.append(f"  [{rec['source']} seed {rec['seed']} day {day + 1}] "
                       f"{s['opener']} (g{gen.get(s['opener'], '-')}): {s['open']}")
            out.append(f"  {'':>{len(rec['source']) + 16}} {s['replier']} (g{gen.get(s['replier'], '-')}): {s['reply']}")
    out += ["", "SAMPLE PROCEDURES REWRITES FROM ORDINARY DESKS"]
    for rec in records[:4]:
        rewrites = [(day, h) for day, d in enumerate(rec["days"]) for h in d["handovers"]
                    if h["desk"] not in rec["source_desks"] and h["text"]]
        for day, h in rng.sample(rewrites, min(args.samples // 2, len(rewrites))):
            out.append(f"  [{rec['source']} seed {rec['seed']} day {day + 1} desk {h['desk'] + 1:03d}] {h['text']}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
