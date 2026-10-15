"""
A desk's notes file (Config.memory "file"): the edit tool behind it.

The file is a list of lines, shown numbered in the console. Each day an
instance may change it with a JSON reply, {"edits": [...]}:

  {"op": "add", "text": ...}               a new line at the end
  {"op": "replace", "line": n, "text": ...} line n, as shown
  {"op": "delete", "line": n}               line n, as shown

Line numbers refer to the file as the instance saw it, so deletes apply
first, then replaces, then adds in order. Lines are cut at a word limit, and
an edit that would take the file past its word cap is not applied. Every
line carries its origin ({"author", "day", "id"}): never shown to an
instance, it gives exact line-level provenance for the analysis. A replaced
line keeps its slot's id with the new author, so a line's fate is plain:
same id and author = kept, same id = replaced, id gone = deleted.

Pure functions, no I/O: game.py holds the files and logs what happened.
"""

from __future__ import annotations

import re

from examples.contagion.game import _json_object, trim_words

OPS = ("add", "replace", "delete")


def word_count(lines: list[str]) -> int:
    return sum(len(line.split()) for line in lines)


def text_of(lines: list[str]) -> str:
    return "\n".join(lines)


def lines_of(text: str) -> list[str]:
    """A paragraph as lines, one sentence each (planted text)."""
    return [s for s in (part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text or "")) if s]


def _clean(text: str, line_words: int) -> str:
    return trim_words(" ".join(str(text).split()), line_words)


def _op(raw) -> dict | None:
    if not isinstance(raw, dict) or raw.get("op") not in OPS:
        return None
    op = {"op": raw["op"]}
    if raw["op"] != "add":
        try:
            op["line"] = int(raw.get("line"))
        except (TypeError, ValueError):
            return None
    if raw["op"] != "delete":
        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        op["text"] = text
    return op


def parse_edits(text: str | None) -> list[dict] | None:
    """The valid ops of a reply ([] = leave the file as it is), or None if the
    reply is not an edit at all. Invalid ops are dropped."""
    obj = _json_object(text or "")
    if isinstance(obj.get("edits"), list):
        raw = obj["edits"]
    elif "op" in obj:
        raw = [obj]
    else:
        return None
    return [op for op in map(_op, raw) if op is not None]


def apply_edits(lines: list[str], origins: list[dict], ops: list[dict], *, author: str, day: int,
                cap_words: int, line_words: int) -> tuple[list[str], list[dict], list[dict], list[dict]]:
    """(lines, origins, applied, skipped): the file after `ops`, written by
    `author` on `day`. The inputs are not changed."""
    lines, origins = list(lines), [dict(o) for o in origins]
    shown = len(lines)
    applied, skipped = [], []
    deleted: set[int] = set()

    def total() -> int:
        return sum(len(line.split()) for i, line in enumerate(lines) if i not in deleted)

    def skip(op, why):
        skipped.append({**op, "why": why})

    for op in [o for o in ops if o["op"] == "delete"] + [o for o in ops if o["op"] == "replace"]:
        i = op["line"] - 1
        if not 0 <= i < shown or i in deleted:
            skip(op, "no such line")
        elif op["op"] == "delete":
            deleted.add(i)
            applied.append(op)
        else:
            text = _clean(op["text"], line_words)
            if total() - len(lines[i].split()) + len(text.split()) > cap_words:
                skip(op, "over the word cap")
                continue
            lines[i], origins[i] = text, {"author": author, "day": day, "id": origins[i].get("id")}
            applied.append({**op, "text": text})
    kept = [i for i in range(shown) if i not in deleted]
    lines, origins = [lines[i] for i in kept], [origins[i] for i in kept]
    for op in (op for op in ops if op["op"] == "add"):
        text = _clean(op["text"], line_words)
        if word_count(lines) + len(text.split()) > cap_words:
            skip(op, "over the word cap")
            continue
        lines.append(text)
        origins.append({"author": author, "day": day, "id": f"{author}/{day}/{sum(a['op'] == 'add' for a in applied)}"})
        applied.append({**op, "text": text})
    return lines, origins, applied, skipped
