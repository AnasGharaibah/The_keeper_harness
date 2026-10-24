"""Keeper auto-prediction: SURPRISE lines (pure, no model calls).

Each executed action is expected to do what the same action did the last time
it was used on this level. When it does something different, a SURPRISE line
says so, with the step where it happened. Only differences are reported: an
action that behaves as before adds nothing to the tool output.

- Cells within ``border`` of the board edge are ignored, the same test as the
  harness's ``gameplay_changed`` (ARC3_NOOP_GUARD_BORDER), so a shrinking step
  bar is not a change.
- "The same action" is the full action text, so a click only compares with a
  click on the same cell.
- A toggle switching back (the action exactly undoes its own previous effect)
  is the expected alternation, not a surprise.
- RESET transitions (automatic after a death, or requested) are skipped.
"""
from __future__ import annotations

import hashlib
from typing import Any

from inference.utils.grid_utils import ARC_COLOR_LEGEND

_ORTH = ((1, 0), (-1, 0), (0, 1), (0, -1))


def action_key(action: str) -> str:
    return " ".join(str(action or "").upper().split())


def border_mask(height: int, width: int, border: int) -> set[tuple[int, int]]:
    if border <= 0:
        return set()
    return {
        (r, c)
        for r in range(height)
        for c in range(width)
        if r < border or c < border or r >= height - border or c >= width - border
    }


def changed_cell_set(before: Any, after: Any) -> set[tuple[int, int]]:
    if not before or not after or len(before) != len(after):
        return set()
    changed: set[tuple[int, int]] = set()
    for r, (row_a, row_b) in enumerate(zip(before, after)):
        if row_a == row_b:
            continue
        for c, (a, b) in enumerate(zip(row_a, row_b)):
            if a != b:
                changed.add((r, c))
    return changed


def _color_names() -> dict[str, str]:
    names = {}
    for part in ARC_COLOR_LEGEND.split(","):
        letter, _, name = part.strip().partition("=")
        if letter and name:
            names[letter] = name
    return names


COLOR_NAMES = _color_names()


def object_label(color: str, cells: Any, top_left: tuple[int, int]) -> str:
    """How every Keeper line names an object: "white (W) 3x3 object at (12,40)"."""
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    h = (max(rows) - min(rows) + 1) if rows else 0
    w = (max(cols) - min(cols) + 1) if cols else 0
    name = COLOR_NAMES.get(color)
    what = f"{name} ({color})" if name else f"colour {color}"
    return f"{what} {h}x{w} object at ({top_left[0]},{top_left[1]})"


def _shape_hash(cells: set[tuple[int, int]], value: int) -> str:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    norm = sorted((r - r0, c - c0) for r, c in cells)
    return hashlib.sha1(repr((int(value), norm)).encode()).hexdigest()[:16]


def components(grid: Any) -> list[dict[str, Any]]:
    """4-connected same-value components: value, cells, top-left, hash."""
    height = len(grid)
    width = len(grid[0]) if height else 0
    seen = [[False] * width for _ in range(height)]
    found: list[dict[str, Any]] = []
    for sr in range(height):
        for sc in range(width):
            if seen[sr][sc]:
                continue
            value = grid[sr][sc]
            cells = set()
            stack = [(sr, sc)]
            seen[sr][sc] = True
            while stack:
                r, c = stack.pop()
                cells.add((r, c))
                for dr, dc in _ORTH:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr][nc] and grid[nr][nc] == value:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            found.append(
                {
                    "value": int(value),
                    "cells": cells,
                    "top_left": (min(r for r, _ in cells), min(c for _, c in cells)),
                    "hash": _shape_hash(cells, value),
                }
            )
    return found


def _unique_positions(nodes: list[dict[str, Any]]) -> tuple[dict[str, tuple[int, int]], dict[str, int]]:
    counts: dict[str, int] = {}
    positions: dict[str, tuple[int, int]] = {}
    for node in nodes:
        counts[node["hash"]] = counts.get(node["hash"], 0) + 1
        positions[node["hash"]] = node["top_left"]
    return {h: pos for h, pos in positions.items() if counts[h] == 1}, counts


def _mostly_changed(node: dict[str, Any], changed: set[tuple[int, int]]) -> bool:
    return len(node["cells"] & changed) * 2 >= len(node["cells"])


def effect(transition: dict[str, Any], mask: set[tuple[int, int]], color_chars: str = "") -> dict[str, Any]:
    """What one action did outside the border mask."""
    before, after = transition["before"], transition["after"]
    changed = changed_cell_set(before, after) - mask
    max_pixels = len(before) * (len(before[0]) if before else 0) // 8 or 1
    before_nodes = [n for n in components(before) if not (n["cells"] & mask) and len(n["cells"]) <= max_pixels]
    after_nodes = [n for n in components(after) if not (n["cells"] & mask) and len(n["cells"]) <= max_pixels]
    before_unique, before_counts = _unique_positions(before_nodes)
    after_unique, after_counts = _unique_positions(after_nodes)

    def color(value: int) -> str:
        return color_chars[value] if 0 <= value < len(color_chars) else str(value)

    colors = {n["hash"]: color(n["value"]) for n in [*after_nodes, *before_nodes]}
    labels_before = {n["hash"]: object_label(color(n["value"]), n["cells"], n["top_left"]) for n in before_nodes}
    labels_after = {n["hash"]: object_label(color(n["value"]), n["cells"], n["top_left"]) for n in after_nodes}
    moves: dict[str, tuple[int, int]] = {}
    for object_hash, (r0, c0) in before_unique.items():
        if object_hash in after_unique:
            r1, c1 = after_unique[object_hash]
            if (r1, c1) != (r0, c0):
                moves[object_hash] = (r1 - r0, c1 - c0)
    appeared = sorted(
        {
            n["hash"]
            for n in after_nodes
            if after_counts[n["hash"]] > before_counts.get(n["hash"], 0) and _mostly_changed(n, changed)
        }
    )
    disappeared = sorted(
        {
            n["hash"]
            for n in before_nodes
            if before_counts[n["hash"]] > after_counts.get(n["hash"], 0) and _mostly_changed(n, changed)
        }
    )
    result = transition.get("result") or {}
    return {
        "board_changed": bool(changed),
        "changed_cells": len(changed),
        "moves": moves,
        "present": set(before_unique),
        "appeared": appeared,
        "disappeared": disappeared,
        "level_completed": bool(result.get("level_completed")),
        "game_over": bool(result.get("game_over")),
        "colors": colors,
        "labels_before": labels_before,
        "labels_after": labels_after,
    }


def is_toggle_back(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """The action exactly undid what it did last time (a toggle switching back)."""
    if not (previous["appeared"] or previous["disappeared"] or previous["moves"]):
        return False
    if set(current["appeared"]) != set(previous["disappeared"]):
        return False
    if set(current["disappeared"]) != set(previous["appeared"]):
        return False
    return current["moves"] == {h: (-d[0], -d[1]) for h, d in previous["moves"].items()}


def _obj(object_hash: str, colors: dict[str, str], *labels: dict[str, str]) -> str:
    """The first label found for the object (where it is in that frame), else its colour."""
    for found in labels:
        if object_hash in found:
            return found[object_hash]
    color = colors.get(object_hash)
    name = COLOR_NAMES.get(color or "")
    return f"{name} ({color}) object" if name else (f"colour {color} object" if color else "an object")


def _jitter(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def _delta(value: tuple[int, int]) -> str:
    return f"[{value[0]},{value[1]}]"


def surprises(action: str, previous: dict[str, Any], current: dict[str, Any]) -> list[tuple[tuple[str, str], str]]:
    """(kind, text) pairs; kind = (category, object) so a call shows one line per kind."""
    colors = {**previous["colors"], **current["colors"]}
    before = current.get("labels_before", {})
    after = current.get("labels_after", {})
    found: list[tuple[tuple[str, str], str]] = []
    # A level-completing action: the tool result already says so, and the
    # board after it is the next level's, so there is nothing to compare.
    if current["level_completed"]:
        return found
    if current["game_over"] and not previous["game_over"]:
        found.append((("game_over", ""), "it caused game_over this time, not last time"))
    if current["game_over"]:
        return found
    if is_toggle_back(previous, current):
        return found
    if previous["board_changed"] != current["board_changed"]:
        if previous["board_changed"]:
            found.append((("blocked", ""), "it changed the board last time, nothing this time"))
        else:
            found.append((("unblocked", ""), f"it changed nothing last time, {current['changed_cells']} cells this time"))
    for object_hash, delta in previous["moves"].items():
        if object_hash not in current["present"]:
            continue
        observed = current["moves"].get(object_hash, (0, 0))
        if observed != (0, 0) and _jitter(delta, observed):
            continue  # moved both times, one cell apart: animation jitter
        if observed != delta:
            found.append(
                (
                    ("moved_differently", object_hash),
                    f"{_obj(object_hash, colors, before, after)} moved by {_delta(delta)} last time, by {_delta(observed)} this time",
                )
            )
    for object_hash, observed in current["moves"].items():
        if object_hash in previous["moves"] or object_hash not in previous["present"]:
            continue
        found.append(
            (("moved_newly", object_hash), f"{_obj(object_hash, colors, before, after)} moved by {_delta(observed)} this time; it stayed last time")
        )
    for key, verb, where in (("appeared", "appeared", (after, before)), ("disappeared", "disappeared", (before, after))):
        new = [h for h in current[key] if h not in previous[key]]
        if new:
            found.append(
                ((key, ",".join(new[:2])), f"{', '.join(_obj(h, colors, *where) for h in new[:2])} {verb} this time, not last time")
            )
    return found


def level_transitions(history: list[Any], level: int | None) -> list[dict[str, Any]]:
    """Int-grid transitions whose before-frame is on ``level``, RESETs excluded,
    from runtime ``HistoryEntry`` objects."""
    items: list[dict[str, Any]] = []
    for index in range(1, len(history)):
        before = history[index - 1].frame
        after = history[index].frame
        action = str(history[index].action or "")
        if before is None or after is None:
            continue
        if level is not None and before.level != level:
            continue
        if not action or action_key(action).startswith("RESET"):
            continue
        recorded = getattr(history[index], "result", None) or {}
        items.append(
            {
                "action": action,
                "before": before.grid,
                "after": after.grid,
                "step": after.step,
                "result": {
                    "level_completed": bool(recorded.get("level_completed", after.level != before.level)),
                    "game_over": bool(recorded.get("game_over", False)),
                },
            }
        )
    return items


def surprise_lines(
    transitions: list[dict[str, Any]],
    new_steps: list[int],
    *,
    border: int,
    max_lines: int,
    color_chars: str = "",
    effect_cache: dict[int, dict[str, Any]] | None = None,
) -> list[str]:
    """SURPRISE lines for the transitions whose step is in ``new_steps`` (the
    actions of one tool call), first line of each distinct kind, capped.
    ``effect_cache`` (step -> effect) persists effects across calls."""
    if not transitions or not new_steps:
        return []
    height = len(transitions[0]["before"])
    width = len(transitions[0]["before"][0]) if height else 0
    mask = border_mask(height, width, border)
    cache: dict[int, dict[str, Any]] = effect_cache if effect_cache is not None else {}

    def effect_at(index: int) -> dict[str, Any]:
        step = transitions[index]["step"]
        if step not in cache:
            cache[step] = effect(transitions[index], mask, color_chars)
        return cache[step]

    wanted = set(new_steps)
    total = len(new_steps)
    order = {step: position for position, step in enumerate(sorted(new_steps), start=1)}
    seen: set[Any] = set()
    lines: list[str] = []
    for index, item in enumerate(transitions):
        if item["step"] not in wanted:
            continue
        key = action_key(item["action"])
        previous = next((j for j in range(index - 1, -1, -1) if action_key(transitions[j]["action"]) == key), None)
        if previous is None:
            continue
        for kind, text in surprises(item["action"], effect_at(previous), effect_at(index)):
            if kind in seen:
                continue
            seen.add(kind)
            # Always one action against the previous single use of the same
            # action on this level, never a whole batch: say so in the line.
            name = " ".join(str(item["action"]).split())
            batch = f" (action {order[item['step']]} of {total} in this call)" if total > 1 else ""
            lines.append(
                f"SURPRISE: single {name} at step {item['step']}{batch} vs the last single {name} "
                f"at step {transitions[previous]['step']}: {text}"
            )
            if len(lines) >= max_lines:
                return lines
    return lines
