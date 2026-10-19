"""Harness-side automatic predictions from the level's own history (no LLM).

For each executed action the harness predicts that it has the same effect it had
the last time the same action was taken on this level, and reports a SURPRISE
when the effect differs. Cells that change on almost every step (timers, step
bars, other HUD) are detected from the history and ignored. Everything here is
pure: grids in, plain dicts/strings out.
"""
from __future__ import annotations

from typing import Any

from inference.agent.prediction import grid_components

VERDICT_FIRST_TIME = "first_time"
VERDICT_AS_BEFORE = "as_before"
VERDICT_SURPRISE = "surprise"

_MIN_TRANSITIONS_FOR_HUD = 3
_MAX_LISTED = 4


def action_key(action: str) -> str:
    """Actions count as "the same" when their text matches, so a MOUSE click only
    matches a click on the same cell."""
    return " ".join(str(action or "").upper().split())


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


def hud_mask(transitions: list[dict[str, Any]], *, change_fraction: float) -> set[tuple[int, int]]:
    """Cells that changed in at least ``change_fraction`` of the transitions,
    regardless of action: timers and step bars rather than gameplay."""
    if len(transitions) < _MIN_TRANSITIONS_FOR_HUD or change_fraction <= 0:
        return set()
    counts: dict[tuple[int, int], int] = {}
    for item in transitions:
        for cell in changed_cell_set(item["before"], item["after"]):
            counts[cell] = counts.get(cell, 0) + 1
    threshold = change_fraction * len(transitions)
    return {cell for cell, count in counts.items() if count >= threshold}


def _objects(grid: Any, mask: set[tuple[int, int]]) -> list[dict[str, Any]]:
    """Components that do not touch the HUD mask."""
    return [node for node in grid_components(grid, with_cells=True) if not (mask and node["cells"] & mask)]


def _unique_positions(nodes: list[dict[str, Any]]) -> tuple[dict[str, tuple[int, int]], dict[str, int]]:
    """(top-left of hashes that occur exactly once, count per hash)."""
    counts: dict[str, int] = {}
    positions: dict[str, tuple[int, int]] = {}
    for node in nodes:
        counts[node["hash"]] = counts.get(node["hash"], 0) + 1
        positions[node["hash"]] = node["top_left"]
    return {h: pos for h, pos in positions.items() if counts[h] == 1}, counts


def _mostly_changed(node: dict[str, Any], changed: set[tuple[int, int]]) -> bool:
    """A reshaped background keeps most of its cells; a real (dis)appearance does not."""
    return len(node["cells"] & changed) * 2 >= len(node["cells"])


def effect(transition: dict[str, Any], mask: set[tuple[int, int]]) -> dict[str, Any]:
    """What one action did, outside the HUD mask."""
    before, after = transition["before"], transition["after"]
    changed = changed_cell_set(before, after) - mask
    before_nodes = _objects(before, mask)
    after_nodes = _objects(after, mask)
    before_unique, before_counts = _unique_positions(before_nodes)
    after_unique, after_counts = _unique_positions(after_nodes)
    colors = {node["hash"]: node["color"] for node in [*after_nodes, *before_nodes]}
    moves: dict[str, tuple[int, int]] = {}
    for object_hash, (r0, c0) in before_unique.items():
        if object_hash in after_unique:
            r1, c1 = after_unique[object_hash]
            if (r1, c1) != (r0, c0):
                moves[object_hash] = (r1 - r0, c1 - c0)
    appeared = sorted(
        {
            node["hash"]
            for node in after_nodes
            if after_counts[node["hash"]] > before_counts.get(node["hash"], 0) and _mostly_changed(node, changed)
        }
    )
    disappeared = sorted(
        {
            node["hash"]
            for node in before_nodes
            if before_counts[node["hash"]] > after_counts.get(node["hash"], 0) and _mostly_changed(node, changed)
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
    }


def _obj(object_hash: str, colors: dict[str, str]) -> str:
    color = colors.get(object_hash)
    return f"obj {object_hash}" + (f" ({color})" if color else "")


def _delta(value: tuple[int, int]) -> str:
    return f"[{value[0]},{value[1]}]"


def describe_effect(item: dict[str, Any]) -> str:
    """Compact one-line description, e.g. ``obj 3f.. (O) moved [0,1]; 12 cells changed``."""
    if item["level_completed"]:
        return "level completed"
    if item["game_over"]:
        return "game over"
    if not item["board_changed"]:
        return "no board change (HUD ignored)"
    parts = []
    moves = list(item["moves"].items())
    for object_hash, delta in moves[:_MAX_LISTED]:
        parts.append(f"{_obj(object_hash, item['colors'])} moved {_delta(delta)}")
    if len(moves) > _MAX_LISTED:
        parts.append(f"+{len(moves) - _MAX_LISTED} more moved")
    for label, key in (("appeared", "appeared"), ("disappeared", "disappeared")):
        hashes = item[key]
        if hashes:
            listed = ", ".join(_obj(h, item["colors"]) for h in hashes[:2])
            more = f" +{len(hashes) - 2}" if len(hashes) > 2 else ""
            parts.append(f"{listed}{more} {label}")
    parts.append(f"{item['changed_cells']} cells changed")
    return "; ".join(parts)


def effect_signature(item: dict[str, Any]) -> tuple[Any, ...]:
    """Hashable summary used to merge consecutive identical effects."""
    return (
        item["board_changed"],
        tuple(sorted(item["moves"].items())),
        tuple(item["appeared"]),
        tuple(item["disappeared"]),
        item["level_completed"],
        item["game_over"],
    )


def _surprises(action: str, previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
    colors = {**previous["colors"], **current["colors"]}
    found: list[str] = []
    for key in ("level_completed", "game_over"):
        if current[key] and not previous[key]:
            found.append(f"SURPRISE: {action} caused {key} (it did not last time)")
    if current["level_completed"] or current["game_over"]:
        return found
    if previous["board_changed"] != current["board_changed"]:
        if previous["board_changed"]:
            found.append(f"SURPRISE: {action} changed the board last time, this time nothing changed")
        else:
            found.append(f"SURPRISE: {action} did nothing last time, this time {current['changed_cells']} cells changed")
    for object_hash, delta in previous["moves"].items():
        if object_hash not in current["present"]:
            continue
        observed = current["moves"].get(object_hash, (0, 0))
        if observed != delta:
            found.append(
                f"SURPRISE: {action} moved {_obj(object_hash, colors)} by {_delta(delta)} last time, "
                f"this time {_delta(observed)}"
            )
    for object_hash, observed in current["moves"].items():
        if object_hash in previous["moves"] or object_hash not in previous["present"]:
            continue
        found.append(
            f"SURPRISE: {action} moved {_obj(object_hash, colors)} by {_delta(observed)}; last time it stayed"
        )
    for key, verb in (("appeared", "appear"), ("disappeared", "disappear")):
        new = [h for h in current[key] if h not in previous[key]]
        if new:
            found.append(
                f"SURPRISE: {action} made {', '.join(_obj(h, colors) for h in new[:2])} {verb} (not last time)"
            )
    return found


def auto_predict(
    transitions: list[dict[str, Any]],
    *,
    change_fraction: float,
    new_count: int = 1,
) -> list[dict[str, Any]]:
    """Auto-prediction verdicts for the last ``new_count`` transitions of a level.

    ``transitions`` items carry ``action``, ``before``/``after`` (int grids),
    ``step`` and ``result``. Each verdict compares the transition with the most
    recent earlier one on the same level that used the same action.
    """
    if not transitions or new_count <= 0:
        return []
    mask = hud_mask(transitions, change_fraction=change_fraction)
    cache: dict[int, dict[str, Any]] = {}

    def effect_at(index: int) -> dict[str, Any]:
        if index not in cache:
            cache[index] = effect(transitions[index], mask)
        return cache[index]

    verdicts: list[dict[str, Any]] = []
    for index in range(max(0, len(transitions) - new_count), len(transitions)):
        action = str(transitions[index]["action"])
        key = action_key(action)
        current = effect_at(index)
        previous_index = next(
            (j for j in range(index - 1, -1, -1) if action_key(transitions[j]["action"]) == key),
            None,
        )
        entry: dict[str, Any] = {
            "source": "auto",
            "step": transitions[index].get("step"),
            "action": action,
            "effect": describe_effect(current),
        }
        if previous_index is None:
            entry["verdict"] = VERDICT_FIRST_TIME
            entry["lines"] = [f"{action} first time on this level: {entry['effect']}"]
        else:
            surprises = _surprises(action, effect_at(previous_index), current)
            entry["compared_with_step"] = transitions[previous_index].get("step")
            entry["verdict"] = VERDICT_SURPRISE if surprises else VERDICT_AS_BEFORE
            entry["lines"] = surprises
        verdicts.append(entry)
    return verdicts


def summarize_auto(verdicts: list[dict[str, Any]], *, max_surprises_shown: int) -> list[str]:
    """Tool-result lines: a count line, then SURPRISE lines, then first-time effects."""
    if not verdicts:
        return []
    counts = {VERDICT_AS_BEFORE: 0, VERDICT_SURPRISE: 0, VERDICT_FIRST_TIME: 0}
    for entry in verdicts:
        counts[entry["verdict"]] = counts.get(entry["verdict"], 0) + 1
    lines = [
        f"auto: {counts[VERDICT_AS_BEFORE]} as before / {counts[VERDICT_SURPRISE]} surprise / "
        f"{counts[VERDICT_FIRST_TIME]} first time (vs the last same action on this level; HUD ignored)"
    ]
    surprises = [line for entry in verdicts if entry["verdict"] == VERDICT_SURPRISE for line in entry["lines"]]
    shown = surprises[: max(0, int(max_surprises_shown))]
    lines.extend(shown)
    if len(surprises) > len(shown):
        lines.append(f"... {len(surprises) - len(shown)} more surprises")
    first = [line for entry in verdicts if entry["verdict"] == VERDICT_FIRST_TIME for line in entry["lines"]]
    lines.extend(first[:2])
    return lines


def grid_transitions_for_level(history: list[Any], level: int | None) -> list[dict[str, Any]]:
    """Int-grid transitions whose before-frame is on ``level`` (RESETs excluded),
    from runtime ``HistoryEntry`` objects."""
    items: list[dict[str, Any]] = []
    for index in range(1, len(history)):
        before = history[index - 1].frame
        after = history[index].frame
        action = str(history[index].action or "")
        if level is not None and before.level != level:
            continue
        if not action or action_key(action).startswith("RESET"):
            continue
        items.append({"action": action, "before": before.grid, "after": after.grid, "step": after.step, "result": {}})
    return items


__all__ = [
    "VERDICT_AS_BEFORE",
    "VERDICT_FIRST_TIME",
    "VERDICT_SURPRISE",
    "action_key",
    "auto_predict",
    "describe_effect",
    "effect",
    "effect_signature",
    "grid_transitions_for_level",
    "hud_mask",
    "summarize_auto",
]
