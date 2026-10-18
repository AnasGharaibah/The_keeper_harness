"""Harness-side checks of the model's stated expectations for an action.

The model may pass ``expect={...}`` to ``action(...)``. After the real action
runs, the harness (never the model) compares that expectation with the actual
before/after boards and result flags, and reports a verdict plus a short diff.
Everything here is pure: grids in, plain dicts/strings out.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from inference.utils.grid_utils import ARC_COLOR_CHARS
from inference.utils.segmentation import _ORTH, _object_hash

VERDICT_CONFIRMED = "confirmed"
VERDICT_WRONG = "wrong"
VERDICT_UNCHECKED = "unchecked"

FLAG_KEYS = ("board_changed", "level_completed", "game_over")
OBJECT_KEYS = ("moves", "appears", "disappears", "cells")
EXPECT_KEYS = (*FLAG_KEYS, *OBJECT_KEYS, "note")

_HASH_RE = re.compile(r"^[0-9a-f]{16}$")
_CELL_KEY_RE = re.compile(r"^\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?$")
_MAX_LINE_CHARS = 240
_MAX_FULL_LINES = 6


def _color_char(value: int) -> str:
    return ARC_COLOR_CHARS[max(0, min(15, int(value)))]


def grid_components(grid: Any) -> list[dict[str, Any]]:
    """4-connected same-value components in the same id order as ``segment_layer``.

    Only what the expectation checks need (id, color, hash, top-left, cells), so
    the expensive containment pass of ``segment_layer`` is skipped. Ids and hashes
    match what the model sees in ``current_frame.segmentation``.
    """
    height = len(grid)
    width = len(grid[0]) if height else 0
    seen = [[False] * width for _ in range(height)]
    components: list[dict[str, Any]] = []
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
            color = _color_char(value)
            components.append(
                {
                    "id": len(components),
                    "color": color,
                    "hash": _object_hash(cells, color),
                    "top_left": (min(r for r, _ in cells), min(c for _, c in cells)),
                    "pixels": len(cells),
                }
            )
    return components


def _object_ref(value: Any) -> tuple[str, Any] | None:
    """Classify an object reference as ("id", int) or ("hash", str)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return ("id", value)
    text = str(value or "").strip()
    if _HASH_RE.match(text):
        return ("hash", text)
    if text.lstrip("-").isdigit():
        return ("id", int(text))
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _parse_cell_color(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 0 <= value <= 15 else None
    text = str(value or "").strip()
    if len(text) == 1 and text in ARC_COLOR_CHARS:
        return ARC_COLOR_CHARS.index(text)
    return None


def normalize_expectation(expect: Any) -> tuple[dict[str, Any], list[str]]:
    """Validate an ``expect`` payload. Returns (normalized, problems).

    Unknown keys and malformed values are dropped and reported as problems; they
    never raise, so a bad expectation can't break a real action.
    """
    if not isinstance(expect, dict):
        return {}, ["expect must be a dict"]
    normalized: dict[str, Any] = {}
    problems: list[str] = []
    for key, value in expect.items():
        key = str(key)
        if key not in EXPECT_KEYS:
            problems.append(f"unknown expect key {key!r} ignored")
            continue
        if key in FLAG_KEYS:
            if isinstance(value, bool):
                normalized[key] = value
            else:
                problems.append(f"{key} must be True/False")
        elif key == "note":
            text = " ".join(str(value or "").split())
            if text:
                normalized["note"] = text[:_MAX_LINE_CHARS]
        elif key == "moves":
            if not isinstance(value, dict):
                problems.append("moves must be {object_id_or_hash: [d_row, d_col]}")
                continue
            moves: list[tuple[tuple[str, Any], tuple[int, int]]] = []
            for ref, delta in value.items():
                parsed_ref = _object_ref(ref)
                try:
                    d_row, d_col = (int(delta[0]), int(delta[1]))
                    if len(delta) != 2:
                        raise ValueError
                except (TypeError, ValueError, IndexError, KeyError):
                    problems.append(f"moves[{ref}] displacement must be [d_row, d_col]")
                    continue
                if parsed_ref is None:
                    problems.append(f"moves key {ref!r} is not an object id or hash")
                    continue
                moves.append((parsed_ref, (d_row, d_col)))
            if moves:
                normalized["moves"] = moves
        elif key == "appears":
            items: list[tuple[str, str]] = []
            for item in _as_list(value):
                text = str(item or "").strip()
                if not text:
                    continue
                if len(text) == 1 and text in ARC_COLOR_CHARS:
                    items.append(("color", text))
                elif _HASH_RE.match(text):
                    items.append(("hash", text))
                else:
                    items.append(("text", text[:80]))
            if items:
                normalized["appears"] = items
        elif key == "disappears":
            refs = []
            for item in _as_list(value):
                parsed_ref = _object_ref(item)
                if parsed_ref is None:
                    problems.append(f"disappears item {item!r} is not an object id or hash")
                    continue
                refs.append(parsed_ref)
            if refs:
                normalized["disappears"] = refs
        elif key == "cells":
            cells: list[tuple[int, int, int]] = []
            raw_items: Iterable[Any]
            if isinstance(value, dict):
                raw_items = [[cell_key, color] for cell_key, color in value.items()]
            else:
                raw_items = _as_list(value)
            for item in raw_items:
                row = col = color = None
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    row, col, color = item
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    cell_key, color = item
                    if isinstance(cell_key, (list, tuple)) and len(cell_key) == 2:
                        row, col = cell_key
                    else:
                        match = _CELL_KEY_RE.match(str(cell_key))
                        if match:
                            row, col = match.group(1), match.group(2)
                parsed_color = _parse_cell_color(color)
                try:
                    parsed_row, parsed_col = int(row), int(col)
                except (TypeError, ValueError):
                    parsed_row = parsed_col = None
                if parsed_row is None or parsed_color is None:
                    problems.append(f"cells item {item!r} must be (row, col): color")
                    continue
                cells.append((parsed_row, parsed_col, parsed_color))
            if cells:
                normalized["cells"] = cells
    return normalized, problems


def _ref_label(ref: tuple[str, Any]) -> str:
    kind, value = ref
    return f"id {value}" if kind == "id" else str(value)


def _resolve_hash(ref: tuple[str, Any], before: list[dict[str, Any]]) -> tuple[str | None, list[tuple[int, int]]]:
    """Return (hash, before top-lefts of the referenced instance(s))."""
    kind, value = ref
    if kind == "id":
        if 0 <= int(value) < len(before):
            node = before[int(value)]
            return node["hash"], [node["top_left"]]
        return None, []
    positions = [node["top_left"] for node in before if node["hash"] == value]
    return (value if positions else None), positions


def _hash_count(components: list[dict[str, Any]], object_hash: str) -> int:
    return sum(1 for node in components if node["hash"] == object_hash)


def compare_expectation(
    expect: Any,
    *,
    before_grid: Any,
    after_grid: Any,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Compare an expectation against what actually happened.

    ``result`` holds the real action flags (``board_changed``, ``level_completed``,
    ``game_over``, ``run_complete``). Returns a dict with ``verdict``
    (confirmed/wrong/unchecked), ``failures`` (concrete expected-vs-observed
    strings), ``checked`` (keys actually compared), ``skipped`` and ``note``.
    """
    normalized, problems = normalize_expectation(expect) if expect is not None else ({}, [])
    failures: list[str] = []
    checked: list[str] = []
    skipped: list[str] = list(problems)

    for key in FLAG_KEYS:
        if key not in normalized:
            continue
        observed = bool(result.get(key))
        checked.append(key)
        if observed != normalized[key]:
            failures.append(f"{key}={observed}, expected {normalized[key]}")

    object_keys = [key for key in OBJECT_KEYS if key in normalized]
    scene_changed = bool(result.get("level_completed") or result.get("run_complete"))
    if object_keys and scene_changed:
        skipped.append(f"{'/'.join(object_keys)} not checked: the level changed")
        object_keys = []
    if object_keys and (not before_grid or not after_grid):
        skipped.append(f"{'/'.join(object_keys)} not checked: board unavailable")
        object_keys = []

    before: list[dict[str, Any]] = []
    after: list[dict[str, Any]] = []
    if any(key in object_keys for key in ("moves", "appears", "disappears")):
        before = grid_components(before_grid)
        after = grid_components(after_grid)

    if "moves" in object_keys:
        checked.append("moves")
        for ref, (d_row, d_col) in normalized["moves"]:
            label = _ref_label(ref)
            object_hash, positions = _resolve_hash(ref, before)
            if object_hash is None:
                failures.append(f"moves[{label}]: no such object before the move")
                continue
            after_positions = {node["top_left"] for node in after if node["hash"] == object_hash}
            expected_positions = {(r + d_row, c + d_col) for r, c in positions}
            if expected_positions <= after_positions:
                continue
            if not after_positions:
                failures.append(
                    f"moves[{label}]: object not found after the move (removed or shape/color changed), "
                    f"expected [{d_row},{d_col}]"
                )
            elif len(positions) == 1:
                r0, c0 = positions[0]
                nearest = min(after_positions, key=lambda pos: (abs(pos[0] - r0) + abs(pos[1] - c0), pos))
                failures.append(
                    f"moves[{label}]: moved [{nearest[0] - r0},{nearest[1] - c0}], expected [{d_row},{d_col}]"
                )
            else:
                matched = len(expected_positions & after_positions)
                failures.append(
                    f"moves[{label}]: {matched}/{len(positions)} instances moved [{d_row},{d_col}]"
                )

    if "appears" in object_keys:
        for kind, value in normalized["appears"]:
            if kind == "text":
                skipped.append(f"appears {value!r}: free text, not machine-checked")
                continue
            if "appears" not in checked:
                checked.append("appears")
            if kind == "color":
                before_count = sum(1 for node in before if node["color"] == value)
                after_count = sum(1 for node in after if node["color"] == value)
            else:
                before_count = _hash_count(before, value)
                after_count = _hash_count(after, value)
            if after_count <= before_count:
                failures.append(
                    f"appears {value}: no new object ({before_count} before, {after_count} after)"
                )

    if "disappears" in object_keys:
        checked.append("disappears")
        for ref in normalized["disappears"]:
            label = _ref_label(ref)
            object_hash, _positions = _resolve_hash(ref, before)
            if object_hash is None:
                failures.append(f"disappears[{label}]: no such object before the move")
                continue
            before_count = _hash_count(before, object_hash)
            after_count = _hash_count(after, object_hash)
            if after_count >= before_count:
                failures.append(
                    f"disappears[{label}]: still present ({before_count} before, {after_count} after)"
                )

    if "cells" in object_keys:
        checked.append("cells")
        height = len(after_grid)
        width = len(after_grid[0]) if height else 0
        for row, col, color in normalized["cells"]:
            if not (0 <= row < height and 0 <= col < width):
                failures.append(f"cells[({row},{col})]: outside the {height}x{width} board")
                continue
            observed = int(after_grid[row][col])
            if observed != color:
                failures.append(
                    f"cells[({row},{col})]={_color_char(observed)}, expected {_color_char(color)}"
                )

    if not checked:
        verdict = VERDICT_UNCHECKED
    elif failures:
        verdict = VERDICT_WRONG
    else:
        verdict = VERDICT_CONFIRMED
    return {
        "verdict": verdict,
        "failures": failures,
        "checked": checked,
        "skipped": skipped,
        "note": normalized.get("note", ""),
    }


def unchecked_prediction(reason: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "verdict": VERDICT_UNCHECKED,
        "failures": [],
        "checked": [],
        "skipped": [],
        "note": "",
    }
    if reason:
        payload["skipped"] = [reason]
    return payload


def _actions_label(actions: list[str]) -> str:
    names = [str(name).strip() for name in actions if str(name).strip()]
    if not names:
        return "action"
    if len(names) <= 3:
        return ",".join(names)
    return f"{','.join(names[:3])} (+{len(names) - 3})"


def _clip(text: str) -> str:
    if len(text) <= _MAX_LINE_CHARS:
        return text
    return f"{text[:_MAX_LINE_CHARS - 3].rstrip()}..."


def format_prediction_line(entry: dict[str, Any]) -> str:
    """One compact line, e.g. ``#12 RIGHT -> WRONG: moves[id 3]: moved [0,0], expected [0,1]``."""
    verdict = str(entry.get("verdict") or VERDICT_UNCHECKED)
    prefix = f"#{entry['action_num']} " if entry.get("action_num") is not None else ""
    head = f"{prefix}{_actions_label(list(entry.get('actions') or []))}"
    if verdict == VERDICT_WRONG:
        return _clip(f"{head} -> WRONG: {'; '.join(entry.get('failures') or [])}")
    if verdict == VERDICT_CONFIRMED:
        return _clip(f"{head} -> confirmed ({','.join(entry.get('checked') or [])})")
    reason = "; ".join(entry.get("skipped") or []) or "no expect="
    return _clip(f"{head} -> unchecked ({reason})")


def prediction_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts = {VERDICT_CONFIRMED: 0, VERDICT_WRONG: 0, VERDICT_UNCHECKED: 0}
    for entry in entries:
        verdict = str(entry.get("verdict") or VERDICT_UNCHECKED)
        counts[verdict if verdict in counts else VERDICT_UNCHECKED] += 1
    return counts


def summarize_predictions(entries: list[dict[str, Any]], *, max_wrong_diffs_shown: int = 5) -> list[str]:
    """Lines for the tool result's ``prediction_check`` block (one per action() call,
    or a count summary plus the first few wrong diffs for long loops)."""
    if not entries:
        return []
    counts = prediction_counts(entries)
    if counts[VERDICT_UNCHECKED] == len(entries):
        return [f"{len(entries)} action() call(s), none passed expect= (unchecked)"]
    if len(entries) <= _MAX_FULL_LINES:
        return [format_prediction_line(entry) for entry in entries]
    lines = [
        f"{len(entries)} action() calls: {counts[VERDICT_CONFIRMED]} confirmed / "
        f"{counts[VERDICT_WRONG]} wrong / {counts[VERDICT_UNCHECKED]} unchecked"
    ]
    wrong = [entry for entry in entries if entry.get("verdict") == VERDICT_WRONG]
    shown = wrong[: max(0, int(max_wrong_diffs_shown))]
    lines.extend(format_prediction_line(entry) for entry in shown)
    if len(wrong) > len(shown):
        lines.append(f"... {len(wrong) - len(shown)} more wrong")
    return lines
