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
VERDICT_RESET = "reset"

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


_HUD_EDGE_BAND = 2
_HUD_MAX_FRACTION = 0.15
_HUD_MAX_THICKNESS = 3


def _regions(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """8-connected groups of cells."""
    remaining = set(cells)
    groups: list[set[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        group = {seed}
        stack = [seed]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    neighbour = (r + dr, c + dc)
                    if neighbour in remaining:
                        remaining.discard(neighbour)
                        group.add(neighbour)
                        stack.append(neighbour)
        groups.append(group)
    return groups


def _hud_shaped(region: set[tuple[int, int]], height: int, width: int) -> bool:
    """Bars, counters and timers are thin (narrow side at most 3 cells) and small."""
    rows = [r for r, _ in region]
    cols = [c for _, c in region]
    thickness = min(max(rows) - min(rows) + 1, max(cols) - min(cols) + 1)
    return thickness <= _HUD_MAX_THICKNESS and len(region) <= _HUD_MAX_FRACTION * height * width


def hud_mask(transitions: list[dict[str, Any]], *, change_fraction: float) -> set[tuple[int, int]]:
    """Timer / step-bar cells, ignored as gameplay.

    Measured over the level's transitions regardless of action, two candidates:
    cells that changed in at least ``change_fraction`` of them; and connected
    regions of ever-changing cells near the board edge that changed (somewhere) in
    at least ``change_fraction`` of them -- a shrinking step bar loses a different
    cell each move, so no single cell qualifies but the bar as a whole does. Only
    HUD-shaped candidates (thin and small) are kept, so a play area that changes on
    every move is never masked."""
    if len(transitions) < _MIN_TRANSITIONS_FOR_HUD or change_fraction <= 0:
        return set()
    per_move = [changed_cell_set(item["before"], item["after"]) for item in transitions]
    counts: dict[tuple[int, int], int] = {}
    for cells in per_move:
        for cell in cells:
            counts[cell] = counts.get(cell, 0) + 1
    threshold = change_fraction * len(transitions)
    board = transitions[0]["before"]
    height = len(board)
    width = len(board[0]) if height else 0

    mask: set[tuple[int, int]] = set()
    frequent = {cell for cell, count in counts.items() if count >= threshold}
    for region in _regions(frequent):
        if _hud_shaped(region, height, width):
            mask |= region
    for region in _regions(set(counts)):
        if not _hud_shaped(region, height, width):
            continue
        near_edge = any(
            r <= _HUD_EDGE_BAND or c <= _HUD_EDGE_BAND or r >= height - 1 - _HUD_EDGE_BAND or c >= width - 1 - _HUD_EDGE_BAND
            for r, c in region
        )
        if near_edge and sum(1 for cells in per_move if cells & region) >= threshold:
            mask |= region
    return mask


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


def is_toggle_back(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """The action exactly undid what it did last time (a toggle switched back):
    what appeared then disappears now and vice versa, and every move is reversed."""
    if not (previous["appeared"] or previous["disappeared"] or previous["moves"]):
        return False
    if set(current["appeared"]) != set(previous["disappeared"]):
        return False
    if set(current["disappeared"]) != set(previous["appeared"]):
        return False
    reversed_moves = {h: (-d[0], -d[1]) for h, d in previous["moves"].items()}
    return current["moves"] == reversed_moves


def _surprises(action: str, previous: dict[str, Any], current: dict[str, Any]) -> list[tuple[tuple[str, str], str]]:
    """(kind, text) pairs; kind = (category, object) so a batch can show one line per distinct kind.
    A toggle switching back is the expected alternation, not a surprise."""
    colors = {**previous["colors"], **current["colors"]}
    found: list[tuple[tuple[str, str], str]] = []
    if not (current["level_completed"] or current["game_over"]) and is_toggle_back(previous, current):
        return found
    for key in ("level_completed", "game_over"):
        if current[key] and not previous[key]:
            found.append(((key, ""), f"SURPRISE: {action} caused {key} (it did not last time)"))
    if current["level_completed"] or current["game_over"]:
        return found
    if previous["board_changed"] != current["board_changed"]:
        if previous["board_changed"]:
            found.append((("blocked", ""), f"SURPRISE: {action} changed the board last time, this time nothing changed"))
        else:
            found.append(
                (("unblocked", ""), f"SURPRISE: {action} did nothing last time, this time {current['changed_cells']} cells changed")
            )
    for object_hash, delta in previous["moves"].items():
        if object_hash not in current["present"]:
            continue
        observed = current["moves"].get(object_hash, (0, 0))
        if observed != delta:
            found.append(
                (
                    ("moved_differently", object_hash),
                    f"SURPRISE: {action} moved {_obj(object_hash, colors)} by {_delta(delta)} last time, "
                    f"this time {_delta(observed)}",
                )
            )
    for object_hash, observed in current["moves"].items():
        if object_hash in previous["moves"] or object_hash not in previous["present"]:
            continue
        found.append(
            (
                ("moved_newly", object_hash),
                f"SURPRISE: {action} moved {_obj(object_hash, colors)} by {_delta(observed)}; last time it stayed",
            )
        )
    for key, verb in (("appeared", "appear"), ("disappeared", "disappear")):
        new = [h for h in current[key] if h not in previous[key]]
        if new:
            found.append(
                (
                    (key, ",".join(new[:2])),
                    f"SURPRISE: {action} made {', '.join(_obj(h, colors) for h in new[:2])} {verb} (not last time)",
                )
            )
    return found


def _full_diff(a: Any, b: Any) -> int:
    if not a or not b or len(a) != len(b):
        return 10**9
    return sum(1 for row_a, row_b in zip(a, b) if row_a != row_b for x, y in zip(row_a, row_b) if x != y)


def detect_resets(transitions: list[dict[str, Any]], mask: set[tuple[int, int]] | None = None) -> set[int]:
    """Indices of transitions that put the board back to the level's start.

    The game exposes no signal for an in-level reset (e.g. a step budget running
    out), so this is a heuristic on the full board (HUD included): after the move
    the board is within a few cells of the level's first board, before it was not,
    and the move changed far more cells than a normal move on this level; and the
    level's HUD (``mask``, e.g. a step bar) is back to its start state after the move
    but was not before -- a refilled step bar. Without a detected HUD nothing is
    labelled RESET: a button or key that restores the board is a real effect of
    that action, and offline replays show it is the common case without a bar."""
    if len(transitions) < 2 or not mask:
        return set()
    start = transitions[0]["before"]
    height = len(start)
    width = len(start[0]) if height else 0
    tolerance = max(12, int(0.01 * height * width))
    changed = [len(changed_cell_set(t["before"], t["after"])) for t in transitions]
    typical = sorted(n for n in changed if n)
    median = typical[len(typical) // 2] if typical else 0
    resets: set[int] = set()
    for index, item in enumerate(transitions):
        result = item.get("result") or {}
        if result.get("level_completed") or result.get("game_over"):
            continue
        if changed[index] < max(10, 2 * median):
            continue
        if not _full_diff(item["after"], start) <= tolerance < _full_diff(item["before"], start):
            continue
        hud_after = sum(1 for r, c in mask if item["after"][r][c] != start[r][c])
        hud_before = sum(1 for r, c in mask if item["before"][r][c] != start[r][c])
        if hud_after <= _HUD_RESET_SLACK < hud_before:
            resets.add(index)
    return resets


def short_effect(item: dict[str, Any]) -> str:
    """Short label for batch summaries: blocked / moved [dr,dc] / N cells changed."""
    if item["level_completed"]:
        return "level completed"
    if item["game_over"]:
        return "game over"
    if not item["board_changed"]:
        return "blocked"
    deltas = sorted(set(item["moves"].values()))
    parts = []
    if len(deltas) == 1:
        count = len(item["moves"])
        parts.append(f"moved {_delta(deltas[0])}" + (f" ({count} objects)" if count > 1 else ""))
    elif deltas:
        parts.append("moved " + ", ".join(_delta(d) for d in deltas[:3]))
    if item["appeared"]:
        parts.append(f"{len(item['appeared'])} appeared")
    if item["disappeared"]:
        parts.append(f"{len(item['disappeared'])} disappeared")
    if not parts:
        parts.append(f"{item['changed_cells']} cells changed")
    return ", ".join(parts)


RESET_LABEL = "RESET (board back to level start)"
_HUD_RESET_SLACK = 2


def auto_predict(
    transitions: list[dict[str, Any]],
    *,
    change_fraction: float,
    new_count: int = 1,
    reset_detection: bool = True,
) -> list[dict[str, Any]]:
    """Auto-prediction verdicts for the last ``new_count`` transitions of a level.

    ``transitions`` items carry ``action``, ``before``/``after`` (int grids),
    ``step`` and ``result``. Each verdict compares the transition with the most
    recent earlier one on the same level that used the same action. A RESET
    transition is labelled as such and is never compared, nor used as the
    reference for later comparisons.
    """
    if not transitions or new_count <= 0:
        return []
    mask = hud_mask(transitions, change_fraction=change_fraction)
    resets = detect_resets(transitions, mask) if reset_detection else set()
    cache: dict[int, dict[str, Any]] = {}

    def effect_at(index: int) -> dict[str, Any]:
        if index not in cache:
            cache[index] = effect(transitions[index], mask)
        return cache[index]

    verdicts: list[dict[str, Any]] = []
    for index in range(max(0, len(transitions) - new_count), len(transitions)):
        action = str(transitions[index]["action"])
        key = action_key(action)
        entry: dict[str, Any] = {
            "source": "auto",
            "step": transitions[index].get("step"),
            "action": action,
        }
        if index in resets:
            entry.update(verdict=VERDICT_RESET, effect=RESET_LABEL, short=RESET_LABEL, blocked=False,
                         lines=[f"{RESET_LABEL} -- not an effect of {action}"], kinds=[("reset", "")])
            verdicts.append(entry)
            continue
        current = effect_at(index)
        entry["effect"] = describe_effect(current)
        entry["short"] = short_effect(current)
        entry["blocked"] = entry["short"] == "blocked"
        previous_index = next(
            (
                j
                for j in range(index - 1, -1, -1)
                if j not in resets and action_key(transitions[j]["action"]) == key
            ),
            None,
        )
        if previous_index is None:
            entry["verdict"] = VERDICT_FIRST_TIME
            entry["lines"] = [f"{action} first time on this level: {entry['effect']}"]
            entry["kinds"] = []
        else:
            surprises = _surprises(action, effect_at(previous_index), current)
            entry["compared_with_step"] = transitions[previous_index].get("step")
            entry["verdict"] = VERDICT_SURPRISE if surprises else VERDICT_AS_BEFORE
            entry["lines"] = [text for _kind, text in surprises]
            entry["kinds"] = [kind for kind, _text in surprises]
        verdicts.append(entry)
    return verdicts


def batch_summary(verdicts: list[dict[str, Any]], *, max_chars: int) -> str:
    """One line for a multi-action call, merging runs of identical effects:
    ``LEFT x15: blocked -> step 48: RESET (...) -> LEFT x3: moved [0,-5] each``."""
    runs: list[list[Any]] = []  # [action, short, count, first_step]
    for entry in verdicts:
        label = str(entry.get("short") or "")
        action = str(entry.get("action") or "")
        if entry.get("verdict") != VERDICT_RESET and runs and runs[-1][0] == action and runs[-1][1] == label:
            runs[-1][2] += 1
            continue
        runs.append([action, label, 1, entry.get("step")])
    parts = []
    for action, label, count, first_step in runs:
        if label == RESET_LABEL:
            parts.append(f"step {first_step}: {label}")
        else:
            each = " each" if count > 1 and label not in ("blocked", "level completed", "game over") else ""
            parts.append(f"{action} x{count}: {label}{each}")
    text = ""
    for index, part in enumerate(parts):
        candidate = part if not text else f"{text} -> {part}"
        if max_chars > 0 and len(candidate) > max_chars and text:
            return f"{text} -> ... (+{len(parts) - index} more)"
        text = candidate
    return text


def _where(entry: dict[str, Any]) -> str:
    batch = entry.get("batch")
    if batch and int(batch[1]) > 1:
        return f"step {entry.get('step')} ({entry.get('action')} {batch[0]}/{batch[1]})"
    return f"step {entry.get('step')} ({entry.get('action')})"


def summarize_auto(
    verdicts: list[dict[str, Any]],
    *,
    max_surprises_shown: int,
    max_perception_shown: int = 0,
) -> list[str]:
    """Tool-result lines: a count line, RESET lines, perception lines, the first
    SURPRISE of each distinct kind (with its step), then first-time effects."""
    if not verdicts:
        return []
    counts = {VERDICT_AS_BEFORE: 0, VERDICT_SURPRISE: 0, VERDICT_FIRST_TIME: 0, VERDICT_RESET: 0}
    for entry in verdicts:
        counts[entry["verdict"]] = counts.get(entry["verdict"], 0) + 1
    head = (
        f"auto: {counts[VERDICT_AS_BEFORE]} as before / {counts[VERDICT_SURPRISE]} surprise / "
        f"{counts[VERDICT_FIRST_TIME]} first time"
    )
    if counts[VERDICT_RESET]:
        head += f" / {counts[VERDICT_RESET]} reset"
    lines = [head + " (vs the last same action on this level; HUD ignored)"]
    lines.extend(
        f"RESET at {_where(entry)}: board back to level start -- not an effect of {entry['action']}"
        for entry in verdicts
        if entry["verdict"] == VERDICT_RESET
    )
    perceived = [
        f"PERCEIVED after {entry['action']} (step {entry.get('step')}): {line}"
        for entry in verdicts
        for line in entry.get("perception") or []
    ]
    shown_perception = max(0, int(max_perception_shown))
    if shown_perception:
        lines.extend(perceived[:shown_perception])
        if len(perceived) > shown_perception:
            lines.append(f"... {len(perceived) - shown_perception} more perception lines")
    seen: set[Any] = set()
    surprise_lines: list[str] = []
    hidden = 0
    for entry in verdicts:
        if entry["verdict"] != VERDICT_SURPRISE:
            continue
        kinds = entry.get("kinds") or [None] * len(entry["lines"])
        for kind, text in zip(kinds, entry["lines"]):
            if kind is not None and kind in seen:
                hidden += 1
                continue
            seen.add(kind)
            if len(surprise_lines) >= max(0, int(max_surprises_shown)):
                hidden += 1
                continue
            body = text[len("SURPRISE: "):] if text.startswith("SURPRISE: ") else text
            surprise_lines.append(f"SURPRISE at {_where(entry)}: {body}")
    lines.extend(surprise_lines)
    if hidden:
        lines.append(f"... {hidden} more surprises (repeats of the kinds above or over the cap)")
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
    "VERDICT_RESET",
    "VERDICT_SURPRISE",
    "batch_summary",
    "detect_resets",
    "short_effect",
    "action_key",
    "auto_predict",
    "describe_effect",
    "effect",
    "effect_signature",
    "grid_transitions_for_level",
    "hud_mask",
    "summarize_auto",
]
