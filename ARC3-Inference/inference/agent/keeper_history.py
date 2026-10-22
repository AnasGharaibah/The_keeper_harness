"""Keeper's own per-game record of recent transitions.

The Keeper used to reload the whole runtime-state file on every tool call (32 MB
for a long game) and rebuild the level's transitions from it. Instead, it now
takes the history the harness has already loaded for that call (a reference,
never a copy), adds only the frames it has not seen, and keeps them compactly:

- grids are stored as a tuple of ``bytes`` rows (64 bytes per row, values 0-15),
  and consecutive transitions share the grid object they have in common;
- only the current level and the one before it are kept (the previous level is
  still needed for its post-level review), each capped at ``max_per_level``
  transitions, oldest dropped first;
- the SURPRISE effect of each transition is computed once and cached.

Transitions have the same shape ``keeper_auto.level_transitions`` returns, so
SURPRISE, perception and the review use them unchanged.
"""
from __future__ import annotations

from typing import Any, Iterable

from inference.agent import keeper_auto

Grid = tuple  # tuple[bytes, ...]


def pack(grid: Any) -> Grid:
    return tuple(bytes(int(v) & 0xFF for v in row) for row in grid)


class KeeperHistory:
    def __init__(self, *, keep_levels: int = 2, max_per_level: int = 600) -> None:
        self.keep_levels = max(1, int(keep_levels))
        self.max_per_level = max(10, int(max_per_level))
        self.levels: dict[int, list[dict[str, Any]]] = {}
        self.effects: dict[int, dict[str, Any]] = {}  # step -> cached SURPRISE effect
        self.last_step: int | None = None
        self._prev: tuple[Grid, int, int] | None = None  # grid, step, level of the last frame seen

    def sync(self, history: Iterable[Any]) -> list[int]:
        """Add the entries of ``history`` (runtime HistoryEntry objects, oldest
        first) newer than the last one seen; return the steps of the new
        non-RESET transitions."""
        entries = list(history)
        new = []
        for entry in reversed(entries):
            frame = getattr(entry, "frame", None)
            if frame is None:
                continue
            if self.last_step is not None and frame.step <= self.last_step:
                break
            new.append(entry)
        new.reverse()
        added: list[int] = []
        for entry in new:
            frame = entry.frame
            grid = pack(frame.grid)
            if self._prev is not None and grid == self._prev[0]:
                grid = self._prev[0]  # share the object: a no-op costs no memory
            action = str(entry.action or "")
            if self._prev is not None and action and not keeper_auto.action_key(action).startswith("RESET"):
                before, _step, before_level = self._prev
                recorded = getattr(entry, "result", None) or {}
                self.levels.setdefault(before_level, []).append({
                    "action": action,
                    "before": before,
                    "after": grid,
                    "step": frame.step,
                    "result": {
                        "level_completed": bool(recorded.get("level_completed", frame.level != before_level)),
                        "game_over": bool(recorded.get("game_over", False)),
                    },
                })
                added.append(frame.step)
            self._prev = (grid, frame.step, frame.level)
            self.last_step = frame.step
        self._trim()
        return added

    def transitions(self, level: int) -> list[dict[str, Any]]:
        return self.levels.get(level, [])

    def levels_of(self, steps: Iterable[int]) -> list[int]:
        wanted = set(steps)
        return sorted(level for level, items in self.levels.items() if any(t["step"] in wanted for t in items))

    def _trim(self) -> None:
        if not self.levels:
            return
        newest = max(self.levels)
        for level in [lv for lv in self.levels if lv <= newest - self.keep_levels]:
            del self.levels[level]
        for items in self.levels.values():
            if len(items) > self.max_per_level:
                del items[: len(items) - self.max_per_level]
        kept = {t["step"] for items in self.levels.values() for t in items}
        for step in [s for s in self.effects if s not in kept]:
            del self.effects[step]

    def clear(self) -> None:
        self.levels.clear()
        self.effects.clear()
