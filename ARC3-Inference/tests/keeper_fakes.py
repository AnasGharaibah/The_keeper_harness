"""Test helpers for the Keeper additions: a fake game that answers the agent's
action calls in the solver's payload format and writes the runtime state after
every action, as HarnessSolver does. No model, no real game engine."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inference.agent.runtime_state import RUNTIME_STATE_FILENAME, Frame, HistoryEntry, write_runtime_state

SIZE = 16
WALL_COL = 11
BAR_ROW = SIZE - 1  # a step bar inside the 4-cell border


def board(col: int, *, step: int, extra: dict[tuple[int, int], int] | None = None) -> tuple[tuple[int, ...], ...]:
    """A 2x2 player (colour 7) at rows 6-7 and column ``col``; a wall (5) at
    WALL_COL; a step bar on the bottom row that shrinks by one cell per step."""
    rows = [[0] * SIZE for _ in range(SIZE)]
    for r in range(SIZE):
        rows[r][WALL_COL] = 5
    for r in (6, 7):
        for c in (col, col + 1):
            rows[r][c] = 7
    for c in range(SIZE):
        rows[BAR_ROW][c] = 4 if c < SIZE - step else 2
    for (r, c), value in (extra or {}).items():
        rows[r][c] = value
    return tuple(tuple(row) for row in rows)


class FakeGame:
    """RIGHT moves the player one column until the wall; anything else does nothing.
    ``toggles`` maps a MOUSE cell to a 2x2 block that a click switches on and off."""

    def __init__(self, state_path: Path, *, start_col: int = 6, toggles: dict[tuple[int, int], tuple[int, int]] | None = None):
        self.state_path = state_path
        self.col = start_col
        self.level = 1
        self.lit: set[tuple[int, int]] = set()
        self.toggles = toggles or {}
        self.history = [HistoryEntry(action="", frame=Frame(grid=self._grid(0), step=0, level=1))]
        self.sent: list[str] = []
        self.complete_at_col: int | None = None
        self._write()

    def _grid(self, step: int) -> tuple[tuple[int, ...], ...]:
        extra = {}
        for (r, c) in self.lit:
            for dr in range(2):
                for dc in range(2):
                    extra[(r + dr, c + dc)] = 9
        return board(self.col, step=step, extra=extra)

    def _write(self) -> None:
        write_runtime_state(self.state_path, current_frame=self.history[-1].frame, history=self.history)

    def __call__(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments.get("query") == "animation":
            return {"record": None}
        per_action, executed = [], []
        for item in arguments["actions"]:
            name = str(item.get("action"))
            display = name if name != "MOUSE" else f"MOUSE(row={item.get('row')}, col={item.get('col')})"
            self.sent.append(display)
            before = self.history[-1].frame
            step = len(self.history)
            if name == "RIGHT" and self.col + 2 < WALL_COL:
                self.col += 1
            if name == "MOUSE":
                block = self.toggles.get((int(item.get("row")), int(item.get("col"))))
                if block is not None:
                    self.lit ^= {block}
            level_completed = self.complete_at_col is not None and self.col == self.complete_at_col and name == "RIGHT"
            if level_completed:
                self.level += 1
                self.col = 2
                self.complete_at_col = None
            after = Frame(grid=self._grid(step), step=step, level=self.level)
            gameplay = any(
                before.grid[r][c] != after.grid[r][c] for r in range(4, SIZE - 4) for c in range(4, SIZE - 4)
            )
            result = {"action_num": step, "action_display": display, "board_changed": before.grid != after.grid,
                      "gameplay_changed": gameplay, "level_completed": level_completed, "game_over": False,
                      "run_complete": False}
            self.history.append(HistoryEntry(action=display, frame=after, result=dict(result)))
            self._write()
            per_action.append(result)
            executed.append(display)
            if level_completed:
                break
        last = per_action[-1]
        return {
            "executed": True, "action_num": last["action_num"], "level": self.level, "score": self.level - 1,
            "state": "NOT_FINISHED", "valid_actions": ["RIGHT", "UP", "MOUSE"],
            "board_changed": any(p["board_changed"] for p in per_action),
            "gameplay_changed": any(p["gameplay_changed"] for p in per_action),
            "gameplay_changed_per_action": [p["gameplay_changed"] for p in per_action],
            "level_completed": last["level_completed"], "game_over": False, "run_complete": False, "done": False,
            "action_display": last["action_display"], "action_name": last["action_display"],
            "executed_actions": executed, "requested_count": len(arguments["actions"]),
            "executed_count": len(per_action), "stopped_early": len(per_action) < len(arguments["actions"]),
            "per_action": per_action,
        }


def make_agent(tmp_path: Path, **game_kwargs: Any):
    from inference.agent.tool_agent import ToolAgent

    agent = ToolAgent(model="test-model")
    state_path = tmp_path / "artifacts" / f"game_p0_{RUNTIME_STATE_FILENAME}"
    game = FakeGame(state_path, **game_kwargs)
    agent._ensure_session(state_path)
    agent._step_env_callback = game
    agent._current_valid_actions = ["RIGHT", "UP", "MOUSE"]
    return agent, state_path, game


def run_code(agent, state_path: Path, code: str) -> dict[str, Any]:
    return json.loads(agent._run_python_tool(state_path, {"code": code}).content)


KEEPER_ENV = ("KEEPER_SURPRISE", "KEEPER_REVIEW", "KEEPER_PERCEPTION")


def clear_keeper_env(monkeypatch) -> None:
    import os

    for key in list(os.environ):
        if key.startswith("KEEPER_"):
            monkeypatch.delenv(key)
    from inference.agent import keeper_guard, keeper_resources

    keeper_guard.GLOBAL.reset()
    keeper_resources.ResourceLogger.reset()
