"""Part 5: RESET detection, per-step batch summaries, step-numbered surprises, early batch stop."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inference.agent.auto_prediction import (
    RESET_LABEL,
    VERDICT_RESET,
    auto_predict,
    batch_summary,
    detect_resets,
    summarize_auto,
)
from inference.agent.layer_config import LevelReviewConfig, PredictionCheckConfig
from inference.agent.level_memory import LevelMemory, level_transitions, merged_move_lines
from inference.agent.perception import PerceptionTracker
from inference.agent.runtime_state import RUNTIME_STATE_FILENAME, Frame, HistoryEntry, write_runtime_state
from inference.agent.tool_agent import ToolAgent


def _board(col: int, *, size: int = 12, wall: int = 9, extra: dict[tuple[int, int], int] | None = None):
    """A 2x2 player (7) at rows 2-3, column ``col``; a wall (5) at column ``wall``."""
    rows = [[0] * size for _ in range(size)]
    for r in range(size):
        rows[r][wall] = 5
    for r in (2, 3):
        for c in (col, col + 1):
            rows[r][c] = 7
    for (r, c), value in (extra or {}).items():
        rows[r][c] = value
    return tuple(tuple(row) for row in rows)


def _t(action: str, before, after, step: int, **result) -> dict:
    return {"action": action, "before": before, "after": after, "step": step, "result": result}


def _decorated(painted: int, col: int, *, bar: int = 12):
    """Board with ``painted`` cells of colour 3 (a lot of change) and a step bar on the
    bottom row: ``bar`` cells of colour 4 left out of 12."""
    extra = {(8 + i // 8, i % 8): 3 for i in range(painted)}
    extra.update({(11, c): (4 if c < bar else 1) for c in range(12)})
    return _board(col, extra=extra)


def _walk_then_reset() -> list[dict]:
    """Five RIGHT moves that paint cells and drain the bar, then a jump back to the start."""
    boards = [_decorated(i * 2, i, bar=12 - i) for i in range(6)]
    walk = [_t("RIGHT", boards[i], boards[i + 1], i + 1) for i in range(5)]
    walk.append(_t("RIGHT", boards[5], boards[0], 6))
    return walk


# ----- D: reset detection --------------------------------------------------------


def test_reset_is_a_big_jump_back_to_the_level_start_with_the_step_bar_refilled():
    from inference.agent.auto_prediction import hud_mask

    walk = _walk_then_reset()
    mask = hud_mask(walk, change_fraction=0.8)
    assert {(11, c) for c in range(6, 12)} & mask  # the bar is HUD
    assert detect_resets(walk, mask) == {5}
    verdicts = auto_predict(walk, change_fraction=0.8, new_count=1)
    assert verdicts[0]["verdict"] == VERDICT_RESET and verdicts[0]["short"] == RESET_LABEL
    assert auto_predict(walk, change_fraction=0.8, new_count=1, reset_detection=False)[0]["verdict"] != VERDICT_RESET


def test_no_hud_means_no_reset_label():
    walk = _walk_then_reset()
    assert detect_resets(walk, set()) == set()


def test_toggle_back_with_a_draining_bar_is_not_a_reset():
    from inference.agent.auto_prediction import hud_mask

    boards = []
    for step in range(6):
        lit = step % 2 == 1
        extra = {(6 + i // 4, 4 + i % 4): 6 for i in range(20)} if lit else {}
        extra.update({(11, c): (4 if c < 12 - step else 1) for c in range(12)})
        boards.append(_board(0, extra=extra))
    transitions = [_t("CLICK", boards[i], boards[i + 1], i + 1) for i in range(5)]
    assert detect_resets(transitions, hud_mask(transitions, change_fraction=0.8)) == set()


def test_reset_is_never_the_reference_for_later_comparisons():
    walk = _walk_then_reset()
    walk.append(_t("RIGHT", walk[-1]["after"], _decorated(0, 1, bar=11), 7))
    (last,) = auto_predict(walk, change_fraction=0.8, new_count=1)
    assert last["compared_with_step"] == 5  # skips the RESET at step 6


# ----- A/B: batch summary and surprise lines -------------------------------------


def test_batch_summary_merges_runs_and_caps_length():
    entries = (
        [{"action": "LEFT", "short": "blocked", "verdict": "as_before", "step": 33 + i} for i in range(15)]
        + [{"action": "LEFT", "short": RESET_LABEL, "verdict": VERDICT_RESET, "step": 48}]
        + [{"action": "LEFT", "short": "moved [0,-5]", "verdict": "as_before", "step": 49 + i} for i in range(3)]
        + [{"action": "LEFT", "short": "blocked", "verdict": "as_before", "step": 52}]
    )
    assert batch_summary(entries, max_chars=400) == (
        "LEFT x15: blocked -> step 48: RESET (board back to level start) -> LEFT x3: moved [0,-5] each -> LEFT x1: blocked"
    )
    assert batch_summary(entries, max_chars=40).endswith("(+3 more)")


def test_surprise_lines_carry_step_and_batch_position_and_dedupe_kinds():
    def surprise(step, index, kind):
        return {"verdict": "surprise", "action": "LEFT", "step": step, "batch": (index, 20),
                "lines": [f"SURPRISE: LEFT thing {kind}"], "kinds": [(kind, "")]}

    verdicts = [surprise(40, 8, "a"), surprise(41, 9, "a"), surprise(42, 10, "b")]
    lines = summarize_auto(verdicts, max_surprises_shown=5)
    assert "SURPRISE at step 40 (LEFT 8/20): LEFT thing a" in lines
    assert "SURPRISE at step 42 (LEFT 10/20): LEFT thing b" in lines
    assert not any("step 41" in line for line in lines)
    assert lines[-1].startswith("... 1 more surprises")


# ----- C: early stop through the agent ---------------------------------------------


class BatchEnv:
    """Mimics HarnessSolver.step_env for batches: executes actions one by one, writes the
    runtime state after each, and honours ``continue_check`` like the real solver."""

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.col = 5
        self.history = [HistoryEntry(action="", frame=Frame(grid=_board(self.col), step=0, level=1))]
        self.sent: list[str] = []
        write_runtime_state(state_path, current_frame=self.history[-1].frame, history=self.history)

    def __call__(self, arguments: dict[str, Any]) -> dict[str, Any]:
        actions = [item["action"] for item in arguments["actions"]]
        check = arguments.get("continue_check")
        per_action, stop_reason, detail = [], None, None
        for index, name in enumerate(actions, start=1):
            self.sent.append(name)
            before = self.history[-1].frame.grid
            if name == "RIGHT" and self.col + 2 < 9:
                self.col += 1
            step = len(self.history)
            after = _board(self.col)
            self.history.append(HistoryEntry(action=name, frame=Frame(grid=after, step=step, level=1)))
            write_runtime_state(self.state_path, current_frame=self.history[-1].frame, history=self.history)
            payload = {"action_num": step, "action_display": name, "board_changed": before != after,
                       "level_completed": False, "game_over": False}
            per_action.append(payload)
            if callable(check) and index < len(actions):
                detail = check(payload, index, len(actions))
                if detail:
                    stop_reason = "early_stop"
                    break
        result = {
            "executed": True, "action_num": per_action[-1]["action_num"], "level": 1,
            "board_changed": any(p["board_changed"] for p in per_action), "level_completed": False,
            "game_over": False, "run_complete": False, "valid_actions": ["RIGHT", "UP"],
            "executed_actions": [p["action_display"] for p in per_action], "per_action": per_action,
            "requested_count": len(actions), "executed_count": len(per_action),
            "stopped_early": len(per_action) < len(actions),
        }
        if stop_reason:
            result["stop_reason"], result["stop_detail"] = stop_reason, detail
        return result


def _agent_with_env(tmp_path: Path, prediction: PredictionCheckConfig) -> tuple[ToolAgent, Path, BatchEnv]:
    agent = ToolAgent(model="m", prediction_check=prediction, level_review=LevelReviewConfig())
    state_path = tmp_path / "artifacts" / f"game_p0_{RUNTIME_STATE_FILENAME}"
    env = BatchEnv(state_path)
    agent._ensure_session(state_path)
    agent._step_env_callback = env
    agent._current_valid_actions = ["RIGHT", "UP"]
    return agent, state_path, env


def test_batch_stops_on_surprise_in_a_repeated_batch_and_does_not_send_the_rest(tmp_path):
    agent, state_path, env = _agent_with_env(tmp_path, PredictionCheckConfig(enabled=True))
    assert "batch_stopped" in agent._system_prompt
    payload = json.loads(agent._run_python_tool(state_path, {"code": "action(['RIGHT'] * 8)"}).content)
    # col 5 -> 6 -> 7, then the wall: the 3rd RIGHT is blocked although RIGHT moved last time.
    assert env.sent == ["RIGHT"] * 3
    assert payload["batch_stopped"] == (
        "BATCH STOPPED at step 3 (3/8): RIGHT changed the board last time, this time nothing changed"
        " \u2014 5 actions not executed"
    )
    assert payload["batch_summary"] == "RIGHT x2: moved [0,1] each -> RIGHT x1: blocked"


def test_batch_stops_after_two_blocked_moves(tmp_path):
    config = PredictionCheckConfig(enabled=True, stop_on_surprise_repeated=False)
    agent, state_path, env = _agent_with_env(tmp_path, config)
    payload = json.loads(agent._run_python_tool(state_path, {"code": "action(['RIGHT'] * 8)"}).content)
    assert env.sent == ["RIGHT"] * 4
    assert payload["batch_stopped"] == "BATCH STOPPED at step 4 (4/8): blocked 2 times in a row \u2014 4 actions not executed"


def test_batch_stop_switch_off_runs_everything(tmp_path):
    agent, state_path, env = _agent_with_env(tmp_path, PredictionCheckConfig(enabled=True, batch_stop=False))
    payload = json.loads(agent._run_python_tool(state_path, {"code": "action(['RIGHT'] * 8)"}).content)
    assert env.sent == ["RIGHT"] * 8 and "batch_stopped" not in payload
    assert payload["batch_summary"] == "RIGHT x2: moved [0,1] each -> RIGHT x6: blocked"


def test_all_part5_switches_off_keep_old_payload(tmp_path):
    off = PredictionCheckConfig(enabled=True, batch_stop=False, batch_summary=False, reset_detection=False)
    agent, state_path, env = _agent_with_env(tmp_path, off)
    assert "batch_stopped" not in agent._system_prompt
    payload = json.loads(agent._run_python_tool(state_path, {"code": "action(['RIGHT'] * 4)"}).content)
    assert env.sent == ["RIGHT"] * 4
    assert "batch_summary" not in payload and "batch_stopped" not in payload


# ----- D: resets are invisible to hypothesis checks and perception ---------------


def test_reset_steps_are_excluded_from_hypothesis_transitions_and_labelled_for_the_reviewer():
    history = [HistoryEntry(action="", frame=Frame(grid=_board(0), step=0, level=1))]
    for step in range(1, 4):
        history.append(HistoryEntry(action="RIGHT", frame=Frame(grid=_board(step), step=step, level=1)))
    history.append(HistoryEntry(action="RIGHT", frame=Frame(grid=_board(0), step=4, level=1)))
    memory = LevelMemory(LevelReviewConfig(enabled=True))
    memory.record_transition_results([{"action_num": s, "board_changed": True} for s in range(1, 5)])
    memory.mark_reset(4)
    memory.record_transition_results([{"action_num": 4, "board_changed": True}])  # flag survives a re-record
    steps = [item["step"] for item in level_transitions(history, 1, memory._results_by_step, cap=300)]
    assert steps == [1, 2, 3]
    lines = merged_move_lines(history, 1, memory._results_by_step, hud_change_fraction=0.8, max_lines=60)
    assert lines[-1] == f"step 4 RIGHT: {RESET_LABEL} -- not an effect of RIGHT"


def test_perception_ignores_reset_transitions():
    tracker = PerceptionTracker(min_size=3)
    assert tracker.observe(_t("RIGHT", _board(3), _board(0), 1), set(), level=1, reset=True) == []
