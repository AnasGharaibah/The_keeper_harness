"""Keeper part 1: SURPRISE lines."""
from __future__ import annotations

import pytest

from inference.agent import keeper_auto
from inference.agent.runtime_state import Frame, HistoryEntry
from keeper_fakes import board, clear_keeper_env, make_agent, run_code


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    clear_keeper_env(monkeypatch)
    monkeypatch.delenv("MULTIMODAL_CONTEXT", raising=False)


def _t(action, before, after, step, **result):
    return {"action": action, "before": before, "after": after, "step": step, "result": result}


# ----- pure logic ---------------------------------------------------------------------


def test_blocked_move_is_a_surprise_and_the_step_bar_is_ignored():
    transitions = [_t("RIGHT", board(6, step=0), board(7, step=1), 1), _t("RIGHT", board(7, step=1), board(7, step=2), 2)]
    lines = keeper_auto.surprise_lines(transitions, [2], border=4, max_lines=3)
    assert lines[0] == (
        "SURPRISE: single RIGHT at step 2 vs the last single RIGHT at step 1: "
        "it changed the board last time, nothing this time"
    )
    # The shrinking bar alone (border 0 = not ignored) would make the blocked move a change.
    assert not any("nothing this time" in line for line in keeper_auto.surprise_lines(transitions, [2], border=0, max_lines=3))


def test_same_effect_as_last_time_adds_nothing():
    transitions = [_t("RIGHT", board(6, step=0), board(7, step=1), 1), _t("RIGHT", board(7, step=1), board(8, step=2), 2)]
    assert keeper_auto.surprise_lines(transitions, [2], border=4, max_lines=3) == []


def test_clicks_on_different_cells_are_different_actions_and_toggles_are_not_surprises():
    lit = {(6, 4): 9, (6, 5): 9, (7, 4): 9, (7, 5): 9}
    off, on = board(2, step=0), board(2, step=0, extra=lit)
    transitions = [
        _t("MOUSE(row=6, col=4)", off, on, 1),
        _t("MOUSE(row=6, col=4)", on, off, 2),   # toggles back: expected
        _t("MOUSE(row=9, col=9)", off, off, 3),  # a different cell: first time, nothing to compare
    ]
    assert keeper_auto.surprise_lines(transitions, [2, 3], border=4, max_lines=3) == []


def test_lines_name_both_steps_for_clicks_and_new_movement():
    lit = {(6, 4): 9, (6, 5): 9, (7, 4): 9, (7, 5): 9}
    transitions = [_t("MOUSE(row=6, col=4)", board(2, step=0), board(2, step=1), 1),
                   _t("UP", board(2, step=1), board(2, step=2), 2),
                   _t("MOUSE(row=6, col=4)", board(2, step=2), board(2, step=3, extra=lit), 3)]
    lines = keeper_auto.surprise_lines(transitions, [3], border=4, max_lines=3)
    prefix = "SURPRISE: single MOUSE(row=6, col=4) at step 3 vs the last single MOUSE(row=6, col=4) at step 1: "
    assert all(line.startswith(prefix) for line in lines)
    assert lines[0].endswith("it changed nothing last time, 4 cells this time")
    assert lines[1].endswith("appeared this time, not last time")


def test_resets_are_skipped_and_kinds_are_deduplicated_and_capped():
    history = [HistoryEntry(action="", frame=Frame(grid=board(6, step=0), step=0, level=1))]
    cols = [7, 7, 7]
    for step, col in enumerate(cols, start=1):
        history.append(HistoryEntry(action="RIGHT", frame=Frame(grid=board(col, step=step), step=step, level=1)))
    history.append(HistoryEntry(action="RESET", frame=Frame(grid=board(6, step=0), step=4, level=1)))
    transitions = keeper_auto.level_transitions(history, 1)
    assert [t["action"] for t in transitions] == ["RIGHT", "RIGHT", "RIGHT"]
    lines = keeper_auto.surprise_lines(transitions, [2, 3], border=4, max_lines=3)
    # Step 2 shows its two distinct kinds; step 3 repeats them and adds nothing.
    assert [line.split(": ")[1] for line in lines] == [
        "single RIGHT at step 2 (action 1 of 2 in this call) vs the last single RIGHT at step 1"
    ] * 2


# ----- agent integration ----------------------------------------------------------------


def test_switch_off_adds_no_prompt_text_no_output_and_runs_no_keeper_code(tmp_path, monkeypatch):
    from inference.agent.tool_agent import ToolAgent, _build_system_prompt

    monkeypatch.setattr(ToolAgent, "_keeper_after_actions", lambda *a, **k: (_ for _ in ()).throw(AssertionError("keeper ran")))
    agent, state_path, game = make_agent(tmp_path)
    assert "surprises" not in agent._system_prompt
    assert agent._system_prompt == _build_system_prompt(tool_output_tokens=agent._tool_output_tokens)
    payload = run_code(agent, state_path, "action(['RIGHT'] * 6)")
    assert "surprises" not in payload
    assert game.sent == ["RIGHT"] * 6


def test_switch_on_reports_the_blocked_move_in_the_tool_result(tmp_path, monkeypatch):
    monkeypatch.setenv("KEEPER_SURPRISE", "1")
    agent, state_path, game = make_agent(tmp_path)
    assert "`surprises`" in agent._system_prompt
    payload = run_code(agent, state_path, "action(['RIGHT'] * 6)")
    # Player at column 6 can move to 7, 8, 9; the 4th RIGHT is blocked by the wall.
    assert payload["surprises"][0] == (
        "SURPRISE: single RIGHT at step 4 (action 4 of 6 in this call) vs the last single RIGHT at step 3: "
        "it changed the board last time, nothing this time"
    )
    assert len(payload["surprises"]) <= 3
    assert list(payload)[-1] == "surprises"  # last field: kept by middle truncation


def test_surprise_lines_name_objects_by_colour_size_and_place_not_by_hash():
    import re

    from inference.agent import keeper_auto
    from inference.utils.grid_utils import ARC_COLOR_CHARS

    def grid(col):
        rows = [[0] * 16 for _ in range(16)]
        for r in (6, 7, 8):
            for c in (col, col + 1, col + 2):
                rows[r][c] = 9
        return tuple(tuple(r) for r in rows)

    ts = [
        {"action": "RIGHT", "before": grid(4), "after": grid(5), "step": 1, "result": {}},
        {"action": "RIGHT", "before": grid(5), "after": grid(5), "step": 2, "result": {}},
    ]
    lines = keeper_auto.surprise_lines(ts, [2], border=4, max_lines=3, color_chars=ARC_COLOR_CHARS)
    text = " ".join(lines)
    assert "blue (b) 3x3 object at (6,5)" in text, lines
    assert not re.search(r"\b[0-9a-f]{16}\b", text)


def _block_grid(r0, c0, size=3, value=9, n=20):
    rows = [[0] * n for _ in range(n)]
    for r in range(r0, r0 + size):
        for c in range(c0, c0 + size):
            rows[r][c] = value
    return tuple(tuple(r) for r in rows)


def test_no_surprise_line_for_a_level_completing_action():
    from inference.agent import keeper_auto

    ts = [
        {"action": "RIGHT", "before": _block_grid(8, 5), "after": _block_grid(8, 6), "step": 1, "result": {}},
        {"action": "RIGHT", "before": _block_grid(8, 6), "after": _block_grid(8, 6), "step": 2,
         "result": {"level_completed": True}},
    ]
    assert keeper_auto.surprise_lines(ts, [2], border=4, max_lines=3) == []


def test_one_cell_animation_jitter_is_not_a_surprise_but_a_block_still_is():
    from inference.agent import keeper_auto

    jitter = [
        {"action": "DOWN", "before": _block_grid(5, 6), "after": _block_grid(12, 5), "step": 1, "result": {}},
        {"action": "DOWN", "before": _block_grid(5, 9), "after": _block_grid(11, 9), "step": 2, "result": {}},
    ]
    assert keeper_auto.surprise_lines(jitter, [2], border=4, max_lines=3) == []
    blocked = [
        {"action": "DOWN", "before": _block_grid(5, 6), "after": _block_grid(6, 6), "step": 1, "result": {}},
        {"action": "DOWN", "before": _block_grid(6, 6), "after": _block_grid(6, 6), "step": 2, "result": {}},
    ]
    assert keeper_auto.surprise_lines(blocked, [2], border=4, max_lines=3)
