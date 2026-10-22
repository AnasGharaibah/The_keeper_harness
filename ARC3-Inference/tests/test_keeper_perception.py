"""Keeper part 3: perception lines for what frame_diff() does not report."""
from __future__ import annotations

import pytest

from inference.agent.keeper_perception import PerceptionTracker, frame_objects
from keeper_fakes import clear_keeper_env, make_agent, run_code

L_SHAPE = {(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 2)}  # two separate pieces once drawn in a box
TARGET = {(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 2)}   # L_SHAPE rotated 90 degrees clockwise


def _rot_cw(cells):
    return {(c, 2 - r) for r, c in cells}


def _board(bottom, *, player_col=14, single=None):
    """24x24: a 5x5 box (5) at (1,1) holding TARGET at 1x; an 8x8 box at (14,1)
    holding ``bottom`` at 2x; a 2x2 player (7) at row 5; optional single L object."""
    rows = [[0] * 24 for _ in range(24)]
    for r in range(1, 6):
        for c in range(1, 6):
            rows[r][c] = 5
    for r, c in TARGET:
        rows[2 + r][2 + c] = 9
    for r in range(14, 22):
        for c in range(1, 9):
            rows[r][c] = 5
    for r, c in bottom:
        for dr in range(2):
            for dc in range(2):
                rows[15 + 2 * r + dr][2 + 2 * c + dc] = 9
    for dr in range(2):
        for dc in range(2):
            rows[5 + dr][player_col + dc] = 7
    for r, c in single or ():
        rows[10 + r][14 + c] = 3
    return tuple(tuple(row) for row in rows)


def _t(action, before, after, step=1):
    return {"action": action, "before": before, "after": after, "step": step, "result": {}}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    clear_keeper_env(monkeypatch)
    monkeypatch.delenv("MULTIMODAL_CONTEXT", raising=False)


def test_two_piece_pattern_is_one_object():
    patterns = [o for o in frame_objects(_board(L_SHAPE), set(), min_size=3) if o.kind == "pattern"]
    # Both are drawn as two separate pieces; each becomes one pattern object.
    assert {(o.top_left, o.scale) for o in patterns} == {((2, 2), 1), ((15, 2), 2)}


def test_pattern_rotation_and_match_are_reported_once():
    tracker = PerceptionTracker(min_size=3, max_lines=3)
    s0, s1 = _board(L_SHAPE), _board(_rot_cw(L_SHAPE), player_col=15)
    lines = tracker.observe(_t("RIGHT", s0, s1), set(), level=1)
    assert any("now MATCHES" in line and "scale 2:1" in line for line in lines)
    assert any("pattern at (15,2) 6x6 ROTATED 90° CW" in line for line in lines)
    s2 = _board(_rot_cw(_rot_cw(L_SHAPE)), player_col=16)
    lines = tracker.observe(_t("RIGHT", s1, s2, 2), set(), level=1)
    assert any("no longer matches" in line for line in lines)


def test_single_object_changes_are_left_to_frame_diff():
    tracker = PerceptionTracker(min_size=3, max_lines=3)
    single = {(0, 0), (0, 1), (0, 2), (1, 0)}
    lines = tracker.observe(
        _t("SPACE", _board(L_SHAPE, single=single), _board(L_SHAPE, single={(c, 2 - r) for r, c in single})),
        set(), level=1,
    )
    assert not any("ROTATED" in line or "moved" in line or "RECOLORED" in line for line in lines)


def test_switch_off_adds_nothing(tmp_path, monkeypatch):
    from inference.agent.tool_agent import ToolAgent

    monkeypatch.setattr(ToolAgent, "_keeper_after_actions", lambda *a, **k: (_ for _ in ()).throw(AssertionError("keeper ran")))
    agent, state_path, _game = make_agent(tmp_path)
    assert "`perceived`" not in agent._system_prompt
    assert "perceived" not in run_code(agent, state_path, "action(['RIGHT'] * 3)")


def test_switch_on_reports_roles_in_the_tool_result(tmp_path, monkeypatch):
    monkeypatch.setenv("KEEPER_PERCEPTION", "1")
    agent, state_path, _game = make_agent(tmp_path, start_col=4)
    assert "`perceived`" in agent._system_prompt
    payload = run_code(agent, state_path, "action(['RIGHT'] * 4)")
    assert payload["perceived"][0].startswith("ROLES: likely player = obj ") and "moved with arrows 3/3" in payload["perceived"][0]
    assert "surprises" not in payload  # independent switches
