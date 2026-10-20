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
    assert lines[0] == "SURPRISE at step 2: RIGHT changed the board last time, this time nothing changed"
    # The shrinking bar alone (border 0 = not ignored) would make the blocked move a change.
    assert not any("nothing changed" in line for line in keeper_auto.surprise_lines(transitions, [2], border=0, max_lines=3))


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
    assert [line.split(":", 1)[0] for line in lines] == ["SURPRISE at step 2, action 1/2"] * 2


# ----- agent integration ----------------------------------------------------------------


def test_switch_off_adds_no_prompt_text_no_output_and_runs_no_keeper_code(tmp_path, monkeypatch):
    from inference.agent.tool_agent import ToolAgent, _build_system_prompt

    monkeypatch.setattr(ToolAgent, "_keeper_observe", lambda *a, **k: (_ for _ in ()).throw(AssertionError("keeper ran")))
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
        "SURPRISE at step 4, action 4/6: RIGHT changed the board last time, this time nothing changed"
    )
    assert len(payload["surprises"]) <= 3
    assert list(payload)[-1] == "surprises"  # last field: kept by middle truncation
