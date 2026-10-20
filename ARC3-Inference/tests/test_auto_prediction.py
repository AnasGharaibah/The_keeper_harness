"""Harness auto-prediction: same action -> same effect, HUD ignored, SURPRISE on difference."""
from __future__ import annotations

from inference.agent.auto_prediction import (
    VERDICT_AS_BEFORE,
    VERDICT_FIRST_TIME,
    VERDICT_SURPRISE,
    action_key,
    auto_predict,
    hud_mask,
    summarize_auto,
)


def _grid(col: int, *, tick: int | None = 0) -> tuple[tuple[int, ...], ...]:
    """8x8 board: a 2x2 blue block at row 2, column ``col``; cell (0, 7) is a timer."""
    rows = [[0] * 8 for _ in range(8)]
    for r in (2, 3):
        for c in (col, col + 1):
            rows[r][c] = 9
    if tick is not None:
        rows[0][7] = 3 + tick % 2
    return tuple(tuple(row) for row in rows)


def _t(action: str, col_before: int, col_after: int, step: int, *, timer: bool = True, **result) -> dict:
    return {
        "action": action,
        "before": _grid(col_before, tick=step - 1 if timer else None),
        "after": _grid(col_after, tick=step if timer else None),
        "step": step,
        "result": result,
    }


def test_first_time_then_as_before():
    transitions = [_t("RIGHT", 0, 1, 1, timer=False)]
    (first,) = auto_predict(transitions, change_fraction=0.8)
    assert first["verdict"] == VERDICT_FIRST_TIME
    assert first["source"] == "auto"
    assert "moved [0,1]" in first["lines"][0]

    transitions.append(_t("RIGHT", 1, 2, 2, timer=False))
    (second,) = auto_predict(transitions, change_fraction=0.8)
    assert second["verdict"] == VERDICT_AS_BEFORE
    assert second["compared_with_step"] == 1
    assert second["lines"] == []


def test_blocked_move_is_a_surprise():
    transitions = [_t("RIGHT", 0, 1, 1), _t("RIGHT", 1, 1, 2)]
    (verdict,) = auto_predict(transitions, change_fraction=0.8)
    assert verdict["verdict"] == VERDICT_SURPRISE
    assert any("by [0,1] last time, this time [0,0]" in line for line in verdict["lines"])


def test_timer_cell_is_detected_as_hud_and_ignored():
    transitions = [_t("UP", 0, 0, 1), _t("RIGHT", 0, 1, 2), _t("UP", 1, 1, 3)]
    assert hud_mask(transitions, change_fraction=0.8) == {(0, 7)}
    # Both UPs only tick the timer: with the HUD ignored they "did nothing" both times.
    (verdict,) = auto_predict(transitions, change_fraction=0.8)
    assert verdict["verdict"] == VERDICT_AS_BEFORE
    assert "no board change" in verdict["effect"]


def test_hud_needs_enough_transitions():
    transitions = [_t("UP", 0, 0, 1), _t("UP", 0, 0, 2)]
    assert hud_mask(transitions, change_fraction=0.8) == set()
    # Too early to call the timer HUD, but its flip back is a toggle switching back,
    # which is not a surprise either.
    (verdict,) = auto_predict(transitions, change_fraction=0.8)
    assert verdict["verdict"] == VERDICT_AS_BEFORE


def test_level_completed_is_a_surprise_when_new():
    transitions = [_t("RIGHT", 0, 1, 1), _t("RIGHT", 1, 2, 2, level_completed=True)]
    (verdict,) = auto_predict(transitions, change_fraction=0.8)
    assert verdict["verdict"] == VERDICT_SURPRISE
    assert verdict["lines"] == ["SURPRISE: RIGHT caused level_completed (it did not last time)"]


def test_mouse_clicks_only_match_the_same_cell():
    assert action_key("MOUSE(row=4, col=7)") == action_key("mouse(row=4,  col=7)")
    assert action_key("MOUSE(row=4, col=7)") != action_key("MOUSE(row=4, col=8)")


def test_new_count_covers_a_batch():
    transitions = [_t("RIGHT", 0, 1, 1), _t("RIGHT", 1, 2, 2), _t("RIGHT", 2, 2, 3)]
    verdicts = auto_predict(transitions, change_fraction=0.8, new_count=3)
    assert [v["verdict"] for v in verdicts] == [VERDICT_FIRST_TIME, VERDICT_AS_BEFORE, VERDICT_SURPRISE]
    lines = summarize_auto(verdicts, max_surprises_shown=3)
    assert lines[0].startswith("auto: 1 as before / 1 surprise / 1 first time")
    assert any(line.startswith("SURPRISE at step 3 (RIGHT): RIGHT") for line in lines)
