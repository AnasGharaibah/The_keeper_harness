"""Perception detectors: transformations, shape relations, roles (pure, no LLM)."""
from __future__ import annotations

from inference.agent.perception import (
    PerceptionTracker,
    PObject,
    classify_change,
    frame_objects,
    normalize,
    reduce_scale,
    relation_between,
)

L_SHAPE = {(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 2)}  # bbb / b.. / b.b
TARGET = {(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 2)}  # bbb / ..b / b.b  (L_SHAPE rotated 90 CW)


def _rot_cw(cells: set[tuple[int, int]], size: int = 3) -> set[tuple[int, int]]:
    return {(c, size - 1 - r) for r, c in cells}


def _board(
    bottom: set[tuple[int, int]],
    *,
    player_col: int = 14,
    top: set[tuple[int, int]] = TARGET,
    extra: dict[tuple[int, int], int] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """24x24 board: a 5x5 box (color 5) at (1,1) holding ``top`` at 1x; an 8x8 box at
    (14,1) holding ``bottom`` at 2x; a 2x2 player (color 7) at row 5."""
    rows = [[0] * 24 for _ in range(24)]
    for r in range(1, 6):
        for c in range(1, 6):
            rows[r][c] = 5
    for r, c in top:
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
    for (r, c), value in (extra or {}).items():
        rows[r][c] = value
    return tuple(tuple(row) for row in rows)


def _t(action: str, before, after, step: int = 1) -> dict:
    return {"action": action, "before": before, "after": after, "step": step, "result": {}}


# ----- shape math --------------------------------------------------------------


def test_reduce_scale_and_relations():
    big = normalize({(2 * r + dr, 2 * c + dc) for r, c in L_SHAPE for dr in range(2) for dc in range(2)})
    reduced, k = reduce_scale(big)
    assert k == 2 and reduced == normalize(L_SHAPE)
    assert relation_between(normalize(L_SHAPE), normalize(TARGET)) == "rotated 90° CW"
    assert relation_between(normalize(TARGET), normalize(TARGET)) == "exactly"
    assert relation_between(normalize(L_SHAPE), normalize({(0, 0), (1, 1), (2, 2)})) is None


def test_classify_change_fixed_set():
    def obj(cells, color="b"):
        return PObject(color=color, cells=frozenset(cells), kind="object")

    base = {(0, 0), (0, 1), (0, 2), (1, 0)}  # an L of 4 cells
    assert classify_change(obj(base), obj({(r + 3, c + 1) for r, c in base})) == "TRANSLATED"
    assert classify_change(obj(base), obj(_rot_cw(base))) == "ROTATED 90° CW"
    assert classify_change(obj(base), obj({(r, 2 - c) for r, c in base})) == "FLIPPED horizontally"
    assert classify_change(obj(base), obj(base, color="Y")) == "RECOLORED b->Y"
    doubled = {(2 * r + dr, 2 * c + dc) for r, c in base for dr in range(2) for dc in range(2)}
    assert classify_change(obj(base), obj(doubled)) == "SCALED x2"
    assert classify_change(obj(base), obj({(0, 0), (1, 1), (2, 2), (3, 3)})) is None


def test_pattern_inside_a_box_is_one_object():
    objects = frame_objects(_board(L_SHAPE), set(), min_size=3)
    patterns = [o for o in objects if o.kind == "pattern"]
    assert {(o.top_left, o.scale) for o in patterns} == {((2, 2), 1), ((15, 2), 2)}


def test_min_size_and_hud_are_ignored():
    board = _board(L_SHAPE, extra={(10, 20): 3, (10, 21): 3})  # a 2-cell blip
    objects = frame_objects(board, set(), min_size=3)
    assert not any(o.color == "g" or o.pixels == 2 for o in objects)
    masked = frame_objects(board, {(15, 2)}, min_size=3)
    assert not any(o.kind == "pattern" and o.top_left == (15, 2) for o in masked)


# ----- tracker: transformations + relations --------------------------------------


def test_rotation_then_match_then_unmatch_reported_once_each():
    tracker = PerceptionTracker(min_size=3, max_lines=3)
    s0 = _board(L_SHAPE, player_col=14)
    s1 = _board(_rot_cw(L_SHAPE), player_col=15)  # now equals TARGET
    s2 = _board(_rot_cw(_rot_cw(L_SHAPE)), player_col=16)
    s3 = _board(_rot_cw(_rot_cw(L_SHAPE)), player_col=17)  # nothing transforms
    lines1 = tracker.observe(_t("RIGHT", s0, s1), set(), level=1)
    assert lines1[0] == (
        "b pattern at (15,2) 6x6 now MATCHES b pattern at (2,2) 3x3 exactly "
        "(same shape and orientation, scale 2:1)"
    )
    assert any("ROTATED 90° CW" in line for line in lines1)
    lines2 = tracker.observe(_t("RIGHT", s1, s2, 2), set(), level=1)
    assert any("no longer matches" in line and "rotated 90° CW relative to it" in line for line in lines2)
    assert not any("MATCHES" in line for line in tracker.observe(_t("RIGHT", s2, s3, 3), set(), level=1))


def test_relation_not_reported_when_nothing_transformed():
    tracker = PerceptionTracker(min_size=3, max_lines=3)
    before = _board(TARGET, player_col=14)
    after = _board(TARGET, player_col=15)
    assert tracker.observe(_t("RIGHT", before, after), set(), level=1) == []


def test_max_lines_cap():
    tracker = PerceptionTracker(min_size=3, max_lines=1)
    lines = tracker.observe(_t("RIGHT", _board(L_SHAPE), _board(_rot_cw(L_SHAPE), player_col=15)), set(), level=1)
    assert len(lines) == 1 and "MATCHES" in lines[0]  # relations have priority


def test_several_recolors_collapse_into_one_line():
    before = _board(TARGET, extra={(10, 12): 4, (10, 13): 4, (11, 12): 4, (12, 16): 6, (12, 17): 6, (13, 16): 6})
    after = _board(TARGET, extra={(10, 12): 6, (10, 13): 6, (11, 12): 6, (12, 16): 4, (12, 17): 4, (13, 16): 4})
    lines = PerceptionTracker(min_size=3).observe(_t("CLICK", before, after), set(), level=1)
    # One line, and no spurious MATCHES / no longer matches from the colour swap.
    assert len(lines) == 1 and lines[0].startswith("2 objects RECOLORED: (10,12)")


# ----- roles -------------------------------------------------------------------------


def test_roles_player_and_not_player_reported_once_and_on_change():
    tracker = PerceptionTracker(min_size=3, max_lines=3)
    role_lines = []
    for step in range(1, 8):
        before = _board(L_SHAPE, player_col=8 + step)
        after = _board(L_SHAPE, player_col=9 + step)
        role_lines += [line for line in tracker.observe(_t("RIGHT", before, after, step), set(), level=1) if line.startswith("ROLES")]
    assert role_lines, "player should be detected"
    assert role_lines[0].startswith("ROLES: likely player = obj ") and "(P) moved with arrows 3/3" in role_lines[0]
    assert len(role_lines) <= 2  # once, plus once when the not-player list fills in
    assert "NOT the player" in role_lines[-1] and "b pattern at (2,2) 3x3" in role_lines[-1]
    # Same assignment on later moves: nothing new to report.
    later = tracker.observe(_t("RIGHT", _board(L_SHAPE, player_col=16), _board(L_SHAPE, player_col=17), 8), set(), level=1)
    assert not any(line.startswith("ROLES") for line in later)


def test_level_change_resets_tracker():
    tracker = PerceptionTracker(min_size=3)
    for step in range(1, 5):
        tracker.observe(_t("RIGHT", _board(L_SHAPE, player_col=8 + step), _board(L_SHAPE, player_col=9 + step), step), set(), level=1)
    assert tracker._movers
    tracker.observe(_t("RIGHT", _board(L_SHAPE), _board(L_SHAPE, player_col=15)), set(), level=2)
    assert tracker.level == 2 and sum(s.consistent for s in tracker._movers.values()) == 1


# ----- integration: tool result + reviewer move summaries -------------------------


def test_perception_lines_in_auto_summary_and_review_moves():
    from inference.agent.auto_prediction import summarize_auto
    from inference.agent.level_memory import merged_move_lines
    from inference.agent.runtime_state import Frame, HistoryEntry

    entry = {"verdict": "as_before", "lines": [], "action": "RIGHT", "step": 4, "perception": ["b pattern ROTATED 90° CW"]}
    lines = summarize_auto([entry], max_surprises_shown=3, max_perception_shown=6)
    assert "PERCEIVED after RIGHT (step 4): b pattern ROTATED 90° CW" in lines
    assert summarize_auto([entry], max_surprises_shown=3)[1:] == []  # perception off -> nothing extra

    boards = [_board(L_SHAPE, player_col=14), _board(_rot_cw(L_SHAPE), player_col=15)]
    history = [HistoryEntry(action="", frame=Frame(grid=boards[0], step=0, level=1)),
               HistoryEntry(action="RIGHT", frame=Frame(grid=boards[1], step=1, level=1))]
    with_perception = merged_move_lines(history, 1, {}, hud_change_fraction=0.8, max_lines=60, perception_max_lines=3)
    assert "PERCEIVED: b pattern at (15,2) 6x6 now MATCHES" in with_perception[0]
    without = merged_move_lines(history, 1, {}, hud_change_fraction=0.8, max_lines=60)
    assert "PERCEIVED" not in without[0]


def test_perception_switch_controls_prompt_and_tracker():
    from inference.agent.layer_config import LevelReviewConfig, PredictionCheckConfig
    from inference.agent.tool_agent import ToolAgent

    off = ToolAgent(model="m", prediction_check=PredictionCheckConfig(enabled=True, perception=False), level_review=LevelReviewConfig())
    on = ToolAgent(model="m", prediction_check=PredictionCheckConfig(enabled=True), level_review=LevelReviewConfig())
    assert off._perception is None and "ROTATED / FLIPPED" not in off._system_prompt
    assert on._perception is not None and "ROTATED / FLIPPED" in on._system_prompt
    no_auto = ToolAgent(model="m", prediction_check=PredictionCheckConfig(enabled=True, auto_predict=False), level_review=LevelReviewConfig())
    assert no_auto._perception is None and "ROTATED / FLIPPED" not in no_auto._system_prompt
