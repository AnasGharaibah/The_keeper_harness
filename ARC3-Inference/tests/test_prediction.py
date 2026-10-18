"""Layer 1: harness-side comparison of a stated expectation vs. the real move."""
from __future__ import annotations

from inference.agent.prediction import (
    compare_expectation,
    grid_components,
    normalize_expectation,
    summarize_predictions,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS
from inference.utils.segmentation import segment_layer

BG = 0  # "W"
BLUE = 9  # "b"
RED = 8  # "R"


def _grid(cells: dict[tuple[int, int], int], size: int = 8) -> list[list[int]]:
    grid = [[BG] * size for _ in range(size)]
    for (row, col), value in cells.items():
        grid[row][col] = value
    return grid


def _block(top: int, left: int, value: int = BLUE) -> dict[tuple[int, int], int]:
    return {(top + dr, left + dc): value for dr in range(2) for dc in range(2)}


BEFORE = _grid(_block(2, 2))
MOVED_RIGHT = _grid(_block(2, 3))
FLAGS_CHANGED = {"board_changed": True, "level_completed": False, "game_over": False}
FLAGS_SAME = {"board_changed": False, "level_completed": False, "game_over": False}


def _blue_node(grid):
    return next(node for node in grid_components(grid) if node["color"] == "b")


def test_components_match_segmentation_ids_and_hashes():
    ours = grid_components(MOVED_RIGHT)
    theirs = segment_layer(MOVED_RIGHT, ARC_COLOR_CHARS)["nodes"]
    assert [(n["id"], n["hash"], n["color"]) for n in ours] == [
        (n["id"], n["hash"], n["color"]) for n in theirs
    ]


def test_no_expectation_is_unchecked():
    report = compare_expectation(None, before_grid=BEFORE, after_grid=MOVED_RIGHT, result=FLAGS_CHANGED)
    assert report["verdict"] == "unchecked"
    assert report["failures"] == []


def test_board_changed_confirmed_and_wrong():
    ok = compare_expectation({"board_changed": True}, before_grid=BEFORE, after_grid=MOVED_RIGHT, result=FLAGS_CHANGED)
    assert ok["verdict"] == "confirmed"
    bad = compare_expectation({"board_changed": True}, before_grid=BEFORE, after_grid=BEFORE, result=FLAGS_SAME)
    assert bad["verdict"] == "wrong"
    assert bad["failures"] == ["board_changed=False, expected True"]


def test_level_completed_and_game_over_flags():
    result = {"board_changed": True, "level_completed": True, "game_over": False}
    report = compare_expectation(
        {"level_completed": False, "game_over": False},
        before_grid=BEFORE,
        after_grid=MOVED_RIGHT,
        result=result,
    )
    assert report["verdict"] == "wrong"
    assert report["failures"] == ["level_completed=True, expected False"]


def test_moves_by_hash_confirmed():
    blue_hash = _blue_node(BEFORE)["hash"]
    report = compare_expectation(
        {"moves": {blue_hash: [0, 1]}}, before_grid=BEFORE, after_grid=MOVED_RIGHT, result=FLAGS_CHANGED
    )
    assert report["verdict"] == "confirmed"
    assert report["checked"] == ["moves"]


def test_moves_by_id_reports_observed_displacement():
    blue_id = _blue_node(BEFORE)["id"]
    report = compare_expectation(
        {"moves": {blue_id: [0, 1]}}, before_grid=BEFORE, after_grid=BEFORE, result=FLAGS_SAME
    )
    assert report["verdict"] == "wrong"
    assert report["failures"] == [f"moves[id {blue_id}]: moved [0,0], expected [0,1]"]


def test_moves_string_id_key_from_json():
    blue_id = _blue_node(BEFORE)["id"]
    report = compare_expectation(
        {"moves": {str(blue_id): [0, 1]}}, before_grid=BEFORE, after_grid=MOVED_RIGHT, result=FLAGS_CHANGED
    )
    assert report["verdict"] == "confirmed"


def test_moves_object_missing_after():
    blue_hash = _blue_node(BEFORE)["hash"]
    after = _grid({})
    report = compare_expectation({"moves": {blue_hash: [0, 1]}}, before_grid=BEFORE, after_grid=after, result=FLAGS_CHANGED)
    assert report["verdict"] == "wrong"
    assert "not found after the move" in report["failures"][0]


def test_moves_unknown_object():
    report = compare_expectation({"moves": {99: [0, 1]}}, before_grid=BEFORE, after_grid=MOVED_RIGHT, result=FLAGS_CHANGED)
    assert report["verdict"] == "wrong"
    assert "no such object" in report["failures"][0]


def test_appears_by_color():
    after = _grid({**_block(2, 2), (6, 6): RED})
    ok = compare_expectation({"appears": ["R"]}, before_grid=BEFORE, after_grid=after, result=FLAGS_CHANGED)
    assert ok["verdict"] == "confirmed"
    bad = compare_expectation({"appears": ["R"]}, before_grid=BEFORE, after_grid=MOVED_RIGHT, result=FLAGS_CHANGED)
    assert bad["verdict"] == "wrong"
    assert bad["failures"][0].startswith("appears R: no new object")


def test_appears_free_text_is_not_checked():
    report = compare_expectation(
        {"appears": ["a door opens"]}, before_grid=BEFORE, after_grid=MOVED_RIGHT, result=FLAGS_CHANGED
    )
    assert report["verdict"] == "unchecked"
    assert any("free text" in item for item in report["skipped"])


def test_disappears_by_hash():
    blue_hash = _blue_node(BEFORE)["hash"]
    gone = _grid({})
    ok = compare_expectation({"disappears": [blue_hash]}, before_grid=BEFORE, after_grid=gone, result=FLAGS_CHANGED)
    assert ok["verdict"] == "confirmed"
    bad = compare_expectation(
        {"disappears": [blue_hash]}, before_grid=BEFORE, after_grid=MOVED_RIGHT, result=FLAGS_CHANGED
    )
    assert bad["verdict"] == "wrong"
    assert "still present" in bad["failures"][0]


def test_cells_all_forms():
    for cells in (
        [[2, 4, "b"]],  # sandbox-normalized list
        {"(2, 4)": "b"},  # str keys from JSON
        {"2,4": "b"},
    ):
        report = compare_expectation({"cells": cells}, before_grid=BEFORE, after_grid=MOVED_RIGHT, result=FLAGS_CHANGED)
        assert report["verdict"] == "confirmed", cells
    bad = compare_expectation({"cells": [[2, 2, "b"]]}, before_grid=BEFORE, after_grid=MOVED_RIGHT, result=FLAGS_CHANGED)
    assert bad["verdict"] == "wrong"
    assert bad["failures"] == ["cells[(2,2)]=W, expected b"]
    outside = compare_expectation({"cells": [[20, 0, "b"]]}, before_grid=BEFORE, after_grid=MOVED_RIGHT, result=FLAGS_CHANGED)
    assert "outside" in outside["failures"][0]


def test_note_is_stored_not_checked():
    report = compare_expectation({"note": "the block slides"}, before_grid=BEFORE, after_grid=MOVED_RIGHT, result=FLAGS_CHANGED)
    assert report["verdict"] == "unchecked"
    assert report["note"] == "the block slides"


def test_object_keys_skipped_when_level_changes():
    result = {"board_changed": True, "level_completed": True}
    report = compare_expectation(
        {"moves": {0: [0, 1]}, "level_completed": True},
        before_grid=BEFORE,
        after_grid=MOVED_RIGHT,
        result=result,
    )
    assert report["verdict"] == "confirmed"
    assert report["checked"] == ["level_completed"]
    assert any("level changed" in item for item in report["skipped"])


def test_bad_values_never_raise():
    normalized, problems = normalize_expectation(
        {"board_changed": "yes", "moves": {"x": "bad"}, "cells": [[1]], "bogus": 1}
    )
    assert normalized == {}
    assert len(problems) == 4
    assert compare_expectation("not a dict", before_grid=BEFORE, after_grid=BEFORE, result={})["verdict"] == "unchecked"


def test_summary_one_line_per_call_and_long_loops():
    wrong = {"action_num": 3, "actions": ["RIGHT"], "verdict": "wrong", "failures": ["board_changed=False, expected True"]}
    ok = {"action_num": 4, "actions": ["UP"], "verdict": "confirmed", "checked": ["moves"]}
    lines = summarize_predictions([wrong, ok])
    assert lines == [
        "#3 RIGHT -> WRONG: board_changed=False, expected True",
        "#4 UP -> confirmed (moves)",
    ]
    many = [dict(ok, action_num=i) for i in range(10)] + [dict(wrong, action_num=20 + i) for i in range(4)]
    summary = summarize_predictions(many, max_wrong_diffs_shown=2)
    assert summary[0] == "14 action() calls: 10 confirmed / 4 wrong / 0 unchecked"
    assert len(summary) == 4 and summary[-1] == "... 2 more wrong"
    unchecked = summarize_predictions([{"verdict": "unchecked", "actions": ["UP"]}] * 3)
    assert unchecked == ["3 action() call(s), none passed expect= (unchecked)"]
