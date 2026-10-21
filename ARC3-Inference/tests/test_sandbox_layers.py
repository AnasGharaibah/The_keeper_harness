"""Sandbox bootstrap changes for the layers, exercised through the real subprocess."""
from __future__ import annotations

from typing import Any

from inference.agent.python_tool_sandbox import run_sandboxed_check, run_sandboxed_python
from inference.utils.grid_utils import format_grid_ascii


def _grid(cells: dict[tuple[int, int], int], size: int = 6) -> list[list[int]]:
    grid = [[0] * size for _ in range(size)]
    for (row, col), value in cells.items():
        grid[row][col] = value
    return grid


def _frame_payload(grid: list[list[int]], *, step: int, level: int = 1) -> dict[str, Any]:
    return {
        "ascii": format_grid_ascii(grid),
        "step": step,
        "level": level,
        "shape": [len(grid), len(grid[0])],
        "grid": grid,
    }


def _state(**extra: Any) -> dict[str, Any]:
    grid = _grid({(1, 1): 9})
    frame = _frame_payload(grid, step=0)
    return {
        "current_frame": frame,
        "history": [{"action": "", "frame": frame}],
        "valid_actions": ["UP", "MOUSE"],
        "last_action_result": {},
        **extra,
    }


def _rows(grid: list[list[int]]) -> list[str]:
    return format_grid_ascii(grid).split("\n")


def _transition(action: str, before: list[list[int]], after: list[list[int]], step: int, **result: Any) -> dict[str, Any]:
    return {
        "action": action,
        "before": _rows(before),
        "after": _rows(after),
        "before_step": step - 1,
        "before_level": 1,
        "step": step,
        "level": 1,
        "result": {"board_changed": before != after, **result},
    }


# ----- backward compatibility + expect passthrough ---------------------------


def test_action_old_signature_unchanged_and_handler_called_without_expect():
    calls: list[list[dict[str, Any]]] = []

    def one_arg_handler(actions):  # the pre-layer handler signature
        calls.append(actions)
        return {"action_result": {"executed": True, "board_changed": False}, "state": _state()}

    result = run_sandboxed_python(
        code="r1 = action(['UP'])\nr2 = action('UP')\nr3 = action([{'action': 'MOUSE', 'row': 1, 'col': 2}])\nresult = [r1['executed'], r2['executed'], r3['executed']]",
        timeout_seconds=10,
        initial_state=_state(),
        action_handler=one_arg_handler,
    )
    assert result["error"] == ""
    assert result["result"] == [True, True, True]
    assert calls == [
        [{"action": "UP"}],
        [{"action": "UP"}],
        [{"action": "MOUSE", "row": 1, "col": 2}],
    ]
    assert result["hypotheses"] == []


def test_action_row_col_keywords_and_expect_forwarded():
    seen: list[tuple[list[dict[str, Any]], Any]] = []

    def handler(actions, expect=None):
        seen.append((actions, expect))
        prediction = {"verdict": "confirmed", "failures": []}
        return {
            "action_result": {"executed": True, "prediction": prediction},
            "state": _state(last_prediction=prediction, prediction_log=[prediction]),
        }

    result = run_sandboxed_python(
        code=(
            "r = action('MOUSE', row=3, col=4, expect={'board_changed': True, 'cells': {(3, 4): 'R'}})\n"
            "result = [last_prediction['verdict'], len(prediction_log)]"
        ),
        timeout_seconds=10,
        initial_state=_state(),
        action_handler=handler,
    )
    assert result["error"] == ""
    assert result["result"] == ["confirmed", 1]
    actions, expect = seen[0]
    assert actions == [{"action": "MOUSE", "row": 3, "col": 4}]
    assert expect == {"board_changed": True, "cells": [[3, 4, "R"]]}


def test_prediction_globals_default_when_layer_off():
    result = run_sandboxed_python(
        code="result = [last_prediction, prediction_log]",
        timeout_seconds=10,
        initial_state=_state(),
        action_handler=lambda actions: {},
    )
    assert result["result"] == [None, []]


def test_expect_must_be_dict_and_row_col_need_single_name():
    result = run_sandboxed_python(
        code="action('UP', expect=[1, 2])",
        timeout_seconds=10,
        initial_state=_state(),
        action_handler=lambda actions, expect=None: {},
    )
    assert "TypeError" in result["error"]
    result = run_sandboxed_python(
        code="action(['MOUSE'], row=1, col=1)",
        timeout_seconds=10,
        initial_state=_state(),
        action_handler=lambda actions: {},
    )
    assert "TypeError" in result["error"]


# ----- hypothesis() ----------------------------------------------------------


def test_hypothesis_records_and_quick_checks():
    after = _grid({(1, 2): 9})
    history = [
        {"action": "", "frame": _frame_payload(_grid({(1, 1): 9}), step=0)},
        {"action": "RIGHT", "frame": _frame_payload(after, step=1)},
    ]
    state = _state(history=history, current_frame=history[-1]["frame"], next_hypothesis_index=4, quick_check_seconds=3)
    code = (
        "src = 'def check_transition(t):\\n    return len(changed_cells(t.before_frame, t.after_frame)) > 0'\n"
        "r = hypothesis('RIGHT changes the board', kind='rule', check_code=src)\n"
        "g = hypothesis('reach the red cell', kind='goal')\n"
        "result = [r['id'], r['quick_check']['status'], r['quick_check']['support'], g['id']]"
    )
    result = run_sandboxed_python(code=code, timeout_seconds=10, initial_state=state, action_handler=lambda a: {})
    assert result["error"] == ""
    assert result["result"] == ["H4", "pass", 1, "H5"]
    assert [item["id"] for item in result["hypotheses"]] == ["H4", "H5"]
    assert result["hypotheses"][1]["check_code"] is None


def test_hypotheses_survive_a_later_error():
    result = run_sandboxed_python(
        code="hypothesis('a claim')\nraise ValueError('boom')",
        timeout_seconds=10,
        initial_state=_state(),
        action_handler=lambda a: {},
    )
    assert "ValueError" in result["error"]
    assert [item["text"] for item in result["hypotheses"]] == ["a claim"]


def test_quick_check_has_its_own_time_cap():
    code = (
        "src = 'def check_transition(t):\\n    while True:\\n        pass'\n"
        "r = hypothesis('slow', check_code=src)\n"
        "result = r['quick_check']['status']"
    )
    history = [
        {"action": "", "frame": _frame_payload(_grid({}), step=0)},
        {"action": "UP", "frame": _frame_payload(_grid({(0, 0): 9}), step=1)},
    ]
    state = _state(history=history, current_frame=history[-1]["frame"], quick_check_seconds=0.5)
    result = run_sandboxed_python(code=code, timeout_seconds=10, initial_state=state, action_handler=lambda a: {})
    assert result["error"] == ""
    assert result["result"] == "skipped"


def test_transition_results_fill_older_transitions():
    history = [
        {"action": "", "frame": _frame_payload(_grid({}), step=0)},
        {"action": "UP", "frame": _frame_payload(_grid({(0, 0): 9}), step=1)},
        {"action": "UP", "frame": _frame_payload(_grid({(0, 1): 9}), step=2)},
    ]
    state = _state(history=history, transition_results={"1": {"board_changed": True, "reward": 0.5}})
    result = run_sandboxed_python(
        code="result = transitions[0].result.get('reward')",
        timeout_seconds=10,
        initial_state=state,
        action_handler=lambda a: {},
    )
    assert result["result"] == 0.5


# ----- check mode -------------------------------------------------------------

BEFORE = _grid({(1, 1): 9})
AFTER_RIGHT = _grid({(1, 2): 9})
MOVES_RIGHT_CHECK = (
    "def check_transition(t):\n"
    "    name, _, _ = parse_action(t.action)\n"
    "    if name != 'RIGHT':\n"
    "        return None\n"
    "    return cell(t.after_frame, 1, 2) == 'b' and cell(t.before_frame, 1, 1) == 'b'\n"
)


def test_check_passes_counts_support_and_is_falsifiable():
    transitions = [
        _transition("RIGHT", BEFORE, AFTER_RIGHT, 1),
        _transition("RIGHT", BEFORE, AFTER_RIGHT, 2),
        _transition("UP", BEFORE, BEFORE, 3),
    ]
    report = run_sandboxed_check(check_code=MOVES_RIGHT_CHECK, transitions=transitions, timeout_seconds=10)
    assert report["status"] == "pass"
    assert report["support"] == 2  # the UP transition returned None: not support, not failure
    assert report["not_applicable"] == 1
    assert report["falsifiable"] is True


def test_check_failure_reports_first_failing_transition():
    transitions = [_transition("RIGHT", BEFORE, AFTER_RIGHT, 1), _transition("RIGHT", BEFORE, BEFORE, 2)]
    report = run_sandboxed_check(check_code=MOVES_RIGHT_CHECK, transitions=transitions, timeout_seconds=10)
    assert report["status"] == "fail"
    assert report["first_failure"] == {"action": "RIGHT", "step": 2}


def test_vacuous_check_is_unfalsifiable():
    transitions = [_transition("RIGHT", BEFORE, AFTER_RIGHT, 1), _transition("RIGHT", BEFORE, AFTER_RIGHT, 2)]
    report = run_sandboxed_check(
        check_code="def check_transition(t):\n    return True\n",
        transitions=transitions,
        timeout_seconds=10,
    )
    assert report["status"] == "pass"
    assert report["falsifiable"] is False


def test_crash_on_perturbed_transition_is_inconclusive():
    # Passes on real moves, raises IndexError on the frozen variant (no changed cells),
    # and returns True on the swapped/flipped variants: never a clean False.
    transitions = [_transition("RIGHT", BEFORE, AFTER_RIGHT, 1), _transition("RIGHT", BEFORE, AFTER_RIGHT, 2)]
    report = run_sandboxed_check(
        check_code="def check_transition(t):\n    return changed_cells(t.before_frame, t.after_frame)[0] is not None\n",
        transitions=transitions,
        timeout_seconds=10,
    )
    assert report["status"] == "pass"
    assert report["falsifiable"] is False


def test_crash_on_one_variant_does_not_hide_clean_false_on_another():
    # Frozen variant of the first transition crashes; frozen variant of the second
    # returns a clean False. (Keyed on before_frame.step, which perturbation keeps.)
    check = (
        "def check_transition(t):\n"
        "    changes = changed_cells(t.before_frame, t.after_frame)\n"
        "    if t.before_frame.step == 0:\n"
        "        return changes[0] is not None\n"
        "    return len(changes) > 0\n"
    )
    transitions = [_transition("RIGHT", BEFORE, AFTER_RIGHT, 1), _transition("RIGHT", BEFORE, AFTER_RIGHT, 2)]
    report = run_sandboxed_check(check_code=check, transitions=transitions, timeout_seconds=10)
    assert report["status"] == "pass"
    assert report["falsifiable"] is True


def test_whole_list_check_counts_one_support():
    transitions = [_transition("RIGHT", BEFORE, AFTER_RIGHT, 1), _transition("RIGHT", BEFORE, AFTER_RIGHT, 2)]
    report = run_sandboxed_check(
        check_code="def check(transitions):\n    return all(t.result['board_changed'] for t in transitions)\n",
        transitions=transitions,
        timeout_seconds=10,
    )
    assert report["status"] == "pass"
    assert report["form"] == "all"
    assert report["support"] == 1
    assert report["falsifiable"] is True  # the frozen perturbation sets board_changed False


def test_check_segmentation_works_from_row_strings():
    report = run_sandboxed_check(
        check_code="def check_transition(t):\n    return len(t.after_frame.segmentation['nodes']) == 2\n",
        transitions=[_transition("RIGHT", BEFORE, AFTER_RIGHT, 1)],
        timeout_seconds=10,
        perturb=False,
    )
    assert report["status"] == "pass"
    assert report["falsifiable"] is None


def test_check_import_allowlist_and_errors():
    transitions = [_transition("RIGHT", BEFORE, AFTER_RIGHT, 1)]
    blocked = run_sandboxed_check(
        check_code="import os\ndef check_transition(t):\n    return True\n",
        transitions=transitions,
        timeout_seconds=10,
    )
    assert blocked["status"] == "error"
    assert "not allowed" in blocked["error"]
    allowed = run_sandboxed_check(
        check_code="import math\ndef check_transition(t):\n    return math.isfinite(1.0)\n",
        transitions=transitions,
        timeout_seconds=10,
        perturb=False,
    )
    assert allowed["status"] == "pass"
    missing = run_sandboxed_check(check_code="x = 1\n", transitions=transitions, timeout_seconds=10)
    assert missing["status"] == "error"
    syntax = run_sandboxed_check(check_code="def check_transition(t)\n", transitions=transitions, timeout_seconds=10)
    assert syntax["status"] == "error"


def test_check_timeout():
    report = run_sandboxed_check(
        check_code="def check_transition(t):\n    while True:\n        pass\n",
        transitions=[_transition("RIGHT", BEFORE, AFTER_RIGHT, 1)],
        timeout_seconds=2,
    )
    assert report["status"] == "timeout"


def test_check_cannot_call_action():
    report = run_sandboxed_check(
        check_code="def check_transition(t):\n    action(['UP'])\n    return True\n",
        transitions=[_transition("RIGHT", BEFORE, AFTER_RIGHT, 1)],
        timeout_seconds=10,
    )
    assert report["status"] == "error"
    assert "NameError" in report["error"]


# ----- world_model() / plan() --------------------------------------------------


def _no_action(_actions: list[dict[str, Any]], **_kwargs: Any) -> dict[str, Any]:
    raise AssertionError("no action expected")


def test_world_model_and_plan_return_notes_when_enabled():
    outcome = run_sandboxed_python(
        code="r = world_model('blue cell is the player')\nplan('go right')\nresult = r",
        timeout_seconds=10,
        initial_state=_state(world_model_enabled=True),
        action_handler=_no_action,
    )
    assert outcome["error"] == ""
    assert outcome["notes"] == {"world_model": "blue cell is the player", "plan": "go right"}
    assert outcome["result"]["stored"] == "world_model"


def test_world_model_absent_when_disabled():
    outcome = run_sandboxed_python(
        code="world_model('x')",
        timeout_seconds=10,
        initial_state=_state(),
        action_handler=_no_action,
    )
    assert "world_model" in outcome["error"] and outcome["notes"] == {}


def test_notes_survive_a_later_error():
    outcome = run_sandboxed_python(
        code="world_model('kept')\nraise ValueError('boom')",
        timeout_seconds=10,
        initial_state=_state(world_model_enabled=True),
        action_handler=_no_action,
    )
    assert "boom" in outcome["error"]
    assert outcome["notes"] == {"world_model": "kept"}
