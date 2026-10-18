"""Layer 2: hypothesis review, fact promotion/demotion, pinned block, persistence."""
from __future__ import annotations

import json
from typing import Any

from inference.agent.layer_config import LevelReviewConfig
from inference.agent.level_memory import (
    LevelMemory,
    level_transitions,
    parse_review_response,
)
from inference.agent.python_tool_sandbox import run_sandboxed_check
from inference.agent.runtime_state import Frame, HistoryEntry


def _frame(cells: dict[tuple[int, int], int], *, step: int, level: int, size: int = 6) -> Frame:
    grid = [[0] * size for _ in range(size)]
    for (row, col), value in cells.items():
        grid[row][col] = value
    return Frame(grid=tuple(tuple(row) for row in grid), step=step, level=level)


def _history(level_actions: list[tuple[int, str]]) -> list[HistoryEntry]:
    """Blue cell walks right one column per RIGHT; UP leaves the board as is."""
    col = 0
    entries = [HistoryEntry(action="", frame=_frame({(1, col): 9}, step=0, level=level_actions[0][0]))]
    for step, (level, action) in enumerate(level_actions, start=1):
        if action == "RIGHT":
            col += 1
        entries.append(HistoryEntry(action=action, frame=_frame({(1, col): 9}, step=step, level=level)))
    return entries


class FakeRunner:
    def __init__(self, outcomes: dict[str, dict[str, Any]]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[str, int, bool]] = []

    def __call__(self, code: str, transitions: list[dict[str, Any]], alt_actions: list[str], perturb: bool):
        self.calls.append((code, len(transitions), perturb))
        outcome = self.outcomes[code]
        return outcome(transitions) if callable(outcome) else dict(outcome)


PASS = {"status": "pass", "support": 3, "failures": 0, "falsifiable": True}


def _memory(**overrides: Any) -> LevelMemory:
    return LevelMemory(LevelReviewConfig(enabled=True, **overrides))


def _review(memory: LevelMemory, history, runner, llm_call=None, budget: float = 30.0):
    return memory.run_pending_reviews(
        history=history,
        prediction_entries=[],
        check_runner=runner,
        llm_call=llm_call,
        time_budget_seconds=budget,
    )


def _add(memory: LevelMemory, text: str, code: str | None = None, *, level: int = 1, kind: str = "rule") -> str:
    hypothesis_id = f"H{memory.next_index}"
    memory.add_hypotheses(
        [{"id": hypothesis_id, "text": text, "kind": kind, "check_code": code, "level": level}],
        default_level=level,
    )
    return hypothesis_id


HISTORY_L1 = _history([(1, "RIGHT"), (1, "RIGHT"), (1, "UP"), (1, "RIGHT")])


def test_passing_falsifiable_check_with_support_becomes_fact():
    memory = _memory()
    hypothesis_id = _add(memory, "RIGHT moves blue right", "C1")
    memory.mark_review_pending(1, "level_completed")
    (summary,) = _review(memory, HISTORY_L1, FakeRunner({"C1": PASS}))
    fact = memory.get(hypothesis_id)
    assert fact.status == "fact"
    assert fact.support == 3 and fact.level_learned == 1 and fact.levels_held == [1]
    assert summary["promoted"] == [hypothesis_id]
    assert summary["transitions"] == 4


def test_low_support_stays_unverified():
    memory = _memory(min_support=5)
    hypothesis_id = _add(memory, "claim", "C1")
    memory.mark_review_pending(1, "level_completed")
    _review(memory, HISTORY_L1, FakeRunner({"C1": PASS}))
    assert memory.get(hypothesis_id).status == "unverified"
    assert "min_support" in memory.get(hypothesis_id).reason


def test_unfalsifiable_check_is_not_a_fact():
    memory = _memory()
    hypothesis_id = _add(memory, "always true", "C1")
    memory.mark_review_pending(1, "level_completed")
    _review(memory, HISTORY_L1, FakeRunner({"C1": dict(PASS, falsifiable=False)}))
    hypothesis = memory.get(hypothesis_id)
    assert hypothesis.status == "unverified"
    assert hypothesis.reason.startswith("unfalsifiable")


def test_perturbation_test_can_be_disabled():
    memory = _memory(perturbation_test=False)
    hypothesis_id = _add(memory, "always true", "C1")
    memory.mark_review_pending(1, "level_completed")
    runner = FakeRunner({"C1": dict(PASS, falsifiable=None)})
    _review(memory, HISTORY_L1, runner)
    assert memory.get(hypothesis_id).status == "fact"
    assert runner.calls[0][2] is False


def test_failing_or_crashing_check_is_refuted():
    memory = _memory()
    failing = _add(memory, "UP moves blue", "C1")
    crashing = _add(memory, "crashes", "C2")
    memory.mark_review_pending(1, "game_over")
    runner = FakeRunner(
        {
            "C1": {"status": "fail", "failures": 1, "first_failure": {"step": 3, "action": "UP"}},
            "C2": {"status": "error", "error": "ZeroDivisionError: division by zero"},
        }
    )
    summary = _review(memory, HISTORY_L1, runner)[0]
    assert memory.get(failing).status == "refuted"
    assert "step 3 (UP)" in memory.get(failing).reason
    assert memory.get(crashing).status == "refuted"
    assert sorted(summary["refuted"]) == sorted([failing, crashing])


def test_timeout_keeps_status():
    memory = _memory()
    hypothesis_id = _add(memory, "slow", "C1")
    memory.mark_review_pending(1, "level_completed")
    _review(memory, HISTORY_L1, FakeRunner({"C1": {"status": "timeout", "error": "Tool timed out"}}))
    assert memory.get(hypothesis_id).status == "pending"


def test_fact_rechecked_on_next_level_and_demoted():
    memory = _memory()
    hypothesis_id = _add(memory, "RIGHT moves blue right", "C1")
    memory.mark_review_pending(1, "level_completed")
    history = _history([(1, "RIGHT"), (1, "RIGHT"), (2, "RIGHT"), (2, "RIGHT")])
    results = iter([PASS, {"status": "fail", "failures": 1, "first_failure": {"step": 4, "action": "RIGHT"}}])
    runner = FakeRunner({"C1": lambda _transitions: next(results)})
    _review(memory, history, runner)
    assert memory.get(hypothesis_id).status == "fact"
    memory.pop_notices()
    memory.mark_review_pending(2, "level_completed")
    _review(memory, history, runner)
    fact = memory.get(hypothesis_id)
    assert fact.status == "refuted" and fact.refuted_at_level == 2
    assert any(notice.startswith("FACT DEMOTED") for notice in memory.pop_notices())
    # facts are re-checked without perturbations
    assert runner.calls[-1][2] is False


def test_support_not_double_counted_across_reviews_of_same_level():
    memory = _memory()
    hypothesis_id = _add(memory, "claim", "C1")
    runner = FakeRunner({"C1": lambda transitions: dict(PASS, support=len(transitions))})
    memory.mark_review_pending(1, "game_over")
    _review(memory, _history([(1, "RIGHT"), (1, "RIGHT")]), runner)
    memory.mark_review_pending(1, "level_completed")
    _review(memory, _history([(1, "RIGHT"), (1, "RIGHT"), (1, "RIGHT")]), runner)
    assert memory.get(hypothesis_id).support == 3
    assert [calls for _code, calls, _perturb in runner.calls] == [2, 1]


def test_llm_writes_missing_check_which_is_executed():
    memory = _memory()
    hypothesis_id = _add(memory, "RIGHT moves blue right")  # no check_code
    memory.mark_review_pending(1, "level_completed")
    seen: list[list[dict[str, Any]]] = []

    def llm_call(messages, timeout):
        seen.append(messages)
        return json.dumps({"checks": [{"id": hypothesis_id, "check_code": "LLM_CODE"}], "contradicted": []})

    runner = FakeRunner({"LLM_CODE": PASS})
    summary = _review(memory, HISTORY_L1, runner, llm_call=llm_call)[0]
    assert summary["llm"] == "ran"
    hypothesis = memory.get(hypothesis_id)
    assert hypothesis.status == "fact" and hypothesis.check_origin == "review"
    assert "no check_code" in seen[0][1]["content"]


def test_llm_never_promotes_and_contradicted_only_refutes_unchecked():
    memory = _memory()
    fact_id = _add(memory, "verified rule", "C1")
    open_id = _add(memory, "unchecked claim")
    passing_unverified = _add(memory, "true but rare", "C2")
    memory.mark_review_pending(1, "level_completed")

    def llm_call(messages, timeout):
        return (
            "<think>sure</think> "
            + json.dumps(
                {
                    "checks": [{"id": fact_id, "check_code": "OVERRIDE"}],
                    "contradicted": [
                        {"id": fact_id, "why": "seems wrong"},
                        {"id": open_id, "why": "UP did nothing"},
                        {"id": passing_unverified, "why": "nope"},
                    ],
                    "promote": [open_id],
                }
            )
        )

    runner = FakeRunner({"C1": PASS, "C2": dict(PASS, support=1)})
    _review(memory, HISTORY_L1, runner, llm_call=llm_call)
    assert memory.get(fact_id).status == "fact"
    assert memory.get(fact_id).check_code == "C1"
    assert memory.get(open_id).status == "refuted"
    assert memory.get(passing_unverified).status == "unverified"


def test_llm_review_without_code_leaves_hypothesis_open():
    memory = _memory()
    hypothesis_id = _add(memory, "claim")
    memory.mark_review_pending(1, "level_completed")
    _review(memory, HISTORY_L1, FakeRunner({}), llm_call=lambda m, t: '{"checks": [], "contradicted": []}')
    assert memory.get(hypothesis_id).status == "pending"


def test_llm_failure_keeps_mechanical_results():
    memory = _memory()
    fact_id = _add(memory, "rule", "C1")
    _add(memory, "claim without code")
    memory.mark_review_pending(1, "level_completed")

    def broken(messages, timeout):
        raise TimeoutError("read timed out")

    summary = _review(memory, HISTORY_L1, FakeRunner({"C1": PASS}), llm_call=broken)[0]
    assert summary["llm"] == "failed: TimeoutError"
    assert memory.get(fact_id).status == "fact"


def test_llm_call_limit_and_time_budget():
    memory = _memory(max_llm_calls_per_level=1)
    _add(memory, "claim without code")
    calls: list[int] = []

    def llm_call(messages, timeout):
        calls.append(1)
        return "{}"

    memory.mark_review_pending(1, "game_over")
    memory.mark_review_pending(1, "level_completed")
    summaries = _review(memory, HISTORY_L1, FakeRunner({}), llm_call=llm_call)
    assert len(calls) == 1
    assert summaries[1]["llm"] == "skipped: per-level call limit"
    memory.mark_review_pending(2, "level_completed")
    summary = _review(memory, HISTORY_L1, FakeRunner({}), llm_call=llm_call, budget=0.0)[0]
    assert summary["llm"] == "skipped: no time left"


def test_duplicate_hypotheses_are_merged_and_ids_stay_aligned():
    memory = _memory()
    first = _add(memory, "RIGHT moves blue right")
    lines = memory.add_hypotheses(
        [{"id": "H2", "text": "right moves  blue right", "kind": "rule", "check_code": "C9"}],
        default_level=1,
    )
    assert "same claim as H1" in lines[0]
    assert memory.get(first).check_code == "C9"
    assert memory.next_index == 3
    assert len(memory.hypotheses) == 1


def test_pinned_block_caps_and_orders_facts(tmp_path):
    memory = LevelMemory(
        LevelReviewConfig(enabled=True, max_facts_in_context=2, max_refuted_in_context=1),
        facts_path=tmp_path / "game_p0_facts.json",
    )
    ids = [_add(memory, f"rule {index}", f"C{index}") for index in range(3)]
    refuted_id = _add(memory, "wrong rule", "BAD")
    memory.mark_review_pending(1, "level_completed")
    runner = FakeRunner(
        {
            "C0": dict(PASS, support=2),
            "C1": dict(PASS, support=9),
            "C2": dict(PASS, support=5),
            "BAD": {"status": "fail", "failures": 1},
        }
    )
    _review(memory, HISTORY_L1, runner)
    block = memory.render_pinned_block()
    assert block.startswith("VERIFIED FACTS")
    assert block.index(f"{ids[1]} [rule]") < block.index(f"{ids[2]} [rule]")
    assert f"{ids[0]} [rule]" not in block  # lowest support dropped by the cap
    assert "REFUTED" in block and refuted_id in block
    saved = json.loads((tmp_path / "game_p0_facts.json").read_text())
    assert {item["id"]: item["status"] for item in saved["hypotheses"]}[ids[1]] == "fact"
    assert saved["level_stats"][0]["level"] == 1


def test_pinned_block_empty_without_facts():
    assert _memory().render_pinned_block() == ""


def test_level_transitions_compact_and_skip_reset():
    history = _history([(1, "RIGHT"), (1, "RESET"), (1, "UP"), (2, "RIGHT")])
    items = level_transitions(history, 1, {1: {"board_changed": True}}, cap=10)
    # the level-completing transition (level 1 -> 2) belongs to level 1; RESET is skipped
    assert [item["action"] for item in items] == ["RIGHT", "UP", "RIGHT"]
    assert items[-1]["level"] == 2
    assert items[0]["before"][1] == "bWWWWW"
    assert isinstance(items[0]["after"][0], str)
    assert items[0]["result"] == {"board_changed": True}
    assert len(level_transitions(history, 1, {}, cap=1)) == 1


def test_parse_review_response_is_lenient():
    checks, contradicted = parse_review_response(
        'text before {"checks": [{"id": "H1", "check_code": "def check_transition(t):\\n    return None"}],'
        ' "contradicted": ["H2", {"id": "H3", "why": "x"}]} after'
    )
    assert checks == [("H1", "def check_transition(t):\n    return None")]
    assert contradicted == [("H2", ""), ("H3", "x")]
    assert parse_review_response("no json here") == ([], [])


def test_end_to_end_with_real_sandbox_checks():
    memory = _memory()
    real = _add(
        memory,
        "RIGHT moves the blue cell one column right",
        "def check_transition(t):\n"
        "    if parse_action(t.action)[0] != 'RIGHT':\n"
        "        return None\n"
        "    before = [c for r, c, old, new in changed_cells(t.before_frame, t.after_frame) if old == 'b']\n"
        "    after = [c for r, c, old, new in changed_cells(t.before_frame, t.after_frame) if new == 'b']\n"
        "    return len(before) == 1 and len(after) == 1 and after[0] == before[0] + 1\n",
    )
    vacuous = _add(memory, "anything goes", "def check_transition(t):\n    return True\n")
    wrong = _add(memory, "UP moves blue", "def check_transition(t):\n    return None if t.action != 'UP' else t.result.get('board_changed', False)\n")
    memory.record_transition_results(
        [{"action_num": step, "board_changed": step != 3} for step in range(1, 5)]
    )
    memory.mark_review_pending(1, "level_completed")

    def runner(code, transitions, alt_actions, perturb):
        return run_sandboxed_check(
            check_code=code, transitions=transitions, timeout_seconds=10, alt_actions=alt_actions, perturb=perturb
        )

    _review(memory, HISTORY_L1, runner)
    assert memory.get(real).status == "fact"
    assert memory.get(real).support == 3
    assert memory.get(vacuous).status == "unverified"
    assert memory.get(wrong).status == "refuted"
