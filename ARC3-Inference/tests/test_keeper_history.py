"""Keeper's own per-game history: same transitions as before, no file reloads, bounded."""
from __future__ import annotations

import pytest

from inference.agent import keeper_auto
from inference.agent.keeper_history import KeeperHistory, pack
from inference.agent.runtime_state import Frame, HistoryEntry, load_runtime_state
from keeper_fakes import board, clear_keeper_env, make_agent, run_code


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    clear_keeper_env(monkeypatch)
    monkeypatch.delenv("MULTIMODAL_CONTEXT", raising=False)


def _history(cols_levels, actions=None):
    entries = [HistoryEntry(action="", frame=Frame(grid=board(cols_levels[0][0], step=0), step=0, level=cols_levels[0][1]))]
    for step, (col, level) in enumerate(cols_levels[1:], start=1):
        action = (actions or {}).get(step, "RIGHT")
        entries.append(HistoryEntry(action=action, frame=Frame(grid=board(col, step=step), step=step, level=level)))
    return entries


def test_incremental_sync_gives_the_same_transitions_as_rebuilding_from_the_whole_history():
    full = _history([(6, 1), (7, 1), (8, 1), (8, 1), (2, 2), (3, 2), (3, 2), (4, 2)], actions={6: "RESET"})
    record = KeeperHistory()
    added = []
    for end in (2, 3, 6, 8):  # four tool calls, each seeing the whole history so far
        added += record.sync(full[:end])
    assert added == [1, 2, 3, 4, 5, 7]  # the RESET at step 6 is not a transition
    for level in (1, 2):
        expected = keeper_auto.level_transitions(full, level)
        got = record.transitions(level)
        assert [(t["action"], t["step"], t["result"]) for t in got] == [(t["action"], t["step"], t["result"]) for t in expected]
        assert [(t["before"], t["after"]) for t in got] == [(pack(t["before"]), pack(t["after"])) for t in expected]
    assert record.sync(full) == []  # nothing new


def test_only_the_current_and_previous_level_are_kept_and_each_level_is_capped():
    cols_levels = [(6, 1)] + [(6, 1)] * 3 + [(6, 2)] * 3 + [(6, 3)] * 30
    record = KeeperHistory(keep_levels=2, max_per_level=10)
    record.sync(_history(cols_levels))
    assert sorted(record.levels) == [2, 3]
    assert len(record.transitions(3)) == 10 and record.transitions(3)[-1]["step"] == len(cols_levels) - 1
    record.effects.update({1: {}, len(cols_levels) - 1: {}})
    record.sync(_history(cols_levels + [(6, 3)]))
    assert 1 not in record.effects and len(cols_levels) - 1 in record.effects


def test_a_no_op_shares_the_previous_grid_object():
    record = KeeperHistory()
    same = board(9, step=0)
    record.sync([HistoryEntry(action="" if s == 0 else "RIGHT", frame=Frame(grid=same, step=s, level=1)) for s in range(3)])
    first, second = record.transitions(1)
    assert first["after"] is first["before"] and second["before"] is first["after"]


def test_the_keeper_never_reloads_the_runtime_state_file(tmp_path, monkeypatch):
    import inference.agent.tool_agent as ta

    def counting(path):
        counting.calls += 1
        return load_runtime_state(path)

    def loads_for(env):
        for key in ("KEEPER_SURPRISE", "KEEPER_REVIEW", "KEEPER_PERCEPTION"):
            monkeypatch.delenv(key, raising=False)
        for key in env:
            monkeypatch.setenv(key, "1")
        counting.calls = 0
        monkeypatch.setattr(ta, "load_runtime_state", counting)
        agent, state_path, game = make_agent(tmp_path / ("on" if env else "off"))
        game.complete_at_col = 8
        payloads = [run_code(agent, state_path, code) for code in
                    ("action(['RIGHT'] * 3)", "action(['UP'])\naction(['RIGHT'] * 2)", "action(['RIGHT'] * 4)")]
        if env:
            agent._keeper_before_turn(state_path, "P", lambda *a: None, 0.0)  # review skipped (time cap): no load either
        return counting.calls, payloads

    off, _ = loads_for([])
    on, payloads = loads_for(["KEEPER_SURPRISE", "KEEPER_REVIEW", "KEEPER_PERCEPTION"])
    assert on == off  # the Keeper adds no file reads
    assert any("surprises" in p for p in payloads)


def test_the_review_uses_the_keeper_history(tmp_path, monkeypatch):
    import json

    import inference.agent.tool_agent as ta

    monkeypatch.setenv("KEEPER_REVIEW", "1")
    seen = []

    def chat(self, messages, **_k):
        seen.append(messages[-1]["content"])
        return ta._ChatCompletionResult(message={"content": json.dumps({"cause": "c", "hypotheses": []})},
                                        finish_reason="stop", usage={"completion_tokens": 1, "prompt_tokens": 1})

    monkeypatch.setattr(ta.ToolAgent, "_chat_completion", chat)
    agent, state_path, game = make_agent(tmp_path)
    game.complete_at_col = 8
    run_code(agent, state_path, "action(['RIGHT', 'RIGHT'])")
    monkeypatch.setattr(ta, "load_runtime_state", lambda *_a: (_ for _ in ()).throw(AssertionError("reloaded")))
    agent._keeper_before_turn(state_path, "P", lambda *a: None, 900.0)
    assert seen and "step 1 RIGHT: moved [0,1]" in seen[0] and "step 2 RIGHT: level completed" in seen[0]
    assert agent._keeper_history.transitions(1)[0]["result"] == {"level_completed": False, "game_over": False}
