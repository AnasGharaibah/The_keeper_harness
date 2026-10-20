"""Keeper part 2: post-level review, mechanical checks, verified facts, notes block."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from inference.agent import keeper_review
from inference.agent.keeper_check_sandbox import run_check
from keeper_fakes import board, clear_keeper_env, make_agent, run_code

# Falsifiable: states where the player ends up. The "crossed" perturbation pairs a
# RIGHT's before-board with another move's after-board, where it does not hold.
RIGHT_MOVES_PLAYER = (
    "def check_transition(t):\n"
    "    if parse_action(t.action)[0] != 'RIGHT' or t.result.get('level_completed'):\n"
    "        return None\n"
    "    def col(frame):\n"
    "        return min(c for c, ch in enumerate(frame.ascii.split('\\n')[6]) if ch == 'P')\n"
    "    before, after = col(t.before_frame), col(t.after_frame)\n"
    "    return after == before + 1 or (after == before and before + 2 >= 10)\n"
)
ALWAYS_TRUE = "def check_transition(t):\n    return True\n"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    clear_keeper_env(monkeypatch)
    monkeypatch.delenv("MULTIMODAL_CONTEXT", raising=False)


def _reply(cause, hypotheses):
    return json.dumps({"cause": cause, "hypotheses": hypotheses})


def _transitions():
    cols = [6, 7, 8, 9, 9]
    out = []
    for step in range(1, len(cols)):
        out.append({"action": "RIGHT", "before": board(cols[step - 1], step=step - 1), "after": board(cols[step], step=step),
                    "step": step, "level": 1, "result": {"board_changed": True}})
    return out


def _check(code, items, perturb):
    return run_check(code, items, perturb=perturb, timeout_seconds=5)


# ----- memory, parsing, verification -------------------------------------------------------


def test_truncated_reply_keeps_cause_and_complete_hypotheses():
    reply = ('{"cause": "the last RIGHT reached the exit", "hypotheses": [{"text": "a", "kind": "rule", '
             '"check_code": "def check_transition(t):\\n    return None"}, {"text": "b", "check_code": "def ch')
    parsed = keeper_review.parse_reply(reply)
    assert parsed["truncated"] and parsed["cause"] == "the last RIGHT reached the exit"
    assert [h["text"] for h in parsed["hypotheses"]] == ["a"]


def test_only_passing_falsifiable_checks_with_support_become_facts():
    memory = keeper_review.KeeperMemory(min_support=2, max_open=10, max_hypotheses=5)
    reply = _reply("RIGHT moves the player", [
        {"text": "RIGHT moves the player one column until the wall", "kind": "rule", "check_code": RIGHT_MOVES_PLAYER},
        {"text": "anything holds", "kind": "rule", "check_code": ALWAYS_TRUE},
        {"text": "broken", "kind": "rule", "check_code": "def check_transition(t):\n    return t.nope\n"},
    ])
    summary = memory.apply(1, reply, _transitions(), _check)
    statuses = {h.text: (h.status, h.reason) for h in memory.hypotheses}
    assert summary["cause"] == "RIGHT moves the player"
    assert statuses["RIGHT moves the player one column until the wall"][0] == "fact"
    assert summary["promoted"] == ["K1"]
    assert statuses["anything holds"][0] == "open" and statuses["anything holds"][1].startswith("unfalsifiable")
    assert statuses["broken"][0] == "dropped"
    block = memory.notes_block()
    assert block.startswith("KEEPER NOTES") and "Level 1, likely cause: RIGHT moves the player" in block
    assert "- RIGHT moves the player one column until the wall (support 4, levels 1)" in block
    assert "anything holds" not in block  # unfalsifiable ideas are not shown


def test_open_hypotheses_carry_to_the_next_level_and_facts_get_demoted():
    memory = keeper_review.KeeperMemory(min_support=10, max_open=10, max_hypotheses=5)
    reply = _reply("c", [{"text": "RIGHT shifts by one", "kind": "rule", "check_code": RIGHT_MOVES_PLAYER}])
    memory.apply(1, reply, _transitions(), _check)
    hypothesis = memory.hypotheses[0]
    first_support = hypothesis.support
    memory.apply(2, _reply("c2", []), _transitions(), _check)
    assert hypothesis.support > first_support and hypothesis.levels_held == [1, 2]
    hypothesis.status = "fact"
    bad = [dict(t) for t in _transitions()]
    bad[0]["after"] = board(8, step=1)  # RIGHT jumped two columns
    summary = memory.apply(3, _reply("c3", []), bad, _check)
    assert hypothesis.status == "refuted" and summary["demoted"] == [hypothesis.id]


def test_request_is_short_and_names_the_level():
    memory = keeper_review.KeeperMemory(min_support=2, max_open=10, max_hypotheses=5)
    text = memory.request_text(3, _transitions(), border=4, max_moves=60)
    assert text.startswith("KEEPER LEVEL REVIEW") and "completed level 3" in text
    assert "steps 1-3 RIGHT x3: moved [0,1]" in text and "step 4 RIGHT: no change" in text
    assert len(text) < 2500


# ----- agent integration ---------------------------------------------------------------------


class _FakeResult(SimpleNamespace):
    pass


def _fake_completion(reply, calls):
    def chat_completion(self, messages, *, tools, request_timeout_seconds=None, max_tokens=None, enable_thinking=None):
        calls.append({"messages": messages, "tools": tools, "max_tokens": max_tokens, "enable_thinking": enable_thinking})
        return _FakeResult(message={"content": reply}, usage={"completion_tokens": 300, "prompt_tokens": 9000})
    return chat_completion


def test_switch_off_makes_no_request_and_adds_nothing(tmp_path, monkeypatch):
    from inference.agent.tool_agent import ToolAgent

    calls = []
    monkeypatch.setattr(ToolAgent, "_chat_completion", _fake_completion("{}", calls))
    monkeypatch.setattr(ToolAgent, "_keeper_before_turn", lambda *a, **k: (_ for _ in ()).throw(AssertionError("keeper ran")))
    agent, state_path, game = make_agent(tmp_path)
    game.complete_at_col = 8
    run_code(agent, state_path, "action(['RIGHT', 'RIGHT'])")
    assert getattr(agent, "_keeper_pending_review", None) is None
    assert "KEEPER" not in agent._system_prompt and calls == []


def test_level_completion_triggers_one_review_on_the_cached_prefix(tmp_path, monkeypatch):
    from inference.agent.tool_agent import ToolAgent

    monkeypatch.setenv("KEEPER_REVIEW", "1")
    calls = []
    reply = _reply("Two RIGHT moves reached column 8, which completes the level.",
                   [{"text": "RIGHT moves the player one column", "kind": "rule", "check_code": RIGHT_MOVES_PLAYER}])
    monkeypatch.setattr(ToolAgent, "_chat_completion", _fake_completion(reply, calls))
    agent, state_path, game = make_agent(tmp_path)
    assert "KEEPER NOTES" in agent._system_prompt
    agent._history_messages = [{"role": "user", "content": "earlier turn"}, {"role": "assistant", "content": "ok"}]
    game.complete_at_col = 8
    run_code(agent, state_path, "action(['RIGHT', 'RIGHT'])")
    assert agent._keeper_pending_review == {"level": 1, "reason": "level_completed"}
    agent._tokens_at_level_start = agent._session_generated_tokens
    transcript = []
    prompt = agent._keeper_before_turn(state_path, "LEVEL UP PROMPT", lambda label, text: transcript.append((label, text)), 900.0)
    (call,) = calls
    # Same system prompt, history and tools as a game request; thinking off; not stored.
    assert call["messages"][0] == {"role": "system", "content": agent._system_prompt}
    assert call["messages"][1:3] == agent._history_messages
    assert call["messages"][-1]["content"].startswith("KEEPER LEVEL REVIEW")
    assert call["tools"] == agent._tools(state_path) and call["enable_thinking"] is False and call["max_tokens"] == 2048
    assert len(agent._history_messages) == 2
    # Tokens: in the session total, not the turn budget, not the level's priority tokens.
    assert agent._session_generated_tokens == 300 and agent._turn_generated_tokens == 0
    assert agent._session_generated_tokens - agent._tokens_at_level_start == 0
    assert prompt.startswith("KEEPER NOTES") and prompt.endswith("LEVEL UP PROMPT")
    assert "Level 1, likely cause: Two RIGHT moves" in prompt
    assert transcript and transcript[-1][0] == "KEEPER REVIEW"
    # Next turn: the block is in history now, so it is not repeated...
    agent._history_messages.append({"role": "user", "content": prompt})
    assert agent._keeper_before_turn(state_path, "NEXT", lambda *a: None, 900.0) == "NEXT"
    # ...until a trim removes the message that carried it.
    agent._history_messages = agent._history_messages[-1:][:0]
    assert agent._keeper_before_turn(state_path, "AFTER TRIM", lambda *a: None, 900.0).startswith("KEEPER NOTES")


def test_review_is_skipped_in_the_tail_window_and_logged(tmp_path, monkeypatch):
    import inference.agent.tool_agent as ta

    monkeypatch.setenv("KEEPER_REVIEW", "1")
    calls = []
    monkeypatch.setattr(ta.ToolAgent, "_chat_completion", _fake_completion(_reply("c", []), calls))
    now = time.monotonic()
    gate = SimpleNamespace(_clock_start=now - 900.0, _clock_deadline=now + 100.0)  # 10% of the run left
    monkeypatch.setattr(ta, "_priority_gate", lambda: gate)
    agent, state_path, game = make_agent(tmp_path)
    game.complete_at_col = 8
    run_code(agent, state_path, "action(['RIGHT', 'RIGHT'])")
    transcript = []
    prompt = agent._keeper_before_turn(state_path, "P", lambda label, text: transcript.append((label, text)), 900.0)
    assert calls == [] and prompt == "P"
    assert transcript[0][0] == "KEEPER REVIEW" and transcript[0][1].startswith("skipped for level 1: final 20%")
    # Too little time left for the budget is skipped as well.
    gate._clock_start, gate._clock_deadline = now - 100.0, now + 200.0
    game.complete_at_col = 4
    agent._keeper_pending_review = {"level": 2, "reason": "level_completed"}
    agent._keeper_before_turn(state_path, "P", lambda label, text: transcript.append((label, text)), 900.0)
    assert calls == [] and "below the 240s review budget" in transcript[-1][1]


def test_game_over_reviews_only_with_their_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("KEEPER_REVIEW", "1")
    agent, state_path, game = make_agent(tmp_path)
    assert agent._keeper.review_on_game_over is False
