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


def test_a_long_cause_is_cut_at_a_sentence_end_not_mid_word():
    first = "The level completed when the dot landed on the cross at (31,52). " * 4
    second = "That needed the yellow bar shortened first so the chain could rise above the purple wall and slide right"
    parsed = keeper_review.parse_reply(_reply(first + second * 5, []))
    assert parsed["cause"] == first.strip()
    assert len(keeper_review.parse_reply(_reply("x" * 300 + ". done", []))["cause"]) == 306  # fits: kept whole
    words = keeper_review.clip_at_sentence("word " * 200, 600)
    assert words.endswith("word ...") and len(words) <= 600


def test_the_notes_drop_low_support_facts_instead_of_cutting_text():
    memory = keeper_review.KeeperMemory(min_support=1, max_open=10, max_hypotheses=5)
    memory.last_level = 1
    memory.causes[1] = "c" * 590 + "."
    for i in range(8):
        memory.hypotheses.append(keeper_review.Hypothesis(id=f"K{i}", text=f"fact {i} " + "x" * 200, kind="rule",
                                                          check_code="", level_proposed=1, status="fact",
                                                          support=10 - i, levels_held=[1]))
    block = memory.notes_block(max_chars=1500)
    assert len(block) <= 1500 and block.endswith("c" * 590 + ".")
    assert "fact 0 " in block and "fact 7 " not in block


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
    assert "- held on level 1: RIGHT moves the player one column until the wall (support 4)" in block
    assert "Verified facts" not in block  # facts are shown as level-scoped, not as general rules
    assert "anything holds" not in block  # unfalsifiable ideas are not shown


def test_open_ideas_are_never_shown_only_facts_and_the_cause():
    memory = keeper_review.KeeperMemory(min_support=50, max_open=10, max_hypotheses=5)  # nothing can become a fact
    memory.apply(1, _reply("the last RIGHT reached the exit",
                           [{"text": "RIGHT moves the player one column", "kind": "rule", "check_code": RIGHT_MOVES_PLAYER}]),
                 _transitions(), _check)
    assert memory.hypotheses[0].status == "open" and memory.hypotheses[0].support > 0
    block = memory.notes_block()
    assert "Level 1, likely cause: the last RIGHT reached the exit" in block
    assert "RIGHT moves the player" not in block and "idea" not in block.lower()


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


def test_a_fact_names_every_level_it_held_on():
    memory = keeper_review.KeeperMemory(min_support=2, max_open=10, max_hypotheses=5)
    memory.apply(1, _reply("c", [{"text": "RIGHT moves the player", "kind": "rule", "check_code": RIGHT_MOVES_PLAYER}]),
                 _transitions(), _check)
    memory.apply(2, _reply("c2", []), _transitions(), _check)
    assert memory.hypotheses[0].status == "fact"
    assert "- held on levels 1 and 2: RIGHT moves the player (support 8)" in memory.notes_block()


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


def test_notes_block_is_not_repeated_when_messages_carry_images(tmp_path, monkeypatch):
    from inference.agent.tool_agent import ToolAgent

    monkeypatch.setenv("KEEPER_REVIEW", "1")
    calls = []
    monkeypatch.setattr(ToolAgent, "_chat_completion", _fake_completion(_reply("cause", []), calls))
    agent, state_path, game = make_agent(tmp_path)
    game.complete_at_col = 8
    run_code(agent, state_path, "action(['RIGHT', 'RIGHT'])")
    opener = agent._keeper_before_turn(state_path, "LEVEL UP", lambda *a: None, 900.0)
    assert opener.startswith("KEEPER NOTES")
    agent._history_messages.append(
        {"role": "user", "content": [{"type": "text", "text": opener}, {"type": "image_url", "image_url": {"url": "data:"}}]}
    )
    assert agent._keeper_before_turn(state_path, "NEXT", lambda *a: None, 900.0) == "NEXT"


# ----- check API examples and the one retry ---------------------------------------------------


def _examples():
    text = keeper_review.CHECK_API
    blocks = text.split("def check_transition(t):")[1:]
    return ["def check_transition(t):" + b.split("\nExample")[0] for b in blocks]


def test_the_prompt_examples_run_as_written():
    rule, goal = _examples()
    assert "frame_diff" in keeper_review.CHECK_API and "four values" in keeper_review.CHECK_API
    report = _check(rule, _transitions()[:3], True)  # player colour 7 is 'P'; RIGHT moves it one column
    assert report["status"] == "pass" and report["support"] == 3 and report["falsifiable"] is True
    still = board(4, step=0)
    lit = board(4, step=1, extra={(6, 8): 9})
    clicks = [{"action": "MOUSE(row=2, col=2)", "before": still, "after": still, "step": 1, "level": 1, "result": {}},
              {"action": "MOUSE(row=3, col=3)", "before": still, "after": lit, "step": 2, "level": 1, "result": {}}]
    assert _check(goal, clicks[:1], False)["status"] == "pass"
    assert _check(goal, clicks, False)["status"] == "fail"


def _scripted_completion(replies, calls):
    def chat_completion(self, messages, *, tools, request_timeout_seconds=None, max_tokens=None, enable_thinking=None):
        calls.append({"messages": messages, "max_tokens": max_tokens, "enable_thinking": enable_thinking})
        return _FakeResult(message={"content": replies[len(calls) - 1]}, usage={"completion_tokens": 100, "prompt_tokens": 9000})
    return chat_completion


def _review_with(tmp_path, monkeypatch, replies):
    from inference.agent.tool_agent import ToolAgent

    monkeypatch.setenv("KEEPER_REVIEW", "1")
    calls = []
    monkeypatch.setattr(ToolAgent, "_chat_completion", _scripted_completion(replies, calls))
    agent, state_path, game = make_agent(tmp_path)
    game.complete_at_col = 8
    run_code(agent, state_path, "action(['RIGHT', 'RIGHT'])")
    transcript = []
    agent._keeper_before_turn(state_path, "P", lambda label, text: transcript.append((label, text)), 900.0)
    return agent, calls, json.loads(transcript[-1][1])


BROKEN = "def check_transition(t):\n    return t.level > 0\n"


def test_a_crashed_check_is_re_asked_once_with_its_error_and_then_judged(tmp_path, monkeypatch):
    first = _reply("cause", [{"text": "RIGHT moves the player one column", "kind": "rule", "check_code": BROKEN}])
    fix = json.dumps({"fixes": [{"id": "K1", "check_code": RIGHT_MOVES_PLAYER}]})
    agent, calls, summary = _review_with(tmp_path, monkeypatch, [first, fix])
    assert len(calls) == 2
    ask = calls[1]["messages"]
    assert ask[-2] == {"role": "assistant", "content": first}  # same conversation, cached prefix
    assert ask[-1]["content"].startswith("KEEPER CHECK FIX") and "'TransitionView' object has no attribute 'level'" in ask[-1]["content"]
    assert calls[1]["enable_thinking"] is False
    (h,) = agent._keeper_memory.hypotheses
    assert h.status in ("open", "fact") and h.check_code == RIGHT_MOVES_PLAYER.strip() and h.support > 0
    assert summary["retried"] == ["K1"] and summary["fixed"] == ["K1"] and summary["dropped"] == []
    # both requests are review tokens: in the session total, not the level's priority tokens
    assert agent._session_generated_tokens - agent._tokens_at_level_start == 0


def test_a_check_that_still_crashes_is_dropped_and_no_third_request_is_made(tmp_path, monkeypatch):
    first = _reply("cause", [{"text": "a", "kind": "rule", "check_code": BROKEN},
                             {"text": "b", "kind": "rule", "check_code": BROKEN.replace("level", "lvl")}])
    fix = json.dumps({"fixes": [{"id": "K1", "check_code": BROKEN}]})  # K2 gets no fix
    agent, calls, summary = _review_with(tmp_path, monkeypatch, [first, fix])
    assert len(calls) == 2
    reasons = {h.id: (h.status, h.reason) for h in agent._keeper_memory.hypotheses}
    assert reasons["K1"][0] == "dropped" and reasons["K1"][1].startswith("check could not run after a retry")
    assert reasons["K2"][0] == "dropped" and reasons["K2"][1].startswith("check could not run (no fix given)")
    assert summary["fixed"] == [] and sorted(summary["dropped"]) == ["K1", "K2"]


def test_no_retry_request_when_every_check_runs(tmp_path, monkeypatch):
    first = _reply("cause", [{"text": "RIGHT moves", "kind": "rule", "check_code": RIGHT_MOVES_PLAYER}])
    _agent, calls, summary = _review_with(tmp_path, monkeypatch, [first])
    assert len(calls) == 1 and "retried" not in summary
