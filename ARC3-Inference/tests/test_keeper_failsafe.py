"""Keeper fail-safe: a failing or slow Keeper hook turns the Keeper off for that
game only; the game keeps playing exactly as it would with the Keeper off."""
from __future__ import annotations

import json
import logging
import time

import pytest

from inference.agent import keeper_auto, keeper_guard, keeper_history, keeper_perception, keeper_resources, keeper_review
from keeper_fakes import clear_keeper_env, make_agent, run_code

ALL = ("KEEPER_SURPRISE", "KEEPER_REVIEW", "KEEPER_PERCEPTION")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    clear_keeper_env(monkeypatch)
    monkeypatch.delenv("MULTIMODAL_CONTEXT", raising=False)
    for key in ALL:
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("KEEPER_HOOK_TIMEOUT_SECONDS", "0.3")


def _boom(*_a, **_k):
    raise RuntimeError("injected failure")


def _slow(*_a, **_k):
    time.sleep(2)
    return []


def _keeper_lines(payload):
    return {k: payload[k] for k in ("surprises", "perceived") if k in payload}


def _disabled_lines(caplog):
    return [r for r in caplog.records if "keeper disabled for" in r.getMessage()]


def _plays_on_without_keeper(agent, state_path, game, calls_before):
    """More actions run, nothing Keeper-shaped is added, the hook is not re-run."""
    sent = len(game.sent)
    payload = run_code(agent, state_path, "action(['UP', 'RIGHT'])")
    assert len(game.sent) == sent + 2 and game.sent[-2:] == ["UP", "RIGHT"]
    assert _keeper_lines(payload) == {}
    return payload


# ----- per-call hooks: SURPRISE, perception, history sync, review trigger -------------------------


HOOKS = {
    "surprise": (keeper_auto, "surprise_lines"),
    "perception": (keeper_perception.PerceptionTracker, "observe"),
    "history sync": (keeper_history.KeeperHistory, "sync"),
}


@pytest.mark.parametrize("hook", sorted(HOOKS))
@pytest.mark.parametrize("fault", ["exception", "timeout"])
def test_a_per_call_hook_failure_turns_keeper_off_for_the_game(tmp_path, monkeypatch, caplog, hook, fault):
    owner, name = HOOKS[hook]
    calls = []
    real = getattr(owner, name)

    def faulty(*a, **k):
        calls.append(1)
        return (_boom if fault == "exception" else _slow)(*a, **k)

    agent, state_path, game = make_agent(tmp_path)
    run_code(agent, state_path, "action(['RIGHT'] * 3)")  # Keeper working first
    monkeypatch.setattr(owner, name, faulty)
    caplog.set_level(logging.WARNING)
    started = time.monotonic()
    payload = run_code(agent, state_path, "action(['RIGHT'] * 6)")  # the blocked RIGHT would be a SURPRISE
    assert time.monotonic() - started < 1.5  # a hung hook is abandoned at the 0.3 s limit
    assert game.sent.count("RIGHT") == 9 and _keeper_lines(payload) == {}
    guard = agent._keeper_guard
    assert guard.disabled_reason.startswith("surprise/perception: ")
    assert ("took longer than" in guard.disabled_reason) == (fault == "timeout")
    assert len(_disabled_lines(caplog)) == 1
    n = len(calls)
    monkeypatch.setattr(owner, name, real)
    _plays_on_without_keeper(agent, state_path, game, n)
    _plays_on_without_keeper(agent, state_path, game, n)
    assert len(calls) == n and len(_disabled_lines(caplog)) == 1  # not re-run, logged once


def test_a_review_trigger_failure_turns_keeper_off(tmp_path, monkeypatch, caplog):
    from inference.agent.tool_agent import ToolAgent

    agent, state_path, game = make_agent(tmp_path)
    monkeypatch.setattr(ToolAgent, "_keeper_review_trigger", _boom)
    caplog.set_level(logging.WARNING)
    game.complete_at_col = 8
    run_code(agent, state_path, "action(['RIGHT', 'RIGHT'])")
    assert agent._keeper_guard.disabled_reason.startswith("review trigger: RuntimeError")
    assert getattr(agent, "_keeper_pending_review", None) is None
    _plays_on_without_keeper(agent, state_path, game, 0)


# ----- review, checks and notes injection ---------------------------------------------------------


def _complete_a_level(tmp_path, monkeypatch, reply="{}"):
    from inference.agent.tool_agent import ToolAgent, _ChatCompletionResult

    calls = []

    def chat(self, messages, **_k):
        calls.append(messages)
        return _ChatCompletionResult(message={"content": reply}, finish_reason="stop",
                                     usage={"completion_tokens": 10, "prompt_tokens": 100})

    monkeypatch.setattr(ToolAgent, "_chat_completion", chat)
    agent, state_path, game = make_agent(tmp_path)
    game.complete_at_col = 8
    run_code(agent, state_path, "action(['RIGHT', 'RIGHT'])")
    assert agent._keeper_pending_review is not None
    return agent, state_path, game, calls


GOOD_REPLY = json.dumps({"cause": "two RIGHTs reached the exit", "hypotheses": [
    {"text": "RIGHT moves", "kind": "rule", "check_code": "def check_transition(t):\n    return True\n"}]})


@pytest.mark.parametrize("where", ["model request", "memory.apply", "check sandbox", "notes injection", "review timeout"])
def test_a_review_failure_returns_the_prompt_unchanged_and_turns_keeper_off(tmp_path, monkeypatch, caplog, where):
    import inference.agent.tool_agent as ta

    agent, state_path, game, calls = _complete_a_level(tmp_path, monkeypatch, GOOD_REPLY)
    if where == "model request":
        monkeypatch.setattr(ta.ToolAgent, "_chat_completion", _boom)
    elif where == "memory.apply":
        monkeypatch.setattr(keeper_review.KeeperMemory, "apply", _boom)
    elif where == "check sandbox":
        monkeypatch.setattr(ta, "keeper_run_check", _boom)
    elif where == "notes injection":
        monkeypatch.setattr(keeper_review.KeeperMemory, "notes_block", _boom)
    elif where == "review timeout":
        monkeypatch.setattr(ta, "_KEEPER_REVIEW_BUDGET_MARGIN_SECONDS", -1e6)  # any review is over budget
    caplog.set_level(logging.WARNING)
    prompt = agent._keeper_before_turn(state_path, "LEVEL UP PROMPT", lambda *a: None, 900.0)
    if where == "review timeout":
        # the review finished, but past its budget: this game's Keeper is now off
        assert "over its" in agent._keeper_guard.disabled_reason
    else:
        assert prompt == "LEVEL UP PROMPT"
        assert agent._keeper_guard.disabled_reason.startswith("review/notes: RuntimeError")
    assert len(_disabled_lines(caplog)) == 1
    n = len(calls)
    assert agent._keeper_before_turn(state_path, "NEXT", lambda *a: None, 900.0) == "NEXT"
    _plays_on_without_keeper(agent, state_path, game, 0)
    assert len(calls) == n  # no further review requests


# ----- one game's failure does not touch another game ---------------------------------------------


def test_disabling_one_game_leaves_another_game_on(tmp_path, monkeypatch):
    a_agent, a_path, _a_game = make_agent(tmp_path / "a")
    b_agent, b_path, _b_game = make_agent(tmp_path / "b")
    a_agent._keeper_guard_for(a_path).disable("test", "injected")
    payload = run_code(b_agent, b_path, "action(['RIGHT'] * 6)")
    assert payload["surprises"] and b_agent._keeper_guard.disabled_reason is None


# ----- process-wide memory guard -------------------------------------------------------------------


def test_memory_pressure_turns_all_keeper_work_off_and_frees_its_histories(tmp_path, monkeypatch, caplog):
    agent, state_path, game = make_agent(tmp_path)
    run_code(agent, state_path, "action(['RIGHT'] * 3)")
    history = agent._keeper_history
    assert history.levels
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(keeper_guard.GLOBAL, "read_limit", lambda: 16 * 2**30)
    monkeypatch.setattr(keeper_guard.GLOBAL, "read_rss", lambda: 13 * 2**30)  # 81% > 75%
    keeper_guard.GLOBAL.check_memory(force=True)
    assert "process RSS 13.0 GiB is above 75% of 16.0 GiB" in keeper_guard.GLOBAL.disabled_reason
    assert history.levels == {}  # dropped
    payload = run_code(agent, state_path, "action(['RIGHT'] * 6)")
    assert _keeper_lines(payload) == {} and game.sent.count("RIGHT") == 9
    agent2, state_path2, _game2 = make_agent(tmp_path / "other")
    assert "surprises" not in run_code(agent2, state_path2, "action(['RIGHT'] * 6)")
    assert len([r for r in caplog.records if "for the rest of the run" in r.getMessage()]) == 1


def test_low_system_memory_also_turns_keeper_off(monkeypatch):
    monkeypatch.setattr(keeper_guard.GLOBAL, "read_rss", lambda: 2**30)
    monkeypatch.setattr(keeper_guard.GLOBAL, "read_limit", lambda: 64 * 2**30)
    monkeypatch.setattr(keeper_guard.GLOBAL, "read_meminfo", lambda: {"MemTotal": 64 * 2**30, "MemAvailable": 3 * 2**30})
    keeper_guard.GLOBAL.check_memory(force=True)
    assert "below 10%" in keeper_guard.GLOBAL.disabled_reason


# ----- a whole analyzer turn with a broken hook ----------------------------------------------------


def test_an_analyzer_turn_completes_and_keeps_acting_after_a_hook_failure(tmp_path, monkeypatch):
    from inference.agent import tool_agent as ta

    codes = ["action(['RIGHT'] * 3)", "action(['RIGHT'] * 3)", "action(['UP'])", None]
    seen = []

    def post(self, messages, **_k):
        seen.append(messages)
        code = codes[min(len(seen) - 1, len(codes) - 1)]
        message = {"role": "assistant", "content": "", "reasoning_content": "go"}
        if code:
            message["tool_calls"] = [{"id": f"c{len(seen)}", "type": "function",
                                      "function": {"name": "python", "arguments": json.dumps({"code": code})}}]
        return ta._ChatCompletionResult(message=message, finish_reason="tool_calls" if code else "stop",
                                        usage={"completion_tokens": 5, "prompt_tokens": 50})

    monkeypatch.setattr(ta.ToolAgent, "_post_chat_completion", post)
    monkeypatch.setattr(keeper_auto, "surprise_lines", _boom)
    monkeypatch.setenv("KEEPER_TIMING", "1")
    monkeypatch.setattr(keeper_resources.ResourceLogger, "start", _boom)  # even a broken resource log
    monkeypatch.setenv("KEEPER_RESOURCE_LOG", "1")
    agent, state_path, game = make_agent(tmp_path)
    for step in range(1, 4):  # one analyzer turn per executed action call
        agent.analyze(state_path, len(game.history) - 1, ["RIGHT", "UP", "MOUSE"], step_env=game,
                      transcript_path=tmp_path / f"t{step}.txt", analysis_step=step, request_timeout_seconds=900.0)
    assert game.sent == ["RIGHT"] * 6 + ["UP"]
    assert agent._keeper_guard.disabled_reason.startswith("surprise/perception: RuntimeError: injected failure")
