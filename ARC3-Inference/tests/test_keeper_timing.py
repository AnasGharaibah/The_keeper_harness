"""Keeper part 4: the per-turn timing log."""
from __future__ import annotations

import json
import random

import pytest

from inference.agent import keeper_timing
from keeper_fakes import clear_keeper_env, make_agent


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    clear_keeper_env(monkeypatch)
    monkeypatch.delenv("MULTIMODAL_CONTEXT", raising=False)


def _scripted(replies, calls):
    from inference.agent import tool_agent as ta

    def post(self, messages, *, tools, request_timeout_seconds=None, max_tokens=None, enable_thinking=None):
        calls.append(messages)
        code = replies[min(len(calls) - 1, len(replies) - 1)]
        message = {"role": "assistant", "content": "", "reasoning_content": "go"}
        if code:
            message["tool_calls"] = [{"id": f"c{len(calls)}", "type": "function",
                                      "function": {"name": "python", "arguments": json.dumps({"code": code})}}]
        return ta._ChatCompletionResult(message=message, finish_reason="tool_calls" if code else "stop",
                                        usage={"completion_tokens": 50, "prompt_tokens": 1200,
                                               "prompt_tokens_details": {"cached_tokens": 1000}})
    return post


def _one_turn(tmp_path, monkeypatch, replies):
    from inference.agent.tool_agent import ToolAgent

    calls = []
    monkeypatch.setattr(ToolAgent, "_post_chat_completion", _scripted(replies, calls))
    agent, state_path, game = make_agent(tmp_path)
    agent.analyze(state_path, 0, ["RIGHT", "UP", "MOUSE"], step_env=game, transcript_path=tmp_path / "t.txt",
                  analysis_step=1, request_timeout_seconds=900.0)
    return agent, state_path, calls


def test_switch_off_writes_no_log(tmp_path, monkeypatch):
    agent, state_path, calls = _one_turn(tmp_path, monkeypatch, ["action(['RIGHT'] * 6)", None])
    assert calls and not keeper_timing.timing_path(state_path).exists()
    assert getattr(agent, "_keeper_timer", None) is None


def test_switch_on_logs_one_line_per_turn_with_requests_tools_and_keeper_text(tmp_path, monkeypatch):
    monkeypatch.setenv("KEEPER_TIMING", "1")
    monkeypatch.setenv("KEEPER_SURPRISE", "1")
    agent, state_path, calls = _one_turn(tmp_path, monkeypatch, ["action(['RIGHT'] * 6)", None])
    (record,) = [json.loads(line) for line in keeper_timing.timing_path(state_path).read_text().splitlines()]
    assert record["analysis_step"] == 1 and record["action_num"] == 0
    assert [r["kind"] for r in record["requests"]] == ["turn"] * len(calls)
    first = record["requests"][0]
    assert (first["prompt_tokens"], first["cached_tokens"], first["generated_tokens"]) == (1200, 1000, 50)
    assert "first_token_seconds" not in first  # not streaming
    assert record["tool_seconds"] > 0 and record["keeper_seconds"] > 0 and record["turn_seconds"] >= record["tool_seconds"]
    assert record["keeper_chars"]["surprises"] > 0  # the blocked RIGHT at the wall
    assert getattr(agent, "_keeper_timer", None) is None


def test_stream_probe_notes_the_first_data_line():
    probe = {}
    lines = list(keeper_timing_lines(probe))
    assert lines == [b": keep-alive", b"data: {}", b"data: [DONE]"] and "at" in probe


def keeper_timing_lines(probe):
    from inference.agent.tool_agent import _note_first_data_line

    return _note_first_data_line(iter([b": keep-alive", b"data: {}", b"data: [DONE]"]), probe)


def test_summary_recovers_prefill_and_generation_costs():
    rng = random.Random(0)
    turns = []
    for _ in range(200):
        prompt, cached, gen = rng.randint(2000, 60000), 0, rng.randint(50, 6000)
        cached = rng.randint(0, prompt)
        seconds = 0.00004 * (prompt - cached) + 0.012 * gen + 0.3
        turns.append({"requests": [{"kind": "turn", "seconds": seconds, "prompt_tokens": prompt,
                                    "cached_tokens": cached, "generated_tokens": gen}],
                      "turn_seconds": seconds + 1, "tool_seconds": 1.0, "keeper_seconds": 0.1,
                      "gate_wait_seconds": 0.0, "keeper_chars": {"notes": 400, "surprises": 200, "perceived": 0}})
    summary = keeper_timing.summarize(turns)
    fit = summary["fit_split"]
    assert fit["seconds_per_uncached_prompt_token"] == pytest.approx(0.00004, rel=1e-6)
    assert fit["seconds_per_generated_token"] == pytest.approx(0.012, rel=1e-6)
    assert summary["keeper_tokens_per_turn_est"] == pytest.approx(150.0)
