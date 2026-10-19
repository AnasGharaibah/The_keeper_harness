"""ToolAgent integration of the layers: prompts, tool results, review, pinned facts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inference.agent.layer_config import LevelReviewConfig, PredictionCheckConfig
from inference.agent.prediction import compare_expectation
from inference.agent.runtime_state import RUNTIME_STATE_FILENAME, Frame, HistoryEntry, write_runtime_state
from inference.agent.tool_agent import ToolAgent, _build_system_prompt, _PYTHON_TOOL_DESCRIPTION

OFF_PREDICTION = PredictionCheckConfig(enabled=False)
OFF_REVIEW = LevelReviewConfig(enabled=False)
ON_PREDICTION = PredictionCheckConfig(enabled=True)
ON_REVIEW = LevelReviewConfig(enabled=True, llm_review=False)


@pytest.fixture(autouse=True)
def _no_image_context(monkeypatch):
    monkeypatch.delenv("MULTIMODAL_CONTEXT", raising=False)


def _agent(prediction=OFF_PREDICTION, review=OFF_REVIEW) -> ToolAgent:
    return ToolAgent(model="test-model", prediction_check=prediction, level_review=review)


def _frame(col: int, *, step: int, level: int = 1) -> Frame:
    grid = [[0] * 6 for _ in range(6)]
    grid[1][col] = 9
    return Frame(grid=tuple(tuple(row) for row in grid), step=step, level=level)


class FakeEnv:
    """Mimics HarnessSolver.step_env: RIGHT moves the blue cell; the harness checks `expect`."""

    def __init__(self, state_path: Path, *, complete_level_at: int | None = None) -> None:
        self.state_path = state_path
        self.col = 0
        self.level = 1
        self.history = [HistoryEntry(action="", frame=_frame(0, step=0))]
        self.complete_level_at = complete_level_at
        self.calls: list[dict[str, Any]] = []
        self._write()

    def _write(self) -> None:
        write_runtime_state(self.state_path, current_frame=self.history[-1].frame, history=self.history)

    def __call__(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(arguments)
        before = self.history[-1].frame
        name = arguments["actions"][0]["action"]
        if name == "RIGHT":
            self.col += 1
        step = len(self.history)
        level_completed = self.complete_level_at == step
        if level_completed:
            self.level += 1
        after = _frame(self.col, step=step, level=self.level)
        self.history.append(HistoryEntry(action=name, frame=after))
        self._write()
        payload = {
            "executed": True,
            "action_num": step,
            "level": self.level,
            "board_changed": before.grid != after.grid,
            "level_completed": level_completed,
            "game_over": False,
            "run_complete": False,
            "valid_actions": ["UP", "RIGHT"],
            "executed_actions": [name],
            "action_display": name,
        }
        payload["per_action"] = [dict(payload)]
        if "expect" in arguments:
            payload["prediction"] = compare_expectation(
                arguments["expect"], before_grid=before.grid, after_grid=after.grid, result=payload
            )
        return payload


def _setup(tmp_path: Path, agent: ToolAgent, **env_kwargs: Any) -> tuple[Path, FakeEnv]:
    state_path = tmp_path / "artifacts" / f"game_p0_{RUNTIME_STATE_FILENAME}"
    env = FakeEnv(state_path, **env_kwargs)
    agent._ensure_session(state_path)
    agent._step_env_callback = env
    agent._current_valid_actions = ["UP", "RIGHT"]
    return state_path, env


def _run(agent: ToolAgent, state_path: Path, code: str) -> dict[str, Any]:
    dispatch = agent._run_python_tool(state_path, {"code": code})
    return json.loads(dispatch.content)


def test_layers_off_keep_baseline_prompt_and_tool_description(tmp_path):
    agent = _agent()
    assert agent._system_prompt == _build_system_prompt(tool_output_tokens=agent._tool_output_tokens)
    assert agent._tool_description == _PYTHON_TOOL_DESCRIPTION
    assert "Prediction checks:" not in agent._system_prompt
    assert "hypothesis(" not in agent._system_prompt
    assert agent._sandbox_layer_state(1) == {}
    state_path, _env = _setup(tmp_path, agent)
    assert agent._system_message_content() == agent._system_prompt


def test_layers_on_extend_prompt_and_tool_description():
    agent = _agent(ON_PREDICTION, ON_REVIEW)
    assert "Prediction checks:" in agent._system_prompt
    assert "Rules and goals memory:" in agent._system_prompt
    assert "expect=" in agent._tool_description and "hypothesis(" in agent._tool_description


def test_expect_ignored_when_prediction_layer_off(tmp_path):
    agent = _agent()
    state_path, env = _setup(tmp_path, agent)
    payload = _run(agent, state_path, "action(['RIGHT'], expect={'board_changed': False})")
    assert "expect" not in env.calls[0]
    assert "prediction_check" not in payload
    assert agent.drain_layer_events() == []


def test_prediction_check_block_in_tool_result(tmp_path):
    agent = _agent(ON_PREDICTION)
    state_path, env = _setup(tmp_path, agent)
    payload = _run(
        agent,
        state_path,
        "action(['RIGHT'], expect={'board_changed': True})\n"
        "action(['UP'], expect={'board_changed': True})\n"
        "action(['UP'])\n"
        "result = [last_prediction['verdict'], [p['verdict'] for p in prediction_log]]",
    )
    assert env.calls[0]["expect"] == {"board_changed": True}
    assert payload["prediction_check"] == [
        "#1 RIGHT -> confirmed (board_changed)",
        "#2 UP -> WRONG: board_changed=False, expected True",
        "#3 UP -> no_prediction (no expect=)",
    ]
    assert payload["result"] == ["no_prediction", ["confirmed", "wrong", "no_prediction"]]
    (event,) = [event for event in agent.drain_layer_events() if event["type"] == "prediction_check"]
    assert event["type"] == "prediction_check"
    assert event["counts"] == {"confirmed": 1, "wrong": 1, "no_prediction": 1}


def test_hypotheses_recorded_reviewed_and_pinned(tmp_path):
    agent = _agent(ON_PREDICTION, ON_REVIEW)
    state_path, _env = _setup(tmp_path, agent, complete_level_at=4)
    check = (
        "def check_transition(t):\n"
        "    if parse_action(t.action)[0] != 'RIGHT':\n"
        "        return None\n"
        "    moved = [c for r, c, old, new in changed_cells(t.before_frame, t.after_frame) if new == 'b']\n"
        "    left = [c for r, c, old, new in changed_cells(t.before_frame, t.after_frame) if old == 'b']\n"
        "    return len(moved) == 1 and len(left) == 1 and moved[0] == left[0] + 1\n"
    )
    payload = _run(
        agent,
        state_path,
        f"r = hypothesis('RIGHT moves the blue cell one column right', check_code={check!r})\n"
        "for _ in range(4):\n"
        "    out = action(['RIGHT'])\n"
        "    if out.get('level_completed'):\n"
        "        break\n"
        "result = r['id']",
    )
    assert payload["result"] == "H1"
    assert payload["hypotheses_recorded"][0].startswith("H1 [rule] recorded")
    assert agent._level_memory.has_pending()

    notices = agent._run_pending_reviews(state_path, time_cap_seconds=60)
    assert notices and "new facts: H1" in notices[0]
    assert agent._level_memory.get("H1").status == "fact"
    system = agent._system_message_content()
    assert system.startswith(agent._system_prompt)
    assert "VERIFIED FACTS" in system and "H1 [rule] RIGHT moves the blue cell" in system
    facts_file = state_path.with_name("game_p0_facts.json")
    assert json.loads(facts_file.read_text())["hypotheses"][0]["status"] == "fact"
    review_events = [event for event in agent.drain_layer_events() if event["type"] == "level_review"]
    assert review_events and review_events[0]["promoted"] == ["H1"]


def test_pinned_facts_block_survives_eviction(tmp_path):
    agent = _agent(review=ON_REVIEW)
    state_path, _env = _setup(tmp_path, agent)
    memory = agent._level_memory
    memory.add_hypotheses([{"id": "H1", "text": "RIGHT moves blue right", "kind": "rule", "level": 1}], default_level=1)
    fact = memory.get("H1")
    fact.status, fact.support, fact.level_learned = "fact", 7, 1
    agent._history_messages = [
        {"role": "user", "content": "u" * 3000},
        {"role": "assistant", "content": "a" * 3000},
    ] * 6
    tools = agent._tools(state_path)
    messages = agent._initial_messages({"role": "user", "content": "current turn"})
    agent._context_budget_tokens = (
        agent._estimate_request_input_tokens([messages[0], messages[-1]], tools=tools) + 50
    )
    trimmed = agent._trim_messages_for_context(messages, tools=tools)
    assert len(trimmed) < len(messages)
    assert trimmed[0]["role"] == "system"
    assert "VERIFIED FACTS" in trimmed[0]["content"]
    assert "H1 [rule] RIGHT moves blue right" in trimmed[0]["content"]
    assert trimmed[-1]["content"] == "current turn"


def test_finalize_reviews_level_in_progress(tmp_path):
    agent = _agent(review=ON_REVIEW)
    state_path, _env = _setup(tmp_path, agent)
    _run(agent, state_path, "hypothesis('UP does nothing')\naction(['RIGHT'])")
    agent.finalize(state_path, request_timeout_seconds=30, allow_llm=False)
    stats = json.loads(state_path.with_name("game_p0_facts.json").read_text())["level_stats"]
    assert stats[-1]["reason"] == "run_end"
    assert stats[-1]["llm"] == "disabled"


BASELINE_PREDICTION = PredictionCheckConfig(enabled=True, auto_predict=False, nudge_after_actions=0)


def test_new_prediction_switches_off_add_nothing(tmp_path):
    agent = _agent(BASELINE_PREDICTION)
    assert "SURPRISE" not in agent._system_prompt and "no_prediction" not in agent._system_prompt
    state_path, _env = _setup(tmp_path, agent)
    payload = _run(agent, state_path, "for _ in range(4):\n    action(['RIGHT'])\naction(['UP'])\naction(['UP'])")
    assert "auto_prediction" not in payload and "prediction_warning" not in payload
    assert [event["type"] for event in agent.drain_layer_events()] == ["prediction_check"]
    assert agent._auto_entries == []


def test_auto_prediction_reported_in_tool_result(tmp_path):
    agent = _agent(ON_PREDICTION)
    assert "SURPRISE:" in agent._system_prompt and "no_prediction" in agent._system_prompt
    state_path, _env = _setup(tmp_path, agent)
    payload = _run(agent, state_path, "action(['RIGHT'])\naction(['RIGHT'])\nresult = last_action_result['auto_prediction']")
    assert payload["auto_prediction"][0].startswith("auto: 1 as before / 0 surprise / 1 first time")
    assert payload["result"][0].startswith("auto: 1 as before")
    assert [entry["source"] for entry in agent._auto_entries] == ["auto", "auto"]
    # UP does nothing on this board; RIGHT moved last time, so a no-op RIGHT would surprise.
    payload = _run(agent, state_path, "action(['UP'])\naction(['UP'])")
    assert payload["auto_prediction"][0].startswith("auto: 1 as before / 0 surprise / 1 first time")
    events = [event for event in agent.drain_layer_events() if event["type"] == "auto_prediction"]
    assert len(events) == 2 and events[0]["counts"] == {"as_before": 1, "surprise": 0, "first_time": 1}


def test_nudge_after_consecutive_moves_without_expect(tmp_path):
    agent = _agent(PredictionCheckConfig(enabled=True, nudge_after_actions=3))
    state_path, _env = _setup(tmp_path, agent)
    payload = _run(agent, state_path, "action(['UP'])\naction(['UP'])")
    assert "prediction_warning" not in payload
    payload = _run(agent, state_path, "action(['UP'])")
    assert payload["prediction_warning"].startswith("NO PREDICTION FOR LAST 3 MOVES")
    # An expect= resets the counter.
    payload = _run(agent, state_path, "action(['UP'])\naction(['UP'])\naction(['UP'], expect={'board_changed': False})")
    assert "prediction_warning" not in payload
    assert agent._moves_without_expect == 0
    assert any(event["type"] == "prediction_nudge" for event in agent.drain_layer_events())
