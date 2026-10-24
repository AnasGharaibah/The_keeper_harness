"""Keeper on odd inputs: hidden games may have any ID format, missing metadata,
other board sizes, other action sets and empty histories. Whatever the input,
nothing may raise out of the Keeper, and the game plays on.

Also: Keeper code outside a game's hooks (startup, config, timing, the tool
result glue) and the last-resort switch (a failure before a game's first turn
has finished turns the Keeper off for the whole run)."""
from __future__ import annotations

import json
import random

import pytest

from inference.agent import keeper_auto, keeper_guard, keeper_history, keeper_perception, keeper_review, keeper_timing
from inference.agent.runtime_state import RUNTIME_STATE_FILENAME, Frame, HistoryEntry, write_runtime_state
from inference.utils.grid_utils import ARC_COLOR_CHARS
from keeper_fakes import clear_keeper_env, run_code

ALL = ("KEEPER_SURPRISE", "KEEPER_REVIEW", "KEEPER_PERCEPTION", "KEEPER_TIMING", "KEEPER_RESOURCE_LOG")

ODD_IDS = [
    "k000", "ls20", "ls20-9607627b", "a/b/c", "..", "ünïcødé-游戏", "x" * 300,
    "123e4567-e89b-12d3-a456-426614174000", "space id", "",
]
ODD_ACTIONS = [
    ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE", "UNDO"],
    ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"],
    ["MOUSE"],
    ["ACTION6"],
    ["", "RESET", "???"],
]
SHAPES = [(64, 64), (1, 1), (5, 9), (30, 45), (128, 128), (64, 1), (0, 0)]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    clear_keeper_env(monkeypatch)
    monkeypatch.delenv("MULTIMODAL_CONTEXT", raising=False)
    for key in ALL:
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("KEEPER_RESOURCE_LOG_INTERVAL_SECONDS", "1")
    yield
    keeper_guard.GLOBAL.reset()


def _grid(rng: random.Random, shape: tuple[int, int], *, odd_values: bool) -> tuple[tuple[int, ...], ...]:
    h, w = shape
    values = [0, 1, 2, 3, 4, 5, 7, 9, 15] + ([16, 255, -1, 1000] if odd_values else [])
    rows = [[rng.choice(values[:4]) for _ in range(w)] for _ in range(h)]
    for _ in range(rng.randint(0, 4)):  # a few blocks so there are objects to track
        if h > 2 and w > 2:
            r, c, v = rng.randrange(h - 2), rng.randrange(w - 2), rng.choice(values)
            for dr in range(2):
                for dc in range(2):
                    rows[r + dr][c + dc] = v
    if odd_values and h and w and rng.random() < 0.3:
        rows[rng.randrange(h)] = rows[0][: max(0, w - 1)]  # a ragged row
    return tuple(tuple(row) for row in rows)


class OddGame:
    """Answers action calls like the solver does, with odd boards, levels and results."""

    def __init__(self, state_path, rng: random.Random, actions: list[str], *, odd_values: bool, shape_changes: bool):
        self.state_path, self.rng, self.actions = state_path, rng, actions
        self.odd_values, self.shape_changes = odd_values, shape_changes
        self.shape = rng.choice(SHAPES)
        self.level = 1
        self.history = [HistoryEntry(action="", frame=Frame(grid=_grid(rng, self.shape, odd_values=odd_values), step=0, level=1))]
        self.sent: list[str] = []
        self._write()

    def _write(self):
        write_runtime_state(self.state_path, current_frame=self.history[-1].frame, history=self.history)

    def __call__(self, arguments):
        if arguments.get("query") == "animation":
            return {"record": None}
        per_action, executed = [], []
        for item in arguments["actions"]:
            name = str(item.get("action"))
            display = name if name != "MOUSE" else f"MOUSE(row={item.get('row')}, col={item.get('col')})"
            self.sent.append(display)
            step = len(self.history)
            if self.shape_changes and self.rng.random() < 0.2:
                self.shape = self.rng.choice(SHAPES)
            completed = self.rng.random() < 0.15
            over = not completed and self.rng.random() < 0.05
            if completed:
                self.level += self.rng.choice([1, 1, 2])
            grid = self.history[-1].frame.grid if self.rng.random() < 0.3 else _grid(self.rng, self.shape, odd_values=self.odd_values)
            frame = Frame(grid=grid, step=step, level=self.level)
            result = {"action_num": step, "action_display": display, "board_changed": True,
                      "gameplay_changed": self.rng.random() < 0.5, "level_completed": completed, "game_over": over,
                      "run_complete": False}
            if self.rng.random() < 0.2:
                result = {}  # results with fields missing
            self.history.append(HistoryEntry(action=display, frame=frame, result=dict(result)))
            self._write()
            per_action.append(result)
            executed.append(display)
            if completed or over:
                break
        last = per_action[-1]
        return {
            "executed": True, "action_num": len(self.history) - 1, "level": self.level, "state": "NOT_FINISHED",
            "valid_actions": self.actions, "board_changed": True, "gameplay_changed": True,
            "level_completed": bool(last.get("level_completed")), "game_over": bool(last.get("game_over")),
            "run_complete": False, "done": False, "action_display": executed[-1], "executed_actions": executed,
            "requested_count": len(arguments["actions"]), "executed_count": len(per_action),
            "stopped_early": len(per_action) < len(arguments["actions"]), "per_action": per_action,
        }


def _agent(tmp_path, game_id: str, actions: list[str], rng, **game_kwargs):
    from inference.agent.tool_agent import ToolAgent
    from inference.framework.solver import artifact_stem

    agent = ToolAgent(model="test-model")
    stem = artifact_stem(f"{game_id}_p0") if game_id else "_p0"
    state_path = tmp_path / "artifacts" / f"{stem[:200]}_{RUNTIME_STATE_FILENAME}"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    game = OddGame(state_path, rng, actions, **game_kwargs)
    agent._ensure_session(state_path)
    agent._step_env_callback = game
    agent._current_valid_actions = actions
    return agent, state_path, game


def _play_code(actions: list[str], n: int) -> str:
    picks = []
    for i in range(n):
        a = actions[i % len(actions)]
        picks.append({"action": "MOUSE", "row": (i * 7) % 70, "col": (i * 13) % 70} if a == "MOUSE" else a)
    return f"action({json.dumps(picks)})"


KEEPER_FIELDS = ("surprises", "perceived")


def _play(tmp_path, monkeypatch, seed: int, keeper_on: bool):
    """Eight tool calls and turn starts on one odd game; returns what the game was
    sent and each tool result without the Keeper's own fields."""
    import inference.agent.tool_agent as ta

    if not keeper_on:
        for key in ALL:
            monkeypatch.delenv(key, raising=False)
    keeper_guard.GLOBAL.reset()
    reviews = []

    def chat(self, messages, **_k):
        reviews.append(1)
        reply = random.Random(seed + len(reviews)).choice([
            "", "not json", "{}", json.dumps({"cause": None, "hypotheses": None}),
            json.dumps({"cause": "c" * 5000, "hypotheses": [{"text": "t", "kind": "rule", "check_code": "def check_transition(t):\n    return t.after.shape"}]}),
            json.dumps({"hypotheses": [{"text": 1, "kind": [], "check_code": 5}, "junk", None]}),
        ])
        return ta._ChatCompletionResult(message={"content": reply}, finish_reason="stop", usage={"completion_tokens": 3})

    monkeypatch.setattr(ta.ToolAgent, "_chat_completion", chat)
    rng = random.Random(seed)
    game_id = ODD_IDS[seed % len(ODD_IDS)]
    actions = ODD_ACTIONS[seed % len(ODD_ACTIONS)]
    agent, state_path, game = _agent(tmp_path / ("on" if keeper_on else "off"), game_id, actions, rng,
                                     odd_values=seed % 2 == 1, shape_changes=seed % 3 == 0)
    results = []
    for call in range(8):
        n = rng.randint(1, 5)
        payload = run_code(agent, state_path, _play_code(actions, n))
        assert isinstance(payload, dict)
        results.append(json.dumps({k: v for k, v in payload.items() if k not in KEEPER_FIELDS}, sort_keys=True, default=str))
        if keeper_on:
            prompt = agent._keeper_before_turn(state_path, f"PROMPT {call}", lambda *a: None, 900.0)
            assert isinstance(prompt, str) and prompt.endswith(f"PROMPT {call}")
            agent._keeper_guard_for(state_path).finish_turn()
    return list(game.sent), results, keeper_guard.GLOBAL.disabled_reason


@pytest.mark.parametrize("seed", range(24))
def test_odd_games_never_raise_and_play_exactly_as_with_keeper_off(tmp_path, monkeypatch, seed):
    sent_on, results_on, _reason = _play(tmp_path, monkeypatch, seed, keeper_on=True)
    sent_off, results_off, _ = _play(tmp_path, monkeypatch, seed, keeper_on=False)
    assert sent_on == sent_off
    assert results_on == results_off


@pytest.mark.parametrize("shape", SHAPES)
def test_hooks_directly_on_odd_boards_return_or_raise_inside_the_guard(shape):
    rng = random.Random(len(shape))
    guard = keeper_guard.GameGuard("odd")
    guard.finish_turn()
    entries = [HistoryEntry(action="", frame=Frame(grid=_grid(rng, shape, odd_values=True), step=0, level=1))]
    for step in range(1, 12):
        entries.append(HistoryEntry(action=rng.choice(["UP", "MOUSE(row=3, col=4)", "ACTION7", "RESET", ""]),
                                    frame=Frame(grid=_grid(rng, rng.choice(SHAPES), odd_values=True), step=step, level=1 + step // 5)))
    record = keeper_history.KeeperHistory()
    guard.call("sync", lambda: record.sync(entries))
    for level in list(record.levels):
        ts = record.transitions(level)
        guard.call("surprise", lambda: keeper_auto.surprise_lines(ts, [t["step"] for t in ts], border=4, max_lines=3,
                                                                    color_chars=ARC_COLOR_CHARS, effect_cache=record.effects))
        tracker = keeper_perception.PerceptionTracker(min_size=3, max_lines=3)
        for t in ts:
            guard.call("perception", lambda t=t: tracker.observe(t, set(), level=level))
        memory = keeper_review.KeeperMemory(min_support=2, max_open=5, max_hypotheses=5)
        guard.call("standalone text", lambda: memory.standalone_text(level, ts, border=4, max_moves=20,
                                                                      color_chars=ARC_COLOR_CHARS, surprises=[], reasoning=""))


def test_empty_and_missing_histories(tmp_path):
    record = keeper_history.KeeperHistory()
    assert record.sync([]) == []
    assert record.sync([HistoryEntry(action="UP", frame=None)]) == []
    assert record.transitions(1) == [] and record.levels_of([]) == []


# ----- Keeper code outside the hooks --------------------------------------------------------------


def test_a_config_failure_at_startup_turns_keeper_off_and_the_agent_works(tmp_path, monkeypatch):
    from inference.agent import keeper_config
    from inference.agent.tool_agent import ToolAgent

    def broken(cls):
        raise ValueError("bad config")

    monkeypatch.setattr(keeper_config.KeeperConfig, "from_env", classmethod(broken))
    agent = ToolAgent(model="test-model")
    assert not agent._keeper.any_switch
    assert "config: ValueError" in keeper_guard.GLOBAL.disabled_reason
    assert "KEEPER NOTES" not in agent._system_prompt and "`surprises`" not in agent._system_prompt


def test_a_tool_result_glue_failure_leaves_the_payload_as_the_baseline(tmp_path, monkeypatch):
    import inference.agent.tool_agent as ta

    rng = random.Random(1)
    agent, state_path, game = _agent(tmp_path, "g", ["UP", "RIGHT"], rng, odd_values=False, shape_changes=False)
    agent._keeper_guard_for(state_path).finish_turn()
    monkeypatch.setattr(ta.ToolAgent, "_keeper_tool_result", lambda *a, **k: (_ for _ in ()).throw(KeyError("glue")))
    payload = run_code(agent, state_path, "action(['UP', 'RIGHT'])")
    assert "surprises" not in payload and "perceived" not in payload and game.sent[0] == "UP"
    assert keeper_guard.GLOBAL.disabled_reason.startswith("tool result: KeyError")


def test_a_timing_failure_never_changes_the_timed_block(monkeypatch):
    timer = keeper_timing.TurnTimer(path=None)  # type: ignore[arg-type]
    errors = []
    monkeypatch.setattr(timer, "add", lambda *a: (_ for _ in ()).throw(KeyError("timer")))
    with pytest.raises(ZeroDivisionError):  # the block's own error passes through
        with keeper_timing.measured(timer, "tool_seconds", on_error=errors.append):
            1 / 0
    with keeper_timing.measured(timer, "tool_seconds", on_error=errors.append):
        pass
    assert len(errors) == 2


def test_a_timing_flush_failure_after_a_turn_keeps_the_turn_result(tmp_path, monkeypatch):
    import inference.agent.tool_agent as ta

    monkeypatch.setattr(ta.ToolAgent, "_analyze_turn", lambda self, *a, **k: "TURN RESULT")
    monkeypatch.setattr(keeper_timing.TurnTimer, "flush", lambda self: (_ for _ in ()).throw(OSError("disk")))
    rng = random.Random(2)
    agent, state_path, _game = _agent(tmp_path, "g", ["UP"], rng, odd_values=False, shape_changes=False)
    assert agent.analyze(state_path, 0, ["UP"]) == "TURN RESULT"
    assert keeper_guard.GLOBAL.disabled_reason.startswith("turn end: OSError")
    assert agent.analyze(state_path, 1, ["UP"]) == "TURN RESULT"  # later turns: plain baseline path


def test_a_turn_start_failure_runs_the_turn_without_keeper(tmp_path, monkeypatch):
    import inference.agent.tool_agent as ta

    seen = []
    monkeypatch.setattr(ta.ToolAgent, "_analyze_turn", lambda self, *a, **k: seen.append(getattr(self, "_keeper_timer", None)) or "OK")
    monkeypatch.setattr(ta.ToolAgent, "_keeper_turn_start", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("start")))
    rng = random.Random(3)
    agent, state_path, _game = _agent(tmp_path, "g", ["UP"], rng, odd_values=False, shape_changes=False)
    assert agent.analyze(state_path, 0, ["UP"]) == "OK"
    assert keeper_guard.GLOBAL.disabled_reason.startswith("turn start: RuntimeError")


def test_a_harness_error_in_the_turn_still_propagates_unchanged(tmp_path, monkeypatch):
    import inference.agent.tool_agent as ta

    monkeypatch.setattr(ta.ToolAgent, "_analyze_turn", lambda self, *a, **k: (_ for _ in ()).throw(ConnectionError("endpoint")))
    rng = random.Random(4)
    agent, state_path, _game = _agent(tmp_path, "g", ["UP"], rng, odd_values=False, shape_changes=False)
    with pytest.raises(ConnectionError):
        agent.analyze(state_path, 0, ["UP"])
    assert keeper_guard.GLOBAL.disabled_reason is None  # not a Keeper error


# ----- last resort: an early failure turns the Keeper off for the run -------------------------------


def test_a_failure_before_the_first_turn_finished_turns_keeper_off_for_every_game(tmp_path):
    a = keeper_guard.GameGuard("a")
    a.call("hook", lambda: 1 / 0)
    assert a.disabled_reason and "a failed before its first turn finished" in keeper_guard.GLOBAL.disabled_reason
    assert not keeper_guard.GameGuard("b").active


def test_a_later_failure_turns_keeper_off_for_that_game_only(tmp_path):
    a = keeper_guard.GameGuard("a")
    a.finish_turn()
    a.call("hook", lambda: 1 / 0)
    assert a.disabled_reason and keeper_guard.GLOBAL.disabled_reason is None
    assert keeper_guard.GameGuard("b").active


def test_after_the_switch_the_tool_result_is_the_baseline(tmp_path):
    rng = random.Random(5)
    agent, state_path, game = _agent(tmp_path, "g", ["UP", "RIGHT"], rng, odd_values=False, shape_changes=False)
    keeper_guard.GLOBAL.disable("test")
    for _ in range(3):
        payload = run_code(agent, state_path, "action(['UP', 'RIGHT', 'UP'])")
        assert "surprises" not in payload and "perceived" not in payload
    assert agent._keeper_before_turn(state_path, "P", lambda *a: None, 900.0) == "P"
    assert getattr(agent, "_keeper_history", None) is None  # no Keeper work at all


# ----- Keeper file names from any game id -----------------------------------------------------------


def test_keeper_file_names_are_safe_short_and_distinct_for_any_id(tmp_path):
    names = set()
    for game_id in ODD_IDS + ["a/b", "ünï", "\udcff"]:
        state = tmp_path / f"{game_id.replace('/', '_')}_p0_tool_runtime_state.json"
        path = keeper_guard.artifact_path(state, "_keeper_facts.json")
        assert path.parent == tmp_path and len(path.name) <= 160
        assert all(ch.isascii() and (ch.isalnum() or ch in "_.-") for ch in path.name)
        names.add(path.name)
        path.write_text("{}")  # the file system accepts it
    assert len(names) == len(ODD_IDS) + 3


def test_public_game_file_names_are_unchanged(tmp_path):
    state = tmp_path / "lp85-305b61c3_p0_tool_runtime_state.json"
    assert keeper_guard.artifact_path(state, "_keeper_facts.json").name == "lp85-305b61c3_p0_tool_runtime_state_keeper_facts.json"
    assert keeper_timing.timing_path(state).name == "lp85-305b61c3_p0_tool_runtime_state_timing.jsonl"


@pytest.mark.parametrize("ending", ["game_over", "level_completed", "run_complete"])
def test_a_game_that_ends_on_its_first_action(tmp_path, monkeypatch, ending):
    import inference.agent.tool_agent as ta

    monkeypatch.setattr(ta.ToolAgent, "_chat_completion", lambda self, m, **k: ta._ChatCompletionResult(
        message={"content": "{}"}, finish_reason="stop", usage={}))
    rng = random.Random(7)
    agent, state_path, game = _agent(tmp_path, "end1", ["UP"], rng, odd_values=False, shape_changes=False)
    real = game.__call__

    def first_action_ends(arguments):
        out = real(arguments)
        out[ending] = True
        if out["per_action"]:
            out["per_action"][-1][ending] = True
        game.history[-1].result[ending] = True
        return out

    agent._step_env_callback = first_action_ends
    payload = run_code(agent, state_path, "action(['UP'])")
    assert isinstance(payload, dict) and game.sent == ["UP"]
    assert agent._keeper_before_turn(state_path, "P", lambda *a: None, 900.0).endswith("P")
    assert keeper_guard.GLOBAL.disabled_reason is None


def test_a_reset_before_any_move(tmp_path):
    rng = random.Random(8)
    grid = _grid(rng, (64, 64), odd_values=False)
    record = keeper_history.KeeperHistory()
    entries = [HistoryEntry(action="", frame=Frame(grid=grid, step=0, level=1)),
               HistoryEntry(action="RESET", frame=Frame(grid=grid, step=1, level=1))]
    assert record.sync(entries) == []
    entries.append(HistoryEntry(action="UP", frame=Frame(grid=_grid(rng, (64, 64), odd_values=False), step=2, level=1)))
    assert record.sync(entries) == [2]
