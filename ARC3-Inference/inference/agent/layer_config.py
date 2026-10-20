"""Switches for the optional duck layers (prediction check, level review).

Values come from ``LOCAL_ANALYZER_*`` env vars, which the Makefile fills from
``analyzer.prediction_check.*`` / ``analyzer.level_review.*`` in the JSON config.
Everything defaults to disabled so configs without these keys behave exactly as
before.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True)
class PredictionCheckConfig:
    enabled: bool = False
    max_wrong_diffs_shown: int = 5
    sandbox_log_size: int = 20
    # Loud reminder after this many consecutive executed actions without expect= (0 = off).
    nudge_after_actions: int = 5
    # Harness predicts "same effect as the last same action on this level" (no LLM).
    auto_predict: bool = True
    # Cells changing in at least this fraction of a level's transitions are HUD and ignored.
    hud_change_fraction: float = 0.8
    max_surprises_shown: int = 5  # distinct surprise kinds shown per call
    # Perception lines (transformations, shape matches, roles) in auto_prediction.
    perception: bool = True
    perception_min_size: int = 3
    perception_max_lines: int = 3
    # Label in-level resets (board back to level start) as RESET, never as an action effect.
    reset_detection: bool = True
    # Per-step summary of multi-action calls ("LEFT x15: blocked -> step 48: RESET -> ...").
    batch_summary: bool = True
    batch_summary_max_chars: int = 400
    # Early stop of a multi-action call; the remaining actions are not sent to the game.
    batch_stop: bool = True
    stop_on_reset: bool = True
    stop_after_blocked: int = 2  # consecutive blocked moves (0 = off)
    stop_on_surprise_repeated: bool = True  # batch of one repeated action
    stop_on_surprise_mixed: bool = False  # batch of mixed actions

    @classmethod
    def from_env(cls) -> "PredictionCheckConfig":
        defaults = cls()
        return cls(
            enabled=_env_bool("LOCAL_ANALYZER_PREDICTION_CHECK_ENABLED", False),
            max_wrong_diffs_shown=max(0, _env_int("LOCAL_ANALYZER_PREDICTION_CHECK_MAX_WRONG_DIFFS", 5)),
            sandbox_log_size=max(0, _env_int("LOCAL_ANALYZER_PREDICTION_CHECK_SANDBOX_LOG_SIZE", 20)),
            nudge_after_actions=max(
                0, _env_int("LOCAL_ANALYZER_PREDICTION_CHECK_NUDGE_AFTER_ACTIONS", defaults.nudge_after_actions)
            ),
            auto_predict=_env_bool("LOCAL_ANALYZER_PREDICTION_CHECK_AUTO_PREDICT", defaults.auto_predict),
            hud_change_fraction=min(
                1.0,
                max(0.0, _env_float("LOCAL_ANALYZER_PREDICTION_CHECK_HUD_CHANGE_FRACTION", defaults.hud_change_fraction)),
            ),
            max_surprises_shown=max(
                0, _env_int("LOCAL_ANALYZER_PREDICTION_CHECK_MAX_SURPRISES_SHOWN", defaults.max_surprises_shown)
            ),
            perception=_env_bool("LOCAL_ANALYZER_PREDICTION_CHECK_PERCEPTION", defaults.perception),
            perception_min_size=max(
                1, _env_int("LOCAL_ANALYZER_PREDICTION_CHECK_PERCEPTION_MIN_SIZE", defaults.perception_min_size)
            ),
            perception_max_lines=max(
                0, _env_int("LOCAL_ANALYZER_PREDICTION_CHECK_PERCEPTION_MAX_LINES", defaults.perception_max_lines)
            ),
            reset_detection=_env_bool("LOCAL_ANALYZER_PREDICTION_CHECK_RESET_DETECTION", defaults.reset_detection),
            batch_summary=_env_bool("LOCAL_ANALYZER_PREDICTION_CHECK_BATCH_SUMMARY", defaults.batch_summary),
            batch_summary_max_chars=max(
                0, _env_int("LOCAL_ANALYZER_PREDICTION_CHECK_BATCH_SUMMARY_MAX_CHARS", defaults.batch_summary_max_chars)
            ),
            batch_stop=_env_bool("LOCAL_ANALYZER_PREDICTION_CHECK_BATCH_STOP", defaults.batch_stop),
            stop_on_reset=_env_bool("LOCAL_ANALYZER_PREDICTION_CHECK_STOP_ON_RESET", defaults.stop_on_reset),
            stop_after_blocked=max(
                0, _env_int("LOCAL_ANALYZER_PREDICTION_CHECK_STOP_AFTER_BLOCKED", defaults.stop_after_blocked)
            ),
            stop_on_surprise_repeated=_env_bool(
                "LOCAL_ANALYZER_PREDICTION_CHECK_STOP_ON_SURPRISE_REPEATED", defaults.stop_on_surprise_repeated
            ),
            stop_on_surprise_mixed=_env_bool(
                "LOCAL_ANALYZER_PREDICTION_CHECK_STOP_ON_SURPRISE_MIXED", defaults.stop_on_surprise_mixed
            ),
        )


_WORLD_MODEL_PREFIX = "LOCAL_ANALYZER_WORLD_MODEL_"


@dataclass(frozen=True)
class WorldModelConfig:
    """``world_model(text)`` / ``plan(text)`` in the sandbox, stored by the harness
    and re-injected every turn."""

    enabled: bool = False
    # Reminder in the user prompt after this many turns without world_model() (0 = off).
    nudge_after_turns: int = 3
    # Without a world_model() call, take the last "World model:" / "Plan:" lines from the reasoning.
    reasoning_fallback: bool = True

    @classmethod
    def from_env(cls) -> "WorldModelConfig":
        defaults = cls()
        return cls(
            enabled=_env_bool(f"{_WORLD_MODEL_PREFIX}ENABLED", defaults.enabled),
            nudge_after_turns=max(0, _env_int(f"{_WORLD_MODEL_PREFIX}NUDGE_AFTER_TURNS", defaults.nudge_after_turns)),
            reasoning_fallback=_env_bool(f"{_WORLD_MODEL_PREFIX}REASONING_FALLBACK", defaults.reasoning_fallback),
        )


_REVIEW_PREFIX = "LOCAL_ANALYZER_LEVEL_REVIEW_"


@dataclass(frozen=True)
class LevelReviewConfig:
    enabled: bool = False
    llm_review: bool = True
    max_review_tokens: int = 2048
    max_llm_calls_per_level: int = 1
    review_thinking: bool = False
    llm_timeout_seconds: float = 60.0
    review_time_budget_seconds: float = 90.0
    max_facts_in_context: int = 15
    max_refuted_in_context: int = 5
    max_block_chars: int = 2500
    min_support: int = 2
    perturbation_test: bool = True
    check_timeout_seconds: int = 10
    quick_check_seconds: float = 3.0
    max_review_transitions: int = 300
    max_hypotheses: int = 60
    # "LEARNED LAST LEVEL" block: review cause sentence plus untested ideas.
    learned_block: bool = True
    max_untested_shown: int = 3
    # One LLM call after every level end, even with no agent hypotheses: states the
    # likely cause of the win/death and proposes hypotheses (source="reviewer").
    post_level_review: bool = True
    max_reviewer_hypotheses: int = 5
    # Per-move summaries in the review prompt (consecutive identical effects merged).
    max_review_moves: int = 60
    # Open hypotheses with check code carried across levels (0 = no cap).
    max_carried_unverified: int = 10

    @classmethod
    def from_env(cls) -> "LevelReviewConfig":
        defaults = cls()
        return cls(
            enabled=_env_bool(f"{_REVIEW_PREFIX}ENABLED", defaults.enabled),
            llm_review=_env_bool(f"{_REVIEW_PREFIX}LLM_REVIEW", defaults.llm_review),
            max_review_tokens=max(64, _env_int(f"{_REVIEW_PREFIX}MAX_REVIEW_TOKENS", defaults.max_review_tokens)),
            max_llm_calls_per_level=max(
                0, _env_int(f"{_REVIEW_PREFIX}MAX_LLM_CALLS_PER_LEVEL", defaults.max_llm_calls_per_level)
            ),
            review_thinking=_env_bool(f"{_REVIEW_PREFIX}REVIEW_THINKING", defaults.review_thinking),
            llm_timeout_seconds=max(
                1.0, _env_float(f"{_REVIEW_PREFIX}LLM_TIMEOUT_SECONDS", defaults.llm_timeout_seconds)
            ),
            review_time_budget_seconds=max(
                1.0, _env_float(f"{_REVIEW_PREFIX}REVIEW_TIME_BUDGET_SECONDS", defaults.review_time_budget_seconds)
            ),
            max_facts_in_context=max(
                0, _env_int(f"{_REVIEW_PREFIX}MAX_FACTS_IN_CONTEXT", defaults.max_facts_in_context)
            ),
            max_refuted_in_context=max(
                0, _env_int(f"{_REVIEW_PREFIX}MAX_REFUTED_IN_CONTEXT", defaults.max_refuted_in_context)
            ),
            max_block_chars=max(200, _env_int(f"{_REVIEW_PREFIX}MAX_BLOCK_CHARS", defaults.max_block_chars)),
            min_support=max(1, _env_int(f"{_REVIEW_PREFIX}MIN_SUPPORT", defaults.min_support)),
            perturbation_test=_env_bool(f"{_REVIEW_PREFIX}PERTURBATION_TEST", defaults.perturbation_test),
            check_timeout_seconds=max(
                1, _env_int(f"{_REVIEW_PREFIX}CHECK_TIMEOUT_SECONDS", defaults.check_timeout_seconds)
            ),
            quick_check_seconds=max(
                0.0, _env_float(f"{_REVIEW_PREFIX}QUICK_CHECK_SECONDS", defaults.quick_check_seconds)
            ),
            max_review_transitions=max(
                1, _env_int(f"{_REVIEW_PREFIX}MAX_REVIEW_TRANSITIONS", defaults.max_review_transitions)
            ),
            max_hypotheses=max(1, _env_int(f"{_REVIEW_PREFIX}MAX_HYPOTHESES", defaults.max_hypotheses)),
            learned_block=_env_bool(f"{_REVIEW_PREFIX}LEARNED_BLOCK", defaults.learned_block),
            max_untested_shown=max(
                0, _env_int(f"{_REVIEW_PREFIX}MAX_UNTESTED_SHOWN", defaults.max_untested_shown)
            ),
            post_level_review=_env_bool(f"{_REVIEW_PREFIX}POST_LEVEL_REVIEW", defaults.post_level_review),
            max_reviewer_hypotheses=max(
                0, _env_int(f"{_REVIEW_PREFIX}MAX_REVIEWER_HYPOTHESES", defaults.max_reviewer_hypotheses)
            ),
            max_review_moves=max(1, _env_int(f"{_REVIEW_PREFIX}MAX_REVIEW_MOVES", defaults.max_review_moves)),
            max_carried_unverified=max(
                0, _env_int(f"{_REVIEW_PREFIX}MAX_CARRIED_UNVERIFIED", defaults.max_carried_unverified)
            ),
        )
