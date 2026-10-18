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

    @classmethod
    def from_env(cls) -> "PredictionCheckConfig":
        return cls(
            enabled=_env_bool("LOCAL_ANALYZER_PREDICTION_CHECK_ENABLED", False),
            max_wrong_diffs_shown=max(0, _env_int("LOCAL_ANALYZER_PREDICTION_CHECK_MAX_WRONG_DIFFS", 5)),
            sandbox_log_size=max(0, _env_int("LOCAL_ANALYZER_PREDICTION_CHECK_SANDBOX_LOG_SIZE", 20)),
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
        )
