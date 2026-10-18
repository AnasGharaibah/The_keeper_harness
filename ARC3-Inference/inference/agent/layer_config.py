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
