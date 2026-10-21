"""Switches for the Keeper additions. Every switch defaults to off.

With all of them off the harness behaves exactly as the published Milestone 2
configuration: no Keeper code runs per action or per turn, nothing is added to
any prompt or tool output, and no extra model request is made.

Read once when a ToolAgent is created (the published notebook sets its
environment before the harness starts).
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True)
class KeeperConfig:
    # SURPRISE lines: a single action did something different from the previous
    # single use of the same action on this level.
    surprise: bool = False
    surprise_max_lines: int = 3
    # Cells this close to the board edge are ignored, like the harness's own
    # gameplay_changed test (ARC3_NOOP_GUARD_BORDER).
    border: int = 4
    # Post-level review: one extra request on the game's own cached prefix after a
    # level is completed; hypotheses are checked mechanically and only verified
    # ones are shown as facts.
    review: bool = False
    review_on_game_over: bool = False
    review_max_hypotheses: int = 5
    review_max_tokens: int = 2048
    review_time_budget_seconds: int = 240
    review_tail_fraction: float = 0.2  # skip in the final fraction of the scheduled runtime
    review_max_moves: int = 60
    review_max_open: int = 10
    review_check_timeout_seconds: int = 10
    facts_min_support: int = 2
    # Perception lines for what frame_diff() does not report: whole-pattern
    # rotations/flips/scale, shape matches between objects, and player roles.
    perception: bool = False
    perception_max_lines: int = 3
    perception_min_size: int = 3

    @property
    def any_observation(self) -> bool:
        return self.surprise or self.perception

    @classmethod
    def from_env(cls) -> "KeeperConfig":
        defaults = cls()
        return cls(
            surprise=_env_bool("KEEPER_SURPRISE", defaults.surprise),
            surprise_max_lines=max(1, _env_int("KEEPER_SURPRISE_MAX_LINES", defaults.surprise_max_lines)),
            border=max(0, _env_int("ARC3_NOOP_GUARD_BORDER", defaults.border)),
            review=_env_bool("KEEPER_REVIEW", defaults.review),
            review_on_game_over=_env_bool("KEEPER_REVIEW_ON_GAME_OVER", defaults.review_on_game_over),
            review_max_hypotheses=max(1, _env_int("KEEPER_REVIEW_MAX_HYPOTHESES", defaults.review_max_hypotheses)),
            review_max_tokens=max(256, _env_int("KEEPER_REVIEW_MAX_TOKENS", defaults.review_max_tokens)),
            review_time_budget_seconds=max(
                10, _env_int("KEEPER_REVIEW_TIME_BUDGET_SECONDS", defaults.review_time_budget_seconds)
            ),
            review_tail_fraction=min(1.0, max(0.0, _env_float("KEEPER_REVIEW_TAIL_FRACTION", defaults.review_tail_fraction))),
            review_max_moves=max(5, _env_int("KEEPER_REVIEW_MAX_MOVES", defaults.review_max_moves)),
            review_max_open=max(0, _env_int("KEEPER_REVIEW_MAX_OPEN", defaults.review_max_open)),
            review_check_timeout_seconds=max(
                1, _env_int("KEEPER_REVIEW_CHECK_TIMEOUT_SECONDS", defaults.review_check_timeout_seconds)
            ),
            facts_min_support=max(1, _env_int("KEEPER_FACTS_MIN_SUPPORT", defaults.facts_min_support)),
            perception=_env_bool("KEEPER_PERCEPTION", defaults.perception),
            perception_max_lines=max(1, _env_int("KEEPER_PERCEPTION_MAX_LINES", defaults.perception_max_lines)),
            perception_min_size=max(1, _env_int("KEEPER_PERCEPTION_MIN_SIZE", defaults.perception_min_size)),
        )
