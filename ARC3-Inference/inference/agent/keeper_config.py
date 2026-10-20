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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True)
class KeeperConfig:
    # SURPRISE lines: an action did something different from the last time it
    # was used on this level.
    surprise: bool = False
    surprise_max_lines: int = 3
    # Cells this close to the board edge are ignored, like the harness's own
    # gameplay_changed test (ARC3_NOOP_GUARD_BORDER).
    border: int = 4

    @property
    def any_observation(self) -> bool:
        return self.surprise

    @classmethod
    def from_env(cls) -> "KeeperConfig":
        defaults = cls()
        return cls(
            surprise=_env_bool("KEEPER_SURPRISE", defaults.surprise),
            surprise_max_lines=max(1, _env_int("KEEPER_SURPRISE_MAX_LINES", defaults.surprise_max_lines)),
            border=max(0, _env_int("ARC3_NOOP_GUARD_BORDER", defaults.border)),
        )
