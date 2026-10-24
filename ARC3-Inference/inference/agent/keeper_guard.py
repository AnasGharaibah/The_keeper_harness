"""Keeper fail-safe: the worst case of turning the Keeper on must be the baseline.

- ``GameGuard`` (one per game): every Keeper hook runs through ``call``. Any
  exception (MemoryError included) or a hook running past its time limit
  disables the Keeper for that game only, logs one line, and returns the
  hook's fallback, so the harness's own code continues as if the Keeper were
  off.
- A process-wide memory guard: when this process's RSS exceeds
  ``KEEPER_MAX_RSS_FRACTION`` of the RAM available to it (cgroup limit or
  MemTotal; default 0.75), or the system's available RAM falls below
  ``KEEPER_MIN_AVAILABLE_FRACTION`` (default 0.10), all Keeper work stops for
  the rest of the run and its per-game histories are dropped.
- ``outside``: Keeper code that is not inside a game's hook (startup, config,
  prompt assembly, the timing and resource logs, the glue around the hooks).
  Any error there turns the Keeper off for the whole run.
- Last resort: a game whose Keeper fails before its first turn has finished
  turns the Keeper off for the whole run, not just for that game; a failure
  that early points at the setup, which every other game shares.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
import weakref
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


class KeeperTimeout(Exception):
    pass


_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def artifact_path(state_path: Path, suffix: str) -> Path:
    """A Keeper file next to the game's runtime state, named from its stem.

    Hidden games can have any id, and the stem carries it: the Keeper's own
    files use ASCII letters, digits, ``_ . -`` only, and stay short. A stem that
    had to be changed or cut gets a hash of the original, so two games never
    share a file."""
    stem = state_path.stem
    safe = _UNSAFE.sub("_", stem).strip(".") or "game"
    if safe != stem or len(safe) > 120:
        safe = f"{safe[:80]}_{hashlib.sha1(stem.encode('utf-8', 'surrogatepass')).hexdigest()[:12]}"
    return state_path.with_name(f"{safe}{suffix}")


# ----- memory readings ------------------------------------------------------------------------


def rss_bytes() -> int:
    try:
        with open("/proc/self/statm") as handle:
            return int(handle.read().split()[1]) * PAGE
    except (OSError, ValueError, IndexError):
        return 0


def peak_rss_bytes() -> int:
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    out[key] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return out


def memory_limit_bytes() -> int:
    """RAM this process may use: the cgroup limit when there is one, else MemTotal."""
    total = meminfo().get("MemTotal", 0)
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw.isdigit() and 0 < int(raw) < (total or int(raw) + 1):
            return int(raw)
    return total


# ----- process-wide guard ---------------------------------------------------------------------


class _Global:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.disabled_reason: str | None = None
        self.max_rss_fraction = 0.75
        self.min_available_fraction = 0.10
        self.check_interval = 2.0
        self._last_check = 0.0
        self.histories: "weakref.WeakSet[Any]" = weakref.WeakSet()
        self.disabled_games: set[str] = set()
        self.read_rss: Callable[[], int] = rss_bytes
        self.read_meminfo: Callable[[], dict[str, int]] = meminfo
        self.read_limit: Callable[[], int] = memory_limit_bytes

    def configure(self, *, max_rss_fraction: float, min_available_fraction: float) -> None:
        self.max_rss_fraction = max_rss_fraction
        self.min_available_fraction = min_available_fraction

    def check_memory(self, *, force: bool = False) -> None:
        """Rate-limited; disables all Keeper work once memory is short."""
        now = time.monotonic()
        if self.disabled_reason is not None or (not force and now - self._last_check < self.check_interval):
            return
        self._last_check = now
        try:
            limit = self.read_limit()
            rss = self.read_rss()
            info = self.read_meminfo()
        except Exception:  # noqa: BLE001
            return
        reason = None
        if limit and self.max_rss_fraction > 0 and rss > self.max_rss_fraction * limit:
            reason = f"process RSS {rss / 2**30:.1f} GiB is above {self.max_rss_fraction:.0%} of {limit / 2**30:.1f} GiB"
        total, available = info.get("MemTotal", 0), info.get("MemAvailable")
        if reason is None and total and available is not None and available < self.min_available_fraction * total:
            reason = f"system available RAM {available / 2**30:.1f} GiB is below {self.min_available_fraction:.0%} of {total / 2**30:.1f} GiB"
        if reason is not None:
            self.disable(reason)

    def disable(self, reason: str) -> None:
        with self.lock:
            if self.disabled_reason is not None:
                return
            self.disabled_reason = reason
        log.warning("keeper disabled for the rest of the run: %s; games continue without Keeper", reason)
        for history in list(self.histories):
            try:
                history.clear()
            except Exception:  # noqa: BLE001
                pass

    @property
    def active(self) -> bool:
        return self.disabled_reason is None

    def reset(self) -> None:  # tests
        self.__init__()


GLOBAL = _Global()


def outside(where: str, fn: Callable[[], Any], fallback: Any = None) -> Any:
    """Run Keeper code that is not inside a game's hook. Any exception turns the
    Keeper off for the rest of the run and returns ``fallback``; once it is off,
    ``fn`` is not run at all."""
    if GLOBAL.disabled_reason is not None:
        return fallback
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - MemoryError included
        GLOBAL.disable(f"{where}: {exc.__class__.__name__}: {str(exc)[:200]}")
        return fallback


# ----- per-game guard ---------------------------------------------------------------------------


def run_with_timeout(fn: Callable[[], Any], timeout: float) -> Any:
    """Run ``fn`` in a daemon thread; KeeperTimeout if it has not returned in time.
    A timed-out thread is abandoned: its game's Keeper is disabled, so nothing
    reads what it still writes."""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            box["error"] = exc

    worker = threading.Thread(target=target, name="keeper-hook", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise KeeperTimeout(f"took longer than {timeout:.0f}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


class GameGuard:
    def __init__(self, game: str) -> None:
        self.game = game
        self.disabled_reason: str | None = None
        self.turns_done = 0  # analyzer turns of this game that have finished

    @property
    def active(self) -> bool:
        if self.disabled_reason is not None:
            return False
        GLOBAL.check_memory()
        return GLOBAL.disabled_reason is None

    def finish_turn(self) -> None:
        self.turns_done += 1

    def disable(self, hook: str, reason: str) -> None:
        if self.disabled_reason is not None:
            return
        self.disabled_reason = f"{hook}: {reason}"
        GLOBAL.disabled_games.add(self.game)
        log.warning("keeper disabled for %s after %s; the game continues without Keeper", self.game, self.disabled_reason)
        if self.turns_done == 0:
            GLOBAL.disable(f"{self.game} failed before its first turn finished ({self.disabled_reason})")

    def call(self, hook: str, fn: Callable[[], Any], fallback: Any = None, *, timeout: float | None = None,
             budget: float | None = None) -> Any:
        """Run one hook. ``timeout``: hard limit (fn runs in a worker thread).
        ``budget``: soft limit checked after fn returns (for hooks that already
        bound themselves, like the review's model request)."""
        if not self.active:
            return fallback
        started = time.monotonic()
        try:
            value = run_with_timeout(fn, timeout) if timeout is not None else fn()
        except KeeperTimeout as exc:
            self.disable(hook, str(exc))
            return fallback
        except Exception as exc:  # noqa: BLE001 - MemoryError included
            self.disable(hook, f"{exc.__class__.__name__}: {str(exc)[:200]}")
            return fallback
        elapsed = time.monotonic() - started
        if budget is not None and elapsed > budget:
            self.disable(hook, f"took {elapsed:.0f}s, over its {budget:.0f}s budget")
        return value
