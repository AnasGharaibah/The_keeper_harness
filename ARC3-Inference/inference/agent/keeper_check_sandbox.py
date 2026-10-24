"""Keeper: run reviewer-written check code against recorded transitions.

A separate, check-only sandbox so the harness's own Python tool sandbox is not
modified. Same isolation: ``python -I -S``, an import allowlist, restricted
builtins, resource limits and a wall-clock timeout. Check code defines
``check_transition(t)`` returning True (claim held), False (violated) or None
(not applicable); the report also says whether the check fails on perturbed
copies of the transitions, so a check that passes on anything is not a fact.
"""
from __future__ import annotations

import atexit
import inspect
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from typing import Any

from inference.utils import segmentation as _segmentation
from inference.utils.grid_utils import ARC_COLOR_CHARS

_BOOTSTRAP = textwrap.dedent(
    r"""
    import builtins
    import json
    import os
    import signal
    import sys

    try:
        import resource
    except ImportError:  # pragma: no cover
        resource = None

    __SEGMENTATION_SOURCE__

    SAFE_MODULES = {"collections", "itertools", "math", "re", "json", "functools", "statistics"}
    SAFE_BUILTINS = {
        "abs", "all", "any", "bool", "chr", "dict", "divmod", "enumerate", "Exception", "filter", "float", "frozenset",
        "getattr", "hasattr", "int", "isinstance", "len", "list", "map", "max", "min", "next", "ord", "range",
        "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip", "KeyError", "IndexError",
        "TypeError", "ValueError",
    }
    COLOR_CHARS = ""


    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if str(name or "").split(".", 1)[0] not in SAFE_MODULES:
            raise ImportError(f"Module '{name}' is not allowed in checks.")
        return builtins.__import__(name, globals, locals, fromlist, level)


    class FrameView:
        def __init__(self, rows, step, level):
            self.ascii = "\n".join(rows)
            self.step = step
            self.level = level
            self.shape = (len(rows), len(rows[0]) if rows else 0)
            self._segmentation = None

        @property
        def segmentation(self):
            if self._segmentation is None:
                grid = [[max(0, COLOR_CHARS.find(ch)) for ch in line] for line in self.ascii.split("\n")]
                self._segmentation = segment_layer(grid, COLOR_CHARS)
            return self._segmentation


    class TransitionView:
        def __init__(self, action, before_frame, after_frame, result):
            self.action = action
            self.before_frame = before_frame
            self.after_frame = after_frame
            self.result = dict(result or {})

        # the level the move was made on, and its step (checks often reach for these)
        @property
        def level(self):
            return self.before_frame.level

        @property
        def step(self):
            return self.after_frame.step


    def cell(frame, row, col):
        lines = frame.ascii.split("\n") if frame is not None else []
        if 0 <= row < len(lines) and 0 <= col < len(lines[row]):
            return lines[row][col]
        return None


    def changed_cells(before, after):
        if before is None or after is None:
            return []
        a_lines, b_lines = before.ascii.split("\n"), after.ascii.split("\n")
        out = []
        for r in range(min(len(a_lines), len(b_lines))):
            for c in range(min(len(a_lines[r]), len(b_lines[r]))):
                if a_lines[r][c] != b_lines[r][c]:
                    out.append((r, c, a_lines[r][c], b_lines[r][c]))
        return out


    def parse_action(text):
        text = str(text or "").strip()
        name = text.split("(", 1)[0].strip().upper()
        row = col = None
        if "(" in text:
            for part in text.split("(", 1)[1].rstrip(")").split(","):
                key, _, value = part.partition("=")
                try:
                    if key.strip() == "row":
                        row = int(value)
                    elif key.strip() == "col":
                        col = int(value)
                except ValueError:
                    pass
        return name, row, col


    ALT_ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "SPACE"]


    def _load(check_code):
        safe = {name: getattr(builtins, name) for name in SAFE_BUILTINS}
        safe["__import__"] = _safe_import
        import collections, itertools, math, re
        namespace = {"__builtins__": safe, "cell": cell, "changed_cells": changed_cells, "parse_action": parse_action,
                     "collections": collections, "itertools": itertools, "math": math, "re": re}
        exec(compile(str(check_code or ""), "<check>", "exec"), namespace)
        fn = namespace.get("check_transition")
        if not callable(fn):
            raise ValueError("check code must define check_transition(t)")
        return fn


    def _tri(value):
        return None if value is None else bool(value)


    def _crossed_after(t, transitions):
        # the outcome of a different recorded move, preferring a different action
        base = parse_action(t.action)
        others = [o for o in transitions if o is not t and o.after_frame.ascii != t.after_frame.ascii]
        different = [o for o in others if parse_action(o.action) != base]
        pool = different or others
        return pool[(t.after_frame.step * 7) % len(pool)] if pool else None


    def _perturbed(t, kind, alt_actions, transitions=()):
        result = dict(t.result)
        action, after = t.action, t.after_frame
        if kind == "crossed":
            other = _crossed_after(t, transitions)
            if other is None:
                return None
            after = other.after_frame
            result = dict(other.result)
        elif kind == "frozen":
            after = t.before_frame
            result.update({"board_changed": False, "level_completed": False})
        elif kind == "swapped":
            base = parse_action(action)[0]
            action = next((a for a in list(alt_actions) + ALT_ACTIONS if parse_action(a)[0] != base), action)
        elif kind == "flipped":
            for key in ("board_changed", "level_completed", "game_over"):
                result[key] = not bool(result.get(key))
        return TransitionView(action, t.before_frame, after, result)


    def evaluate(check_code, transitions, alt_actions, perturb):
        fn = _load(check_code)
        outcomes = [_tri(fn(t)) for t in transitions]
        report = {
            "status": "pass",
            "support": sum(1 for o in outcomes if o is True),
            "failures": sum(1 for o in outcomes if o is False),
            "falsifiable": None,
        }
        if report["failures"]:
            failed = transitions[outcomes.index(False)]
            report["status"] = "fail"
            report["first_failure"] = {"action": failed.action, "step": failed.after_frame.step}
            return report
        if report["support"] == 0:
            report["status"] = "not_applicable"
            return report
        if perturb:
            applied = [t for t, o in zip(transitions, outcomes) if o is True][:50]
            report["falsifiable"] = False
            for kind in ("crossed", "frozen", "swapped", "flipped"):
                for t in applied:
                    variant = _perturbed(t, kind, alt_actions, transitions)
                    if variant is None:
                        continue
                    try:
                        if fn(variant) is False:
                            report["falsifiable"] = True
                            break
                    except Exception:
                        continue
                if report["falsifiable"]:
                    break
        return report


    def main():
        try:
            os.nice(10)  # checks yield the CPU to the game and the model server
        except (AttributeError, OSError):
            pass
        request = json.loads(sys.stdin.readline())
        global COLOR_CHARS
        COLOR_CHARS = str(request.get("color_chars") or "")
        timeout = max(1, int(request.get("timeout_seconds", 10)))
        if resource is not None:
            for limit, value in ((getattr(resource, "RLIMIT_CPU", None), timeout + 1),
                                 (getattr(resource, "RLIMIT_FSIZE", None), 1_000_000),
                                 (getattr(resource, "RLIMIT_NOFILE", None), 32)):
                if limit is not None:
                    try:
                        resource.setrlimit(limit, (value, value))
                    except (OSError, ValueError):
                        pass
        transitions = [
            TransitionView(
                str(item.get("action", "")),
                FrameView(item.get("before") or [], int(item.get("before_step") or 0), int(item.get("before_level") or 0)),
                FrameView(item.get("after") or [], int(item.get("step") or 0), int(item.get("level") or 0)),
                item.get("result") or {},
            )
            for item in request.get("transitions") or []
        ]
        try:
            report = evaluate(request.get("check_code", ""), transitions,
                              request.get("alt_actions") or [], bool(request.get("perturb", True)))
        except Exception as exc:
            report = {"status": "error", "error": f"{exc.__class__.__name__}: {exc}"}
        sys.stdout.write(json.dumps(report) + "\n")
        sys.stdout.flush()


    main()
    """
).replace("__SEGMENTATION_SOURCE__\n", inspect.getsource(_segmentation))


def _rows(grid: Any) -> list[str]:
    return ["".join(ARC_COLOR_CHARS[max(0, min(15, int(v)))] for v in row) for row in grid]


# Check processes still running. Each one is waited for (or killed) by the call
# that started it; this set only catches a process whose caller was abandoned
# (a review past its hard limit) when the interpreter exits.
_LIVE: set[subprocess.Popen] = set()
_LIVE_LOCK = threading.Lock()


def _kill(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            process.kill()
        except OSError:
            pass


@atexit.register
def kill_live_checks() -> None:
    with _LIVE_LOCK:
        live = list(_LIVE)
        _LIVE.clear()
    for process in live:
        _kill(process)


def run_check(
    check_code: str,
    transitions: list[dict[str, Any]],
    *,
    alt_actions: list[str] | None = None,
    perturb: bool = True,
    timeout_seconds: int = 10,
) -> dict[str, Any]:
    """Run one check. ``transitions`` items carry ``action``, int-grid
    ``before``/``after``, ``step`` and ``result``. Never raises."""
    payload = {
        "check_code": str(check_code or ""),
        "transitions": [
            {
                "action": t["action"],
                "before": _rows(t["before"]),
                "after": _rows(t["after"]),
                "before_step": int(t.get("step", 1)) - 1,
                "step": int(t.get("step", 0)),
                "before_level": int(t.get("level", 0) or 0),
                "level": int(t.get("level", 0) or 0),
                "result": dict(t.get("result") or {}),
            }
            for t in transitions
        ],
        "alt_actions": list(alt_actions or []),
        "perturb": bool(perturb),
        "timeout_seconds": int(timeout_seconds),
        "color_chars": ARC_COLOR_CHARS,
    }
    with tempfile.TemporaryDirectory(prefix="keeper_check_") as cwd:
        try:
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", _BOOTSTRAP],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
                start_new_session=True,
            )
        except OSError:
            return {"status": "error", "error": "check sandbox could not start"}
        with _LIVE_LOCK:
            _LIVE.add(process)
        started = time.monotonic()
        try:
            out, _err = process.communicate(json.dumps(payload) + "\n", timeout=max(1, int(timeout_seconds)) + 2)
        except subprocess.TimeoutExpired:
            _kill(process)
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            return {"status": "timeout", "error": f"check exceeded {timeout_seconds}s"}
        finally:
            with _LIVE_LOCK:
                _LIVE.discard(process)
        try:
            report = json.loads(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            if process.returncode is not None and process.returncode < 0:
                # killed by the CPU limit
                return {"status": "timeout", "error": f"check exceeded {timeout_seconds}s"}
            return {"status": "error", "error": "check sandbox returned no report"}
        report["seconds"] = round(time.monotonic() - started, 2)
        return report
