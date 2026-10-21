"""Lightweight isolated runner for analyzer Python tool calls."""
from __future__ import annotations

import inspect
import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import textwrap
import time
from typing import Any, Callable

from inference.utils import segmentation as _segmentation
from inference.utils.grid_utils import ARC_COLOR_CHARS


_SANDBOX_BOOTSTRAP = textwrap.dedent(
    r"""
    import builtins
    import contextlib
    import io
    import json
    import os
    import signal
    import sys
    import time
    import traceback

    try:
        import resource
    except ImportError:  # pragma: no cover
        resource = None

    COLOR_CHARS = ""

    __SEGMENTATION_SOURCE__

    HOST_STDOUT = sys.stdout

    SAFE_MODULES = {
        "bisect",
        "collections",
        "copy",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
    }
    SAFE_BUILTINS = {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "complex",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "Exception",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "TypeError",
        "type",
        "ValueError",
        "RuntimeError",
        "zip",
    }


    def _send(payload):
        HOST_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
        HOST_STDOUT.flush()


    def _recv():
        line = sys.stdin.readline()
        if not line:
            raise EOFError("sandbox input closed")
        return json.loads(line)


    class FrameView:
        def __init__(self, *, ascii, step, level, shape, grid):
            self.ascii = ascii
            self.step = step
            self.level = level
            self.shape = tuple(shape)
            self._grid = grid
            self._segmentation = None

        @property
        def segmentation(self):
            if self._segmentation is None:
                grid = self._grid
                if grid is None:  # check mode ships compact row strings, not int grids
                    grid = [[max(0, COLOR_CHARS.find(ch)) for ch in line] for line in self.ascii.split("\n")]
                self._segmentation = segment_layer(grid, COLOR_CHARS)
            return self._segmentation

        def __str__(self):
            rows, cols = self.shape
            return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"

        __repr__ = __str__


    class HistoryEntryView:
        def __init__(self, *, action, frame):
            self.action = action
            self.frame = frame

        def __str__(self):
            return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"

        __repr__ = __str__


    class TransitionView:
        def __init__(self, *, action, before_frame, after_frame, result):
            self.action = action
            self.before_frame = before_frame
            self.after_frame = after_frame
            self.frame = after_frame
            self.result = dict(result) if isinstance(result, dict) else {}

        def __str__(self):
            return (
                "ActionTransitionView("
                f"action={self.action!r}, "
                f"before_frame={self.before_frame}, "
                f"after_frame={self.after_frame})"
            )

        __repr__ = __str__


    def _frame_from_payload(payload):
        if not isinstance(payload, dict):
            return None
        return FrameView(
            ascii=str(payload.get("ascii", "")),
            step=int(payload.get("step", 0)),
            level=int(payload.get("level", 0)),
            shape=payload.get("shape", [0, 0]),
            grid=payload.get("grid", []),
        )


    def _history_from_payload(payload):
        items = []
        for entry in payload or []:
            if not isinstance(entry, dict):
                continue
            items.append(
                HistoryEntryView(
                    action=str(entry.get("action", "")),
                    frame=_frame_from_payload(entry.get("frame")),
                )
            )
        return items


    def _transitions_from_history(history, last_action_result):
        transitions = []
        for index, entry in enumerate(history):
            action = str(getattr(entry, "action", "") or "").strip()
            if not action:
                continue
            before_frame = history[index - 1].frame if index > 0 else None
            transitions.append(
                TransitionView(
                    action=action,
                    before_frame=before_frame,
                    after_frame=entry.frame,
                    result={},
                )
            )
        if transitions and isinstance(last_action_result, dict):
            transitions[-1].result = dict(last_action_result)
        return transitions


    def _json_safe(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        return str(value)


    def _sanitize_exception(exc):
        extracted = traceback.extract_tb(exc.__traceback__)
        user_frames = [frame for frame in extracted if frame.filename == "<python_tool>"]
        lines = ["Traceback (most recent call last):"]
        for frame in user_frames or extracted[-1:]:
            lines.append(f'  File "<python_tool>", line {frame.lineno}, in {frame.name}')
        lines.append(f"{exc.__class__.__name__}: {exc}")
        return "\n".join(lines)


    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name or "").split(".", 1)[0]
        if root not in SAFE_MODULES:
            raise ImportError(f"Module '{name}' is not allowed in the sandbox.")
        return builtins.__import__(name, globals, locals, fromlist, level)


    def _set_limits(timeout_seconds):
        if resource is None:
            return
        cpu_limit = max(1, int(timeout_seconds)) + 1
        for limit, value in (
            (getattr(resource, "RLIMIT_CPU", None), cpu_limit),
            (getattr(resource, "RLIMIT_FSIZE", None), 1_000_000),
            (getattr(resource, "RLIMIT_NOFILE", None), 32),
        ):
            if limit is None:
                continue
            try:
                resource.setrlimit(limit, (value, value))
            except (OSError, ValueError):
                pass


    def _normalize_actions(actions):
        if isinstance(actions, str):
            items = [actions]
        elif isinstance(actions, dict):
            items = [actions]
        elif isinstance(actions, (list, tuple)):
            items = list(actions)
        else:
            raise TypeError(
                "action(actions) expects a string, an action object, or a list of action strings/objects."
            )
        if not items:
            raise ValueError("action(actions) requires at least one action.")

        normalized = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, str):
                action_name = item.strip()
                if not action_name:
                    raise ValueError(f"Action {index} is empty.")
                normalized.append({"action": action_name})
                continue
            if isinstance(item, dict):
                action_name = str(item.get("action", "")).strip()
                if not action_name:
                    raise ValueError(f"Action {index} is missing an `action` field.")
                entry = {"action": action_name}
                if action_name.upper() == "MOUSE" and ("x" in item or "y" in item):
                    raise ValueError(
                        f"Action {index} uses legacy MOUSE x/y fields; use row and col."
                    )
                if "row" in item:
                    entry["row"] = item.get("row")
                if "col" in item:
                    entry["col"] = item.get("col")
                normalized.append(entry)
                continue
            raise TypeError(f"Action {index} must be a string or a dict.")
        return normalized


    def _normalize_expect(expect):
        if not isinstance(expect, dict):
            raise TypeError("action(..., expect=...) expects a dict, e.g. {'board_changed': True}.")
        normalized = dict(expect)
        cells = normalized.get("cells")
        if isinstance(cells, dict):
            items = []
            for key, color in cells.items():
                if isinstance(key, (tuple, list)) and len(key) == 2:
                    items.append([key[0], key[1], color])
                else:
                    items.append([str(key), color])
            normalized["cells"] = items
        return _json_safe(normalized)


    def cell(frame, row, col):
        if frame is None:
            return None
        lines = frame.ascii.split("\n")
        if 0 <= row < len(lines) and 0 <= col < len(lines[row]):
            return lines[row][col]
        return None


    def changed_cells(before, after):
        if before is None or after is None:
            return []
        a_lines = before.ascii.split("\n")
        b_lines = after.ascii.split("\n")
        changes = []
        for r in range(min(len(a_lines), len(b_lines))):
            a_row, b_row = a_lines[r], b_lines[r]
            for c in range(min(len(a_row), len(b_row))):
                if a_row[c] != b_row[c]:
                    changes.append((r, c, a_row[c], b_row[c]))
        return changes


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


    class _CheckTimeout(BaseException):
        pass


    _PERTURB_LIMIT = 50
    _DEFAULT_ALT_ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "SPACE"]


    def _load_check(check_code):
        safe_builtins = {name: getattr(builtins, name) for name in SAFE_BUILTINS}
        safe_builtins["__import__"] = _safe_import
        namespace = {
            "__builtins__": safe_builtins,
            "cell": cell,
            "changed_cells": changed_cells,
            "parse_action": parse_action,
        }
        exec(compile(str(check_code or ""), "<check>", "exec"), namespace)
        if callable(namespace.get("check_transition")):
            return "each", namespace["check_transition"]
        if callable(namespace.get("check")):
            return "all", namespace["check"]
        raise ValueError("check_code must define check_transition(t) or check(transitions).")


    def _tri(value):
        return None if value is None else bool(value)


    def _outcomes(form, fn, transitions):
        if form == "each":
            return [_tri(fn(t)) for t in transitions]
        value = fn(list(transitions))
        if isinstance(value, (list, tuple)):
            return [_tri(item) for item in value]
        return [_tri(value)]


    def _perturbed(transition, kind, alt_actions):
        result = dict(transition.result)
        action = transition.action
        after = transition.after_frame
        if kind == "frozen":
            after = transition.before_frame
            result.update({"board_changed": False, "level_completed": False})
        elif kind == "swapped":
            base = parse_action(action)[0]
            action = next(
                (alt for alt in list(alt_actions) + _DEFAULT_ALT_ACTIONS if parse_action(alt)[0] != base),
                action,
            )
        elif kind == "flipped":
            for key in ("board_changed", "level_completed", "game_over"):
                result[key] = not bool(result.get(key))
        return TransitionView(
            action=action,
            before_frame=transition.before_frame,
            after_frame=after,
            result=result,
        )


    def _evaluate_check_inner(check_code, transitions, alt_actions, perturb):
        form, fn = _load_check(check_code)
        outcomes = _outcomes(form, fn, transitions)
        report = {
            "status": "pass",
            "form": form,
            "support": sum(1 for item in outcomes if item is True),
            "failures": sum(1 for item in outcomes if item is False),
            "not_applicable": sum(1 for item in outcomes if item is None),
            "falsifiable": None,
        }
        if report["failures"]:
            report["status"] = "fail"
            index = outcomes.index(False)
            if form == "each" or len(outcomes) == len(transitions):
                failed = transitions[index] if index < len(transitions) else None
                if failed is not None:
                    report["first_failure"] = {
                        "action": failed.action,
                        "step": failed.after_frame.step if failed.after_frame is not None else None,
                    }
            return report
        if report["support"] == 0:
            report["status"] = "not_applicable"
            return report
        if not perturb:
            return report
        if form == "each":
            applied = [t for t, item in zip(transitions, outcomes) if item is True][:_PERTURB_LIMIT]
        else:
            applied = list(transitions)
        # Falsifiable only on a clean False. A crash on a perturbed transition is
        # inconclusive: skip it. "each" checks run per variant so one crash can't
        # hide a clean False on another variant.
        falsifiable = False
        for kind in ("frozen", "swapped", "flipped"):
            variants = [_perturbed(t, kind, alt_actions) for t in applied]
            groups = [[variant] for variant in variants] if form == "each" else [variants]
            for group in groups:
                try:
                    perturbed_outcomes = _outcomes(form, fn, group)
                except Exception:
                    continue
                if False in perturbed_outcomes:
                    falsifiable = True
                    break
            if falsifiable:
                break
        report["falsifiable"] = falsifiable
        return report


    def _evaluate_check(check_code, transitions, alt_actions=(), perturb=True, time_limit=None):
        if time_limit is not None and time_limit <= 0:
            return {"status": "skipped", "error": "quick-check time budget used up"}
        use_timer = time_limit is not None and hasattr(signal, "setitimer")
        previous_handler = None
        if use_timer:

            def _on_alarm(signum, frame):
                raise _CheckTimeout()

            previous_handler = signal.signal(signal.SIGALRM, _on_alarm)
            signal.setitimer(signal.ITIMER_REAL, float(time_limit))
        try:
            return _evaluate_check_inner(check_code, transitions, alt_actions, perturb)
        except _CheckTimeout:
            return {"status": "skipped", "error": f"quick check exceeded its {float(time_limit):.1f}s budget"}
        except Exception as exc:
            return {"status": "error", "error": f"{exc.__class__.__name__}: {exc}"}
        finally:
            if use_timer:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)


    def _frame_from_rows(rows, step, level):
        rows = [str(row) for row in rows or []]
        return FrameView(
            ascii="\n".join(rows),
            step=int(step or 0),
            level=int(level or 0),
            shape=[len(rows), len(rows[0]) if rows else 0],
            grid=None,
        )


    def _transition_from_check_payload(payload):
        return TransitionView(
            action=str(payload.get("action", "")),
            before_frame=_frame_from_rows(payload.get("before"), payload.get("before_step"), payload.get("before_level")),
            after_frame=_frame_from_rows(payload.get("after"), payload.get("step"), payload.get("level")),
            result=payload.get("result") or {},
        )


    def main():
        initial = _recv()
        global COLOR_CHARS
        COLOR_CHARS = str(initial.get("color_chars") or "")
        timeout_seconds = max(1, int(initial.get("timeout_seconds", 30)))
        sandbox_cwd = str(initial.get("sandbox_cwd", "")).strip()
        if sandbox_cwd:
            os.chdir(sandbox_cwd)
        _set_limits(timeout_seconds)

        if initial.get("mode") == "check":
            check_transitions = [
                _transition_from_check_payload(item)
                for item in initial.get("transitions") or []
                if isinstance(item, dict)
            ]
            outcome = _evaluate_check(
                initial.get("check_code", ""),
                check_transitions,
                alt_actions=[str(item) for item in initial.get("alt_actions") or []],
                perturb=bool(initial.get("perturb", True)),
            )
            _send({"type": "final", "stdout": "", "result": _json_safe(outcome), "action_results": []})
            return

        action_results = []
        stdout = io.StringIO()
        runtime_globals = {
            "__builtins__": {
                name: getattr(builtins, name)
                for name in SAFE_BUILTINS
            },
            "result": None,
        }
        runtime_globals["__builtins__"]["__import__"] = _safe_import

        def _refresh_state(state_payload):
            current_frame = _frame_from_payload(state_payload.get("current_frame"))
            history = _history_from_payload(state_payload.get("history"))
            last_action_result = state_payload.get("last_action_result")
            action_result = (
                dict(last_action_result) if isinstance(last_action_result, dict) else {}
            )
            transitions = _transitions_from_history(history, action_result)
            transition_results = state_payload.get("transition_results")
            if isinstance(transition_results, dict):
                for transition in transitions:
                    stored = transition_results.get(str(transition.after_frame.step)) if transition.after_frame else None
                    if not transition.result and isinstance(stored, dict):
                        transition.result = dict(stored)
            last_transition = transitions[-1] if transitions else None

            runtime_globals["current_frame"] = current_frame
            runtime_globals["latest_frame"] = current_frame
            runtime_globals["history"] = history
            runtime_globals["transitions"] = transitions
            runtime_globals["last_transition"] = last_transition
            runtime_globals["previous_frame"] = (
                last_transition.before_frame if last_transition is not None else None
            )
            runtime_globals["last_action_frame"] = (
                last_transition.after_frame if last_transition is not None else None
            )
            runtime_globals["last_action"] = last_transition.action if last_transition is not None else None
            runtime_globals["valid_actions"] = [str(item) for item in state_payload.get("valid_actions", [])]
            runtime_globals["last_action_result"] = action_result
            runtime_globals["last_prediction"] = state_payload.get("last_prediction")
            runtime_globals["prediction_log"] = list(state_payload.get("prediction_log") or [])

        def action(actions, row=None, col=None, expect=None):
            if row is not None or col is not None:
                if not isinstance(actions, str):
                    raise TypeError("row/col keywords need one action name, e.g. action('MOUSE', row=4, col=7).")
                actions = {"action": actions, "row": row, "col": col}
            normalized_actions = _normalize_actions(actions)
            message = {"type": "action", "actions": normalized_actions}
            if expect is not None:
                message["expect"] = _normalize_expect(expect)
            _send(message)
            reply = _recv()
            if reply.get("type") == "action_error":
                raise RuntimeError(str(reply.get("error", "action failed")))
            if reply.get("type") != "action_result":
                raise RuntimeError("Invalid action response from sandbox host.")
            action_result = reply.get("action_result") or {}
            action_results.append(action_result)
            _refresh_state(reply.get("state") or {})
            return action_result

        hypotheses = []
        initial_state = initial.get("state") or {}
        next_hypothesis_index = int(initial_state.get("next_hypothesis_index") or 1)
        quick_budget = [float(initial_state.get("quick_check_seconds") or 0.0)]

        def hypothesis(text, kind="rule", check_code=None):
            text = " ".join(str(text or "").split())
            if not text:
                raise ValueError("hypothesis(text) needs a non-empty claim.")
            kind = str(kind or "rule").strip().lower()
            if kind not in ("rule", "goal"):
                raise ValueError("hypothesis kind must be 'rule' or 'goal'.")
            if check_code is not None and not isinstance(check_code, str):
                raise TypeError("check_code must be a string of Python source.")
            frame = runtime_globals.get("current_frame")
            level = frame.level if frame is not None else 0
            record = {
                "id": f"H{next_hypothesis_index + len(hypotheses)}",
                "text": text[:400],
                "kind": kind,
                "check_code": check_code or None,
                "level": level,
            }
            reply = {"id": record["id"], "recorded": True}
            if check_code:
                level_transitions = [
                    t
                    for t in runtime_globals.get("transitions") or []
                    if t.before_frame is not None
                    and t.before_frame.level == level
                    and parse_action(t.action)[0] != "RESET"
                    and not (t.result or {}).get("reset")
                ]
                started = time.monotonic()
                reply["quick_check"] = _evaluate_check(
                    check_code, level_transitions, perturb=False, time_limit=quick_budget[0]
                )
                quick_budget[0] -= time.monotonic() - started
                record["quick_check"] = reply["quick_check"]
            hypotheses.append(record)
            return reply

        notes = {}

        def _store_note(key, text):
            text = str(text or "").strip()
            if not text:
                raise ValueError(f"{key}(text) needs non-empty text.")
            notes[key] = text[:2000]
            return {"stored": key, "chars": len(notes[key])}

        def world_model(text):
            return _store_note("world_model", text)

        def plan(text):
            return _store_note("plan", text)

        runtime_globals["action"] = action
        runtime_globals["hypothesis"] = hypothesis
        if initial_state.get("world_model_enabled"):
            runtime_globals["world_model"] = world_model
            runtime_globals["plan"] = plan
        runtime_globals["cell"] = cell
        runtime_globals["changed_cells"] = changed_cells
        runtime_globals["parse_action"] = parse_action
        _refresh_state(initial.get("state") or {})

        try:
            compiled = compile(str(initial.get("code", "")), "<python_tool>", "exec")
            with contextlib.redirect_stdout(stdout):
                exec(compiled, runtime_globals, runtime_globals)
            _send(
                {
                    "type": "final",
                    "stdout": stdout.getvalue(),
                    "result": _json_safe(runtime_globals.get("result")),
                    "action_results": _json_safe(action_results),
                    "hypotheses": _json_safe(hypotheses),
                    "notes": _json_safe(notes),
                }
            )
        except Exception as exc:
            _send(
                {
                    "type": "error",
                    "error": _sanitize_exception(exc),
                    "stdout": stdout.getvalue(),
                    "action_results": _json_safe(action_results),
                    "hypotheses": _json_safe(hypotheses),
                    "notes": _json_safe(notes),
                }
            )


    if __name__ == "__main__":
        main()
    """
).replace("__SEGMENTATION_SOURCE__\n", inspect.getsource(_segmentation))


def _sanitize_host_error_text(text: str) -> str:
    if not str(text or "").strip():
        return "Sandbox process exited unexpectedly."
    return "Sandbox process exited unexpectedly."


def _sandbox_env() -> dict[str, str]:
    return {
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PATH": os.environ.get("PATH", ""),
    }


def _send_json_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        try:
            process.kill()
        except OSError:
            pass


def _wait_for_process_exit(process: subprocess.Popen[str], *, timeout: float = 1.0) -> None:
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
    except OSError:
        return

    try:
        process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        pass


def run_sandboxed_python(
    *,
    code: str,
    timeout_seconds: int,
    initial_state: dict[str, Any],
    action_handler: Callable[..., dict[str, Any]],
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rgb_python_tool_") as sandbox_dir:
        host_action_results: list[dict[str, Any]] = []
        try:
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", _SANDBOX_BOOTSTRAP],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                cwd=sandbox_dir,
                env=_sandbox_env(),
                start_new_session=True,
            )
        except OSError:
            return {
                "error": "Sandbox process could not start.",
                "stdout": "",
                "action_results": [],
            }
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None

        stdout_queue: queue.Queue[str | None] = queue.Queue()

        def _stdout_reader() -> None:
            for raw_line in process.stdout:
                stdout_queue.put(raw_line)
            stdout_queue.put(None)

        threading.Thread(target=_stdout_reader, daemon=True).start()

        _send_json_line(
            process.stdin,
            {
                **(extra_payload or {}),
                "code": code,
                "timeout_seconds": timeout_seconds,
                "sandbox_cwd": sandbox_dir,
                "state": initial_state,
                "color_chars": ARC_COLOR_CHARS,
            },
        )

        deadline = time.monotonic() + max(1, int(timeout_seconds))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": f"Tool timed out after {timeout_seconds}s",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                line = stdout_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                stderr = process.stderr.read()
                _wait_for_process_exit(process)
                return {
                    "error": _sanitize_host_error_text(stderr),
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                stderr = process.stderr.read()
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": "Sandbox process returned an invalid response.",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            msg_type = str(message.get("type", "")).strip()
            if msg_type == "action":
                try:
                    if message.get("expect") is None:
                        action_result_payload = action_handler(list(message.get("actions") or []))
                    else:
                        action_result_payload = action_handler(
                            list(message.get("actions") or []),
                            expect=message.get("expect"),
                        )
                except Exception:  # noqa: BLE001
                    _send_json_line(
                        process.stdin,
                        {
                            "type": "action_error",
                            "error": "action failed in sandbox host.",
                        },
                    )
                    continue
                raw_action_result = action_result_payload.get("action_result") or {}
                if isinstance(raw_action_result, dict):
                    host_action_results.append(dict(raw_action_result))
                _send_json_line(
                    process.stdin,
                    {
                        "type": "action_result",
                        "action_result": raw_action_result,
                        "state": action_result_payload.get("state") or {},
                    },
                )
                continue

            if msg_type in {"final", "error"}:
                _wait_for_process_exit(process)
                return {
                    "stdout": str(message.get("stdout", "") or ""),
                    "result": message.get("result"),
                    "error": str(message.get("error", "") or ""),
                    "action_results": list(message.get("action_results") or host_action_results),
                    "hypotheses": list(message.get("hypotheses") or []),
                    "notes": dict(message.get("notes") or {}),
                }

            _wait_for_process_exit(process)
            return {
                "error": "Sandbox process returned an unknown message type.",
                "stdout": "",
                "action_results": list(host_action_results),
            }


def run_sandboxed_check(
    *,
    check_code: str,
    transitions: list[dict[str, Any]],
    timeout_seconds: int,
    alt_actions: list[str] | None = None,
    perturb: bool = True,
) -> dict[str, Any]:
    """Run hypothesis check code against recorded transitions in the same
    restricted sandbox (import allowlist, builtins, rlimits, timeout).

    ``transitions`` items are compact dicts: ``action``, ``before``/``after``
    (lists of row strings in ARC color chars), ``before_step``, ``before_level``,
    ``step``, ``level`` and ``result``. Returns the sandbox's check report with
    ``status`` pass/fail/not_applicable/error/timeout.
    """

    def _no_actions(_actions: list[dict[str, Any]], **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("action() is not available while checking a hypothesis.")

    outcome = run_sandboxed_python(
        code="",
        timeout_seconds=timeout_seconds,
        initial_state={},
        action_handler=_no_actions,
        extra_payload={
            "mode": "check",
            "check_code": str(check_code or ""),
            "transitions": transitions,
            "alt_actions": list(alt_actions or []),
            "perturb": bool(perturb),
        },
    )
    error = str(outcome.get("error", "") or "")
    if error:
        status = "timeout" if "timed out" in error else "error"
        return {"status": status, "error": error}
    report = outcome.get("result")
    if not isinstance(report, dict):
        return {"status": "error", "error": "check returned no report"}
    return report
