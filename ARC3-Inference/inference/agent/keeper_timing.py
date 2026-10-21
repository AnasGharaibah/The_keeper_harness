"""Keeper timing log: where a turn's wall time and tokens go.

With KEEPER_TIMING on, every analyzer turn appends one JSON line to
``<runtime state stem>_timing.jsonl`` next to the other run artifacts:

- ``requests``: one entry per model request (kind "turn", "summary", "review"
  or "review_fix") with wall seconds, prompt / cached / generated tokens, and,
  only when KEEPER_TIMING_STREAM is on, ``first_token_seconds`` (queue +
  prefill) so ``seconds - first_token_seconds`` is generation time;
- ``gate_wait_seconds``: time spent waiting for a priority-gate slot;
- ``tool_seconds``: time inside tool calls (game actions and the sandbox),
  of which ``keeper_seconds`` was spent in Keeper code (SURPRISE, perception,
  review checks);
- ``keeper_chars``: characters the Keeper added to this turn's messages
  (notes block, ``surprises`` and ``perceived`` fields).

Without streaming the server reports only total time per request, so
``summarize`` splits prefill from generation by a least-squares fit of
seconds = a * uncached prompt tokens + b * generated tokens + c over many
requests. Logging never changes a request. Never raises into the game.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

CHARS_PER_TOKEN = 4.0  # rough, for reporting Keeper-added text in tokens


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def usage_numbers(usage: Any) -> dict[str, int | None]:
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    cached = _int(details.get("cached_tokens"))
    if cached is None:
        cached = _int(usage.get("cached_tokens"))
    return {
        "prompt_tokens": _int(usage.get("prompt_tokens")),
        "cached_tokens": cached,
        "generated_tokens": _int(usage.get("completion_tokens") or usage.get("output_tokens")),
    }


class TurnTimer:
    """One analyzer turn's timing record."""

    def __init__(self, path: Path, *, analysis_step: Any = None, action_num: Any = None) -> None:
        self.path = path
        self.started = time.monotonic()
        self.record: dict[str, Any] = {
            "analysis_step": analysis_step,
            "action_num": action_num,
            "wall_clock": round(time.time(), 3),
            "requests": [],
            "gate_wait_seconds": 0.0,
            "tool_seconds": 0.0,
            "keeper_seconds": 0.0,
            "keeper_chars": {"notes": 0, "surprises": 0, "perceived": 0},
        }

    def add_request(self, kind: str, seconds: float, usage: Any, first_token_seconds: float | None = None) -> None:
        entry: dict[str, Any] = {"kind": kind, "seconds": round(seconds, 3), **usage_numbers(usage)}
        if first_token_seconds is not None:
            entry["first_token_seconds"] = round(first_token_seconds, 3)
        self.record["requests"].append(entry)

    def add(self, key: str, seconds: float) -> None:
        self.record[key] = round(self.record[key] + seconds, 3)

    def add_chars(self, key: str, text: Any) -> None:
        if not text:
            return
        size = len(text) if isinstance(text, str) else len(json.dumps(text, ensure_ascii=False))
        self.record["keeper_chars"][key] += size

    @contextmanager
    def measure(self, key: str):
        started = time.monotonic()
        try:
            yield
        finally:
            self.add(key, time.monotonic() - started)

    def flush(self) -> None:
        self.record["turn_seconds"] = round(time.monotonic() - self.started, 3)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(self.record) + "\n")
        except OSError:
            pass


def timing_path(state_path: Path) -> Path:
    return state_path.with_name(f"{state_path.stem}_timing.jsonl")


@contextmanager
def nothing():
    yield


# ----- offline summary ------------------------------------------------------------------------


def _solve3(rows: list[tuple[float, float, float]]) -> tuple[float, float, float] | None:
    """Least squares for y = a*x1 + b*x2 + c; rows are (x1, x2, y)."""
    n = len(rows)
    if n < 10:
        return None
    s = [[0.0] * 4 for _ in range(3)]
    for x1, x2, y in rows:
        v = (x1, x2, 1.0)
        for i in range(3):
            for j in range(3):
                s[i][j] += v[i] * v[j]
            s[i][3] += v[i] * y
    for col in range(3):  # Gauss-Jordan with partial pivoting
        pivot = max(range(col, 3), key=lambda r: abs(s[r][col]))
        if abs(s[pivot][col]) < 1e-12:
            return None
        s[col], s[pivot] = s[pivot], s[col]
        for r in range(3):
            if r != col:
                f = s[r][col] / s[col][col]
                s[r] = [a - f * b for a, b in zip(s[r], s[col])]
    return tuple(s[i][3] / s[i][i] for i in range(3))  # type: ignore[return-value]


def summarize(lines: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Totals and the prefill/generation split over many turn records."""
    turns = list(lines)
    requests = [r for t in turns for r in t.get("requests", [])]
    by_kind: dict[str, dict[str, float]] = {}
    for r in requests:
        k = by_kind.setdefault(r["kind"], {"requests": 0, "seconds": 0.0, "prompt_tokens": 0, "generated_tokens": 0})
        k["requests"] += 1
        k["seconds"] += r["seconds"]
        k["prompt_tokens"] += r.get("prompt_tokens") or 0
        k["generated_tokens"] += r.get("generated_tokens") or 0
    rows = []
    for r in requests:
        if r.get("prompt_tokens") is None or r.get("generated_tokens") is None:
            continue
        uncached = r["prompt_tokens"] - (r.get("cached_tokens") or 0)
        rows.append((float(uncached), float(r["generated_tokens"]), float(r["seconds"])))
    fit = _solve3(rows)
    split = None
    if fit is not None:
        a, b, c = fit
        prefill = sum(a * x1 for x1, _x2, _y in rows)
        generation = sum(b * x2 for _x1, x2, _y in rows)
        split = {"seconds_per_uncached_prompt_token": a, "seconds_per_generated_token": b, "seconds_per_request": c,
                 "prefill_seconds": prefill, "generation_seconds": generation, "overhead_seconds": c * len(rows)}
    measured = [r for r in requests if "first_token_seconds" in r]
    keeper_chars = {key: sum(t.get("keeper_chars", {}).get(key, 0) for t in turns) for key in ("notes", "surprises", "perceived")}
    return {
        "turns": len(turns),
        "turn_seconds": sum(t.get("turn_seconds", 0.0) for t in turns),
        "gate_wait_seconds": sum(t.get("gate_wait_seconds", 0.0) for t in turns),
        "tool_seconds": sum(t.get("tool_seconds", 0.0) for t in turns),
        "keeper_seconds": sum(t.get("keeper_seconds", 0.0) for t in turns),
        "requests_by_kind": by_kind,
        "cached_tokens_reported": any(r.get("cached_tokens") is not None for r in requests),
        "fit_split": split,
        "measured_first_token_seconds": sum(r["first_token_seconds"] for r in measured) if measured else None,
        "measured_generation_seconds": sum(r["seconds"] - r["first_token_seconds"] for r in measured) if measured else None,
        "keeper_chars": keeper_chars,
        "keeper_tokens_per_turn_est": sum(keeper_chars.values()) / CHARS_PER_TOKEN / max(1, len(turns)),
    }


def summarize_files(paths: Iterable[Path]) -> dict[str, Any]:
    lines = []
    for path in paths:
        for raw in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                lines.append(json.loads(raw))
            except ValueError:
                continue
    return summarize(lines)


if __name__ == "__main__":  # python -m inference.agent.keeper_timing artifacts/*_timing.jsonl
    import sys

    print(json.dumps(summarize_files(Path(p) for p in sys.argv[1:]), indent=2))
