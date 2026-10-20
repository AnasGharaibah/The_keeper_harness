"""Keeper: post-level review, mechanically verified facts, and the notes block.

After a level is completed the agent makes one extra request on its own cached
prefix (see ToolAgent._keeper_before_turn) asking for the likely cause and a few
hypotheses with check code. Here the reply is parsed and every hypothesis is
checked against the level's recorded transitions in the check sandbox. A
hypothesis becomes a FACT only if its check passes, applies to at least
``min_support`` transitions (summed across levels) and fails on perturbed
transitions. The model never promotes anything by opinion. Facts and open
hypotheses are re-checked on every later reviewed level; a fact that fails is
demoted.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from inference.agent import keeper_auto

FACT, OPEN, REFUTED, DROPPED = "fact", "open", "refuted", "dropped"

CheckRunner = Callable[[str, list[dict[str, Any]], bool], dict[str, Any]]

NOTES_HEADER = "KEEPER NOTES"

REQUEST_TEMPLATE = """KEEPER LEVEL REVIEW (a harness request: do not call tools, reply with one JSON object only).
You just completed level {level}. Its moves (border cells ignored):
{moves}
{facts}Write:
(a) "cause": one sentence - what finally completed the level, and which earlier events it needed.
(b) "hypotheses": up to {max_hypotheses} rules or goals that explain these moves, especially ones that connect events. Each needs "check_code" defining check_transition(t) that returns False on a move that violates it, True when it holds, None when it does not apply. t.action is e.g. 'RIGHT' or 'MOUSE(row=4, col=7)'; t.before_frame / t.after_frame have .ascii (rows of colour letters), .segmentation, .level, .step; t.result has level_completed, game_over, board_changed. Helpers: cell(frame, row, col), changed_cells(a, b) -> [(row, col, old, new)], parse_action(text) -> (name, row, col). Imports: collections, itertools, math, re, json, functools, statistics. At most 12 lines of code each, no comments.
Reply: {{"cause": "...", "hypotheses": [{{"text": "...", "kind": "rule", "check_code": "def check_transition(t):\\n    ..."}}]}}"""


@dataclass
class Hypothesis:
    id: str
    text: str
    kind: str
    check_code: str
    level_proposed: int
    status: str = OPEN
    support: int = 0
    levels_held: list[int] = field(default_factory=list)
    falsifiable_seen: bool = False
    level_learned: int | None = None
    reason: str = ""


# ----- reply parsing ----------------------------------------------------------------------

_CAUSE_RE = re.compile(r'"cause"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _salvage_objects(raw: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    index = raw.find("{", 1)
    while index >= 0:
        try:
            value, end = decoder.raw_decode(raw, index)
        except ValueError:
            index = raw.find("{", index + 1)
            continue
        if isinstance(value, dict):
            found.append(value)
        index = raw.find("{", end)
    return found


def parse_reply(text: str) -> dict[str, Any]:
    """{"cause", "hypotheses": [{text, kind, check_code}], "truncated"}; a reply cut
    off by the token limit keeps its cause and every complete hypothesis."""
    raw = str(text or "")
    start, end = raw.find("{"), raw.rfind("}")
    data: Any = {}
    if 0 <= start < end:
        try:
            data = json.loads(raw[start : end + 1])
        except ValueError:
            data = {}
    truncated = not isinstance(data, dict) or not data
    if truncated and start >= 0:
        match = _CAUSE_RE.search(raw)
        cause = ""
        if match:
            try:
                cause = json.loads(f'"{match.group(1)}"')
            except ValueError:
                cause = match.group(1)
        data = {"cause": cause, "hypotheses": [o for o in _salvage_objects(raw[start:]) if "text" in o]}
    if not isinstance(data, dict):
        data = {}
    hypotheses = []
    for item in data.get("hypotheses") or []:
        if not isinstance(item, dict):
            continue
        claim = " ".join(str(item.get("text") or "").split())
        code = str(item.get("check_code") or "").strip()
        if not claim or not code:
            continue
        kind = str(item.get("kind") or "rule").strip().lower()
        hypotheses.append({"text": claim[:300], "kind": kind if kind in ("rule", "goal") else "rule", "check_code": code})
    return {
        "cause": " ".join(str(data.get("cause") or "").split())[:300],
        "hypotheses": hypotheses,
        "truncated": bool(truncated and start >= 0),
    }


# ----- move summary for the request ---------------------------------------------------------


def _short(item: dict[str, Any]) -> str:
    if item["level_completed"]:
        return "level completed"
    if item["game_over"]:
        return "game over"
    if not item["board_changed"]:
        return "no change"
    deltas = sorted(set(item["moves"].values()))
    parts = []
    if len(deltas) == 1:
        count = len(item["moves"])
        parts.append(f"moved [{deltas[0][0]},{deltas[0][1]}]" + (f" ({count} objects)" if count > 1 else ""))
    elif deltas:
        parts.append("moved " + ", ".join(f"[{d[0]},{d[1]}]" for d in deltas[:3]))
    if item["appeared"]:
        parts.append(f"{len(item['appeared'])} appeared")
    if item["disappeared"]:
        parts.append(f"{len(item['disappeared'])} disappeared")
    return ", ".join(parts) or f"{item['changed_cells']} cells changed"


def move_lines(transitions: list[dict[str, Any]], *, border: int, max_lines: int, color_chars: str = "") -> list[str]:
    """Runs of the same action with the same effect merged; the last ``max_lines``."""
    if not transitions:
        return []
    height = len(transitions[0]["before"])
    width = len(transitions[0]["before"][0]) if height else 0
    mask = keeper_auto.border_mask(height, width, border)
    groups: list[list[Any]] = []  # [action, label, count, first, last]
    for item in transitions:
        label = _short(keeper_auto.effect(item, mask, color_chars))
        if groups and groups[-1][0] == item["action"] and groups[-1][1] == label:
            groups[-1][2] += 1
            groups[-1][4] = item["step"]
            continue
        groups.append([item["action"], label, 1, item["step"], item["step"]])
    out = [
        f"step {first} {action}: {label}" if count == 1 else f"steps {first}-{last} {action} x{count}: {label}"
        for action, label, count, first, last in groups
    ]
    if len(out) > max_lines:
        out = [f"... {len(out) - max_lines} earlier groups omitted", *out[-max_lines:]]
    return out


# ----- memory ----------------------------------------------------------------------------------


class KeeperMemory:
    def __init__(self, *, min_support: int, max_open: int, max_hypotheses: int, path: Path | None = None) -> None:
        self.min_support = max(1, int(min_support))
        self.max_open = max(0, int(max_open))
        self.max_hypotheses = max(1, int(max_hypotheses))
        self.path = path
        self.hypotheses: list[Hypothesis] = []
        self.causes: dict[int, str] = {}
        self.reviews: list[dict[str, Any]] = []
        self.last_level: int | None = None
        self._next = 1

    def request_text(self, level: int, transitions: list[dict[str, Any]], *, border: int, max_moves: int, color_chars: str = "") -> str:
        facts = [h for h in self.hypotheses if h.status == FACT]
        facts_text = ""
        if facts:
            facts_text = "Already verified facts (do not repeat them):\n" + "\n".join(f"- {h.text}" for h in facts[:8]) + "\n"
        return REQUEST_TEMPLATE.format(
            level=level,
            moves="\n".join(move_lines(transitions, border=border, max_lines=max_moves, color_chars=color_chars)),
            facts=facts_text,
            max_hypotheses=self.max_hypotheses,
        )

    def apply(
        self,
        level: int,
        reply_text: str,
        transitions: list[dict[str, Any]],
        run_check: CheckRunner,
        *,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        """Parse the reply, add new hypotheses, re-check everything open or
        verified on this level's transitions, and decide each status."""
        parsed = parse_reply(reply_text)
        summary: dict[str, Any] = {"level": level, "truncated": parsed["truncated"], "proposed": [],
                                   "promoted": [], "refuted": [], "demoted": [], "dropped": [], "skipped_checks": 0}
        if parsed["cause"]:
            self.causes[level] = parsed["cause"]
            summary["cause"] = parsed["cause"]
        known = {h.text.lower() for h in self.hypotheses}
        for item in parsed["hypotheses"][: self.max_hypotheses]:
            if item["text"].lower() in known:
                continue
            hypothesis = Hypothesis(id=f"K{self._next}", text=item["text"], kind=item["kind"],
                                    check_code=item["check_code"], level_proposed=level)
            self._next += 1
            self.hypotheses.append(hypothesis)
            known.add(hypothesis.text.lower())
            summary["proposed"].append(hypothesis.id)
        for hypothesis in [h for h in self.hypotheses if h.status in (FACT, OPEN)]:
            if deadline is not None and time.monotonic() >= deadline:
                summary["skipped_checks"] += 1
                continue
            self._decide(hypothesis, level, transitions, run_check, summary)
        self._enforce_open_cap(summary)
        self.last_level = level
        self.reviews.append(summary)
        self.save()
        return summary

    def _decide(self, h: Hypothesis, level: int, transitions: list[dict[str, Any]], run_check: CheckRunner,
                summary: dict[str, Any]) -> None:
        perturb = h.status != FACT and not h.falsifiable_seen
        report = run_check(h.check_code, transitions, perturb)
        status = str(report.get("status") or "error")
        if status == "timeout":
            return
        if status == "error":
            if h.status == FACT:
                return
            h.status, h.reason = DROPPED, f"check could not run: {str(report.get('error'))[:120]}"
            summary["dropped"].append(h.id)
            return
        if status == "fail":
            failure = report.get("first_failure") or {}
            was_fact = h.status == FACT
            h.status, h.reason = REFUTED, f"failed at step {failure.get('step')} ({failure.get('action')}) on level {level}"
            summary["demoted" if was_fact else "refuted"].append(h.id)
            return
        support = int(report.get("support") or 0)
        h.support += support
        if support and level not in h.levels_held:
            h.levels_held.append(level)
        if report.get("falsifiable") is True:
            h.falsifiable_seen = True
        if h.status == FACT:
            return
        if h.support >= self.min_support and h.falsifiable_seen:
            h.status, h.level_learned, h.reason = FACT, level, ""
            summary["promoted"].append(h.id)
        elif not h.falsifiable_seen and report.get("falsifiable") is False:
            h.reason = "unfalsifiable: also held on perturbed moves"
        else:
            h.reason = f"support {h.support} < {self.min_support}"

    def _enforce_open_cap(self, summary: dict[str, Any]) -> None:
        if self.max_open <= 0:
            return
        order = {h.id: i for i, h in enumerate(self.hypotheses)}
        open_ones = sorted((h for h in self.hypotheses if h.status == OPEN), key=lambda h: (-h.support, -order[h.id]))
        for h in open_ones[self.max_open:]:
            h.status, h.reason = DROPPED, f"more than {self.max_open} open hypotheses"
            summary["dropped"].append(h.id)

    def notes_block(self, *, max_chars: int = 1500) -> str:
        """The block added to the next level's opening message. Empty when there
        is nothing verified and no reviewed level yet."""
        if self.last_level is None:
            return ""
        facts = sorted((h for h in self.hypotheses if h.status == FACT), key=lambda h: (-h.support, h.id))[:8]
        ideas = sorted(
            (h for h in self.hypotheses if h.status == OPEN and not h.reason.startswith("unfalsifiable")),
            key=lambda h: -h.support,
        )[:3]
        lines = [f"{NOTES_HEADER} (facts are checked by the harness against your recorded moves; the rest is not verified)"]
        if facts:
            lines.append("Verified facts:")
            lines += [f"- {h.text} (support {h.support}, levels {','.join(map(str, h.levels_held))})" for h in facts]
        cause = self.causes.get(self.last_level)
        if cause:
            lines.append(f"Level {self.last_level}, likely cause: {cause}")
        if ideas:
            lines.append("Untested ideas:")
            lines += [f"- {h.text} (support {h.support})" for h in ideas]
        if len(lines) == 1:
            return ""
        block = "\n".join(lines)
        return block if len(block) <= max_chars else block[: max_chars - 4].rstrip() + " ..."

    def save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"hypotheses": [asdict(h) for h in self.hypotheses],
                                             "causes": self.causes, "reviews": self.reviews}, indent=2, default=str))
        except OSError:
            pass
