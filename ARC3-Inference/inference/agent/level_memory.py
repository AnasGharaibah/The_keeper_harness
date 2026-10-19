"""Per-level log, hypothesis review, and persistent verified-fact memory.

The model records claims with ``hypothesis(text, kind, check_code)``. When a
level ends (or the game is over / the run ends) the harness reviews them:
check code runs mechanically in the restricted sandbox against the recorded
transitions, an optional single LLM call may write missing check code or flag
contradictions, and only claims whose checks pass (with enough support and a
falsifiable check) become FACTS. Facts are re-checked on later levels and
demoted when they fail. The LLM never promotes anything by opinion.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from inference.agent.layer_config import LevelReviewConfig
from inference.agent.prediction import VERDICT_WRONG, format_prediction_line, prediction_counts
from inference.agent.runtime_state import HistoryEntry

STATUS_PENDING = "pending"
STATUS_FACT = "fact"
STATUS_REFUTED = "refuted"
STATUS_UNVERIFIED = "unverified"

# (check_code, transitions, alt_actions, perturb) -> sandbox check report
CheckRunner = Callable[[str, list[dict[str, Any]], list[str], bool], dict[str, Any]]
# (messages, timeout_seconds) -> assistant text; raises on failure
LlmCall = Callable[[list[dict[str, Any]], float], str]

_MIN_LLM_SECONDS = 5.0
_STANDARD_ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "SPACE"]

REVIEW_SYSTEM_PROMPT = (
    "You review claims made by an agent playing a grid puzzle game. "
    "You only write Python check code and flag contradictions; a harness runs the code on recorded "
    "transitions and decides what is true. Reply with one JSON object and nothing else."
)

CHECK_API_TEXT = (
    "Check API: write `def check_transition(t):` returning True (claim held on this transition), "
    "False (claim violated) or None (claim does not apply). `t.action` is a string like 'RIGHT' or "
    "'MOUSE(row=4, col=7)'; `t.before_frame` / `t.after_frame` expose `.ascii` (newline-separated rows of "
    "color chars), `.segmentation` ({'nodes': [{id, color, hash, pixels, boundary, children}], 'adjacency_list'}), "
    "`.level`, `.step`; `t.result` is a dict with board_changed, level_completed, game_over, reward. "
    "Helpers: `cell(frame, row, col)` -> color char, `changed_cells(a, b)` -> [(row, col, old, new)], "
    "`parse_action(t.action)` -> (name, row, col). Allowed imports: collections, itertools, math, re, json, "
    "functools, statistics. Keep checks short and specific; return None when the claim is not about t."
)


@dataclass
class Hypothesis:
    id: str
    text: str
    kind: str
    level: int
    check_code: str | None = None
    check_origin: str = "model"
    status: str = STATUS_PENDING
    reason: str = ""
    support: int = 0
    levels_held: list[int] = field(default_factory=list)
    level_learned: int | None = None
    refuted_at_level: int | None = None
    falsifiable_seen: bool = False
    checked_through_step: int = 0
    quick_check: dict[str, Any] | None = None

    def tag(self) -> str:
        return f"{self.id} [{self.kind}]"


def _base_action(action: str) -> str:
    return str(action or "").split("(", 1)[0].strip().upper()


def level_transitions(
    history: list[HistoryEntry],
    level: int,
    results_by_step: dict[int, dict[str, Any]],
    *,
    cap: int,
) -> list[dict[str, Any]]:
    """Compact transitions whose before-frame is on ``level`` (RESETs excluded).

    Frames travel as lists of row strings in ARC color chars, not int grids."""
    items: list[dict[str, Any]] = []
    for index in range(1, len(history)):
        before = history[index - 1].frame
        after = history[index].frame
        action = history[index].action
        if before.level != level or not action or _base_action(action) == "RESET":
            continue
        items.append(
            {
                "action": action,
                "before": before.ascii.split("\n"),
                "after": after.ascii.split("\n"),
                "before_step": before.step,
                "before_level": before.level,
                "step": after.step,
                "level": after.level,
                "result": dict(results_by_step.get(after.step) or {}),
            }
        )
    return items[-cap:] if cap > 0 else items


def parse_review_response(text: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Parse ``{"checks": [{id, check_code}], "contradicted": [{id, why}]}`` leniently."""
    raw = str(text or "")
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return [], []
    try:
        data = json.loads(raw[start : end + 1])
    except (TypeError, ValueError):
        return [], []
    if not isinstance(data, dict):
        return [], []
    checks: list[tuple[str, str]] = []
    for item in data.get("checks") or []:
        if isinstance(item, dict) and str(item.get("id", "")).strip() and str(item.get("check_code", "")).strip():
            checks.append((str(item["id"]).strip(), str(item["check_code"])))
    contradicted: list[tuple[str, str]] = []
    for item in data.get("contradicted") or []:
        if isinstance(item, dict) and str(item.get("id", "")).strip():
            contradicted.append((str(item["id"]).strip(), " ".join(str(item.get("why", "")).split())[:200]))
        elif isinstance(item, str) and item.strip():
            contradicted.append((item.strip(), ""))
    return checks, contradicted


class LevelMemory:
    """Hypotheses, per-level stats and verified facts for one game run."""

    def __init__(self, config: LevelReviewConfig, facts_path: Path | None = None) -> None:
        self.config = config
        self.facts_path = facts_path
        self.hypotheses: list[Hypothesis] = []
        self.level_stats: list[dict[str, Any]] = []
        self.next_index = 1
        self._results_by_step: dict[int, dict[str, Any]] = {}
        self._pending: list[tuple[int, str]] = []
        self._llm_calls_by_level: dict[int, int] = {}
        self._notices: list[str] = []

    # ----- recording ---------------------------------------------------

    def record_transition_results(self, per_action: Any) -> None:
        for item in per_action or []:
            if not isinstance(item, dict):
                continue
            try:
                step = int(item.get("action_num"))
            except (TypeError, ValueError):
                continue
            self._results_by_step[step] = {
                key: item.get(key)
                for key in ("board_changed", "level_completed", "game_over", "run_complete", "reward")
                if key in item
            }

    def transition_results_payload(self) -> dict[str, dict[str, Any]]:
        return {str(step): dict(result) for step, result in self._results_by_step.items()}

    def get(self, hypothesis_id: str) -> Hypothesis | None:
        return next((item for item in self.hypotheses if item.id == hypothesis_id), None)

    def add_hypotheses(self, raw_items: Any, *, default_level: int | None) -> list[str]:
        """Record hypotheses returned by one sandbox call. Returns one line per item
        for the tool result. Ids follow the sandbox's numbering so the model's ids
        stay valid; duplicates are merged into the earlier hypothesis."""
        lines: list[str] = []
        items = [item for item in raw_items or [] if isinstance(item, dict)]
        for offset, item in enumerate(items):
            proposed_id = str(item.get("id") or "").strip() or f"H{self.next_index + offset}"
            text = " ".join(str(item.get("text") or "").split())
            kind = str(item.get("kind") or "rule").strip().lower()
            if kind not in ("rule", "goal"):
                kind = "rule"
            check_code = item.get("check_code") or None
            if not text:
                continue
            duplicate = next(
                (h for h in self.hypotheses if h.text.lower() == text.lower() and h.kind == kind),
                None,
            )
            if duplicate is not None:
                if check_code and duplicate.status not in (STATUS_FACT, STATUS_REFUTED):
                    duplicate.check_code = str(check_code)
                    duplicate.check_origin = "model"
                    duplicate.status = STATUS_PENDING
                    duplicate.reason = ""
                lines.append(f"{proposed_id} is the same claim as {duplicate.id} ({duplicate.status}); merged")
                continue
            if len(self.hypotheses) >= self.config.max_hypotheses:
                lines.append(f"{proposed_id} ignored: hypothesis limit {self.config.max_hypotheses} reached")
                continue
            try:
                level = int(item.get("level") or default_level or 1)
            except (TypeError, ValueError):
                level = int(default_level or 1)
            hypothesis = Hypothesis(
                id=proposed_id,
                text=text[:400],
                kind=kind,
                level=level,
                check_code=str(check_code) if check_code else None,
                quick_check=item.get("quick_check") if isinstance(item.get("quick_check"), dict) else None,
            )
            self.hypotheses.append(hypothesis)
            lines.append(self._recorded_line(hypothesis))
        self.next_index += len(items)
        return lines

    def _recorded_line(self, hypothesis: Hypothesis) -> str:
        if not hypothesis.check_code:
            return f"{hypothesis.tag()} recorded without check_code (review may write one)"
        quick = hypothesis.quick_check or {}
        status = str(quick.get("status") or "not run")
        detail = ""
        if quick.get("support") is not None:
            detail = f", support {quick.get('support')}"
        if quick.get("error"):
            detail = f": {str(quick.get('error'))[:160]}"
        return f"{hypothesis.tag()} recorded; quick check on this level so far: {status}{detail}"

    def mark_review_pending(self, level: int | None, reason: str) -> None:
        if level is None:
            return
        entry = (int(level), str(reason))
        if entry not in self._pending:
            self._pending.append(entry)

    def has_pending(self) -> bool:
        return bool(self._pending)

    def pop_notices(self) -> list[str]:
        notices, self._notices = self._notices, []
        return notices

    # ----- review ------------------------------------------------------

    def run_pending_reviews(
        self,
        *,
        history: list[HistoryEntry],
        prediction_entries: list[dict[str, Any]],
        check_runner: CheckRunner,
        llm_call: LlmCall | None,
        time_budget_seconds: float,
    ) -> list[dict[str, Any]]:
        deadline = time.monotonic() + max(0.0, float(time_budget_seconds))
        summaries: list[dict[str, Any]] = []
        pending, self._pending = self._pending, []
        for level, reason in pending:
            summaries.append(
                self._review_level(
                    level,
                    reason,
                    history=history,
                    prediction_entries=prediction_entries,
                    check_runner=check_runner,
                    llm_call=llm_call,
                    deadline=deadline,
                )
            )
        self.save()
        return summaries

    def _review_level(
        self,
        level: int,
        reason: str,
        *,
        history: list[HistoryEntry],
        prediction_entries: list[dict[str, Any]],
        check_runner: CheckRunner,
        llm_call: LlmCall | None,
        deadline: float,
    ) -> dict[str, Any]:
        started = time.monotonic()
        transitions = level_transitions(
            history, level, self._results_by_step, cap=self.config.max_review_transitions
        )
        max_step = max((int(item["step"]) for item in transitions), default=0)
        alt_actions = sorted({_base_action(item["action"]) for item in transitions} | set(_STANDARD_ACTIONS))
        level_predictions = [entry for entry in prediction_entries if entry.get("level") == level]
        summary: dict[str, Any] = {
            "level": level,
            "reason": reason,
            "transitions": len(transitions),
            "predictions": prediction_counts(level_predictions),
            "promoted": [],
            "refuted": [],
            "unverified": [],
            "llm": "disabled" if llm_call is None else "not needed",
        }

        # Facts first (demotion), then every open hypothesis that has check code.
        for hypothesis in self._checkable(level):
            if time.monotonic() >= deadline:
                summary["budget_exhausted"] = True
                break
            self._check_and_decide(hypothesis, level, transitions, max_step, alt_actions, check_runner, summary)

        candidates = [
            h for h in self.hypotheses
            if h.level <= level and h.status not in (STATUS_FACT, STATUS_REFUTED)
        ]
        needs_code = [h for h in candidates if not h.check_code]
        remaining = deadline - time.monotonic()
        if llm_call is not None and candidates:
            used = self._llm_calls_by_level.get(level, 0)
            if used >= self.config.max_llm_calls_per_level:
                summary["llm"] = "skipped: per-level call limit"
            elif remaining < _MIN_LLM_SECONDS:
                summary["llm"] = "skipped: no time left"
            else:
                self._llm_calls_by_level[level] = used + 1
                timeout = min(self.config.llm_timeout_seconds, remaining - 1.0)
                messages = self.build_review_messages(level, transitions, level_predictions, candidates)
                try:
                    reply = llm_call(messages, timeout)
                except Exception as exc:  # noqa: BLE001 - any failure just skips the LLM step
                    summary["llm"] = f"failed: {type(exc).__name__}"
                else:
                    summary["llm"] = "ran"
                    checks, contradicted = parse_review_response(reply)
                    candidate_ids = {h.id for h in needs_code}
                    for hypothesis_id, check_code in checks:
                        hypothesis = self.get(hypothesis_id)
                        if hypothesis is None or hypothesis.id not in candidate_ids:
                            continue
                        hypothesis.check_code = check_code
                        hypothesis.check_origin = "review"
                        hypothesis.checked_through_step = 0
                        if time.monotonic() >= deadline:
                            summary["budget_exhausted"] = True
                            continue
                        self._check_and_decide(
                            hypothesis, level, transitions, max_step, alt_actions, check_runner, summary
                        )
                    shown_ids = {h.id for h in candidates}
                    for hypothesis_id, why in contradicted:
                        hypothesis = self.get(hypothesis_id)
                        # "Contradicted" may refute a claim with no passing check; it never
                        # overrides a check that passed (checked_through_step > 0) or a fact.
                        if (
                            hypothesis is None
                            or hypothesis.id not in shown_ids
                            or hypothesis.status in (STATUS_FACT, STATUS_REFUTED)
                            or hypothesis.checked_through_step > 0
                        ):
                            continue
                        self._refute(hypothesis, level, f"review: contradicted by the evidence. {why}".strip(), summary)

        summary["facts"] = sum(1 for h in self.hypotheses if h.status == STATUS_FACT)
        summary["seconds"] = round(time.monotonic() - started, 2)
        self.level_stats.append(summary)
        self._notices.append(self._summary_notice(summary))
        return summary

    def _checkable(self, level: int) -> list[Hypothesis]:
        facts = [h for h in self.hypotheses if h.status == STATUS_FACT and h.check_code]
        others = [
            h for h in self.hypotheses
            if h.status in (STATUS_PENDING, STATUS_UNVERIFIED) and h.check_code and h.level <= level
        ]
        return facts + others

    def _check_and_decide(
        self,
        hypothesis: Hypothesis,
        level: int,
        transitions: list[dict[str, Any]],
        max_step: int,
        alt_actions: list[str],
        check_runner: CheckRunner,
        summary: dict[str, Any],
    ) -> None:
        new = [item for item in transitions if int(item["step"]) > hypothesis.checked_through_step]
        if not new:
            return
        perturb = self.config.perturbation_test and hypothesis.status != STATUS_FACT and not hypothesis.falsifiable_seen
        try:
            outcome = check_runner(str(hypothesis.check_code), new, alt_actions, perturb)
        except Exception as exc:  # noqa: BLE001
            outcome = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        status = str(outcome.get("status") or "error")

        if status in ("timeout", "skipped"):
            hypothesis.reason = f"check timed out on level {level}; kept as {hypothesis.status}"
            return
        if status == "error":
            if hypothesis.check_origin == "review" and hypothesis.status != STATUS_FACT:
                hypothesis.check_code = None
                hypothesis.reason = f"review-written check failed to run: {str(outcome.get('error'))[:160]}"
                return
            self._refute(hypothesis, level, f"check raised {str(outcome.get('error'))[:160]}", summary)
            return
        if status == "fail" or int(outcome.get("failures") or 0) > 0:
            failure = outcome.get("first_failure") or {}
            where = ""
            if failure:
                where = f" at step {failure.get('step')} ({failure.get('action')})"
            self._refute(hypothesis, level, f"check failed on level {level}{where}", summary)
            return

        hypothesis.checked_through_step = max(hypothesis.checked_through_step, max_step)
        support = int(outcome.get("support") or 0)
        hypothesis.support += support
        if support > 0 and level not in hypothesis.levels_held:
            hypothesis.levels_held.append(level)
        if outcome.get("falsifiable") is True:
            hypothesis.falsifiable_seen = True
        if hypothesis.status == STATUS_FACT:
            return

        if hypothesis.support < self.config.min_support:
            hypothesis.status = STATUS_UNVERIFIED
            hypothesis.reason = f"support {hypothesis.support} < min_support {self.config.min_support}"
            summary["unverified"].append(hypothesis.id)
        elif self.config.perturbation_test and not hypothesis.falsifiable_seen:
            hypothesis.status = STATUS_UNVERIFIED
            hypothesis.reason = "unfalsifiable: also passed on harness-perturbed transitions"
            summary["unverified"].append(hypothesis.id)
        else:
            hypothesis.status = STATUS_FACT
            hypothesis.reason = ""
            hypothesis.level_learned = level
            summary["promoted"].append(hypothesis.id)

    def _refute(self, hypothesis: Hypothesis, level: int, reason: str, summary: dict[str, Any]) -> None:
        was_fact = hypothesis.status == STATUS_FACT
        hypothesis.status = STATUS_REFUTED
        hypothesis.reason = reason
        hypothesis.refuted_at_level = level
        summary["refuted"].append(hypothesis.id)
        if was_fact:
            self._notices.append(f"FACT DEMOTED: {hypothesis.tag()} {hypothesis.text} -- {reason}.")

    def _summary_notice(self, summary: dict[str, Any]) -> str:
        parts = [f"LEVEL REVIEW (level {summary['level']}, {summary['reason']}, {summary['transitions']} transitions):"]
        counts = summary.get("predictions") or {}
        if any(counts.values()):
            parts.append(
                f"predictions {counts.get('confirmed', 0)} confirmed / {counts.get('wrong', 0)} wrong / "
                f"{counts.get('no_prediction', 0)} no_prediction;"
            )
        for key, label in (("promoted", "new facts"), ("refuted", "refuted"), ("unverified", "unverified")):
            ids = summary.get(key) or []
            if ids:
                details = []
                for hypothesis_id in ids:
                    hypothesis = self.get(hypothesis_id)
                    if key == "promoted" or hypothesis is None or not hypothesis.reason:
                        details.append(hypothesis_id)
                    else:
                        details.append(f"{hypothesis_id} ({hypothesis.reason[:120]})")
                parts.append(f"{label}: {', '.join(details)};")
        if len(parts) == 1 or (len(parts) == 2 and any(counts.values())):
            parts.append("no hypothesis changed status;")
        return " ".join(parts).rstrip(";") + "."

    def build_review_messages(
        self,
        level: int,
        transitions: list[dict[str, Any]],
        level_predictions: list[dict[str, Any]],
        candidates: list[Hypothesis],
    ) -> list[dict[str, Any]]:
        actions = [str(item["action"]) for item in transitions]
        compressed: list[str] = []
        for action in actions:
            if compressed and compressed[-1].split(" x", 1)[0] == action:
                head, _, count = compressed[-1].partition(" x")
                compressed[-1] = f"{head} x{int(count or 1) + 1}"
            else:
                compressed.append(action)
        events = [
            f"step {item['step']} {item['action']}: "
            + ", ".join(key for key in ("level_completed", "game_over") if (item.get("result") or {}).get(key))
            for item in transitions
            if (item.get("result") or {}).get("level_completed") or (item.get("result") or {}).get("game_over")
        ]
        wrong = [format_prediction_line(entry) for entry in level_predictions if entry.get("verdict") == VERDICT_WRONG]
        facts = [h for h in self.hypotheses if h.status == STATUS_FACT]
        lines = [
            f"Level {level}: {len(transitions)} recorded transitions.",
            f"Action sequence: {', '.join(compressed[:80])}{' ...' if len(compressed) > 80 else ''}",
        ]
        if events:
            lines.append("Notable outcomes: " + "; ".join(events[:10]))
        if wrong:
            lines.append("Predictions the harness found WRONG:")
            lines.extend(f"- {line}" for line in wrong[:10])
        if facts:
            lines.append("Already verified facts (context only):")
            lines.extend(f"- {h.tag()} {h.text}" for h in facts[:10])
        lines.append("Open hypotheses:")
        for hypothesis in candidates:
            state = "no check_code" if not hypothesis.check_code else f"{hypothesis.status}: {hypothesis.reason or 'awaiting evidence'}"
            lines.append(f"- {hypothesis.tag()} {hypothesis.text} ({state})")
        lines.extend(
            [
                "",
                CHECK_API_TEXT,
                "Tasks: (i) for each open hypothesis marked 'no check_code', write check code that would detect "
                "a violation of the claim; (ii) list hypotheses the evidence above contradicts.",
                'Reply JSON: {"checks": [{"id": "H1", "check_code": "def check_transition(t):\\n    ..."}], '
                '"contradicted": [{"id": "H2", "why": "short reason"}]}',
            ]
        )
        return [
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ]

    # ----- context + persistence --------------------------------------

    def render_pinned_block(self) -> str:
        facts = sorted(
            (h for h in self.hypotheses if h.status == STATUS_FACT),
            key=lambda h: (-h.support, -len(h.levels_held), h.id),
        )[: self.config.max_facts_in_context]
        refuted = sorted(
            (h for h in self.hypotheses if h.status == STATUS_REFUTED),
            key=lambda h: (-(h.refuted_at_level or 0), h.id),
        )[: self.config.max_refuted_in_context]
        if not facts and not refuted:
            return ""
        lines = [
            "VERIFIED FACTS (harness-checked against recorded transitions of this game; "
            "they survive context trimming; re-checked on every new level):"
        ]
        if facts:
            for h in facts:
                held = f", held on {len(h.levels_held)} level(s)" if len(h.levels_held) > 1 else ""
                lines.append(f"- {h.tag()} {h.text} (learned L{h.level_learned}, support {h.support}{held})")
        else:
            lines.append("- none yet")
        if refuted:
            lines.append("REFUTED (do not rely on these):")
            for h in refuted:
                lines.append(f"- {h.tag()} {h.text} -- {h.reason[:120]}")
        block = "\n".join(lines)
        if len(block) > self.config.max_block_chars:
            block = block[: self.config.max_block_chars - 20].rstrip() + "\n- ... (truncated)"
        return block

    def to_payload(self) -> dict[str, Any]:
        return {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": asdict(self.config),
            "hypotheses": [asdict(h) for h in self.hypotheses],
            "level_stats": list(self.level_stats),
        }

    def save(self) -> None:
        if self.facts_path is None:
            return
        try:
            self.facts_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.facts_path.with_suffix(f"{self.facts_path.suffix}.tmp")
            tmp_path.write_text(json.dumps(self.to_payload(), indent=2, default=str), encoding="utf-8")
            tmp_path.replace(self.facts_path)
        except OSError:
            pass
