"""Harness-side perception over consecutive frames (no LLM).

Three detectors, all general (no game-specific rules):

* Transformations: an object that changed is classified against a fixed set --
  rotate 90/180/270, flip H/V, recolor, scale, translate (alone or combined with
  one of the others). Objects are tracked across the change by position overlap
  and color, not by hash, because a rotated object gets a new hash.
* Relations: two separate objects whose shapes match up to rotation/flip (after
  removing an integer scale). Reported only when a match starts, ends or changes
  orientation, and only when one of the two objects transformed on that move.
* Roles: objects that move consistently with the arrow keys are "likely player";
  similar-sized objects that never move with the arrows are "not the player".
  Reported once, and again only when the assignment changes.

An "object" is a 4-connected same-color component, or -- when a container holds
several same-color pieces (a pattern drawn inside a box) -- the group of those
pieces. Everything here is pure: grids in, plain strings out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from inference.agent.prediction import grid_components

Cell = tuple[int, int]
Shape = frozenset[Cell]

ARROWS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
_MAX_OBJECT_FRACTION = 0.125  # larger components are background / walls
_ROLE_MIN_PRESENCE = 5
_ROLE_MIN_CONSISTENT = 3
_ROLE_MAX_LISTED = 3
_ROLE_MIN_SHARE = 0.3  # a player moves on at least this share of board-changing arrow moves


# ----- shapes ------------------------------------------------------------------


def normalize(cells: set[Cell] | frozenset[Cell]) -> Shape:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _dims(shape: Shape) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _rot90(shape: Shape) -> Shape:
    h, _w = _dims(shape)
    return normalize({(c, h - 1 - r) for r, c in shape})


def _rot180(shape: Shape) -> Shape:
    h, w = _dims(shape)
    return normalize({(h - 1 - r, w - 1 - c) for r, c in shape})


def _rot270(shape: Shape) -> Shape:
    _h, w = _dims(shape)
    return normalize({(w - 1 - c, r) for r, c in shape})


def _flip_h(shape: Shape) -> Shape:
    _h, w = _dims(shape)
    return normalize({(r, w - 1 - c) for r, c in shape})


def _flip_v(shape: Shape) -> Shape:
    h, _w = _dims(shape)
    return normalize({(h - 1 - r, c) for r, c in shape})


# Fixed order: rotations before flips (a shape can satisfy both).
DIHEDRAL: list[tuple[str, Callable[[Shape], Shape]]] = [
    ("ROTATED 90° CW", _rot90),
    ("ROTATED 180°", _rot180),
    ("ROTATED 90° CCW", _rot270),
    ("FLIPPED horizontally", _flip_h),
    ("FLIPPED vertically", _flip_v),
]


def reduce_scale(shape: Shape) -> tuple[Shape, int]:
    """Largest integer k such that the shape is made of aligned k x k blocks."""
    h, w = _dims(shape)
    for k in range(min(h, w), 1, -1):
        if h % k or w % k:
            continue
        blocks = {(r // k, c // k) for r, c in shape}
        if len(blocks) * k * k != len(shape):
            continue
        if all((br * k + dr, bc * k + dc) in shape for br, bc in blocks for dr in range(k) for dc in range(k)):
            return frozenset(blocks), k
    return shape, 1


_RELATION_NAMES = {
    "ROTATED 90° CW": "rotated 90° CW",
    "ROTATED 180°": "rotated 180°",
    "ROTATED 90° CCW": "rotated 90° CCW",
    "FLIPPED horizontally": "flipped horizontally",
    "FLIPPED vertically": "flipped vertically",
}


def relation_between(a: Shape, b: Shape) -> str | None:
    """How ``b`` relates to ``a`` (both scale-reduced): 'exactly' or a dihedral name."""
    if a == b:
        return "exactly"
    for name, fn in DIHEDRAL:
        if fn(a) == b:
            return _RELATION_NAMES[name]
    return None


# ----- objects -----------------------------------------------------------------


@dataclass
class PObject:
    color: str
    cells: frozenset[Cell]
    kind: str  # "object" (one component) or "pattern" (same-color pieces in one container)
    hash: str | None = None
    shape: Shape = field(init=False)
    reduced: Shape = field(init=False)
    scale: int = field(init=False)
    top_left: Cell = field(init=False)
    dims: tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        self.shape = normalize(self.cells)
        self.reduced, self.scale = reduce_scale(self.shape)
        self.top_left = (min(r for r, _ in self.cells), min(c for _, c in self.cells))
        self.dims = _dims(self.shape)

    @property
    def pixels(self) -> int:
        return len(self.cells)

    @property
    def center(self) -> tuple[int, int]:
        h, w = self.dims
        return (round(self.top_left[0] + (h - 1) / 2), round(self.top_left[1] + (w - 1) / 2))

    @property
    def key(self) -> tuple[str, int, int]:
        """Identity that survives rotation in place: color plus bounding-box center."""
        return (self.color, *self.center)

    @property
    def place(self) -> tuple[int, int]:
        """Position-only identity for shape relations (a recolor keeps the relation)."""
        return self.center

    @property
    def solid(self) -> bool:
        h, w = self.dims
        return self.pixels == h * w

    def bbox_overlaps(self, other: "PObject") -> bool:
        (r0, c0), (h0, w0) = self.top_left, self.dims
        (r1, c1), (h1, w1) = other.top_left, other.dims
        return r0 < r1 + h1 and r1 < r0 + h0 and c0 < c1 + w1 and c1 < c0 + w0

    def label(self) -> str:
        h, w = self.dims
        where = f"at ({self.top_left[0]},{self.top_left[1]}) {h}x{w}"
        if self.kind == "pattern":
            return f"{self.color} pattern {where}"
        return f"obj {self.hash} ({self.color}) {where}" if self.hash else f"{self.color} object {where}"


def frame_objects(grid: Any, mask: set[Cell], *, min_size: int) -> list[PObject]:
    """Perception objects of one frame, ignoring HUD cells, tiny and background-sized ones."""
    height = len(grid)
    width = len(grid[0]) if height else 0
    max_pixels = max(min_size, int(height * width * _MAX_OBJECT_FRACTION))
    components = grid_components(grid, with_cells=True)
    owner = [[0] * width for _ in range(height)]
    for comp in components:
        for r, c in comp["cells"]:
            owner[r][c] = comp["id"]
    # A piece's container is the one other component holding all its outside neighbours.
    by_container: dict[tuple[int, str], list[int]] = {}
    for comp in components:
        if comp["pixels"] > max_pixels:
            continue
        neighbours: set[int] = set()
        touches_edge = False
        for r, c in comp["cells"]:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < height and 0 <= nc < width):
                    touches_edge = True
                    continue
                other = owner[nr][nc]
                if other != comp["id"]:
                    neighbours.add(other)
            if touches_edge or len(neighbours) > 1:
                break
        if not touches_edge and len(neighbours) == 1:
            by_container.setdefault((neighbours.pop(), comp["color"]), []).append(comp["id"])
    grouped: set[int] = set()
    objects: list[PObject] = []
    for (_container, color), ids in by_container.items():
        if len(ids) < 2:
            continue
        cells = frozenset(cell for i in ids for cell in components[i]["cells"])
        grouped.update(ids)
        objects.append(PObject(color=color, cells=cells, kind="pattern"))
    for comp in components:
        if comp["id"] in grouped:
            continue
        objects.append(PObject(color=comp["color"], cells=frozenset(comp["cells"]), kind="object", hash=comp["hash"]))
    return [
        obj
        for obj in objects
        if min_size <= obj.pixels <= max_pixels and not (mask and obj.cells & mask)
    ]


# ----- transformations ----------------------------------------------------------


def _delta_text(before: PObject, after: PObject) -> str:
    dr = after.top_left[0] - before.top_left[0]
    dc = after.top_left[1] - before.top_left[1]
    return f" and moved [{dr},{dc}]" if (dr, dc) != (0, 0) else ""


def classify_change(before: PObject, after: PObject) -> str | None:
    """Transformation name for a tracked pair, or None when it is none of the fixed set."""
    if before.cells == after.cells:
        return f"RECOLORED {before.color}->{after.color}" if before.color != after.color else None
    if before.color != after.color:
        return None
    if before.shape == after.shape:
        return "TRANSLATED"
    if before.pixels == after.pixels:
        for name, fn in DIHEDRAL:
            if fn(before.shape) == after.shape:
                return name
    if before.reduced == after.reduced and before.scale != after.scale:
        return f"SCALED x{after.scale}/{before.scale}" if after.scale % before.scale else f"SCALED x{after.scale // before.scale}"
    return None


def detect_transformations(before: list[PObject], after: list[PObject]) -> list[tuple[PObject, PObject, str]]:
    """Pairs (before, after, name) for objects that changed, tracked by bbox overlap + color
    (or identical cells for a recolor)."""
    unchanged_before = {(o.color, o.cells) for o in after}
    unchanged_after = {(o.color, o.cells) for o in before}
    changed_before = [o for o in before if (o.color, o.cells) not in unchanged_before]
    changed_after = [o for o in after if (o.color, o.cells) not in unchanged_after]
    used: set[int] = set()
    found: list[tuple[PObject, PObject, str]] = []
    for old in changed_before:
        candidates = [
            (index, new)
            for index, new in enumerate(changed_after)
            if index not in used and old.bbox_overlaps(new) and (new.color == old.color or new.cells == old.cells)
        ]
        for index, new in candidates:
            name = classify_change(old, new)
            if name is None:
                continue
            # Single components that only translate are already reported as moves.
            if name == "TRANSLATED" and old.kind == "object":
                break
            used.add(index)
            found.append((old, new, name))
            break
    # A container whose holes follow the pattern inside it makes the same change;
    # keep only the innermost object.
    return [
        item
        for item in found
        if not any(
            other is not item and other[2] == item[2] and _contains(item[1], other[1])
            for other in found
        )
    ]


def _contains(outer: PObject, inner: PObject) -> bool:
    (r0, c0), (h0, w0) = outer.top_left, outer.dims
    (r1, c1), (h1, w1) = inner.top_left, inner.dims
    return (r0 <= r1 and c0 <= c1 and r1 + h1 <= r0 + h0 and c1 + w1 <= c0 + w0) and outer.pixels > inner.pixels


# ----- relations ------------------------------------------------------------------


def _relation_candidates(objects: list[PObject]) -> list[PObject]:
    return [o for o in objects if not o.solid and len(o.reduced) >= 2]


def relation_state(objects: list[PObject]) -> dict[tuple[Any, Any], tuple[str, PObject, PObject]]:
    """{(key_a, key_b): (relation, a, b)} for every pair whose shapes match up to rotation/flip."""
    candidates = sorted(_relation_candidates(objects), key=lambda o: o.place)
    by_size: dict[int, list[PObject]] = {}
    for obj in candidates:
        by_size.setdefault(len(obj.reduced), []).append(obj)
    state: dict[tuple[Any, Any], tuple[str, PObject, PObject]] = {}
    for group in by_size.values():
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                if a.place == b.place:
                    continue
                relation = relation_between(a.reduced, b.reduced)
                if relation is not None:
                    state[(a.place, b.place)] = (relation, a, b)
    return state


def _scale_note(a: PObject, b: PObject) -> str:
    return f", scale {a.scale}:{b.scale}" if a.scale != b.scale else ""


# ----- roles ------------------------------------------------------------------------


@dataclass
class _RoleStats:
    consistent: int = 0
    inconsistent: int = 0
    color: str = ""
    pixels: int = 0


class PerceptionTracker:
    """Replays a level move by move and returns new perception lines per move."""

    def __init__(self, *, min_size: int = 3, max_lines: int = 3) -> None:
        self.min_size = max(1, int(min_size))
        self.max_lines = max(0, int(max_lines))
        self._reset(None)

    def _reset(self, level: int | None) -> None:
        self.level = level
        self._relations: dict[tuple[Any, Any], tuple[str, PObject, PObject]] | None = None
        self._movers: dict[str, _RoleStats] = {}
        self._static: dict[tuple[str, int, int], dict[str, Any]] = {}
        self._role_state: tuple[Any, ...] | None = None
        self._arrow_changes = 0
        self._cache: dict[int, list[PObject]] = {}  # frame content -> objects

    def _objects(self, grid: Any, mask: set[Cell]) -> list[PObject]:
        key = hash((tuple(map(tuple, grid)), frozenset(mask)))
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[key] = frame_objects(grid, mask, min_size=self.min_size)
        return self._cache[key]

    def observe(self, transition: dict[str, Any], mask: set[Cell], *, level: int | None = None) -> list[str]:
        """Lines for one transition (``action``, ``before``/``after`` int grids, ``result``),
        at most ``max_lines``: relations first, then transformations, then roles."""
        if level != self.level:
            self._reset(level)
        before = self._objects(transition["before"], mask)
        after = self._objects(transition["after"], mask)
        result = transition.get("result") or {}
        if result.get("level_completed") or result.get("game_over"):
            return []

        transformed = detect_transformations(before, after)
        transform_lines = []
        recolors = [(old, new, name) for old, new, name in transformed if name.startswith("RECOLORED")]
        for old, new, name in transformed:
            if name == "TRANSLATED":
                dr, dc = new.top_left[0] - old.top_left[0], new.top_left[1] - old.top_left[1]
                transform_lines.append(f"{new.label()} moved [{dr},{dc}]")
            elif name.startswith("RECOLORED") and len(recolors) > 1:
                continue
            else:
                transform_lines.append(f"{new.label()} {name}{_delta_text(old, new)}")
        if len(recolors) > 1:
            parts = [f"({new.top_left[0]},{new.top_left[1]}) {old.color}->{new.color}" for old, new, _ in recolors[:4]]
            more = f" +{len(recolors) - 4} more" if len(recolors) > 4 else ""
            transform_lines.append(f"{len(recolors)} objects RECOLORED: {', '.join(parts)}{more}")
        touched = {o.place for pair in transformed for o in pair[:2] if pair[2] != "TRANSLATED"}

        relation_lines = self._relation_lines(before, after, touched)
        role_lines = self._role_lines(transition, before, after, mask)
        lines = [*relation_lines, *transform_lines, *role_lines]
        return lines[: self.max_lines]

    def _relation_lines(self, before: list[PObject], after: list[PObject], touched: set[Any]) -> list[str]:
        previous = self._relations if self._relations is not None else relation_state(before)
        current = relation_state(after)
        self._relations = current
        if not touched:
            return []
        lines: list[str] = []
        for pair in sorted(set(previous) | set(current), key=str):
            if pair[0] not in touched and pair[1] not in touched:
                continue
            old = previous.get(pair)
            new = current.get(pair)
            if old and new and old[0] == new[0]:
                continue
            # Name the transformed member first; describe it relative to the other one.
            if new:
                _relation, a, b = new
                if a.place not in touched:
                    a, b = b, a
                relation = relation_between(b.reduced, a.reduced) or _relation
                if relation == "exactly":
                    lines.append(f"{a.label()} now MATCHES {b.label()} exactly (same shape and orientation{_scale_note(a, b)})")
                elif old and old[0] == "exactly":
                    lines.append(f"{a.label()} no longer matches {b.label()} exactly (it is now {relation} relative to it)")
                elif old:
                    lines.append(f"{a.label()} is now {relation} relative to {b.label()} (same shape, not yet matching)")
                else:
                    lines.append(f"{a.label()} now matches {b.label()} up to rotation/flip ({relation}{_scale_note(a, b)})")
            elif old:
                _relation, a, b = old
                if a.place not in touched:
                    a, b = b, a
                lines.append(f"{a.label()} no longer matches {b.label()}")
        return lines

    def _role_lines(self, transition: dict[str, Any], before: list[PObject], after: list[PObject], mask: set[Cell]) -> list[str]:
        name = str(transition.get("action") or "").split("(", 1)[0].strip().upper()
        if name not in ARROWS:
            return []
        if transition["before"] == transition["after"]:
            return []
        want = ARROWS[name]
        self._arrow_changes += 1
        before_hashes: dict[str, PObject] = {}
        counts: dict[str, int] = {}
        for obj in before:
            if obj.hash:
                counts[obj.hash] = counts.get(obj.hash, 0) + 1
                before_hashes[obj.hash] = obj
        after_by_hash: dict[str, list[PObject]] = {}
        for obj in after:
            if obj.hash:
                after_by_hash.setdefault(obj.hash, []).append(obj)
        moved_keys: set[Any] = set()
        for object_hash, obj in before_hashes.items():
            if counts[object_hash] != 1 or len(after_by_hash.get(object_hash, [])) != 1:
                continue
            new = after_by_hash[object_hash][0]
            dr, dc = new.top_left[0] - obj.top_left[0], new.top_left[1] - obj.top_left[1]
            if (dr, dc) == (0, 0):
                continue
            moved_keys.add(obj.key)
            stats = self._movers.setdefault(object_hash, _RoleStats(color=obj.color, pixels=obj.pixels))
            same_direction = (dr > 0) - (dr < 0) == want[0] and (dc > 0) - (dc < 0) == want[1]
            if same_direction:
                stats.consistent += 1
            else:
                stats.inconsistent += 1
        after_keys = {o.key for o in after}
        for obj in before:
            if obj.key in moved_keys or obj.key not in after_keys:
                continue
            entry = self._static.setdefault(obj.key, {"present": 0, "obj": obj})
            entry["present"] += 1
            entry["obj"] = obj

        players = sorted(
            (
                h
                for h, s in self._movers.items()
                if s.consistent >= _ROLE_MIN_CONSISTENT
                and s.inconsistent * 5 <= s.consistent
                and s.consistent >= _ROLE_MIN_SHARE * self._arrow_changes
            ),
            key=lambda h: (-self._movers[h].consistent, h),
        )[:_ROLE_MAX_LISTED]
        players.sort()
        if not players:
            return []
        player_pixels = sum(self._movers[h].pixels for h in players)
        mover_keys = {o.key for o in before if o.hash in players} | {o.key for o in after if o.hash in players}
        never_moved = [
            entry["obj"]
            for key, entry in self._static.items()
            if entry["present"] >= _ROLE_MIN_PRESENCE
            and key not in mover_keys
            and player_pixels / 2 <= entry["obj"].pixels <= player_pixels * 2
        ]
        never_moved.sort(key=lambda o: (o.solid, abs(o.pixels - player_pixels), o.key))
        listed = never_moved[:_ROLE_MAX_LISTED]
        state = (tuple(players), tuple(o.key for o in listed))
        if state == self._role_state:
            return []
        self._role_state = state
        moves = ", ".join(
            f"obj {h} ({self._movers[h].color}) moved with arrows {self._movers[h].consistent}/"
            f"{self._movers[h].consistent + self._movers[h].inconsistent}"
            for h in players
        )
        line = f"ROLES: likely player = {moves}"
        if listed:
            line += "; NOT the player (similar size, never moved with arrows): " + ", ".join(o.label() for o in listed)
        return [line]


def replay_level(
    transitions: list[dict[str, Any]],
    mask: set[Cell],
    *,
    level: int | None,
    min_size: int,
    max_lines: int,
) -> list[list[str]]:
    """Perception lines for every transition of one level, in order."""
    tracker = PerceptionTracker(min_size=min_size, max_lines=max_lines)
    return [tracker.observe(item, mask, level=level) for item in transitions]


__all__ = [
    "DIHEDRAL",
    "PObject",
    "PerceptionTracker",
    "classify_change",
    "detect_transformations",
    "frame_objects",
    "normalize",
    "reduce_scale",
    "relation_between",
    "relation_state",
    "replay_level",
]
