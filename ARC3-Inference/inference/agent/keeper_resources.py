"""Keeper resource log (KEEPER_RESOURCE_LOG): every 60 s one JSON line in
<artifacts>/keeper_resources.jsonl with this process's RSS and peak RSS, the
system's free RAM, CPU use and load, how many games are active, the priority
gate's queue, the model server's queue (from its /metrics), and which games
have the Keeper turned off. One sampler thread per process; never raises.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from inference.agent.keeper_guard import GLOBAL, memory_limit_bytes, meminfo, peak_rss_bytes, rss_bytes


class ResourceLogger:
    """One process-wide sampler thread; started on the first game's first turn."""

    _instance: "ResourceLogger | None" = None
    _lock = threading.Lock()

    def __init__(self, path: Path, interval: float, gate_getter: Callable[[], Any], server_url: str) -> None:
        self.path = path
        self.interval = interval
        self.gate_getter = gate_getter
        self.metrics_url = server_url.rstrip("/").removesuffix("/v1") + "/metrics" if server_url else ""
        self.started = time.monotonic()
        self.games: dict[str, float] = {}  # game -> last turn start (monotonic)
        self.in_turn: set[str] = set()
        self._cpu = (time.monotonic(), time.process_time())
        self._stop = threading.Event()

    @classmethod
    def start(cls, path: Path, *, interval: float, gate_getter: Callable[[], Any], server_url: str) -> "ResourceLogger":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(path, interval, gate_getter, server_url)
                threading.Thread(target=cls._instance._loop, name="keeper-resources", daemon=True).start()
            return cls._instance

    def turn_started(self, game: str) -> None:
        self.games[game] = time.monotonic()
        self.in_turn.add(game)

    def turn_ended(self, game: str) -> None:
        self.in_turn.discard(game)

    def _server_queue(self) -> dict[str, float] | None:
        if not self.metrics_url:
            return None
        try:
            with urllib.request.urlopen(self.metrics_url, timeout=2) as response:
                text = response.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return None
        wanted = {"sglang:num_queue_reqs": "waiting", "sglang:num_running_reqs": "running",
                  "vllm:num_requests_waiting": "waiting", "vllm:num_requests_running": "running"}
        out: dict[str, float] = {}
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            name = line.split("{", 1)[0].split(" ", 1)[0]
            if name in wanted:
                try:
                    out[wanted[name]] = out.get(wanted[name], 0.0) + float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
        return out or None

    def sample(self) -> dict[str, Any]:
        now = time.monotonic()
        cpu_now = time.process_time()
        wall0, cpu0 = self._cpu
        self._cpu = (now, cpu_now)
        info = meminfo()
        gate = self.gate_getter()
        record: dict[str, Any] = {
            "wall_clock": round(time.time(), 1),
            "elapsed_s": round(now - self.started, 1),
            "rss_mb": round(rss_bytes() / 2**20, 1),
            "peak_rss_mb": round(peak_rss_bytes() / 2**20, 1),
            "mem_limit_mb": round(memory_limit_bytes() / 2**20, 1),
            "sys_mem_available_mb": round(info.get("MemAvailable", 0) / 2**20, 1),
            "sys_mem_total_mb": round(info.get("MemTotal", 0) / 2**20, 1),
            "process_cpu_percent": round(100 * (cpu_now - cpu0) / max(1e-6, now - wall0), 1),
            "load_avg": [round(v, 2) for v in os.getloadavg()] if hasattr(os, "getloadavg") else None,
            "cpu_count": os.cpu_count(),
            "games_seen": len(self.games),
            "games_active_10min": sum(1 for t in self.games.values() if now - t < 600),
            "games_in_turn": len(self.in_turn),
            "keeper_disabled_games": sorted(GLOBAL.disabled_games),
            "keeper_disabled_globally": GLOBAL.disabled_reason,
            "gate_free_slots": getattr(gate, "_free", None),
            "gate_waiting": len(getattr(gate, "_waiting", []) or []) if gate is not None else None,
            "server_queue": self._server_queue(),
        }
        return record

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                record = self.sample()
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
            except Exception:  # noqa: BLE001 - logging must never break a run
                continue

    @classmethod
    def reset(cls) -> None:  # tests
        with cls._lock:
            if cls._instance is not None:
                cls._instance._stop.set()
            cls._instance = None
