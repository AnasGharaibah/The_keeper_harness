"""Keeper resource log: one JSON line per interval in the run artifacts."""
from __future__ import annotations

import json
import time

import pytest

from inference.agent import keeper_resources
from keeper_fakes import clear_keeper_env


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    clear_keeper_env(monkeypatch)


def test_a_sample_has_memory_cpu_games_gate_and_server_queue(tmp_path):
    logger = keeper_resources.ResourceLogger(tmp_path / "r.jsonl", 60.0, lambda: None, "http://127.0.0.1:9/v1")
    logger.turn_started("g1")
    logger.turn_started("g2")
    logger.turn_ended("g2")
    record = logger.sample()
    for key in ("rss_mb", "peak_rss_mb", "mem_limit_mb", "sys_mem_available_mb", "process_cpu_percent", "load_avg",
                "games_seen", "games_active_10min", "games_in_turn", "gate_free_slots", "gate_waiting",
                "server_queue", "keeper_disabled_games", "keeper_disabled_globally"):
        assert key in record
    assert record["rss_mb"] > 0 and record["peak_rss_mb"] >= record["rss_mb"]
    assert (record["games_seen"], record["games_in_turn"]) == (2, 1)
    assert record["server_queue"] is None  # nothing listening: recorded as unknown, not an error
    assert logger.metrics_url == "http://127.0.0.1:9/metrics"


def test_the_sampler_thread_appends_lines_and_there_is_one_per_process(tmp_path):
    path = tmp_path / "keeper_resources.jsonl"
    first = keeper_resources.ResourceLogger.start(path, interval=0.05, gate_getter=lambda: None, server_url="")
    second = keeper_resources.ResourceLogger.start(tmp_path / "other.jsonl", interval=0.05, gate_getter=lambda: None,
                                                   server_url="")
    assert first is second
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and (not path.exists() or len(path.read_text().splitlines()) < 2):
        time.sleep(0.05)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) >= 2 and all("rss_mb" in line for line in lines)
    assert not (tmp_path / "other.jsonl").exists()
