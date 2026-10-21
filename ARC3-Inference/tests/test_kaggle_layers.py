"""The Kaggle notebook setup passes the launcher's layer switches through."""
from __future__ import annotations

import ast

from inference.framework.kaggle import duck_kaggle_setup_command


def _script() -> str:
    return duck_kaggle_setup_command().split("<<'PYSETUP'\n", 1)[1].rsplit("\nPYSETUP", 1)[0]


def test_layer_env_is_embedded(monkeypatch):
    monkeypatch.setenv("LOCAL_ANALYZER_PREDICTION_CHECK_ENABLED", "true")
    monkeypatch.setenv("LOCAL_ANALYZER_LEVEL_REVIEW_LLM_TIMEOUT_SECONDS", "300")
    monkeypatch.setenv("LOCAL_ANALYZER_WORLD_MODEL_ENABLED", "true")
    monkeypatch.setenv("LOCAL_ANALYZER_LEVEL_REVIEW_MAX_HYPOTHESES", "")
    monkeypatch.setenv("UNRELATED_SECRET", "nope")
    script = _script()
    ast.parse(script)
    assert "'LOCAL_ANALYZER_PREDICTION_CHECK_ENABLED': 'true'" in script
    assert "'LOCAL_ANALYZER_LEVEL_REVIEW_LLM_TIMEOUT_SECONDS': '300'" in script
    assert "'LOCAL_ANALYZER_WORLD_MODEL_ENABLED': 'true'" in script
    assert "MAX_HYPOTHESES" not in script  # empty values are not forwarded
    assert "UNRELATED_SECRET" not in script


def test_no_layer_env_keeps_layers_off(monkeypatch):
    import os

    for key in list(os.environ):
        if key.startswith(("LOCAL_ANALYZER_PREDICTION_CHECK_", "LOCAL_ANALYZER_LEVEL_REVIEW_", "LOCAL_ANALYZER_WORLD_MODEL_")):
            monkeypatch.delenv(key)
    script = _script()
    ast.parse(script)
    assert "PREDICTION_CHECK" not in script and "LEVEL_REVIEW" not in script
