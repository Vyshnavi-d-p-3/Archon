"""AgentConfig / evaluation defaults."""

from __future__ import annotations

import pytest

from config.settings import AgentConfig, EvalConfig, _default_eval_models


def test_default_eval_model_count() -> None:
    m = _default_eval_models()
    assert len(m) == 9
    assert "meta-llama/Meta-Llama-3.1-8B-Instruct" in m


def test_from_env_uses_default_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARCHON_EVAL_MODELS", raising=False)
    cfg = AgentConfig.from_env()
    assert len(cfg.evaluation.models_to_compare) == 9
    assert cfg.evaluation.models_to_compare[0] == "meta-llama/Meta-Llama-3-8B-Instruct"


def test_from_env_archon_eval_models_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "ARCHON_EVAL_MODELS",
        " org/model-a ,  org/model-b ",
    )
    try:
        cfg = AgentConfig.from_env()
    finally:
        monkeypatch.delenv("ARCHON_EVAL_MODELS", raising=False)
    assert cfg.evaluation.models_to_compare == ["org/model-a", "org/model-b"]


def test_evalconfig_empty_list_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whitespace-only ARCHON_EVAL_MODELS is treated as empty → full default list."""
    monkeypatch.setenv("ARCHON_EVAL_MODELS", "  ,  , ")
    try:
        cfg = AgentConfig.from_env()
    finally:
        monkeypatch.delenv("ARCHON_EVAL_MODELS", raising=False)
    assert len(cfg.evaluation.models_to_compare) == 9


def test_default_evalconfig_matches_helper() -> None:
    assert EvalConfig().models_to_compare == _default_eval_models()
