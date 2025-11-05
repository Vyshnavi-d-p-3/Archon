"""Tests for dashboard readiness checks."""

from __future__ import annotations

from pathlib import Path

from scripts.serve_dashboard import _readiness_check


def test_readiness_passes_for_writable_tmp(tmp_path: Path) -> None:
    result = _readiness_check(tmp_path)
    assert result["ready"] is True
    assert result["traces_dir_exists"] is True
    assert result["traces_dir_writable"] is True
    assert result["rag_store_writable"] is True
