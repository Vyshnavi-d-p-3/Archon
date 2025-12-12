"""Single source of truth for the installed package version (CLI, traces, run manifests)."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

# Bumped when `AgentTrace` JSON (or dashboard normalization) changes incompatibly.
TRACE_SCHEMA_VERSION = "1.1"


def package_version() -> str:
    """Installed package version from importlib, or a dev fallback if not in site-packages."""
    try:
        return version("archon")
    except PackageNotFoundError:
        return "0.0.0+dev"
