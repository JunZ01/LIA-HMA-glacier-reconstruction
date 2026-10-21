"""Explicit path contract for the independent ice-surface module.

The repository and shared project data trees remain separate. New code should
use anchor dictionaries in config; string paths are accepted only for legacy
compatibility with the original validation config.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    override = os.environ.get("LIA_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def project_root() -> Path:
    override = os.environ.get("LIA_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return repo_root().parents[1]


def validation_root() -> Path:
    override = os.environ.get("LIA_VALIDATION_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return project_root() / "validation_data"


def _anchors() -> dict[str, Path]:
    return {
        "repo": repo_root(),
        "project": project_root(),
        "validation": validation_root(),
    }


def resolve(spec: Any) -> Path:
    """Resolve an explicit anchor/subpath or a legacy config path.

    New format::

        {"anchor": "validation", "sub": "results/report"}

    Legacy relative strings are resolved against ``repo_root`` exactly as the
    previous drivers did; absolute strings remain absolute.
    """
    if isinstance(spec, dict):
        anchor = str(spec.get("anchor", ""))
        if anchor not in _anchors():
            raise ValueError(f"Unknown path anchor: {anchor!r}")
        sub = spec.get("sub", "")
        if not isinstance(sub, str):
            raise TypeError("Path spec 'sub' must be a string")
        candidate = Path(sub)
        if candidate.is_absolute():
            raise ValueError("Anchored path subpaths must be relative")
        return (_anchors()[anchor] / candidate).resolve()

    path = Path(spec).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root() / path).resolve()


__all__ = ["repo_root", "project_root", "validation_root", "resolve"]
