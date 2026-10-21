"""Tests for the module path anchors and resolver."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


MODULE_ROOT = Path(__file__).resolve().parents[2]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from ice_surface_reconstruction import paths  # noqa: E402


def test_default_repo_root_is_two_levels_above_module() -> None:
    expected = Path(paths.__file__).resolve().parents[1]
    assert paths.repo_root() == expected
    assert paths.repo_root().name == "With_Rec_climate_organized"


def test_default_project_and_validation_relationship() -> None:
    assert paths.project_root() == paths.repo_root().parent.parent
    assert paths.validation_root() == paths.project_root() / "validation_data"


def test_env_overrides_each_anchor(monkeypatch, tmp_path) -> None:
    repo = tmp_path / "repo"
    project = tmp_path / "proj"
    validation = tmp_path / "val"
    monkeypatch.setenv("LIA_REPO_ROOT", str(repo))
    monkeypatch.setenv("LIA_PROJECT_ROOT", str(project))
    monkeypatch.setenv("LIA_VALIDATION_ROOT", str(validation))
    assert paths.repo_root() == repo.resolve()
    assert paths.project_root() == project.resolve()
    assert paths.validation_root() == validation.resolve()


def test_validation_default_follows_project_override(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LIA_PROJECT_ROOT", str(tmp_path / "proj"))
    monkeypatch.delenv("LIA_VALIDATION_ROOT", raising=False)
    assert paths.validation_root() == (tmp_path / "proj").resolve() / "validation_data"


def test_resolve_new_format_each_anchor(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LIA_REPO_ROOT", str(tmp_path / "repo"))
    monkeypatch.setenv("LIA_VALIDATION_ROOT", str(tmp_path / "val"))
    assert paths.resolve({"anchor": "repo", "sub": "outputs/x.csv"}) == (
        tmp_path / "repo" / "outputs" / "x.csv"
    ).resolve()
    assert paths.resolve({"anchor": "validation", "sub": "public/g"}) == (
        tmp_path / "val" / "public" / "g"
    ).resolve()


def test_resolve_legacy_relative_matches_repo_root() -> None:
    spec = "outputs/run/thing.csv"
    assert paths.resolve(spec) == (paths.repo_root() / spec).resolve()


def test_resolve_legacy_absolute_returned_as_is(tmp_path) -> None:
    absolute = tmp_path / "a.csv"
    assert paths.resolve(str(absolute)) == absolute.resolve()


def test_resolve_rejects_unknown_anchor() -> None:
    with pytest.raises(ValueError, match="[Uu]nknown.*anchor"):
        paths.resolve({"anchor": "nope", "sub": "x"})


def test_resolve_rejects_absolute_sub(tmp_path) -> None:
    with pytest.raises(ValueError, match="must be relative"):
        paths.resolve({"anchor": "repo", "sub": str(tmp_path / "abs")})


def test_resolve_rejects_missing_anchor() -> None:
    with pytest.raises(ValueError, match="[Uu]nknown.*anchor"):
        paths.resolve({"sub": "x"})


def test_resolve_rejects_bad_type() -> None:
    with pytest.raises(TypeError):
        paths.resolve(123)
