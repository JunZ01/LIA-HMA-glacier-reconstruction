"""Config field-equivalence test: canonical vs legacy paleo-surface config.

The two copies use different formats (canonical `{anchor,sub}`, legacy strings).
This guards against the silent-fallback bug where a driver reading the canonical
config sees no `velocity_mosaic_dir` and quietly degrades to legality-only. It
checks that both resolve to equivalent key fields, rather than relying on manual
sync.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ice_surface_reconstruction.paths import resolve  # noqa: E402


CANONICAL = REPO / "ice_surface_reconstruction/config/paleo_surface_validation_v0.json"
LEGACY = REPO / "config/paleo_surface_validation_v0.json"


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def test_both_configs_present() -> None:
    assert CANONICAL.exists() and LEGACY.exists()


def test_velocity_and_calibration_present_in_canonical() -> None:
    canon = _load(CANONICAL)
    # The whole point: canonical must NOT be missing velocity, or drivers silently
    # fall back to no-velocity mode.
    assert "velocity_mosaic_dir" in canon
    assert "connector_calibration" in canon


def test_resolved_paths_equivalent() -> None:
    canon, legacy = _load(CANONICAL), _load(LEGACY)
    for key in ["velocity_mosaic_dir", "output_dir", "stage0_gdir_root",
                "pilot_selection_path"]:
        assert resolve(canon[key]) == resolve(legacy[key]), f"{key} mismatch"


def test_calibration_grid_and_bed_scenarios_equivalent() -> None:
    canon, legacy = _load(CANONICAL), _load(LEGACY)
    assert canon["connector_calibration"] == legacy["connector_calibration"]
    assert canon["bed_scenarios"] == legacy["bed_scenarios"]
