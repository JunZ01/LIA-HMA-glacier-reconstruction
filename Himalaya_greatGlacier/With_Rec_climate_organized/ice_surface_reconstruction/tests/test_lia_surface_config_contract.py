"""Path-contract checks for production and generated scale-out configs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ice_surface_reconstruction.paths import repo_root, resolve, validation_root


EXPECTED_SURFACE_DEM = Path(
    "D:/01 文件整理/04 数据/HMA_newregion/DEM_clip/Himalaya_3regions_DEM.tif"
)
CONFIG_PATHS = [
    repo_root() / "ice_surface_reconstruction/config/lia_surface_production_v0.json",
    validation_root()
    / "results/paleo_surface_validation_v0/scaleout_batch1_v1/lia_surface_scaleout_batch1_v1.json",
    validation_root()
    / "results/paleo_surface_validation_v0/scaleout_batch2_v1/lia_surface_scaleout_batch2_v1.json",
]


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda path: path.stem)
def test_surface_dem_path_is_explicit_and_consistent(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert Path(config["surface_dem_path"]) == EXPECTED_SURFACE_DEM


def test_production_grid_and_connector_policies_are_explicit() -> None:
    config = json.loads(CONFIG_PATHS[0].read_text(encoding="utf-8"))
    assert config["lia_system_grid"]["resolution_factor"] == 4
    assert config["production_connector"]["avoid_existing_network_cells"] is True
    assert config["lia_3d_system_grid_pilot_v1"]["harmonic_method"] == (
        "conjugate-gradient"
    )
    assert config["lia_3d_system_grid_pilot_v1"]["cg_relative_tolerance"] == pytest.approx(
        1e-10
    )


def test_production_selection_is_core_191() -> None:
    config = json.loads(CONFIG_PATHS[0].read_text(encoding="utf-8"))
    selection_path = resolve(config["pilot_selection_path"])
    selection = pd.read_csv(selection_path)
    assert len(selection) == 191
    assert selection["main_rgi_id"].nunique() == 191
    assert set(selection["main_rgi_id"].astype(str))
    assert selection_path.name == "core191_selection_v1.csv"


def test_production_tau_calibration_is_core_191() -> None:
    config = json.loads(CONFIG_PATHS[0].read_text(encoding="utf-8"))
    calibration_path = (
        resolve(config["output_dir"])
        / config["modern_tau_calibration"]["output_subdir"]
        / "modern_tau_calibration_v0.csv"
    )
    calibration = pd.read_csv(calibration_path)
    assert len(calibration) == 382
    assert calibration["rgi_id"].nunique() == 191
    assert calibration["tau_kpa"].notna().sum() == 377
