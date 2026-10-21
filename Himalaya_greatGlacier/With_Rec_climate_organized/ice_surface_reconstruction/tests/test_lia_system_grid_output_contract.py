from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ice_surface_reconstruction.drivers.lia_3d_system_grid_pilot_v1 import (
    _atomic_write_netcdf,
    _build_output_dataset,
    _build_run_manifest,
    is_publishable_member_status,
)
from ice_surface_reconstruction.drivers.lia_system_grid import SystemGrid


def _grid(tmp_path: Path) -> SystemGrid:
    return SystemGrid(
        crs="EPSG:32643",
        x0=500_000.0,
        y0=3_400_000.0,
        nx=3,
        ny=2,
        dx=100.0,
        dy=-100.0,
        json_path=tmp_path / "system_grid.json",
    )


def _dataset(tmp_path: Path) -> xr.Dataset:
    thickness = np.asarray([[0.0, 10.0, np.nan], [0.0, 5.0, np.nan]])
    bed = np.asarray([[4000.0, 3990.0, 3980.0], [3990.0, 3980.0, 3970.0]])
    valid = np.isfinite(thickness)
    return _build_output_dataset(
        grid=_grid(tmp_path),
        rgi_id="RGI60-15.00001",
        lia_unique_id=1,
        member_ids=["farinotti2019"],
        member_statuses=["success"],
        thickness_arrays=[thickness],
        surface_arrays=[bed + thickness],
        bed_arrays=[bed],
        valid_masks=[valid],
        lia_mask=valid,
        excluded_mask=np.zeros_like(valid),
        topography_metadata={
            "topography_source_policy": "regional-dem-only-no-gdir-fallback",
            "topography_source_path": "D:/regional_dem.tif",
            "topography_source_crs": "EPSG:4326",
        },
    )


def test_publishable_status_gate_is_explicit() -> None:
    assert is_publishable_member_status("success")
    assert is_publishable_member_status("success-with-documented-exclusion")
    assert not is_publishable_member_status("partial-component-review")
    assert not is_publishable_member_status("component-reconstruction-review")


def test_output_dataset_rejects_partial_member(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    thickness = dataset["thickness_m"].values[0]
    bed = dataset["bed_elevation_m"].values[0]
    with pytest.raises(ValueError, match="non-publishable member"):
        _build_output_dataset(
            grid=_grid(tmp_path),
            rgi_id="RGI60-15.00001",
            lia_unique_id=1,
            member_ids=["farinotti2019"],
            member_statuses=["partial-component-review"],
            thickness_arrays=[thickness],
            surface_arrays=[bed + thickness],
            bed_arrays=[bed],
            valid_masks=[np.isfinite(thickness)],
            lia_mask=np.isfinite(thickness),
            excluded_mask=np.zeros_like(thickness, dtype=bool),
        )


def test_output_dataset_has_cf_grid_and_member_validity(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    np.testing.assert_allclose(dataset.x.values, [500_000.0, 500_100.0, 500_200.0])
    np.testing.assert_allclose(dataset.y.values, [3_400_000.0, 3_399_900.0])
    assert dataset.x.attrs["standard_name"] == "projection_x_coordinate"
    assert dataset.y.attrs["standard_name"] == "projection_y_coordinate"
    assert dataset["member_status"].item() == "success"
    assert dataset["valid_mask"].dtype == np.int8
    assert dataset["thickness_m"].attrs["grid_mapping"] == "spatial_ref"
    assert "spatial_ref" in dataset["spatial_ref"].attrs
    assert dataset["spatial_ref"].attrs["GeoTransform"].split() == [
        "499950.0", "100.0", "0.0", "3400050.0", "0.0", "-100.0"
    ]
    assert dataset.attrs["topography_source_policy"] == (
        "regional-dem-only-no-gdir-fallback"
    )
    assert dataset.attrs["topography_source_path"] == "D:/regional_dem.tif"


def test_atomic_netcdf_roundtrip_preserves_contract(tmp_path: Path) -> None:
    output = tmp_path / "candidate.nc"
    _atomic_write_netcdf(_dataset(tmp_path), output)
    assert output.exists()
    assert list(tmp_path.glob(".candidate.*.tmp.nc")) == []
    with xr.open_dataset(output, engine="netcdf4") as reopened:
        assert reopened["member_status"].item() == "success"
        assert reopened["valid_mask"].shape == (1, 2, 3)
        assert reopened["spatial_ref"].attrs["GeoTransform"].endswith("-100.0")
        np.testing.assert_allclose(reopened.x.values, [500_000.0, 500_100.0, 500_200.0])


def test_run_manifest_never_publishes_partial_member() -> None:
    systems = pd.DataFrame([
        {
            "rgi_id": "RGI60-15.00001",
            "lia_unique_id": 1,
            "bed_scenario": "farinotti2019",
            "status": "success",
        },
        {
            "rgi_id": "RGI60-15.00001",
            "lia_unique_id": 1,
            "bed_scenario": "millan2022",
            "status": "partial-component-review",
        },
        {
            "rgi_id": "RGI60-15.00002",
            "lia_unique_id": 2,
            "bed_scenario": "farinotti2019",
            "status": "success-with-documented-exclusion",
        },
    ])
    manifest = _build_run_manifest(
        systems,
        {
            ("RGI60-15.00001", 1): "lia_1.nc",
            ("RGI60-15.00002", 2): "lia_2.nc",
        },
        {
            ("RGI60-15.00001", 1): {
                "topography_source_policy": "regional-dem-only-no-gdir-fallback",
                "topography_source_path": "D:/regional_dem.tif",
            },
            ("RGI60-15.00002", 2): {
                "topography_source_policy": "regional-dem-only-no-gdir-fallback",
                "topography_source_path": "D:/regional_dem.tif",
            },
        },
    )
    partial = manifest[manifest["bed_scenario"].eq("millan2022")].iloc[0]
    assert not partial["publishable"]
    assert not partial["published_to_netcdf"]
    assert partial["netcdf_file"] == ""
    published = manifest[manifest["published_to_netcdf"]]
    assert set(published["netcdf_file"]) == {"lia_1.nc", "lia_2.nc"}
    assert set(published["topography_source_policy"]) == {
        "regional-dem-only-no-gdir-fallback"
    }
