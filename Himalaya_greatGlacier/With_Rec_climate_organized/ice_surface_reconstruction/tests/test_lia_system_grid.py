from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from rasterio.transform import Affine
from shapely.geometry import MultiPolygon, Polygon

from ice_surface_reconstruction.drivers.lia_system_grid import (
    INTERIOR_DISTANCE_POLICY,
    MAIN_MEMBER_FIRST_POLICY,
    SystemGrid,
    _load_source,
    _load_system_topography,
    _merge_checked,
    _merge_by_quality,
    _target_grid,
    mosaic_system_bed,
    normalize_overlap_resolution_policy,
    normalize_lia_components,
)
from ice_surface_reconstruction.drivers.mosaic_overlap_policy_sensitivity_v1 import (
    _comparison_table,
)
from ice_surface_reconstruction.drivers.lia_3d_system_grid_pilot_v1 import (
    make_component_masks_disjoint,
    should_exclude_small_component,
)
from ice_surface_reconstruction.drivers.system_grid_resolution_sensitivity_v1 import (
    _comparison_table as _resolution_comparison_table,
)


def test_normalize_lia_components_orders_and_preserves_disconnected_parts() -> None:
    geometry = MultiPolygon([
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(3, 0), (6, 0), (6, 1), (3, 1)]),
    ])
    parts = normalize_lia_components(geometry)
    assert len(parts) == 2
    assert parts[0].area == pytest.approx(3.0)
    assert parts[1].area == pytest.approx(1.0)


def test_merge_checked_keeps_first_source_and_reports_conflicts() -> None:
    current = np.asarray([[100.0, np.nan]])
    incoming = np.asarray([[120.0, 80.0]])
    merged, overlaps, maximum, conflicts = _merge_checked(
        current, incoming, tolerance_m=5.0, label="test"
    )
    np.testing.assert_allclose(merged, [[100.0, 80.0]])
    assert overlaps == 1
    assert maximum == pytest.approx(20.0)
    assert conflicts == 1


def test_merge_by_quality_selects_one_deeper_source_without_averaging() -> None:
    current = np.asarray([[100.0, 100.0]])
    incoming = np.asarray([[120.0, 120.0]])
    merged, quality, overlaps, maximum, conflicts = _merge_by_quality(
        current,
        np.asarray([[10.0, 30.0]]),
        incoming,
        np.asarray([[20.0, 5.0]]),
        tolerance_m=5.0,
    )
    np.testing.assert_allclose(merged, [[120.0, 100.0]])
    np.testing.assert_allclose(quality, [[20.0, 30.0]])
    assert overlaps == 2
    assert maximum == pytest.approx(20.0)
    assert conflicts == 2


def test_overlap_policy_names_are_strict_and_legacy_config_is_compatible() -> None:
    assert normalize_overlap_resolution_policy(None) == INTERIOR_DISTANCE_POLICY
    assert normalize_overlap_resolution_policy(
        "maximum-source-interior-distance-wins-no-average"
    ) == INTERIOR_DISTANCE_POLICY
    assert normalize_overlap_resolution_policy(
        MAIN_MEMBER_FIRST_POLICY
    ) == MAIN_MEMBER_FIRST_POLICY
    with pytest.raises(ValueError, match="unsupported-mosaic-overlap-policy"):
        normalize_overlap_resolution_policy("average-overlaps")


def test_policy_comparison_reports_signed_and_relative_changes() -> None:
    rows = []
    for policy, volume, mean, maximum in [
        (INTERIOR_DISTANCE_POLICY, 2.0, 100.0, 300.0),
        (MAIN_MEMBER_FIRST_POLICY, 2.01, 100.5, 295.0),
    ]:
        rows.append({
            "rgi_id": "RGI60-15.00001",
            "lia_unique_id": 1,
            "bed_scenario": "farinotti2019",
            "policy": policy,
            "status": "success",
            "volume_km3": volume,
            "mean_thickness_m": mean,
            "maximum_thickness_m": maximum,
        })
    comparison = _comparison_table(pd.DataFrame(rows))
    assert comparison.loc[0, "status_match"]
    assert comparison.loc[
        0, "main_member_first_minus_interior_distance_volume_km3"
    ] == pytest.approx(0.01)
    assert comparison.loc[0, "relative_change_volume_km3"] == pytest.approx(0.005)


def test_resolution_comparison_requires_factor2_factor4_convergence() -> None:
    rows = []
    for factor, status, volume, mean, maximum in [
        (1, "component-reconstruction-review", np.nan, np.nan, np.nan),
        (2, "success", 2.0, 100.0, 300.0),
        (4, "success", 2.004, 100.1, 303.0),
        (8, "success", 2.006, 100.2, 304.0),
    ]:
        rows.append({
            "rgi_id": "RGI60-15.00001",
            "lia_unique_id": 1,
            "bed_scenario": "farinotti2019",
            "resolution_factor": factor,
            "system_grid_dx_m": 100.0 / factor,
            "status": status,
            "failure_reason": "conflict" if factor == 1 else "",
            "volume_km3_finite_components": volume,
            "mean_thickness_m_finite_components": mean,
            "maximum_thickness_m_finite_components": maximum,
        })
    comparison = _resolution_comparison_table(pd.DataFrame(rows))
    assert comparison.loc[0, "first_success_factor"] == 2
    assert comparison.loc[0, "resolution_verdict"] == (
        "factor4-converged-against-factor8"
    )
    assert comparison.loc[
        0, "factor4_vs_factor2_volume_relative_fraction"
    ] == pytest.approx((2.004 - 2.0) / 2.004)


def test_resolution_comparison_does_not_claim_convergence_without_factor8() -> None:
    rows = pd.DataFrame([{
        "rgi_id": "RGI60-15.00001",
        "lia_unique_id": 1,
        "bed_scenario": "farinotti2019",
        "resolution_factor": factor,
        "system_grid_dx_m": 100.0 / factor,
        "status": "success" if factor == 4 else "component-reconstruction-review",
        "failure_reason": "conflict" if factor != 4 else "",
        "volume_km3_finite_components": 2.0 if factor == 4 else np.nan,
        "mean_thickness_m_finite_components": 100.0 if factor == 4 else np.nan,
        "maximum_thickness_m_finite_components": 300.0 if factor == 4 else np.nan,
    } for factor in (1, 2, 4, 8)])
    comparison = _resolution_comparison_table(rows)
    assert comparison.loc[0, "resolution_verdict"] == (
        "conflict-resolved-but-convergence-not-demonstrated"
    )


def test_target_grid_negative_dy_aligns_top_above_buffered_geometry(tmp_path) -> None:
    main_grid = {
        "proj": "EPSG:32645",
        "x0y0": [100.0, 1000.0],
        "dxdy": [100.0, -100.0],
    }
    geometry = Polygon([(150, 850), (250, 850), (250, 1150), (150, 1150)])
    target = _target_grid(
        main_grid,
        geometry,
        tmp_path / "system_grid.json",
        buffer_m=50.0,
    )
    assert target.y0 == pytest.approx(1200.0)
    assert target.y0 >= geometry.bounds[3] + 50.0
    assert (main_grid["x0y0"][1] - target.y0) / 100.0 == pytest.approx(-2.0)


def test_target_grid_refinement_nests_pixel_edges_and_covers_geometry(tmp_path) -> None:
    main_grid = {
        "proj": "EPSG:32645",
        "x0y0": [100.0, 1000.0],
        "dxdy": [100.0, -100.0],
    }
    geometry = Polygon([(151, 851), (249, 851), (249, 1149), (151, 1149)])
    coarse = _target_grid(
        main_grid, geometry, tmp_path / "coarse.json", buffer_m=50.0
    )
    fine = _target_grid(
        main_grid,
        geometry,
        tmp_path / "fine.json",
        buffer_m=50.0,
        resolution_factor=2,
    )
    assert fine.dx == pytest.approx(50.0)
    assert (fine.x0 - 25.0 - (main_grid["x0y0"][0] - 50.0)) % 100.0 == pytest.approx(0.0)
    assert (fine.y0 + 25.0 - (main_grid["x0y0"][1] + 50.0)) % 100.0 == pytest.approx(0.0)
    minx, miny, maxx, maxy = geometry.bounds
    assert fine.x0 - 0.5 * fine.dx <= minx - 50.0
    assert fine.x0 + (fine.nx - 0.5) * fine.dx >= maxx + 50.0
    assert fine.y0 + 0.5 * abs(fine.dy) >= maxy + 50.0
    assert fine.y0 - (fine.ny - 0.5) * abs(fine.dy) <= miny - 50.0
    payload = __import__("json").loads(fine.json_path.read_text())
    assert payload["dxdy"] == pytest.approx([50.0, -50.0])
    assert payload.keys() == {"proj", "x0y0", "nxny", "dxdy", "pixel_ref"}
    assert coarse.dx == pytest.approx(100.0)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_target_grid_rejects_invalid_resolution_factor(tmp_path, value) -> None:
    main_grid = {
        "proj": "EPSG:32645",
        "x0y0": [0.0, 0.0],
        "dxdy": [100.0, -100.0],
    }
    geometry = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    with pytest.raises(ValueError, match="resolution-factor"):
        _target_grid(
            main_grid,
            geometry,
            tmp_path / "invalid.json",
            buffer_m=0.0,
            resolution_factor=value,
        )


def test_small_component_exclusion_requires_size_fraction_and_typed_failure() -> None:
    policy = {"maximum_area_km2": 0.25, "maximum_system_area_fraction": 0.005}
    assert should_exclude_small_component(
        0.179, 65.65, "GeometryError: This centerline is too short", policy
    )
    assert not should_exclude_small_component(
        0.30, 65.65, "GeometryError: This centerline is too short", policy
    )
    assert not should_exclude_small_component(
        0.179, 20.0, "GeometryError: This centerline is too short", policy
    )
    assert not should_exclude_small_component(
        0.179, 65.65, "ValueError: no minimum-cost path", policy
    )
    assert should_exclude_small_component(
        0.179, 65.65, "all component constraints were excluded", policy
    )


def test_component_masks_are_disjoint_with_first_component_precedence() -> None:
    first = np.asarray([[True, True, False]])
    second = np.asarray([[False, True, True]])
    output = make_component_masks_disjoint([first, second])
    np.testing.assert_array_equal(output[0], [[True, True, False]])
    np.testing.assert_array_equal(output[1], [[False, False, True]])
    assert not np.logical_and(output[0], output[1]).any()


def test_load_source_rejects_thickness_nodata_inside_modern_mask(tmp_path) -> None:
    (tmp_path / "glacier_grid.json").write_text(
        '{"proj":"EPSG:32645","x0y0":[0,10],"nxny":[2,2],'
        '"dxdy":[10,-10],"pixel_ref":"center"}'
    )
    xr.Dataset({
        "topo": (("y", "x"), np.asarray([[100.0, 101.0], [102.0, 103.0]])),
        "glacier_mask": (("y", "x"), np.asarray([[1, 0], [0, 0]], dtype=np.int8)),
        "distributed_thickness_f19_fs0_full_v0": (
            ("y", "x"), np.asarray([[np.nan, np.nan], [np.nan, np.nan]])
        ),
    }).to_netcdf(tmp_path / "gridded_data.nc")
    with pytest.raises(ValueError, match="modern-mask-thickness-nodata"):
        _load_source(tmp_path, "farinotti2019")


def _small_system_grid(tmp_path) -> SystemGrid:
    return SystemGrid(
        crs="EPSG:32645",
        x0=5.0,
        y0=15.0,
        nx=2,
        ny=2,
        dx=10.0,
        dy=-10.0,
        json_path=tmp_path / "system_grid.json",
    )


def test_load_system_topography_streams_and_preserves_nodata(tmp_path) -> None:
    path = tmp_path / "regional_dem.tif"
    values = np.asarray([[200.0, -9999.0], [202.0, 203.0]], dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:32645",
        transform=Affine(10.0, 0.0, 0.0, 0.0, -10.0, 20.0),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(values, 1)
    topography, metadata = _load_system_topography(
        path, _small_system_grid(tmp_path)
    )
    assert topography[0, 0] == pytest.approx(200.0)
    assert np.isnan(topography[0, 1])
    assert metadata["topography_source_policy"] == (
        "regional-dem-only-no-gdir-fallback"
    )
    assert metadata["topography_target_finite_cell_count"] == 3


def test_system_dem_controls_exposed_bed_and_gdir_controls_masked_bed(
    tmp_path,
) -> None:
    gdir = tmp_path / "gdir"
    gdir.mkdir()
    (gdir / "glacier_grid.json").write_text(
        '{"proj":"EPSG:32645","x0y0":[5,15],"nxny":[2,2],'
        '"dxdy":[10,-10],"pixel_ref":"center"}'
    )
    xr.Dataset({
        "topo": (("y", "x"), np.asarray([[100.0, 101.0], [102.0, 103.0]])),
        "glacier_mask": (
            ("y", "x"), np.asarray([[1, 0], [0, 0]], dtype=np.int8)
        ),
        "distributed_thickness_f19_fs0_full_v0": (
            ("y", "x"), np.asarray([[10.0, 0.0], [0.0, 0.0]])
        ),
    }).to_netcdf(gdir / "gridded_data.nc")
    regional = np.asarray([[200.0, 201.0], [202.0, 203.0]])
    bed, topography, mask, metadata = mosaic_system_bed(
        [gdir],
        _small_system_grid(tmp_path),
        "farinotti2019",
        system_topography_m=regional,
    )
    np.testing.assert_allclose(topography, regional)
    np.testing.assert_allclose(bed, [[90.0, 201.0], [202.0, 203.0]])
    np.testing.assert_array_equal(mask, [[True, False], [False, False]])
    assert metadata["gdir_minus_system_dem_median_m"] == pytest.approx(-100.0)
    assert metadata["gdir_minus_system_dem_rmse_m"] == pytest.approx(100.0)

    regional_with_hole = regional.copy()
    regional_with_hole[1, 1] = np.nan
    bed_with_hole, _, _, _ = mosaic_system_bed(
        [gdir],
        _small_system_grid(tmp_path),
        "farinotti2019",
        system_topography_m=regional_with_hole,
    )
    assert np.isnan(bed_with_hole[1, 1])
