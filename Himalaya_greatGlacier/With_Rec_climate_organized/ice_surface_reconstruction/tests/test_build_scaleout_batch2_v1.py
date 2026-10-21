from __future__ import annotations

import pandas as pd

from ice_surface_reconstruction.drivers.build_scaleout_batch2_v1 import (
    _select_region,
)


def _candidate(index: int) -> dict:
    return {
        "main_rgi_id": f"RGI60-15.{index:05d}",
        "lia_area_km2": float(index + 1),
        "map_dx_m": float(50 + index),
        "declared_member_count": index % 5 + 1,
        "lia_component_count": 1,
        "lia_source_invalid": False,
        "lia_polygon_parts": 1,
        "geometry_repair_flag": False,
        "main_gdir_polygon_coverage_fraction": 1.0,
        "main_gdir_outside_area_km2": 0.0,
    }


def test_select_region_is_unique_deterministic_and_covers_special_risks():
    rows = [_candidate(index) for index in range(14)]
    rows[3].update(lia_component_count=2, lia_polygon_parts=2)
    rows[4].update(
        lia_source_invalid=True,
        derived_geometry_repair="lia-make-valid",
        geometry_repair_flag=True,
    )
    rows[5].update(
        main_gdir_polygon_coverage_fraction=0.91,
        main_gdir_outside_area_km2=4.0,
    )
    frame = pd.DataFrame(rows)

    first = _select_region(frame)
    second = _select_region(frame.sample(frac=1.0, random_state=5))

    assert len(first) == 10
    assert first["main_rgi_id"].is_unique
    assert first["main_rgi_id"].tolist() == second["main_rgi_id"].tolist()
    by_slot = first.set_index("batch_selection_slot")
    assert by_slot.loc["multipolygon-risk", "lia_component_count"] == 2
    assert bool(by_slot.loc["geometry-repair-risk", "geometry_repair_flag"])
    assert (
        by_slot.loc[
            "lowest-main-gdir-coverage", "main_gdir_polygon_coverage_fraction"
        ]
        < 1.0
    )


def test_select_region_records_special_slot_fallbacks():
    frame = pd.DataFrame([_candidate(index) for index in range(12)])
    selected = _select_region(frame)

    statuses = selected.set_index("batch_selection_slot")["selection_criterion_status"]
    assert statuses.loc["multipolygon-risk"] == "fallback-no-eligible-special-case"
    assert statuses.loc["geometry-repair-risk"] == "fallback-no-eligible-special-case"
