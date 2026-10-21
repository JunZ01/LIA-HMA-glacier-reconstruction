"""Select the next 12 unattempted polygon-valid synthetic-system cases."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    queue = pd.read_csv(
        root
        / "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/"
        "diagnostics/unresolved_master_router_v1/unresolved_master_queue.csv"
    )
    queue["lia_unique_id"] = pd.to_numeric(queue["lia_unique_id"]).astype(int)
    attempted: set[int] = set()
    config_root = (
        root
        / "Himalaya_greatGlacier/With_Rec_climate_organized/"
        "ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
    )
    for path in config_root.glob("geometry_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            attempted.add(int(payload["system_id"]))
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    frozen = pd.read_csv(
        root
        / "validation_data/results/paleo_surface_validation_v0/"
        "lia_191_frozen_curve_registry_v1/frozen_candidates.csv"
    )
    excluded = attempted | set(pd.to_numeric(frozen["lia_unique_id"]).astype(int)) | {843, 1044}

    outlines = gpd.read_file(root / "Himalaya_greatGlacier/Him_10_lia_with_large.shp")
    outlines["UniqueID"] = pd.to_numeric(outlines["UniqueID"]).astype(int)
    valid_ids: set[int] = set()
    for uid, group in outlines.groupby("UniqueID"):
        if len(group) != 1:
            continue
        geom = group.iloc[0].geometry
        if geom is not None and geom.geom_type == "Polygon" and geom.is_valid and not geom.is_empty:
            valid_ids.add(int(uid))

    candidates = queue.loc[
        ~queue["lia_unique_id"].isin(excluded)
        & queue["lia_unique_id"].isin(valid_ids)
        & (pd.to_numeric(queue["lia_target_m"]) > pd.to_numeric(queue["rgi_target_m"]))
    ].copy()
    candidates["region"] = candidates["rgi_id"].astype(str).str.split("-").str[1].str.split(".").str[0]
    candidates["objective_sep"] = pd.to_numeric(candidates["objective_sep"])
    candidates = candidates.sort_values(["region", "objective_sep", "lia_unique_id"])

    chosen = pd.concat(
        [group.head(4) for _, group in candidates.groupby("region", sort=True)],
        ignore_index=True,
    ).sort_values(["region", "objective_sep", "lia_unique_id"])
    if len(chosen) < 12:
        remaining = candidates.loc[~candidates["lia_unique_id"].isin(chosen["lia_unique_id"])]
        chosen = pd.concat(
            [chosen, remaining.sort_values(["objective_sep", "lia_unique_id"]).head(12 - len(chosen))],
            ignore_index=True,
        )
    chosen = chosen.sort_values(["region", "objective_sep", "lia_unique_id"])
    rows = chosen[
        ["lia_unique_id", "rgi_id", "region", "lia_target_m", "rgi_target_m", "objective_sep"]
    ].to_dict(orient="records")
    print(
        json.dumps(
            {
                "attempted_count": len(attempted),
                "frozen_baseline_count": len(frozen),
                "eligible_unattempted_count": len(candidates),
                "selected_count": len(rows),
                "selected": rows,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
