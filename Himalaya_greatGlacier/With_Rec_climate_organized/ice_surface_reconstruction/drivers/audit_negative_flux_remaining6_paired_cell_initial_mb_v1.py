"""Read-only B-window mass-balance screening for six negative-flux holds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr
from oggm import cfg, utils
from oggm.core.massbalance import MonthlyTIModel

from .audit_branch_rigid_vertical_datum_hold3_v1 import sha256
from ..paths import resolve


PROJECT = Path(__file__).resolve().parents[4]
OUTPUT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_negative_flux_remaining6_paired_cell_initial_mb_v1"
)
ROOTS = {
    890: "w456inv890v1",
    988: "w427inv988v1",
    1026: "w337inv1026v1",
    1743: "w583inv1743v1",
    2689: "w586inv2689v1",
    2793: "w555inv2793v1",
}
MAX_ABSOLUTE_LINEAR_50Y_FRACTION = 0.10


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    rows = []
    for system_id, root_name in ROOTS.items():
        root = PROJECT / root_name
        config = json.loads((root / "config_snapshot.json").read_text(encoding="utf-8"))
        formal = json.loads((root / "inversion_summary_formal.json").read_text(encoding="utf-8"))
        if formal["status"] != "inversion-pass":
            raise RuntimeError(f"formal-inversion-not-pass:{system_id}")
        paired = resolve(config["paired_grid"])
        with xr.open_dataset(paired) as ds:
            h = np.asarray(ds.thickness_m.isel(member=0), dtype=float)
            s = np.asarray(ds.surface_elevation_m.isel(member=0), dtype=float)
            valid = np.asarray(ds.valid_mask.isel(member=0), dtype=bool)
            lia = np.asarray(ds.lia_mask, dtype=bool)
            x = np.asarray(ds.x, dtype=float)
            y = np.asarray(ds.y, dtype=float)
        positive = valid & lia & np.isfinite(h) & np.isfinite(s) & (h > 1e-6)
        cell_area = float(abs(np.median(np.diff(x))) * abs(np.median(np.diff(y))))
        volume = float(np.sum(h[positive]) * cell_area)
        gdir = utils.GlacierDirectory(config["synthetic_rgi_id"], base_dir=root / "synthetic_gdir", reset=False)
        stage2 = config["stage2"]
        cfg.PARAMS["temp_melt"] = float(stage2["temp_melt_c"])
        model = MonthlyTIModel(
            gdir,
            filename="gcm_data",
            input_filesuffix="_batch1_1441_1470_prcp_off",
            melt_f=float(stage2["melt_f"]),
            prcp_fac=float(stage2["prcp_fac"]),
            temp_bias=float(stage2["base_temp_bias_c"]) + float(stage2["delta_t_c"]),
            ys=1441,
            ye=1470,
            repeat=True,
            check_calib_params=False,
        )
        annual = []
        heights = s[positive]
        for year in range(30):
            mb = np.asarray(model.get_annual_mb(heights, year=float(year)), dtype=float)
            annual.append(float(np.sum(mb) * cfg.SEC_IN_YEAR * cell_area))
        annual = np.asarray(annual)
        mean = float(np.mean(annual))
        annual_fraction = mean / volume
        projected = 50.0 * annual_fraction
        climate_path = Path(gdir.get_filepath("gcm_data", filesuffix="_batch1_1441_1470_prcp_off"))
        rows.append(
            {
                "system_id": system_id,
                "inversion_root": root_name,
                "paired_positive_cell_count": int(np.count_nonzero(positive)),
                "paired_volume_m3": volume,
                "mean_initial_annual_mb_m3": mean,
                "mean_initial_annual_volume_fraction": annual_fraction,
                "linear_50y_volume_fraction": projected,
                "absolute_linear_50y_volume_fraction": abs(projected),
                "minimum_annual_mb_m3": float(np.min(annual)),
                "maximum_annual_mb_m3": float(np.max(annual)),
                "near_neutral_screen_pass": abs(projected) <= MAX_ABSOLUTE_LINEAR_50Y_FRACTION,
                "paired_grid_sha256": sha256(paired),
                "climate_sha256": sha256(climate_path),
            }
        )
    ranked = sorted(rows, key=lambda row: (row["absolute_linear_50y_volume_fraction"], row["system_id"]))
    eligible = [row for row in ranked if row["near_neutral_screen_pass"]]
    selected = eligible[0]["system_id"] if eligible else None
    OUTPUT.mkdir(parents=True)
    import pandas as pd

    pd.DataFrame(ranked).to_csv(OUTPUT / "systems.csv", index=False)
    summary = {
        "schema": "lia-191-negative-flux-remaining6-paired-cell-initial-mb-v1",
        "status": "representative-selected-static-not-run" if selected is not None else "no-near-neutral-subclass",
        "system_count": len(rows),
        "screen": "absolute 50 times mean initial annual B-window MB divided by paired initial volume <= 0.10",
        "maximum_absolute_linear_50y_fraction": MAX_ABSOLUTE_LINEAR_50Y_FRACTION,
        "eligible_system_ids": [row["system_id"] for row in eligible],
        "selected_representative": selected,
        "ranking": [{"system_id": row["system_id"], "linear_50y_volume_fraction": row["linear_50y_volume_fraction"]} for row in ranked],
        "dynamic_run": False,
        "historical_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
