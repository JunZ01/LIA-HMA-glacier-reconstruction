"""Checkpoint 3, part 1: export every modern member of all legacy systems.

One NPZ per modern member gdir, named fullchain_inputs_<uid>_<rgi_id>.npz,
on the system's saved LIA grid, with exactly the arrays and checks of
export_fullchain_inputs_batch_20261021.export_one. Multi-member systems are
therefore covered without a new exporter. The main member of each system is
taken from the production driver selection (config pilot_selection_path,
column main_rgi_id), so the LIA end and the modern end use the same tau owner.
A failing member is recorded, never dropped. Read-only; needs oggm_env2.

Usage (repo root, oggm_env2; same config as checkpoint 2):
    python export_fullchain_inputs_members_20261022.py --config CONFIG.json --out-dir OUT_DIR [--workers N]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(key, "1")

import numpy as np
import pandas as pd
import xarray as xr

import diagnose_extension_operators_20261021 as diag
import export_fullchain_inputs_batch_20261021 as batch
from select_pilot_systems_20261021 import MEMBERS, PAIRING, REGIONS


def population(config: dict) -> pd.DataFrame:
    from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.paths import resolve
    pairing = pd.read_csv(PAIRING)
    regions = pd.read_csv(REGIONS)[["lia_unique_id", "region", "region_zh"]]
    members = pd.read_csv(MEMBERS).drop_duplicates(["lia_unique_id", "rgi_id"])
    selection = pd.read_csv(resolve(config["pilot_selection_path"]))[["lia_unique_id", "main_rgi_id"]]
    systems = pairing.merge(regions, on="lia_unique_id", how="left", validate="one_to_one").merge(
        selection, on="lia_unique_id", how="left", validate="one_to_one")
    rows = members.merge(systems[["lia_unique_id", "region", "region_zh", "main_rgi_id", "modern_member_count",
                                  "lia_netcdf_path", "modern_reference_native_sum_km3"]],
                         on="lia_unique_id", how="inner", validate="many_to_one")
    rows["is_main"] = rows.rgi_id.astype(str) == rows.main_rgi_id.astype(str)
    rows["gdir"] = rows.source_path.map(lambda p: str(Path(p).parent))
    return systems, rows


def native_member_volume(gdir: Path) -> float:
    grid = json.loads((gdir / "glacier_grid.json").read_text(encoding="utf-8"))
    sx, sy = grid["dxdy"]
    with xr.open_dataset(gdir / "gridded_data.nc") as ds:
        h = np.asarray(ds[diag.THICK_VAR], float)
        mask = np.asarray(ds.glacier_mask, bool)
    return float(h[mask].sum() * abs(sx * sy) / 1e9)


def export_member(task):
    row, config, out_dir = task
    tau_table = pd.read_csv(batch.TAU)
    uid, rgi = int(row["lia_unique_id"]), str(row["rgi_id"])
    base = {"lia_unique_id": uid, "rgi_id": rgi, "is_main": bool(row["is_main"]),
            "region": row["region"], "modern_member_count": int(row["modern_member_count"])}
    try:
        native = native_member_volume(Path(row["gdir"]))
        case = {"lia_unique_id": uid, "region": row["region"], "rgi_id": rgi, "gdir": row["gdir"],
                "lia_netcdf": row["lia_netcdf_path"], "native_modern_reference_volume_km3": native}
        record = batch.export_one(case, config, tau_table, out_dir, file_stem=f"{uid}_{rgi}",
                                  require_tau=False, check_prior=int(row["modern_member_count"]) == 1)
        return {**base, "status": "exported", "file": record["file"], "native_volume_km3": native,
                "flowlines": record["flowlines"], "sha256": record["sha256"]}
    except Exception as error:  # recorded, never silently dropped
        return {**base, "status": "export-failed", "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    systems, members = population(config)
    no_main = systems[~systems.lia_unique_id.isin(members.loc[members.is_main, "lia_unique_id"])]
    tasks = [(r, config, args.out_dir) for r in members.to_dict("records")]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            records = list(pool.map(export_member, tasks))
    else:
        records = [export_member(t) for t in tasks]
    table = pd.DataFrame(records)
    table.drop(columns=["traceback"], errors="ignore").to_csv(args.out_dir / "members_manifest.csv", index=False)
    systems.to_csv(args.out_dir / "systems_population.csv", index=False)
    exported = table[table.status == "exported"]
    native_sum = exported.groupby("lia_unique_id").native_volume_km3.sum()
    check = systems.set_index("lia_unique_id").modern_reference_native_sum_km3
    diff = (native_sum - check.reindex(native_sum.index)).abs()
    summary = {"status": "completed", "systems": int(len(systems)), "members": int(len(table)),
               "exported": int(len(exported)), "failed": int((table.status != "exported").sum()),
               "systems_without_main_member_row": no_main.lia_unique_id.astype(int).tolist(),
               "systems_where_member_native_sum_differs_from_pairing_gt_1e-6_km3":
                   diff[diff > 1e-6].index.astype(int).tolist(),
               "failures": table.loc[table.status != "exported", ["lia_unique_id", "rgi_id", "error"]].to_dict("records"),
               "scope": "read-only export; no tau calibration, PP, LIA or dynamic solve"}
    (args.out_dir / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for f in (__file__, batch.__file__):
        shutil.copy2(f, args.out_dir / Path(f).name)
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
