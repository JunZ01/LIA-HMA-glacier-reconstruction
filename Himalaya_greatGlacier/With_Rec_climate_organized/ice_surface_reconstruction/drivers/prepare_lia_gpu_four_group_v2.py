"""Compile fixed inputs and group 1 without importing a dynamics backend."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from lia_gpu_four_group_search_v2 import ROOT, DESIGN, load_search, file_hash, digest
from lia_gpu_matrix_io_v2 import atomic_json

HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_gpu_four_group_implementation_v2"
SOURCES = ("lia_gpu_four_group_search_v2.py", "lia_gpu_matrix_io_v2.py",
           "lia_gpu_matrix_scoring_v2.py", "torch_flowline_1d_matrix_v2.py",
           "run_lia_gpu_four_group_v2.py", "prepare_lia_gpu_four_group_v2.py",
           "torch_flowline_1d_batched_semiimplicit_v1.py",
           "torch_flowline_1d_batched_semiimplicit_annual_mb_v2.py",
           "torch_flowline_1d_batched_semiimplicit_oggm_calendar_v3.py",
           "torch_flowline_1d_batched_spinup_repeat_v1.py",
           "score_lia_all191_forward_essential_standard_v3.py",
           "run_lia_191_gpu_spinup_v10_239.py", "launch_lia_191_gpu_spinup_v10_239.py")

CLARIFICATIONS = {
    "parameter_grid": "unchanged v10.262: 21 / 11 / 9 paired / 10; beam 10",
    "spinup_parent_ranking": ["converged", "domain valid", "LIA relative error", "90y volume drift", "90y length drift", "fingerprint"],
    "forward_parent_ranking": "frozen lexicographic hard-gate/error/shape ordering",
    "no_converged_state": "skip forward; carry up to ten scientific parameter failures to group 3, then 4; fresh initial geometry for new spinups",
    "nonconverged_state_never_forwarded": True,
    "technical_failures": "not science failures; unrecovered cells leave system technical-incomplete",
    "spinup_identity": "system, input/implementation hashes, delta, melt, prcp, glen; excludes forward correction and search group",
    "forward_identity": "spinup identity plus committed state SHA and forward parameters",
    "candidate_pass": "preserve all passes in current group, skip later groups for solved systems; reproduction remains required",
    "duplicate_multiplier_baselines": "reuse exact matching committed result, including recorded scientific failure",
    "forward_bias": "base_temp_bias + forward_end*(year-1470)/530; no delta_t_eq",
    "spinup_forward_temperature_switch": "intentional initial-condition experiment; record bias jump -delta_t_eq, not climate reconstruction evidence",
    "lag_definition": "existing v3 volume-change-equivalent MB proxy; best lag exactly 100 remains boundary-fail",
    "growth_targets": "LIA gate remains the frozen global peak; no undisclosed switch to a minimum",
    "checkpoints": "spinup boundary confirmations 330/630/930/1230/1500; forward every 30 years and end; restart cohort grouped by model year",
    "preflight": "one technical 24-lane 30y run with year-15 restore and injected status-write failure; one-year timing at 24/128/512 lanes in the same preflight; no scientific success requirement",
    "launch_authorized": False,
}


def prepare(output):
    import numpy as np
    import pandas as pd
    import xarray as xr
    from build_lia_191_gpu_spinup_fixture_v1 import FIXTURE_ROOT, PIXEL_TABLE, RECONSTRUCTION
    search = load_search("static-preparation")
    old = json.loads((FIXTURE_ROOT / "manifest.json").read_text())
    if file_hash(ROOT / old["npz_path"]) != old["npz_sha256"]:
        raise ValueError("static geometry/climate fixture hash mismatch")
    if file_hash(RECONSTRUCTION) != old["reconstruction_climate_sha256"]:
        raise ValueError("reconstruction climate hash mismatch")
    hashes = json.loads((ROOT / old["input_hashes_path"]).read_text())
    if file_hash(ROOT / old["input_hashes_path"]) != old["input_hashes_sha256"]:
        raise ValueError("input provenance hash mismatch")
    with np.load(ROOT / old["npz_path"], allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files if k.startswith("system_") or k == "climate_years"}
    systems = old["systems"]
    ids = [int(s["lia_unique_id"]) for s in systems]
    if set(ids) != set(search.baselines) or len(ids) != 191:
        raise ValueError("static fixture population mismatch")
    pixels = pd.read_csv(PIXEL_TABLE)
    years = np.arange(1470, 2001)
    temp = np.zeros((191, len(years), 12))
    prcp = np.zeros_like(temp)
    with xr.open_dataset(RECONSTRUCTION, use_cftime=True) as rec:
        for index, system in enumerate(systems):
            sid = int(system["lia_unique_id"])
            rows = pixels[(pixels.lia_unique_id == sid) & (pixels.rgi_id == system["rgi_id"]) &
                          (pixels.climate_id == "ensemble_rec_ref")]
            if len(rows) != 1:
                raise ValueError("nonunique climate pixel")
            row = rows.iloc[0]
            series = rec.temp.isel(lat=int(row.climate_lat_index), lon=int(row.climate_lon_index))
            ref = series.sel(time=slice("1961-01-01", "1990-12-31")).groupby("time.month").mean("time")
            period = series.sel(time=slice("1470-01-01", "2000-12-31"))
            expected_years = np.repeat(years, 12)
            if not np.array_equal(period.time.dt.year.values, expected_years) or not np.array_equal(
                    period.time.dt.month.values, np.tile(np.arange(1, 13), len(years))):
                raise ValueError("historical climate calendar gap")
            path = Path(system["historical_climate_path"])
            if file_hash(path) != system["historical_climate_sha256"]:
                raise ValueError("W5E5 baseline changed")
            with xr.open_dataset(path, use_cftime=True) as historical:
                base = historical.sel(time=slice("1961-01-01", "1990-12-31"))
                t = base.temp.groupby("time.month").mean("time").values
                p = base.prcp.groupby("time.month").mean("time").values
            temp[index] = t[None, :] + (period.groupby("time.month") - ref).values.reshape(len(years), 12)
            prcp[index] = p[None, :]
    if not np.isfinite(temp).all() or not np.isfinite(prcp).all() or (prcp < 0).any():
        raise ValueError("invalid compiled climate")
    arrays.update(forward_years=years, forward_climate_temp=temp, forward_climate_prcp=prcp,
                  forward_climate_gradient=np.full_like(temp, -.0065))
    targets_path = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/core191_length_references.csv"
    targets = pd.read_csv(targets_path).set_index("lia_unique_id")
    if not targets.index.is_unique or set(targets.index) != set(ids):
        raise ValueError("target population mismatch")
    manifest = {"schema": "lia-gpu-matrix-inputs-v2", "launch_authorized": False,
        "dynamics_performed": False, "source_files": {s: file_hash(HERE / s) for s in SOURCES},
        "design_summary_sha256": file_hash(DESIGN / "summary.json"),
        "old_static_fixture_sha256": old["npz_sha256"], "target_sha256": file_hash(targets_path),
        "climate_sha256": old["reconstruction_climate_sha256"], "clarifications": CLARIFICATIONS,
        "systems": [{"system": sid, "nx": int(arrays["system_nx"][i]),
                     "target_lia": float(targets.loc[sid, "lia_registered_length_m"]),
                     "target_rgi": float(targets.loc[sid, "rgi_registered_length_m"])} for i, sid in enumerate(ids)]}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "inputs.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    manifest["inputs_sha256"] = file_hash(output / "inputs.npz")
    manifest["contract"] = digest(manifest)
    search.contract = manifest["contract"]
    first = search.candidates(1, [])
    atomic_json(output / "group1_matrix.json", [{"candidate": asdict(c), "spin_key": search.spin_key(c)} for c in first])
    manifest["group1_matrix_sha256"] = file_hash(output / "group1_matrix.json")
    manifest["group1_unique_spinups"] = len({search.spin_key(c) for c in first})
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"status": "static-inputs-ready", "systems": len(ids), "group1_cells": len(first),
                      "dynamics_performed": False, "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    prepare(parser.parse_args().output)
